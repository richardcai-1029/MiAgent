"""跨请求共享的资源槽：按优先级排队的计数信号量。

多个请求同时在飞时，它们争用的是同一份外部资源：MiClaw 的并发调用配额
是握手时按会话下发的，端侧推理通常只有一个实例。每个请求各自按配额派发，
合起来就会超出配额。资源槽把这类资源收成一处，所有请求从同一个池里取。

等待者按优先级键出队，而不是按到达先后：键小的先拿到槽，键相同的按排队先后。
键不由调用处传入，取自 request_key —— 由运行时在请求开始时设定。LangGraph
派发节点时会复制调用方的上下文，并行分支里读到的仍是发起请求的那个键，
因此节点只写 `with slots.hold():`，不需要知道自己属于哪个请求。

★ 使用约定：槽只在一次调用期间持有，持有期间不再申请别的槽。
  模型调用与工具调用分在不同节点里，这条天然成立；它保证了不会出现
  「持有一个、等另一个」的循环等待。

本模块只依赖标准库。
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# 请求的优先级键，越小越先。运行时之外直接调用图时取默认值，
# 此时所有调用同键，退化为按排队先后。
request_key: ContextVar[tuple[int, ...]] = ContextVar("request_key", default=())


class SlotPool:
    """容量为 capacity 的资源槽。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"容量至少为 1，收到 {capacity}")
        self.capacity = capacity
        self._free = capacity
        self._cond = threading.Condition()
        self._waiting: list[tuple[tuple[int, ...], int]] = []
        self._tickets = itertools.count()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """取一个槽，退出时归还。没有空槽时阻塞，按优先级键排队。"""
        ticket = (request_key.get(), next(self._tickets))
        with self._cond:
            heapq.heappush(self._waiting, ticket)
            while not (self._free and self._waiting[0] == ticket):
                self._cond.wait()
            heapq.heappop(self._waiting)
            self._free -= 1
            # 空槽可能不止一个：让下一位也检查一次
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._free += 1
                self._cond.notify_all()

    @property
    def waiting(self) -> int:
        """正在排队的调用数，供观测与测试。"""
        with self._cond:
            return len(self._waiting)
