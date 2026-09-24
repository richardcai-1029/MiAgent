"""多请求运行时：同时进来的多份请求怎样并发执行。

图的一次 invoke 处理一个请求。多个用户同时发起请求时，每个请求仍是一次
独立的 invoke，各有各的状态：一个请求中止、重规划，不影响别的请求。
运行时在图之上管三件事：

  · 会话内串行：同一会话的请求按提交先后逐个执行。下一轮要承接上一轮的
    结论（见 memory.session），两轮同时跑，后一轮就看不到前一轮。
  · 准入：同时在飞的请求数不超过 max_inflight，其余排队。
  · 优先级：排队的请求按 (优先级, 提交先后) 出队；同一个键也设进
    slots.request_key，请求在飞期间争用共享资源槽时按它排队。

跨请求共享的资源（MiClaw 并发配额、模型推理）由 slots.SlotPool 仲裁，
分别经 build_agent(miclaw_slots=...) 与 LLM.slots 接入，见各自说明。

请求在线程池里执行：沿用同步实现，不引入事件循环。

★ 本模块不 import langgraph：只要求 agent 有 invoke 方法。
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..graph.state import AgentState
from ..memory.session import Agent, Session
from .slots import request_key


@dataclass
class _Job:
    key: tuple[int, int]
    session: str
    request: str
    future: Future[AgentState] = field(default_factory=Future)


class Runtime:
    """接收多份请求并发执行。

    max_inflight 是同时在飞的请求上限，由调用方按设备内存等约束给定 ——
    每个在飞请求各持一份任务图与记忆，常驻内存随它线性增长。
    """

    def __init__(self, agent: Agent, max_inflight: int,
                 config: dict[str, Any] | None = None) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight 至少为 1，收到 {max_inflight}")
        self._agent = agent
        self._config = config
        self.max_inflight = max_inflight
        self._pool = ThreadPoolExecutor(max_workers=max_inflight,
                                        thread_name_prefix="miagent-request")
        self._cond = threading.Condition()
        self._seq = itertools.count()
        self._sessions: dict[str, Session] = {}
        self._queues: dict[str, deque[_Job]] = {}
        self._busy: set[str] = set()      # 有请求在飞的会话
        self._inflight = 0
        self._closed = False

    def submit(self, request: str, session: str = "default",
               priority: int = 0) -> Future[AgentState]:
        """提交一个请求，返回它的 Future，结果是收尾后的 AgentState。

        priority 越大越先执行，默认 0；同优先级按提交先后。
        排队期间可以 cancel()；已开始执行的请求不可取消。
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("运行时已关闭")
            job = _Job(key=(-priority, next(self._seq)), session=session,
                       request=request)
            self._queues.setdefault(session, deque()).append(job)
            self._admit()
            return job.future

    def session(self, name: str = "default") -> Session:
        """取某个会话，可查看它的轮次记录。"""
        with self._cond:
            return self._session_locked(name)

    def close(self) -> None:
        """不再接收新请求，等已提交的全部结束。"""
        with self._cond:
            self._closed = True
            self._cond.wait_for(lambda: not self._inflight and not self._queues)
        self._pool.shutdown()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------

    def _session_locked(self, name: str) -> Session:
        if name not in self._sessions:
            self._sessions[name] = Session(self._agent, self._config)
        return self._sessions[name]

    def _admit(self) -> None:
        """在配额内放行排队的请求。调用方须持有锁。

        候选只取各个空闲会话的队首：同一会话的后一个请求要等前一个结束。
        """
        while self._inflight < self.max_inflight:
            heads = [q[0] for s, q in self._queues.items() if s not in self._busy]
            if not heads:
                break
            job = min(heads, key=lambda j: j.key)
            queue = self._queues[job.session]
            queue.popleft()
            if not queue:
                del self._queues[job.session]
            if not job.future.set_running_or_notify_cancel():
                continue                  # 排队期间被取消
            self._busy.add(job.session)
            self._inflight += 1
            self._pool.submit(self._run, job, self._session_locked(job.session))
        self._cond.notify_all()

    def _run(self, job: _Job, session: Session) -> None:
        token = request_key.set(job.key)
        try:
            job.future.set_result(session.run(job.request))
        except BaseException as e:        # 结果交给 Future，线程本身不能死
            job.future.set_exception(e)
        finally:
            request_key.reset(token)
            with self._cond:
                self._busy.discard(job.session)
                self._inflight -= 1
                self._admit()
                self._cond.notify_all()
