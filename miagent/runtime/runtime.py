"""多请求运行时：同时进来的多份请求怎样并发执行。

图的一次 invoke 处理一个请求。多个用户同时发起请求时，每个请求仍是一次
独立的 invoke，各有各的状态：一个请求中止、重规划，不影响别的请求。
运行时在图之上管三件事：

  · 会话内串行：同一会话的请求按提交先后逐个执行。下一轮要承接上一轮的
    结论（见 memory.session），两轮同时跑，后一轮就看不到前一轮。
  · 准入：同时在飞的请求数不超过 max_inflight，其余排队。
  · 优先级：排队的请求按 (优先级, 提交先后) 出队；同一个键也设进
    slots.request_key，请求在飞期间争用共享资源槽时按它排队。
  · 会话回收：会话各持一份轮次记录，对话标识不断出现新值时常驻内存只增不减。
    收到对话结束通知即回收；另外空闲会话数超过 max_idle_sessions 时，
    回收最久没用的。在飞或有请求排队的会话不算空闲。被回收的会话再来请求
    时从空白开始 —— 规划器少了上文，不影响执行的正确性。

跨请求共享的资源（MiClaw 并发配额、模型推理）由 slots.SlotPool 仲裁，
分别经 build_agent(miclaw_slots=...) 与 LLM.slots 接入；build_runtime 把
这几处一并装好。

MiClaw 经 miclaw/task.dispatch 派来的请求由 dispatch 受理：对话标识即会话，
协议里的请求优先级映射为这里的优先级；miclaw/conversation.end 由 end 受理。
serve 把这两者登记到客户端上，报文从管道到运行时这一段就接通了。

请求在线程池里执行：沿用同步实现，不引入事件循环。

★ 本模块不 import langgraph：只要求 agent 有 invoke 方法。
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..graph.state import AgentState
from ..memory.session import Agent, Session
from ..protocol import (AgentMethod, ConversationEndParams, ErrorCode, MiClawError,
                        RequestPriority, ResourceBudget, TaskDispatchParams,
                        TaskDispatchResult)
from .slots import SlotPool, request_key

if TYPE_CHECKING:
    from ..client import MiClawClient
    from ..llm import LLM
    from ..tools import ToolRegistry

# 同时在飞的请求上限。暂定值，由项目给定，尚无设备内存实测作依据。
MAX_INFLIGHT = 10

# 空闲会话数上限。暂定值：尚无每个会话常驻内存的实测，取与 MAX_INFLIGHT 相同，
# 使空闲会话的常驻量与在飞请求保持同一量级。实测后按设备内存校准。
MAX_IDLE_SESSIONS = MAX_INFLIGHT

# 推理槽容量。端侧推理是单个模型实例；槽数超过推理栈实际能并行服务的数目时，
# 多出的调用会在推理栈内部按到达先后排队，优先级在那里不再起作用。
# 1 对任何推理栈都成立。推理栈确实能并行服务多路时，按其实际能力调大。
INFERENCE_SLOTS = 1

_P = TypeVar("_P", bound=BaseModel)

# 协议优先级到运行时优先级的映射：前台先于后台。
PRIORITY_RANK = {RequestPriority.FOREGROUND: 1, RequestPriority.BACKGROUND: 0}


@dataclass
class _Job:
    key: tuple[int, int]
    session: str
    # 提交时就绑定会话对象：对话结束后同名的新请求拿到的是新会话，
    # 结束前已提交的请求仍在原会话上执行。
    conversation: Session
    request: str
    future: Future[AgentState] = field(default_factory=Future)


class Runtime:
    """接收多份请求并发执行。

    max_inflight 是同时在飞的请求上限 —— 每个在飞请求各持一份任务图与记忆，
    常驻内存随它线性增长。max_idle_sessions 是保留的空闲会话数上限。
    """

    def __init__(self, agent: Agent, max_inflight: int = MAX_INFLIGHT,
                 config: dict[str, Any] | None = None,
                 max_idle_sessions: int = MAX_IDLE_SESSIONS) -> None:
        if max_inflight < 1:
            raise ValueError(f"max_inflight 至少为 1，收到 {max_inflight}")
        if max_idle_sessions < 0:
            raise ValueError(f"max_idle_sessions 不能为负，收到 {max_idle_sessions}")
        self.max_idle_sessions = max_idle_sessions
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
        # 空闲会话，按最近一次用完的先后排列，最久没用的在最前
        self._idle: OrderedDict[str, None] = OrderedDict()
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
            if session not in self._sessions:
                self._sessions[session] = Session(self._agent, self._config)
            self._idle.pop(session, None)
            job = _Job(key=(-priority, next(self._seq)), session=session,
                       conversation=self._sessions[session], request=request)
            self._queues.setdefault(session, deque()).append(job)
            self._admit()
            return job.future

    def dispatch(self, params: TaskDispatchParams) -> Future[AgentState]:
        """受理一个 miclaw/task.dispatch 请求。结果用 dispatch_result 转成响应。"""
        return self.submit(params.request, session=params.conversationId,
                           priority=PRIORITY_RANK[params.priority])

    def end(self, session: str) -> None:
        """对话结束，回收它的会话。

        已提交的请求照常在原会话上执行完；此后同名的请求从一个新会话开始。
        """
        with self._cond:
            self._sessions.pop(session, None)
            self._idle.pop(session, None)

    def end_conversation(self, params: ConversationEndParams) -> None:
        """受理一个 miclaw/conversation.end 通知。"""
        self.end(params.conversationId)

    def session(self, name: str = "default") -> Session | None:
        """取某个会话，可查看它的轮次记录。不存在或已回收时为 None。"""
        with self._cond:
            return self._sessions.get(name)

    @property
    def sessions(self) -> int:
        """保留中的会话数，含在飞与空闲的。"""
        with self._cond:
            return len(self._sessions)

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
                self._release(job)        # 排队期间被取消
                continue
            self._busy.add(job.session)
            self._inflight += 1
            self._pool.submit(self._run, job)
        self._cond.notify_all()

    def _release(self, job: _Job) -> None:
        """一个请求离开之后，它的会话若已空闲，记入空闲队列并按上限回收。
        调用方须持有锁。"""
        name = job.session
        if (name in self._busy or name in self._queues
                or self._sessions.get(name) is not job.conversation):
            return                        # 仍有请求，或已结束、已换成新会话
        self._idle[name] = None
        self._idle.move_to_end(name)
        while len(self._idle) > self.max_idle_sessions:
            oldest, _ = self._idle.popitem(last=False)
            del self._sessions[oldest]

    def _run(self, job: _Job) -> None:
        token = request_key.set(job.key)
        out: AgentState | None = None
        err: BaseException | None = None
        try:
            out = job.conversation.run(job.request)
        except BaseException as e:        # 结果交给 Future，线程本身不能死
            err = e
        finally:
            request_key.reset(token)
        # 先登记再交付：调用方拿到结果时，会话的在飞、空闲与回收状态已经落定。
        with self._cond:
            self._busy.discard(job.session)
            self._inflight -= 1
            self._admit()
            self._release(job)
            self._cond.notify_all()
        if err is None:
            job.future.set_result(out)
        else:
            job.future.set_exception(err)


def dispatch_result(state: AgentState) -> TaskDispatchResult:
    """把收尾后的状态转成 miclaw/task.dispatch 的响应结果。"""
    return TaskDispatchResult(answer=state["final_answer"],
                              completed=state.get("failure") is None)


def build_runtime(llm: LLM, registry: ToolRegistry, budget: ResourceBudget, *,
                  max_inflight: int = MAX_INFLIGHT,
                  max_idle_sessions: int = MAX_IDLE_SESSIONS,
                  inference_slots: int = INFERENCE_SLOTS,
                  config: dict[str, Any] | None = None,
                  **build_kwargs: Any) -> Runtime:
    """构图并装好跨请求共享的资源槽，返回运行时。

    MiClaw 调用槽的容量取握手下发的 max_concurrent_calls，由全部请求共用；
    推理槽装在 llm 上，同一个 llm 对象的所有调用都经它排队。
    手工装配时漏掉任何一处都不会报错，只会让并发超出配额，故收在这一处。
    """
    from ..graph import build_agent        # 惰性导入：本模块不依赖图引擎

    llm.slots = SlotPool(inference_slots)
    agent = build_agent(llm, registry,
                        max_concurrent_miclaw=budget.max_concurrent_calls,
                        miclaw_slots=SlotPool(budget.max_concurrent_calls),
                        **build_kwargs)
    return Runtime(agent, max_inflight=max_inflight, config=config,
                   max_idle_sessions=max_idle_sessions)


def serve(runtime: Runtime, client: MiClawClient) -> None:
    """把运行时登记到客户端上，受理 MiClaw 派来的请求与对话结束通知。

    参数不合协议时回 MC-2003（协议层严格模式：字段缺失、多出或取值非法都算）。
    派发请求立刻得到一个 Future，客户端在它完成后回复，读线程不被占住。
    """

    def on_dispatch(raw: dict[str, Any]) -> Future[dict[str, Any]]:
        params = _parse(TaskDispatchParams, raw)
        reply: Future[dict[str, Any]] = Future()

        def done(f: Future[AgentState]) -> None:
            if f.exception() is not None:
                reply.set_exception(f.exception())
            else:
                reply.set_result(dispatch_result(f.result()).model_dump())

        runtime.dispatch(params).add_done_callback(done)
        return reply

    def on_end(raw: dict[str, Any]) -> None:
        runtime.end_conversation(_parse(ConversationEndParams, raw))

    client.on_request(AgentMethod.TASK_DISPATCH, on_dispatch)
    client.on_notification(AgentMethod.CONVERSATION_END, on_end)


def _parse(model: type[_P], raw: dict[str, Any]) -> _P:
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise MiClawError(ErrorCode.MC_INVALID_MESSAGE, f"{model.__name__} 不合协议",
                          detail={"errors": e.errors(include_url=False,
                                                     include_context=False)}) from None
