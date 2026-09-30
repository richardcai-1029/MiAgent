"""MiClaw 客户端：Agent 侧的协议实现。

职责边界要清楚 —— 这一层只管「把协议说对」：
  · 构造合法请求、配对响应、把服务端错误还原成异常
  · 做本地时序预检，非法调用不发出去
  · 受理 MiClaw 主动发来的请求与通知，转交登记好的处理函数
它不管「该调哪个工具」（那是 agent 层的事），也不碰大模型。

收发分流：同一条管道上既有自己请求的响应，也有 MiClaw 主动发来的请求与通知。
一个读线程独占接收，按报文种类分流 ——

    带 method              → MiClaw 发来的请求（有 id）或通知（无 id），交给处理函数
    不带 method、带 id      → 自己某个请求的响应，按 id 交给等它的那个调用

发出请求的线程只负责写一行、再等自己的那个 id，不碰接收。多个线程因此可以
同时各有一个请求在途，响应先到先交付，不必按发出的顺序。

用线程而不引入 asyncio：没有事件循环就省掉一整套调度机制和常驻内存，
这也是「轻量化」的一部分。
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from ..protocol import (
    PROTOCOL_VERSION,
    AgentError,
    ErrorCode,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    Method,
    MiClawError,
    RequestId,
    ResourceBudget,
    ToolDescriptor,
    error_response,
    parse_incoming,
    success_response,
)
from ..transport import SubprocessTransport, log

DEFAULT_SERVER_COMMAND = [sys.executable, "-m", "miagent.mock_server"]


# 用哨兵而非 None 作 timeout 的缺省值：None 是合法取值，表示"一直等"。
_UNSET: Any = object()

# MiClaw 发来的请求的处理函数：收 params，返回 result，或返回一个将来给出
# result 的 Future（耗时的处理必须这样做，不能占住读线程）。
RequestHandler = Callable[[dict[str, Any]], "dict[str, Any] | Future[dict[str, Any]]"]
NotificationHandler = Callable[[dict[str, Any]], None]


class MiClawClient:
    """Agent 侧的 MiClaw 协议客户端：握手、注册、工具调用、受理 MiClaw 发来的请求与通知。"""

    def __init__(
        self,
        command: list[str] | None = None,
        timeout: float | None = 10.0,
        transport: Any = None,
    ) -> None:
        # 传输层可注入：默认拉起子进程，测试与调试时可换成进程内回环。
        # 客户端只依赖 send/receive/close 三个方法，不关心底下是管道还是别的。
        self._transport = transport or SubprocessTransport(command or DEFAULT_SERVER_COMMAND)
        self._timeout = timeout
        self._next_id = 0
        # 两把锁各管一件事，都只在一瞬间持有，等响应时不持锁：
        # id 分配不是原子操作（两个线程拿到同一个 id 会让响应配对错乱）；
        # 两个线程同时写管道可能字节交错，破坏按行分帧。
        self._id_lock = threading.Lock()
        self._send_lock = threading.Lock()

        # 在途请求：id → 等它响应的 Future。读线程按 id 交付。
        self._pending: dict[RequestId, Future[JsonRpcResponse]] = {}
        self._pending_lock = threading.Lock()
        self._closed_error: MiClawError | None = None

        self._request_handlers: dict[str, RequestHandler] = {}
        self._notification_handlers: dict[str, NotificationHandler] = {}

        # 本地会话状态。服务端才是权威，这里只是为了「非法调用不出门」，
        # 省一次进程间往返 —— 前端表单校验和后端校验的关系。
        self._handshaked = False
        self._registered = False

        # 握手/注册时服务端下发的约束，上层要据此决策
        self.budget: ResourceBudget | None = None
        self.granted_permissions: list[str] = []
        self.denied_permissions: list[dict[str, Any]] = []
        self.session_id: str | None = None

        self._reader = threading.Thread(target=self._read_loop, name="miclaw-reader",
                                        daemon=True)
        self._reader.start()

    # ------------------------------------------------------------
    # 底层收发
    # ------------------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        with self._send_lock:
            self._transport.send(msg)

    def _request(self, method: Method, params: dict[str, Any] | None = None,
                 timeout: float | None = _UNSET) -> dict[str, Any]:
        """发一个请求并等它的响应。

        timeout 缺省时用连接级超时；工具调用会传入握手时下发的单次调用配额。
        """
        with self._id_lock:
            self._next_id += 1
            request_id = self._next_id

        # 先登记再发送：响应可能在 send 返回之前就被读线程收到。
        waiter: Future[JsonRpcResponse] = Future()
        with self._pending_lock:
            if self._closed_error is not None:
                raise self._closed_error
            self._pending[request_id] = waiter
        try:
            self._send(JsonRpcRequest(id=request_id, method=method, params=params)
                       .model_dump(exclude_none=True))
            response = self._wait(waiter, self._timeout if timeout is _UNSET else timeout)
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if response.error is not None:
            raise self._to_exception(response.error)
        return response.result or {}

    def _wait(self, waiter: Future[JsonRpcResponse], timeout: float | None) -> JsonRpcResponse:
        """等一个在途请求的响应。超时抛 MC-1003；超时之后才到的响应由读线程丢弃。"""
        try:
            return waiter.result(timeout=timeout)
        except FutureTimeout:
            raise MiClawError(ErrorCode.MC_REQUEST_TIMEOUT, f"等待响应超过 {timeout}s",
                              detail={"timeout_s": timeout}) from None

    def _notify(self, method: Method, params: dict[str, Any] | None = None) -> None:
        """发一条通知。没有 id，不等回复。"""
        self._send(JsonRpcNotification(method=method, params=params)
                   .model_dump(exclude_none=True))

    # ------------------------------------------------------------
    # 接收与分流：只在读线程里运行
    # ------------------------------------------------------------

    def _read_loop(self) -> None:
        reason = "服务端在响应前关闭了连接"
        while True:
            try:
                raw = self._transport.receive()
            except MiClawError as e:
                # 一行不是合法 JSON：拿不到 id，无从回复也无从交付，丢弃这一行继续读
                log(f"[client] 丢弃无法解析的报文: {e}")
                continue
            except Exception as e:        # 传输本身坏了，按连接关闭处理
                reason = f"读取报文失败: {e!r}"
                break
            if raw is None:
                break
            try:
                self._route(raw)
            except Exception as e:        # 分流出错不能让读线程死掉
                log(f"[client] 处理报文时出错（已丢弃）: {e!r}")

        closed = MiClawError(ErrorCode.MC_TRANSPORT_CLOSED, reason)
        with self._pending_lock:
            self._closed_error = closed
            waiters, self._pending = list(self._pending.values()), {}
        for w in waiters:
            if not w.done():
                w.set_exception(closed)

    def _route(self, raw: dict[str, Any]) -> None:
        if "method" in raw:
            self._on_incoming(raw)
            return
        response = JsonRpcResponse.model_validate(raw)
        with self._pending_lock:
            waiter = self._pending.get(response.id)
        if waiter is None or waiter.done():
            # 超时后才到的响应：等它的调用已经放弃，交给谁都是张冠李戴
            log(f"[client] 丢弃过期响应 id={response.id}")
            return
        waiter.set_result(response)

    def _on_incoming(self, raw: dict[str, Any]) -> None:
        """MiClaw 发来的请求或通知。

        这些方法都要求本 Agent 已注册：派发请求、通知对话结束，前提都是系统已经
        认得这个 Agent。未登记处理函数的请求回 MC-2004，通知则忽略。
        """
        try:
            msg = parse_incoming(raw)
        except MiClawError as e:
            if raw.get("id") is not None:
                self._reply(error_response(raw["id"], e))
            return

        if isinstance(msg, JsonRpcNotification):
            handler = self._notification_handlers.get(msg.method)
            if handler is None or not self._registered:
                log(f"[client] 忽略通知 {msg.method}")
                return
            try:
                handler(msg.params or {})
            except Exception as e:
                log(f"[client] 处理通知 {msg.method} 出错: {e!r}")
            return

        handler = self._request_handlers.get(msg.method)
        if handler is None:
            self._reply(error_response(msg.id, MiClawError(
                ErrorCode.MC_METHOD_NOT_FOUND, detail={"method": msg.method})))
            return
        if not self._registered:
            self._reply(error_response(msg.id, MiClawError(
                ErrorCode.MC_NOT_INITIALIZED, f"Agent 尚未注册，不能受理 {msg.method}",
                detail={"method": msg.method})))
            return
        try:
            outcome = handler(msg.params or {})
        except Exception as e:
            self._reply(self._failure(msg.id, e))
            return
        if isinstance(outcome, Future):
            outcome.add_done_callback(lambda f, rid=msg.id: self._reply(
                self._failure(rid, f.exception()) if f.exception()
                else success_response(rid, f.result())))
        else:
            self._reply(success_response(msg.id, outcome))

    @staticmethod
    def _failure(request_id: RequestId, exc: BaseException) -> JsonRpcResponse:
        """处理函数失败时的错误响应。

        MC- 码原样回给对端；其余一律回 JSON-RPC 的 Internal error，不带 data ——
        AG-* 码只在 Agent 侧，不上网络。
        """
        if isinstance(exc, MiClawError):
            return error_response(request_id, exc)
        log(f"[client] 受理请求 {request_id} 失败: {exc!r}")
        return JsonRpcResponse(id=request_id,
                               error=JsonRpcError(code=-32603, message="Agent 内部错误"))

    def _reply(self, response: JsonRpcResponse) -> None:
        try:
            self._send(response.model_dump(exclude_none=True))
        except MiClawError as e:          # 连接已断，回复送不出去
            log(f"[client] 回复 {response.id} 未能送出: {e}")

    def on_request(self, method: str, handler: RequestHandler) -> None:
        """登记 MiClaw 发来的某种请求的处理函数。

        处理函数在读线程里被调用：耗时的处理必须立刻返回一个 Future，
        结果就绪后客户端再回复 —— 占住读线程，别的响应就都收不到了。
        """
        self._request_handlers[method] = handler

    def on_notification(self, method: str, handler: NotificationHandler) -> None:
        """登记 MiClaw 发来的某种通知的处理函数。在读线程里调用，必须很快返回。"""
        self._notification_handlers[method] = handler

    @staticmethod
    def _to_exception(error: JsonRpcError) -> MiClawError:
        """把网络上的错误对象还原成本地异常。

        这一步让 MC- 错误码完成了闭环：服务端抛出 -> 序列化过网络 ->
        客户端还原成同一个 ErrorCode，调用方直接 except MiClawError 即可。
        """
        try:
            code = ErrorCode(error.miclaw_code)
        except (ValueError, TypeError):
            # 服务端版本比我们新，返回了不认识的码。不能崩，兜底成执行失败。
            log(f"[client] 未知错误码 {error.miclaw_code!r}，按执行失败处理")
            code = ErrorCode.MC_TOOL_EXECUTION_FAILED
        return MiClawError(code, error.message, detail=(error.data or {}).get("detail"))

    def _require(self, flag: bool, what: str) -> None:
        if not flag:
            raise AgentError(ErrorCode.AG_INVALID_CALL_ORDER, f"必须先{what}", detail={"need": what})

    # ------------------------------------------------------------
    # 协议动作
    # ------------------------------------------------------------

    def initialize(self) -> ResourceBudget:
        result = self._request(Method.INITIALIZE, {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": "MiAgent", "version": "0.1.0"},
        })
        self._handshaked = True
        self.budget = ResourceBudget.model_validate(result["resourceBudget"])
        self._notify(Method.INITIALIZED)      # 告知服务端握手已完成
        return self.budget

    def register(
        self,
        agent_id: str,
        permissions: list[str],
        intents: list[str] | None = None,
    ) -> list[str]:
        """向系统注册。返回**实际批准**的权限 —— 可能少于申请的。"""
        self._require(self._handshaked, "完成 initialize 握手")
        result = self._request(Method.AGENT_REGISTER, {
            "agent": {"agentId": agent_id, "name": "MiAgent", "version": "0.1.0"},
            "capabilities": {"intents": intents or []},
            "permissions": permissions,
        })
        self._registered = True
        self.session_id = result.get("sessionId")
        self.granted_permissions = result.get("grantedPermissions", [])
        self.denied_permissions = result.get("deniedPermissions", [])
        if "resourceBudget" in result:
            self.budget = ResourceBudget.model_validate(result["resourceBudget"])
        return self.granted_permissions

    def list_tools(self) -> list[ToolDescriptor]:
        self._require(self._registered, "完成 agent.register 注册")
        result = self._request(Method.TOOLS_LIST)
        return [ToolDescriptor.model_validate(t) for t in result.get("tools", [])]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用一个系统工具，返回文本结果。

        MCP 的 isError 表示「工具跑了，但业务上失败」（如查不到联系人），
        与 JSON-RPC error（调用根本没成立）语义不同。这里把它转成
        MC-3003 抛出 —— 段位 3 对应「重试无用，换个方案」，正是该有的行为。
        丢掉这个字段会让业务失败被当成成功，是很隐蔽的 bug。
        """
        self._require(self._registered, "完成 agent.register 注册")
        result = self._request(Method.TOOLS_CALL,
                               {"name": name, "arguments": arguments or {}},
                               timeout=self.call_timeout)
        parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        text = "\n".join(parts)
        if result.get("isError"):
            raise MiClawError(ErrorCode.MC_TOOL_EXECUTION_FAILED, text,
                              detail={"tool": name, "kind": "business_failure"})
        return text

    @property
    def call_timeout(self) -> float | None:
        """单次工具调用的超时秒数，取自握手时下发的资源配额。

        与连接级超时是两回事：后者约束的是"这条链路多久没动静就认为断了"，
        前者约束的是"系统愿意为一次工具调用等多久"。一个工具卡住不应该
        拖到整条链路的超时才被发现 —— 端侧的调用配额本就更紧。

        握手尚未完成、或服务端未下发配额时退回连接级超时。
        """
        if self.budget is None:
            return self._timeout
        return self.budget.max_call_timeout_ms / 1000

    def query_resource(self) -> dict[str, Any]:
        self._require(self._handshaked, "完成 initialize 握手")
        return self._request(Method.RESOURCE_QUERY)

    def unregister(self) -> None:
        if self._registered:
            self._request(Method.AGENT_UNREGISTER)
            self._registered = False

    def close(self) -> None:
        try:
            self.unregister()
        except Exception as e:
            log(f"[client] 注销时出错（忽略）: {e}")
        self._transport.close()
        self._reader.join(timeout=self._timeout)

    # ------------------------------------------------------------
    # 便捷入口
    # ------------------------------------------------------------

    def connect(self, agent_id: str, permissions: list[str], intents: list[str] | None = None):
        """握手 + 注册，一步到位。返回批准的权限。"""
        self.initialize()
        return self.register(agent_id, permissions, intents)

    def __enter__(self) -> MiClawClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
