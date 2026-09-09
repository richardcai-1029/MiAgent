"""MiClaw 客户端：Agent 侧的协议实现。

职责边界要清楚 —— 这一层只管「把协议说对」：
  · 构造合法请求、配对响应、把服务端错误还原成异常
  · 做本地时序预检，非法调用不发出去
它不管「该调哪个工具」（那是 graph 层的事），也不碰大模型。

同步阻塞实现，不引入 asyncio。端侧的取舍：
Agent 的调用天然是串行的（模型想一步、调一次），没有事件循环
就省掉一整套调度机制和常驻内存，这也是「轻量化」的一部分。
"""

from __future__ import annotations

import sys
import threading
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
    ResourceBudget,
    ToolDescriptor,
)
from ..transport import SubprocessTransport, log

DEFAULT_SERVER_COMMAND = [sys.executable, "-m", "miagent.mock_server"]


# 用哨兵而非 None 作 timeout 的缺省值：None 是合法取值，表示"一直等"。
_UNSET: Any = object()


class MiClawClient:
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
        # 图层的并行派发会让多个线程同时进来。客户端本身是同步阻塞的：
        # id 分配不是原子操作（两个线程可能拿到同一个 id，导致响应配对错乱），
        # 管道写入也可能字节交错破坏按行分帧。加锁保证正确性。
        #
        # 代价：MiClaw 调用因此被串行化，图层的并行拿不到实际提速。
        # 要拿到真实并发，需要传输层支持多路复用（连发多个请求不等回复，
        # 按 id 配对响应）且服务端并发处理，两项均尚未实现。
        self._lock = threading.RLock()

        # 本地会话状态。服务端才是权威，这里只是为了「非法调用不出门」，
        # 省一次进程间往返 —— 前端表单校验和后端校验的关系。
        self._handshaked = False
        self._registered = False

        # 握手/注册时服务端下发的约束，上层要据此决策
        self.budget: ResourceBudget | None = None
        self.granted_permissions: list[str] = []
        self.denied_permissions: list[dict[str, Any]] = []
        self.session_id: str | None = None

    # ------------------------------------------------------------
    # 底层收发
    # ------------------------------------------------------------

    def _request(self, method: Method, params: dict[str, Any] | None = None,
                 timeout: float | None = _UNSET) -> dict[str, Any]:
        """发一个请求并等它的响应。整段临界区加锁，见 __init__ 的说明。

        timeout 缺省时用连接级超时；工具调用会传入握手时下发的单次调用配额。
        """
        with self._lock:
            return self._request_locked(
                method, params, self._timeout if timeout is _UNSET else timeout)

    def _request_locked(self, method: Method, params: dict[str, Any] | None,
                        timeout: float | None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id

        self._transport.send(
            JsonRpcRequest(id=request_id, method=method, params=params).model_dump(exclude_none=True)
        )

        while True:
            raw = self._transport.receive(timeout=timeout)
            if raw is None:
                raise MiClawError(ErrorCode.MC_TRANSPORT_CLOSED, "服务端在响应前关闭了连接")

            response = JsonRpcResponse.model_validate(raw)

            # ★ id 配对。同步收发也不能省这一步：上次超时的请求，
            #   它的响应可能现在才到，不核对就会张冠李戴。
            if response.id != request_id:
                log(f"[client] 丢弃过期响应 id={response.id}（正在等 id={request_id}）")
                continue

            if response.error is not None:
                raise self._to_exception(response.error)
            return response.result or {}

    def _notify(self, method: Method, params: dict[str, Any] | None = None) -> None:
        """发一条通知。没有 id，不等回复。"""
        with self._lock:
            self._transport.send(
                JsonRpcNotification(method=method, params=params).model_dump(exclude_none=True)
            )

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

    # ------------------------------------------------------------

    def connect(self, agent_id: str, permissions: list[str], intents: list[str] | None = None):
        """握手 + 注册，一步到位。返回批准的权限。"""
        self.initialize()
        return self.register(agent_id, permissions, intents)

    def __enter__(self) -> MiClawClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
