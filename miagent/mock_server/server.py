"""MiClaw mock 服务端：握手状态机 + 方法分发 + 工具执行。

真实 MiClaw 我们接触不到，这个 mock 按我们自定的规约实现，
用来验证客户端和整条链路。将来换成真 MiClaw，客户端一行不用改。
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ..protocol import (
    PROTOCOL_VERSION,
    AgentMethod,
    ErrorCode,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    Method,
    MiClawError,
    ResourceBudget,
    ToolCallResult,
    ToolDescriptor,
    error_response,
    parse_incoming,
    success_response,
    text_content,
)
from ..transport import log
from .tools import TOOL_REGISTRY


class SessionState(StrEnum):
    CONNECTED = "connected"      # 连接刚建立，还没握手
    HANDSHAKED = "handshaked"    # initialize 完成，协议版本已对齐
    REGISTERED = "registered"    # agent.register 完成，可以干活了
    CLOSED = "closed"            # 已注销


# ★ 任务书要求的「任务调度时序逻辑」，就是下面这张表。
#
# 把「什么方法必须在什么状态下才能调」写成数据而不是散落的 if，
# 好处有三：一眼能看全整个时序；加新方法时必须在这里登记（漏了测试会失败）；
# 时序规则可以被单独测试，不用跑完整链路。
_REQUIRED_STATE: dict[Method, set[SessionState]] = {
    Method.INITIALIZE:       {SessionState.CONNECTED},
    Method.INITIALIZED:      {SessionState.HANDSHAKED},
    Method.AGENT_REGISTER:   {SessionState.HANDSHAKED},
    Method.AGENT_UNREGISTER: {SessionState.REGISTERED},
    Method.TOOLS_LIST:       {SessionState.REGISTERED},
    Method.TOOLS_CALL:       {SessionState.REGISTERED},
    Method.RESOURCE_QUERY:   {SessionState.HANDSHAKED, SessionState.REGISTERED},
    Method.PING:             {SessionState.CONNECTED, SessionState.HANDSHAKED,
                              SessionState.REGISTERED},
}


class MiClawMockServer:
    def __init__(
        self,
        budget: ResourceBudget | None = None,
        user_denied: set[str] | None = None,
    ) -> None:
        self.state = SessionState.CONNECTED
        # mock 用的配额值，无外部依据；取得偏紧只为让 capture_screen
        # 这类重工具能触发 MC-4001，便于验证资源约束路径
        self.budget = budget or ResourceBudget(max_memory_mb=64)
        # 模拟「用户在授权弹窗里拒绝了通讯录」
        self.user_denied = user_denied if user_denied is not None else {"contacts.read"}
        self.granted: set[str] = set()
        self.session_id: str | None = None

        # 服务端主动发给 Agent 的报文走这条通道，由传输在 attach 时给出。
        self._send: Callable[[dict[str, Any]], None] | None = None
        self._out_ids = itertools.count(1)
        # Agent 对服务端请求的回复，按请求 id 存放
        self._replies: dict[str, dict[str, Any]] = {}
        self._replies_cond = threading.Condition()

    # ------------------------------------------------------------
    # 服务端 → Agent：派发请求、通知对话结束
    # ------------------------------------------------------------

    def attach(self, send: Callable[[dict[str, Any]], None]) -> None:
        """接上发往 Agent 的通道。"""
        self._send = send

    def dispatch_task(self, conversation_id: str, request: str,
                      priority: str | None = None) -> str:
        """把一个用户请求派给 Agent，返回这次请求的 id。回复用 reply_to 取。"""
        params: dict[str, Any] = {"conversationId": conversation_id, "request": request}
        if priority is not None:
            params["priority"] = priority
        request_id = f"miclaw-{next(self._out_ids)}"
        self._push(JsonRpcRequest(id=request_id, method=AgentMethod.TASK_DISPATCH,
                                  params=params))
        return request_id

    def end_conversation(self, conversation_id: str) -> None:
        self._push(JsonRpcNotification(method=AgentMethod.CONVERSATION_END,
                                       params={"conversationId": conversation_id}))

    def reply_to(self, request_id: str, timeout: float | None = None) -> dict[str, Any]:
        """等 Agent 对某个请求的回复，返回整条响应报文。"""
        with self._replies_cond:
            if not self._replies_cond.wait_for(lambda: request_id in self._replies, timeout):
                raise TimeoutError(f"Agent 未在 {timeout}s 内回复 {request_id}")
            return self._replies.pop(request_id)

    def _push(self, msg: JsonRpcRequest | JsonRpcNotification) -> None:
        if self._send is None:
            raise RuntimeError("没有接上发往 Agent 的通道")
        if self.state is not SessionState.REGISTERED:
            raise RuntimeError(f"Agent 尚未注册（当前 {self.state}），不能向它派发")
        log(f"[server] → Agent {msg.method}")
        self._send(msg.model_dump(exclude_none=True))

    # ------------------------------------------------------------
    # 入口：处理一条报文
    # ------------------------------------------------------------

    def handle(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """返回要回给对端的报文；返回 None 表示这是通知，不需要回复。"""
        req_id = raw.get("id")
        if "method" not in raw and req_id is not None:
            self._on_reply(raw)                            # Agent 对服务端请求的回复
            return None
        try:
            msg = parse_incoming(raw)                      # 结构合法性 -> MC-2003
            method = self._resolve_method(msg.method)      # 方法是否存在 -> MC-2004
            self._check_state(method)                      # 时序是否合法 -> MC-2002

            if isinstance(msg, JsonRpcNotification):
                self._handle_notification(method)
                return None

            result = self._dispatch(method, msg.params or {})
            return success_response(msg.id, result).model_dump(exclude_none=True)

        except MiClawError as e:
            log(f"[server] ✗ {e}")
            return error_response(req_id, e).model_dump(exclude_none=True)
        except Exception as e:  # 兜底：任何没预料到的异常都不能让服务端崩掉
            log(f"[server] ✗ 未捕获异常: {e!r}")
            return error_response(
                req_id,
                MiClawError(ErrorCode.MC_TOOL_EXECUTION_FAILED, detail={"reason": repr(e)}),
            ).model_dump(exclude_none=True)

    def _on_reply(self, raw: dict[str, Any]) -> None:
        try:
            JsonRpcResponse.model_validate(raw)
        except ValueError as e:
            log(f"[server] ✗ 丢弃不合规的回复: {e}")
            return
        with self._replies_cond:
            self._replies[str(raw["id"])] = raw
            self._replies_cond.notify_all()

    def _resolve_method(self, raw_method: str) -> Method:
        try:
            return Method(raw_method)
        except ValueError:
            raise MiClawError(
                ErrorCode.MC_METHOD_NOT_FOUND,
                detail={"method": raw_method},
            ) from None

    def _check_state(self, method: Method) -> None:
        allowed = _REQUIRED_STATE[method]
        if self.state not in allowed:
            raise MiClawError(
                ErrorCode.MC_NOT_INITIALIZED,
                f"当前状态 {self.state} 不允许调用 {method}",
                detail={"current": self.state, "required": sorted(allowed), "method": method},
            )

    def _handle_notification(self, method: Method) -> None:
        if method is Method.INITIALIZED:
            log("[server] 客户端确认握手完成")

    def _dispatch(self, method: Method, params: dict[str, Any]) -> dict[str, Any]:
        return {
            Method.INITIALIZE:       self._on_initialize,
            Method.AGENT_REGISTER:   self._on_register,
            Method.AGENT_UNREGISTER: self._on_unregister,
            Method.TOOLS_LIST:       self._on_tools_list,
            Method.TOOLS_CALL:       self._on_tools_call,
            Method.RESOURCE_QUERY:   self._on_resource_query,
            Method.PING:             lambda p: {},
        }[method](params)

    # ------------------------------------------------------------
    # 各方法的处理逻辑
    # ------------------------------------------------------------

    def _on_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        client_version = params.get("protocolVersion")
        if client_version != PROTOCOL_VERSION:
            raise MiClawError(
                ErrorCode.MC_VERSION_MISMATCH,
                detail={"client": client_version, "server": PROTOCOL_VERSION},
            )
        self.state = SessionState.HANDSHAKED
        log(f"[server] 握手完成，协议版本 {PROTOCOL_VERSION}")
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "MiClawMock", "version": "0.1.0"},
            "resourceBudget": self.budget.model_dump(),
        }

    def _on_register(self, params: dict[str, Any]) -> dict[str, Any]:
        """注册是一次协商：Agent 申请，系统批准，两者可能不一致。"""
        requested = set(params.get("permissions", []))
        self.granted = requested - self.user_denied
        denied = [
            {"permission": p, "code": ErrorCode.MC_USER_REJECTED.value, "reason": "用户未授权"}
            for p in sorted(requested & self.user_denied)
        ]
        self.session_id = "sess-mock-001"
        self.state = SessionState.REGISTERED
        log(f"[server] 注册完成，批准权限 {sorted(self.granted)}，拒绝 {[d['permission'] for d in denied]}")
        return {
            "sessionId": self.session_id,
            "grantedPermissions": sorted(self.granted),
            "deniedPermissions": denied,
            "resourceBudget": self.budget.model_dump(),
        }

    def _on_unregister(self, params: dict[str, Any]) -> dict[str, Any]:
        self.state = SessionState.CLOSED
        self.granted.clear()
        log("[server] Agent 已注销，系统侧资源已释放")
        return {"unregistered": True}

    def _on_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """只返回本 Agent 有权使用的工具。

        为什么在服务端过滤，而不是全都返回让 Agent 自己判断：
        Agent 会把这份列表原样喂给大模型。如果列表里有它无权调用的工具，
        模型就会规划出注定失败的步骤 —— 白烧一轮推理，端侧尤其浪费。
        """
        visible, hidden = [], []
        for tool in TOOL_REGISTRY.values():
            if tool.required_permission is None or tool.required_permission in self.granted:
                visible.append(
                    ToolDescriptor(
                        name=tool.name,
                        description=tool.description,
                        inputSchema=tool.input_schema,
                    ).model_dump()
                )
            else:
                hidden.append(tool.name)
        if hidden:
            log(f"[server] 因权限不足隐藏了工具：{hidden}")
        return {"tools": visible}

    def _on_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """工具执行前的四道检查，每道对应一个错误码。"""
        name = params.get("name")
        args = params.get("arguments") or {}

        # ① 工具存在吗
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            raise MiClawError(ErrorCode.MC_TOOL_NOT_FOUND, detail={"name": name})

        # ② 必填参数齐吗
        missing = [f for f in tool.input_schema.get("required", []) if f not in args]
        if missing:
            raise MiClawError(ErrorCode.MC_TOOL_INVALID_PARAMS, detail={"missing": missing})

        # ③ 有权限吗
        if tool.required_permission and tool.required_permission not in self.granted:
            raise MiClawError(
                ErrorCode.MC_PERMISSION_DENIED,
                detail={"required": tool.required_permission, "granted": sorted(self.granted)},
            )

        # ④ 内存够吗（端侧特有的一道检查）
        if tool.estimated_memory_mb > self.budget.max_memory_mb:
            raise MiClawError(
                ErrorCode.MC_RESOURCE_MEMORY_LIMIT,
                detail={"limit_mb": self.budget.max_memory_mb,
                        "requested_mb": tool.estimated_memory_mb},
            )

        log(f"[server] 执行工具 {name}({args})")
        output = tool.handler(args)
        if output is None:
            # 业务失败：工具跑通了，只是没查到结果。这不是 JSON-RPC error ——
            # 调用本身成立了，所以用 isError 标记，让上层区别对待。
            return ToolCallResult(content=text_content("未查询到结果"), isError=True).model_dump()
        return ToolCallResult(content=text_content(output)).model_dump()

    def _on_resource_query(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "budget": self.budget.model_dump(),
            "usage": {"memory_mb": 38, "active_calls": 0},   # mock 数据
        }
