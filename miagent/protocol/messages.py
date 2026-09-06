"""JSON-RPC 2.0 报文结构 + MiClaw 方法集定义。

本模块是 Agent（客户端）与 MiClaw（服务端）之间唯一的"共同语言"。
客户端、mock 服务端、工具层都只依赖这里，不互相依赖。

JSON-RPC 2.0 一共只有三种报文：
    Request       有 id，要求对方必须回一个 Response
    Notification  没有 id，单向通知，对方不回
    Response      带着请求方的 id 回来，result 和 error 二选一

MCP 就是在这三种报文之上，约定了一批固定的 method 名。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ErrorCode, MiAgentError, jsonrpc_code

# 协议版本。沿用 MCP 的日期版本号格式，握手时双方比对。
PROTOCOL_VERSION = "2026-01-01"

# JSON-RPC 的 id 允许是字符串或数字（标准也允许 null，但仅用于解析失败的响应）
RequestId = str | int


class Method:
    """所有支持的 method 名。

    前半部分是标准 MCP，任何 MCP 客户端都认识；
    带 miclaw/ 前缀的是我们为小米生态定义的扩展。
    """

    # ---- 标准 MCP ----
    INITIALIZE = "initialize"                      # 握手：交换版本与能力
    INITIALIZED = "notifications/initialized"      # 通知：握手完成（无 id）
    TOOLS_LIST = "tools/list"                      # 列出服务端提供的工具
    TOOLS_CALL = "tools/call"                      # 调用一个工具
    PING = "ping"                                  # 保活探测

    # ---- MiClaw 扩展 ----
    AGENT_REGISTER = "miclaw/agent.register"       # Agent 向系统注册身份与声明
    AGENT_UNREGISTER = "miclaw/agent.unregister"   # 注销，释放系统侧资源
    RESOURCE_QUERY = "miclaw/resource.query"       # 查询当前端侧资源配额与占用


# ============================================================
# 一、JSON-RPC 2.0 三种报文
# ============================================================


class JsonRpcError(BaseModel):
    """JSON-RPC 错误对象。

    code/message 是标准字段；我们的 MC-* 字符串码放在 data.code 里。
    """

    code: int
    message: str
    data: dict[str, Any] | None = None

    @property
    def miclaw_code(self) -> str | None:
        """取出我们自定义的 MC-* 码。"""
        return (self.data or {}).get("code")


class JsonRpcRequest(BaseModel):
    # extra="forbid"：出现协议未定义的字段直接报错。
    # 协议层宁可严格 —— 悄悄忽略未知字段会让版本不一致的问题拖到线上才暴露。
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: RequestId
    method: str
    params: dict[str, Any] | None = None


class JsonRpcNotification(BaseModel):
    """没有 id 的单向通知，对方不得回复。"""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: RequestId | None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None

    @model_validator(mode="after")
    def _exactly_one_of_result_or_error(self) -> JsonRpcResponse:
        """JSON-RPC 2.0 硬性规定：result 和 error 必须恰好出现一个。

        这条校验很重要 —— 两个都有或都没有的响应是协议损坏，
        必须在解析阶段拦下，不能让它流到业务逻辑里。
        """
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError("响应中 result 与 error 必须恰好存在一个")
        return self


# ============================================================
# 二、常用负载结构（params / result 的内容）
# ============================================================


class ResourceBudget(BaseModel):
    """端侧资源配额。握手时由 MiClaw 下发，Agent 必须在此约束内运行。

    这是端侧场景相对云端框架多出来的一层约束：云端 Agent 不关心内存和电量，
    端侧必须关心 —— 超了就是 MC-4xxx。
    """

    max_memory_mb: int = Field(default=256, description="Agent 进程内存上限")
    max_concurrent_calls: int = Field(default=2, description="并发工具调用上限")
    max_call_timeout_ms: int = Field(default=5000, description="单次工具调用超时")
    power_saving: bool = Field(default=False, description="设备是否处于省电模式")


class ToolDescriptor(BaseModel):
    """tools/list 返回的单个工具描述。

    inputSchema 用 JSON Schema 描述参数 —— 这是 MCP 的规定，
    好处是它能直接喂给大模型做 function calling，不用二次转换。
    """

    name: str
    description: str
    inputSchema: dict[str, Any]  # noqa: N815  字段名由 MCP 标准规定，保持驼峰


class ToolCallResult(BaseModel):
    """tools/call 的返回。

    isError 为 True 表示"工具跑了但业务上失败了"（比如查不到联系人），
    与 JSON-RPC 层的 error 不同 —— 后者表示"调用本身就没成立"。
    这两者不要混，混了会导致重试策略判断错误。
    """

    content: list[dict[str, Any]]  # MCP 规定的多模态内容块列表
    isError: bool = False  # noqa: N815


# ============================================================
# 三、构造与解析辅助函数
# ============================================================


def success_response(request_id: RequestId, result: dict[str, Any]) -> JsonRpcResponse:
    return JsonRpcResponse(id=request_id, result=result)


def error_response(request_id: RequestId | None, exc: MiAgentError) -> JsonRpcResponse:
    """把一个 MiAgentError 转成合规的 JSON-RPC 错误响应。"""
    int_code = jsonrpc_code(exc.code)
    if int_code is None:
        # AG-* 错误不该出现在网络报文里。真出现了说明代码写错了，
        # 兜底成 -32603 Internal error，但保留原始码方便定位。
        int_code = -32603
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(
            code=int_code,
            message=exc.message,
            data=exc.to_error_data(),
        ),
    )


def text_content(text: str) -> list[dict[str, Any]]:
    """构造 MCP 的文本内容块。工具返回纯文本时用这个包一层。"""
    return [{"type": "text", "text": text}]


def parse_incoming(raw: dict[str, Any]) -> JsonRpcRequest | JsonRpcNotification:
    """服务端侧：把收到的原始 dict 解析成请求或通知。

    区分依据就一条：有没有 id 字段。
    """
    from .errors import MiClawError  # 局部导入避免循环引用

    try:
        if "id" in raw:
            return JsonRpcRequest.model_validate(raw)
        return JsonRpcNotification.model_validate(raw)
    except Exception as e:
        raise MiClawError(
            ErrorCode.MC_INVALID_MESSAGE,
            detail={"reason": str(e)},
        ) from e
