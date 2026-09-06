"""MiAgent <-> MiClaw 协议层。

对外只暴露这些名字，其他模块一律从 `miagent.protocol` 导入，
不要深入到子模块 —— 这样以后重构内部文件结构不会影响调用方。
"""

from .errors import (
    AgentError,
    ErrorCode,
    MiAgentError,
    MiClawError,
    describe,
    is_transport_layer,
    jsonrpc_code,
)
from .messages import (
    PROTOCOL_VERSION,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    Method,
    RequestId,
    ResourceBudget,
    ToolCallResult,
    ToolDescriptor,
    error_response,
    parse_incoming,
    success_response,
    text_content,
)

__all__ = [
    "PROTOCOL_VERSION",
    "AgentError",
    "ErrorCode",
    "JsonRpcError",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "Method",
    "MiAgentError",
    "MiClawError",
    "RequestId",
    "ResourceBudget",
    "ToolCallResult",
    "ToolDescriptor",
    "describe",
    "error_response",
    "is_transport_layer",
    "jsonrpc_code",
    "parse_incoming",
    "success_response",
    "text_content",
]
