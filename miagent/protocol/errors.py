"""MiAgent <-> MiClaw 协议错误码。

分层原则（这是整套错误体系的核心）：

    MC-*  底层错误。由 MiClaw 服务端产生，经 JSON-RPC 报文回传给 Agent。
          排查时看系统侧日志。
    AG-*  上层错误。由 Agent 自身产生（规划、调度、上下文管理）。
          永远不会出现在网络报文中，排查时看 Agent 侧日志。

与 JSON-RPC 2.0 的关系：
    标准要求 error.code 必须是整数，因此每个 MC-* 码都映射到一个标准整数码，
    而我们的字符串码放进 error.data.code。两者并存，互不冲突：

        {"code": -32001,
         "message": "内存超出端侧配额",
         "data": {"code": "MC-4001", "detail": {...}}}

    AG-* 码没有整数映射（值为 None），因为它们从不上网络。
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """全部错误码。

    用 StrEnum（Python 3.11+）而非 (str, Enum)：两者都能当字符串用，但
    StrEnum 的 f"{code}" 会得到 "MC-4001"，老写法会漏出 "ErrorCode.MC_..."。
    错误码常被直接插进日志，这个差别很致命。
    """

    # ===== MC-1xxx 传输层：连接、通道、超时 =====
    MC_CONNECTION_FAILED = "MC-1001"
    MC_TRANSPORT_CLOSED = "MC-1002"
    MC_REQUEST_TIMEOUT = "MC-1003"

    # ===== MC-2xxx 协议层：握手、版本、报文合法性 =====
    MC_VERSION_MISMATCH = "MC-2001"
    MC_NOT_INITIALIZED = "MC-2002"
    MC_INVALID_MESSAGE = "MC-2003"
    MC_METHOD_NOT_FOUND = "MC-2004"

    # ===== MC-3xxx 工具执行层 =====
    MC_TOOL_NOT_FOUND = "MC-3001"
    MC_TOOL_INVALID_PARAMS = "MC-3002"
    MC_TOOL_EXECUTION_FAILED = "MC-3003"

    # ===== MC-4xxx 端侧资源约束（端侧场景特有，云端框架没有这一类）=====
    MC_RESOURCE_MEMORY_LIMIT = "MC-4001"
    MC_RESOURCE_BUSY = "MC-4002"
    MC_RESOURCE_POWER_SAVING = "MC-4003"

    # ===== MC-5xxx 权限与用户授权 =====
    MC_PERMISSION_DENIED = "MC-5001"
    MC_USER_REJECTED = "MC-5002"

    # ===== AG-1xxx 规划层：模型输出到可执行计划的转换 =====
    AG_PLAN_PARSE_FAILED = "AG-1001"
    AG_PLAN_MAX_STEPS_EXCEEDED = "AG-1002"
    AG_PLAN_NO_PROGRESS = "AG-1003"

    # ===== AG-2xxx 工具调度层 =====
    AG_TOOL_NOT_REGISTERED = "AG-2001"
    AG_TOOL_SCHEMA_INVALID = "AG-2002"
    AG_TOOL_RESULT_UNPARSABLE = "AG-2003"

    # ===== AG-3xxx 上下文管理层 =====
    AG_CONTEXT_OVERFLOW = "AG-3001"
    AG_STATE_CORRUPTED = "AG-3002"

    # ===== AG-4xxx 任务生命周期 =====
    AG_TASK_CANCELLED = "AG-4001"
    AG_TASK_BUDGET_EXCEEDED = "AG-4002"
    AG_INVALID_CALL_ORDER = "AG-4003"

    # ===== AG-5xxx 模型层 =====
    AG_LLM_UNAVAILABLE = "AG-5001"
    AG_LLM_INVALID_RESPONSE = "AG-5002"


# 错误码规格表：错误码 -> (JSON-RPC 整数码, 中文描述)
# JSON-RPC 整数码为 None 表示该错误不上网络（所有 AG-* 都是这样）。
#
# 整数码取值遵循 JSON-RPC 2.0：
#   -32600 Invalid Request / -32601 Method not found / -32602 Invalid params
#   -32000 ~ -32099 是标准留给实现方自定义的区间，我们用它表达端侧特有语义。
_SPEC: dict[ErrorCode, tuple[int | None, str]] = {
    ErrorCode.MC_CONNECTION_FAILED:     (-32010, "与 MiClaw 服务端建立连接失败"),
    ErrorCode.MC_TRANSPORT_CLOSED:      (-32011, "传输通道已关闭"),
    ErrorCode.MC_REQUEST_TIMEOUT:       (-32012, "请求超时未收到响应"),

    ErrorCode.MC_VERSION_MISMATCH:      (-32020, "协议版本不匹配"),
    ErrorCode.MC_NOT_INITIALIZED:       (-32021, "尚未完成 initialize 握手"),
    ErrorCode.MC_INVALID_MESSAGE:       (-32600, "报文不符合 JSON-RPC 2.0 规范"),
    ErrorCode.MC_METHOD_NOT_FOUND:      (-32601, "服务端不支持该 method"),

    ErrorCode.MC_TOOL_NOT_FOUND:        (-32030, "服务端未注册该工具"),
    ErrorCode.MC_TOOL_INVALID_PARAMS:   (-32602, "工具入参不合法"),
    ErrorCode.MC_TOOL_EXECUTION_FAILED: (-32031, "工具执行过程中失败"),

    ErrorCode.MC_RESOURCE_MEMORY_LIMIT: (-32040, "内存占用超出端侧配额"),
    ErrorCode.MC_RESOURCE_BUSY:         (-32041, "系统资源被占用，请稍后重试"),
    ErrorCode.MC_RESOURCE_POWER_SAVING: (-32042, "设备处于低电量/省电模式，拒绝执行"),

    ErrorCode.MC_PERMISSION_DENIED:     (-32050, "权限不足"),
    ErrorCode.MC_USER_REJECTED:         (-32051, "用户拒绝了本次授权"),

    ErrorCode.AG_PLAN_PARSE_FAILED:     (None, "模型输出无法解析为可执行计划"),
    ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED: (None, "超出单任务最大步数"),
    ErrorCode.AG_PLAN_NO_PROGRESS:      (None, "连续重复同一调用，判定为无进展循环"),

    ErrorCode.AG_TOOL_NOT_REGISTERED:   (None, "Agent 本地未注册该工具"),
    ErrorCode.AG_TOOL_SCHEMA_INVALID:   (None, "工具参数未通过 schema 校验"),
    ErrorCode.AG_TOOL_RESULT_UNPARSABLE: (None, "工具返回内容无法解析"),

    ErrorCode.AG_CONTEXT_OVERFLOW:      (None, "上下文长度超出模型窗口"),
    ErrorCode.AG_STATE_CORRUPTED:       (None, "Agent 状态非法"),

    ErrorCode.AG_TASK_CANCELLED:        (None, "任务被取消"),
    ErrorCode.AG_TASK_BUDGET_EXCEEDED:  (None, "超出任务耗时或 token 预算"),
    ErrorCode.AG_INVALID_CALL_ORDER:    (None, "在错误的会话阶段发起调用"),

    ErrorCode.AG_LLM_UNAVAILABLE:       (None, "模型服务不可用"),
    ErrorCode.AG_LLM_INVALID_RESPONSE:  (None, "模型返回内容异常"),
}


def describe(code: ErrorCode) -> str:
    """取错误码的中文描述。"""
    return _SPEC[code][1]


def jsonrpc_code(code: ErrorCode) -> int | None:
    """取对应的 JSON-RPC 整数码；AG-* 返回 None（不上网络）。"""
    return _SPEC[code][0]


def is_transport_layer(code: ErrorCode) -> bool:
    """是否是底层（MiClaw 侧）错误。用于日志分流和重试策略判断。"""
    return code.value.startswith("MC-")


class MiAgentError(Exception):
    """所有 MiAgent 异常的基类。

    统一携带三样东西：错误码、可读信息、结构化细节。
    detail 用来放机器可读的补充信息（比如超限时的实际值），
    不要把它塞进 message 字符串里 —— 那样上层就没法程序化处理了。
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message or describe(code)
        self.detail = detail or {}
        super().__init__(f"[{code.value}] {self.message}")

    def to_error_data(self) -> dict:
        """转成 JSON-RPC error.data 的内容。"""
        data: dict = {"code": self.code.value}
        if self.detail:
            data["detail"] = self.detail
        return data


class MiClawError(MiAgentError):
    """底层错误（MC-*）。由服务端产生并回传。"""


class AgentError(MiAgentError):
    """上层错误（AG-*）。由 Agent 自身产生，不出现在网络报文中。"""
