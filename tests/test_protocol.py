"""协议层测试。

这些测试是我们那份"自定规约"的可执行版本 —— 文档会过时，测试不会。
"""

import pytest
from pydantic import ValidationError

from miagent.protocol import (
    PROTOCOL_VERSION,
    AgentError,
    ErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    Method,
    MiClawError,
    error_response,
    is_transport_layer,
    jsonrpc_code,
    parse_incoming,
    success_response,
)


class TestErrorCodes:
    def test_mc_codes_have_jsonrpc_mapping(self):
        """所有 MC-* 都要能上网络，必须有整数码。"""
        for code in ErrorCode:
            if code.value.startswith("MC-"):
                assert jsonrpc_code(code) is not None, f"{code.value} 缺少整数码"

    def test_ag_codes_never_go_on_wire(self):
        """所有 AG-* 都不上网络，整数码必须为 None。"""
        for code in ErrorCode:
            if code.value.startswith("AG-"):
                assert jsonrpc_code(code) is None, f"{code.value} 不应有整数码"

    def test_layer_split(self):
        assert is_transport_layer(ErrorCode.MC_RESOURCE_MEMORY_LIMIT)
        assert not is_transport_layer(ErrorCode.AG_PLAN_PARSE_FAILED)

    def test_code_interpolates_as_plain_string(self):
        """错误码常被直接插进日志。用 StrEnum 保证 f"{code}" 得到 "MC-4001"，
        而不是老写法 (str, Enum) 那样漏出 "ErrorCode.MC_RESOURCE_MEMORY_LIMIT"。
        """
        code = ErrorCode.MC_RESOURCE_MEMORY_LIMIT
        assert f"{code}" == "MC-4001"
        assert str(code) == "MC-4001"

    def test_error_carries_structured_detail(self):
        """细节要放 detail 里，不能只拼进 message 字符串。"""
        exc = MiClawError(
            ErrorCode.MC_RESOURCE_MEMORY_LIMIT,
            detail={"limit_mb": 128, "requested_mb": 210},
        )
        assert exc.to_error_data() == {
            "code": "MC-4001",
            "detail": {"limit_mb": 128, "requested_mb": 210},
        }


class TestJsonRpcEnvelope:
    def test_request_roundtrip(self):
        req = JsonRpcRequest(
            id=1,
            method=Method.TOOLS_CALL,
            params={"name": "send_sms", "arguments": {"to": "10086"}},
        )
        raw = req.model_dump(exclude_none=True)
        assert raw["jsonrpc"] == "2.0"
        assert JsonRpcRequest.model_validate(raw) == req

    def test_response_rejects_both_result_and_error(self):
        with pytest.raises(ValidationError):
            JsonRpcResponse(id=1, result={}, error={"code": -1, "message": "x"})

    def test_response_rejects_neither(self):
        with pytest.raises(ValidationError):
            JsonRpcResponse(id=1)

    def test_unknown_field_is_rejected(self):
        """协议层严格模式：未知字段直接失败，不静默忽略。"""
        with pytest.raises(ValidationError):
            JsonRpcRequest.model_validate(
                {"jsonrpc": "2.0", "id": 1, "method": "ping", "extra": 1}
            )

    def test_notification_has_no_id(self):
        msg = parse_incoming({"jsonrpc": "2.0", "method": Method.INITIALIZED})
        assert not hasattr(msg, "id")

    def test_malformed_message_becomes_mc_2003(self):
        with pytest.raises(MiClawError) as ei:
            parse_incoming({"jsonrpc": "1.0", "id": 1, "method": "ping"})
        assert ei.value.code == ErrorCode.MC_INVALID_MESSAGE


class TestResponseHelpers:
    def test_success(self):
        resp = success_response(7, {"protocolVersion": PROTOCOL_VERSION})
        assert resp.error is None
        assert resp.result["protocolVersion"] == PROTOCOL_VERSION

    def test_mc_error_keeps_both_codes(self):
        """整数码合规 + 字符串码承载语义，两者并存。"""
        resp = error_response(7, MiClawError(ErrorCode.MC_PERMISSION_DENIED))
        assert resp.error.code == -32050
        assert resp.error.miclaw_code == "MC-5001"

    def test_ag_error_falls_back_to_internal(self):
        """AG-* 本不该上网络；万一发生，兜底为 -32603 但保留原始码。"""
        resp = error_response(7, AgentError(ErrorCode.AG_PLAN_PARSE_FAILED))
        assert resp.error.code == -32603
        assert resp.error.miclaw_code == "AG-1001"
