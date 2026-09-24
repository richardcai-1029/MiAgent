"""mock 服务端测试：时序、权限、资源约束三类规则。"""

import pytest

from miagent.mock_server import MiClawMockServer, SessionState
from miagent.mock_server.server import _REQUIRED_STATE
from miagent.protocol import PROTOCOL_VERSION, ErrorCode, Method, ResourceBudget


def req(id_, method, **params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def code_of(resp) -> str:
    """从错误响应里取出我们的字符串码。"""
    return resp["error"]["data"]["code"]


def make_registered(**kw) -> MiClawMockServer:
    """造一个已经走完 initialize + register 的服务端。"""
    s = MiClawMockServer(**kw)
    s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
    s.handle(req(2, Method.AGENT_REGISTER,
                 permissions=["sms.send", "alarm.write", "contacts.read", "screen.capture"]))
    return s


class TestStateMachine:
    """任务书要求的「任务调度时序逻辑」。"""

    def test_every_method_is_declared_in_state_table(self):
        """单一事实来源：加了新 method 却忘了登记时序，这条测试会失败。"""
        assert set(_REQUIRED_STATE) == set(Method)

    def test_call_before_handshake_rejected(self):
        s = MiClawMockServer()
        assert code_of(s.handle(req(1, Method.TOOLS_CALL, name="system.get_battery"))) == "MC-2002"

    def test_register_before_handshake_rejected(self):
        s = MiClawMockServer()
        assert code_of(s.handle(req(1, Method.AGENT_REGISTER, permissions=[]))) == "MC-2002"

    def test_call_after_handshake_but_before_register_rejected(self):
        """握手完了也不行，必须注册过 —— 这是两道独立的门。"""
        s = MiClawMockServer()
        s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
        assert s.state is SessionState.HANDSHAKED
        assert code_of(s.handle(req(2, Method.TOOLS_LIST))) == "MC-2002"

    def test_happy_path_state_transitions(self):
        s = MiClawMockServer()
        assert s.state is SessionState.CONNECTED
        s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
        assert s.state is SessionState.HANDSHAKED
        s.handle(req(2, Method.AGENT_REGISTER, permissions=[]))
        assert s.state is SessionState.REGISTERED
        s.handle(req(3, Method.AGENT_UNREGISTER))
        assert s.state is SessionState.CLOSED

    def test_double_initialize_rejected(self):
        """握手只能做一次，重复握手是状态错误。"""
        s = MiClawMockServer()
        s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
        assert code_of(s.handle(req(2, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))) == "MC-2002"


class TestProtocolLevel:
    def test_version_mismatch(self):
        s = MiClawMockServer()
        assert code_of(s.handle(req(1, Method.INITIALIZE, protocolVersion="1999-01-01"))) == "MC-2001"

    def test_unknown_method(self):
        s = MiClawMockServer()
        assert code_of(s.handle(req(1, "tools/delete"))) == "MC-2004"

    def test_agent_side_method_is_not_accepted_by_server(self):
        """task.dispatch 由 MiClaw 发给 Agent；反方向发给服务端按不支持处理。"""
        from miagent.protocol import AgentMethod
        s = make_registered()
        resp = s.handle(req(9, AgentMethod.TASK_DISPATCH,
                            conversationId="c", request="今晚有空吗"))
        assert code_of(resp) == "MC-2004"

    def test_notification_gets_no_response(self):
        s = MiClawMockServer()
        s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
        assert s.handle({"jsonrpc": "2.0", "method": Method.INITIALIZED}) is None


class TestPermissionNegotiation:
    def test_denied_permission_is_reported(self):
        s = make_registered()
        resp = s.handle(req(9, Method.TOOLS_CALL, name="system.read_contacts",
                            arguments={"name": "张三"}))
        assert code_of(resp) == "MC-5001"

    def test_granted_permission_works(self):
        s = make_registered()
        resp = s.handle(req(9, Method.TOOLS_CALL, name="system.send_sms",
                            arguments={"to": "10086", "text": "hi"}))
        assert resp["result"]["isError"] is False

    def test_tools_list_hides_unpermitted_tools(self):
        """无权限的工具不应出现在列表里，否则模型会规划出注定失败的步骤。"""
        s = make_registered()
        names = {t["name"] for t in s.handle(req(9, Method.TOOLS_LIST))["result"]["tools"]}
        assert "system.send_sms" in names          # 有权限
        assert "system.read_contacts" not in names # 被用户拒绝的
        assert "system.get_battery" in names       # 不需要权限

    def test_unregistered_agent_sees_only_public_tools(self):
        s = MiClawMockServer()
        s.handle(req(1, Method.INITIALIZE, protocolVersion=PROTOCOL_VERSION))
        s.handle(req(2, Method.AGENT_REGISTER, permissions=[]))
        names = {t["name"] for t in s.handle(req(3, Method.TOOLS_LIST))["result"]["tools"]}
        # 只剩下 required_permission 为 None 的公开工具
        from miagent.mock_server.tools import SYSTEM_TOOLS
        assert names == {t.name for t in SYSTEM_TOOLS if t.required_permission is None}


class TestToolCallChecks:
    """工具执行前的四道检查，各对应一个错误码。"""

    def test_tool_not_found(self):
        s = make_registered()
        assert code_of(s.handle(req(9, Method.TOOLS_CALL, name="system.nope"))) == "MC-3001"

    def test_missing_required_param(self):
        s = make_registered()
        resp = s.handle(req(9, Method.TOOLS_CALL, name="system.send_sms",
                            arguments={"to": "10086"}))   # 缺 text
        assert code_of(resp) == "MC-3002"
        assert resp["error"]["data"]["detail"]["missing"] == ["text"]

    def test_memory_limit(self):
        """端侧特有：工具申报 200MB，配额只有 64MB。"""
        s = make_registered()
        resp = s.handle(req(9, Method.TOOLS_CALL, name="system.capture_screen"))
        assert code_of(resp) == "MC-4001"
        assert resp["error"]["data"]["detail"] == {"limit_mb": 64, "requested_mb": 200}

    def test_bigger_budget_allows_heavy_tool(self):
        """同一个工具，配额放宽就能跑 —— 说明拦截确实来自配额而非别的原因。"""
        s = make_registered(budget=ResourceBudget(max_memory_mb=512))
        resp = s.handle(req(9, Method.TOOLS_CALL, name="system.capture_screen"))
        assert resp["result"]["isError"] is False
