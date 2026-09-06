"""客户端测试。全部走真实子进程 + 真实管道，是名副其实的端到端测试。"""

import pytest

from miagent.client import MiClawClient
from miagent.protocol import AgentError, ErrorCode, MiClawError

PERMS = ["sms.send", "alarm.write", "contacts.read", "screen.capture"]


@pytest.fixture
def client():
    c = MiClawClient(timeout=10.0)
    yield c
    c.close()


@pytest.fixture
def ready(client):
    """已完成握手 + 注册的客户端。"""
    client.connect("com.xiaomi.miagent.test", PERMS, intents=["alarm.create"])
    return client


class TestLocalPrecheck:
    """本地时序预检：非法调用不出门，省一次进程间往返。"""

    def test_call_tool_before_register(self, client):
        with pytest.raises(AgentError) as ei:
            client.call_tool("system.get_battery")
        assert ei.value.code == ErrorCode.AG_INVALID_CALL_ORDER

    def test_register_before_initialize(self, client):
        with pytest.raises(AgentError) as ei:
            client.register("x", [])
        assert ei.value.code == ErrorCode.AG_INVALID_CALL_ORDER


class TestHandshakeAndRegister:
    def test_initialize_returns_budget(self, client):
        budget = client.initialize()
        assert budget.max_memory_mb == 64
        assert client.budget is budget

    def test_register_is_a_negotiation(self, ready):
        """申请四个权限，只批到三个 —— 批准结果不等于申请。"""
        assert set(ready.granted_permissions) == {"sms.send", "alarm.write", "screen.capture"}
        assert [d["permission"] for d in ready.denied_permissions] == ["contacts.read"]
        assert ready.session_id == "sess-mock-001"


class TestToolCalls:
    def test_list_tools_is_permission_filtered(self, ready):
        names = {t.name for t in ready.list_tools()}
        assert "system.send_sms" in names
        assert "system.read_contacts" not in names   # 权限被拒，服务端已隐藏

    def test_tool_schema_survives_the_wire(self, ready):
        """inputSchema 要能原样传回来 —— 它将来要直接喂给模型做 function calling。"""
        sms = next(t for t in ready.list_tools() if t.name == "system.send_sms")
        assert sms.inputSchema["required"] == ["to", "text"]

    def test_successful_call(self, ready):
        assert "08:00" in ready.call_tool("system.set_alarm", {"time": "08:00"})


class TestErrorsSurviveTheWire:
    """服务端抛出的 MC- 码，要能在客户端还原成同一个异常。"""

    @pytest.mark.parametrize(
        "name, args, expected",
        [
            ("system.nope",           {},                 ErrorCode.MC_TOOL_NOT_FOUND),
            ("system.send_sms",       {"to": "10086"},    ErrorCode.MC_TOOL_INVALID_PARAMS),
            ("system.read_contacts",  {"name": "张三"},    ErrorCode.MC_PERMISSION_DENIED),
            ("system.capture_screen", {},                 ErrorCode.MC_RESOURCE_MEMORY_LIMIT),
        ],
    )
    def test_error_code_roundtrip(self, ready, name, args, expected):
        with pytest.raises(MiClawError) as ei:
            ready.call_tool(name, args)
        assert ei.value.code is expected

    def test_detail_survives_the_wire(self, ready):
        """结构化的 detail 也要完整传回来，上层才能程序化处理。"""
        with pytest.raises(MiClawError) as ei:
            ready.call_tool("system.capture_screen")
        assert ei.value.detail == {"limit_mb": 64, "requested_mb": 200}

    def test_permission_and_resource_are_separate_gates(self, ready):
        """screen.capture 权限批准了，但内存不够 —— 两道门是独立的。"""
        assert "screen.capture" in ready.granted_permissions
        with pytest.raises(MiClawError) as ei:
            ready.call_tool("system.capture_screen")
        assert ei.value.code is ErrorCode.MC_RESOURCE_MEMORY_LIMIT


class TestLifecycle:
    def test_context_manager_reaps_subprocess(self):
        with MiClawClient() as c:
            c.connect("x", [])
            proc = c._transport._proc
            assert proc.poll() is None      # 还活着
        assert proc.poll() is not None      # 退出 with 后已被回收

    def test_request_ids_increment(self, ready):
        before = ready._next_id
        ready.call_tool("system.get_battery")
        assert ready._next_id == before + 1
