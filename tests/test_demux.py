"""收发分流：一条管道上同时有自己请求的响应与 MiClaw 主动发来的请求、通知。"""

import queue
import sys
import threading
from concurrent.futures import Future

import pytest

from miagent.client import MiClawClient
from miagent.mock_server import MiClawMockServer
from miagent.protocol import ErrorCode, MiClawError
from miagent.runtime import Runtime, build_runtime, serve
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

from .test_graph import llm_for, plan, task
from .test_runtime import StubAgent, wait_until

WAIT = 5


class ScriptedTransport:
    """由测试手动扮演服务端：记录客户端发出的报文，由测试决定何时回什么。"""

    def __init__(self):
        self.sent: queue.Queue = queue.Queue()
        self._inbox: queue.Queue = queue.Queue()

    def send(self, msg):
        self.sent.put(msg)

    def receive(self, timeout=None):
        return self._inbox.get()

    def deliver(self, msg):
        self._inbox.put(msg)

    def close(self):
        self._inbox.put(None)

    def next_sent(self):
        return self.sent.get(timeout=WAIT)


@pytest.fixture
def scripted():
    t = ScriptedTransport()
    c = MiClawClient(timeout=WAIT, transport=t)
    yield c, t
    t.close()


def in_thread(fn, *args):
    """在另一个线程里调用，返回装着结果或异常的 Future。"""
    out: Future = Future()

    def run():
        try:
            out.set_result(fn(*args))
        except BaseException as e:
            out.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    return out


# ============================================================
# 自己请求的响应：按 id 交付
# ============================================================


class TestResponses:
    def test_concurrent_requests_are_paired_by_id_even_out_of_order(self, scripted):
        client, t = scripted
        a = in_thread(client._request, "test/a", {"n": 1})
        b = in_thread(client._request, "test/b", {"n": 2})
        first, second = t.next_sent(), t.next_sent()      # 两个请求同时在途
        for req in (second, first):                        # 倒序回复
            t.deliver({"jsonrpc": "2.0", "id": req["id"],
                       "result": {"echo": req["params"]["n"]}})
        assert a.result(WAIT) == {"echo": 1}
        assert b.result(WAIT) == {"echo": 2}

    def test_timeout_then_late_response_is_discarded(self, scripted):
        client, t = scripted
        with pytest.raises(MiClawError) as ei:
            client._request("test/slow", timeout=0.05)
        assert ei.value.code is ErrorCode.MC_REQUEST_TIMEOUT
        late = t.next_sent()

        pending = in_thread(client._request, "test/next")
        nxt = t.next_sent()
        t.deliver({"jsonrpc": "2.0", "id": late["id"], "result": {"stale": True}})
        t.deliver({"jsonrpc": "2.0", "id": nxt["id"], "result": {"fresh": True}})
        assert pending.result(WAIT) == {"fresh": True}

    def test_closed_connection_fails_every_waiter_and_later_calls(self, scripted):
        client, t = scripted
        waiting = [in_thread(client._request, "test/x") for _ in range(2)]
        t.next_sent(), t.next_sent()
        t.close()
        for w in waiting:
            with pytest.raises(MiClawError) as ei:
                w.result(WAIT)
            assert ei.value.code is ErrorCode.MC_TRANSPORT_CLOSED
        with pytest.raises(MiClawError) as ei:
            client._request("test/after")
        assert ei.value.code is ErrorCode.MC_TRANSPORT_CLOSED

    def test_error_response_is_still_raised(self, scripted):
        client, t = scripted
        pending = in_thread(client._request, "test/x")
        req = t.next_sent()
        t.deliver({"jsonrpc": "2.0", "id": req["id"],
                   "error": {"code": -32050, "message": "权限不足",
                             "data": {"code": "MC-5001"}}})
        with pytest.raises(MiClawError) as ei:
            pending.result(WAIT)
        assert ei.value.code is ErrorCode.MC_PERMISSION_DENIED


# ============================================================
# MiClaw 发来的请求与通知：交给处理函数
# ============================================================


def incoming(id_, method, **params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


class TestIncoming:
    @pytest.fixture
    def registered(self, scripted):
        client, t = scripted
        client._registered = True          # 只测分流，跳过握手与注册的往返
        return client, t

    def test_unhandled_request_gets_mc_2004(self, registered):
        _, t = registered
        t.deliver(incoming("s1", "miclaw/unknown"))
        reply = t.next_sent()
        assert reply["id"] == "s1" and reply["error"]["data"]["code"] == "MC-2004"

    def test_request_before_registration_gets_mc_2002(self, scripted):
        client, t = scripted
        client.on_request("miclaw/x", lambda p: {})
        t.deliver(incoming("s1", "miclaw/x"))
        assert t.next_sent()["error"]["data"]["code"] == "MC-2002"

    def test_plain_result_is_replied_at_once(self, registered):
        client, t = registered
        client.on_request("miclaw/x", lambda p: {"got": p["v"]})
        t.deliver(incoming("s1", "miclaw/x", v=3))
        assert t.next_sent() == {"jsonrpc": "2.0", "id": "s1", "result": {"got": 3}}

    def test_future_result_does_not_block_other_traffic(self, registered):
        client, t = registered
        slow: Future = Future()
        client.on_request("miclaw/slow", lambda p: slow)
        t.deliver(incoming("s1", "miclaw/slow"))

        # 处理函数还没给出结果，读线程照样交付别的响应
        pending = in_thread(client._request, "test/ping")
        req = t.next_sent()
        t.deliver({"jsonrpc": "2.0", "id": req["id"], "result": {}})
        assert pending.result(WAIT) == {}

        slow.set_result({"done": True})
        assert t.next_sent() == {"jsonrpc": "2.0", "id": "s1", "result": {"done": True}}

    def test_miclaw_error_keeps_its_code(self, registered):
        client, t = registered

        def bad(p):
            raise MiClawError(ErrorCode.MC_INVALID_MESSAGE, "参数不合协议")

        client.on_request("miclaw/x", bad)
        t.deliver(incoming("s1", "miclaw/x"))
        assert t.next_sent()["error"]["data"]["code"] == "MC-2003"

    def test_other_failures_do_not_leak_agent_codes(self, registered):
        """AG-* 码只在 Agent 侧：处理函数的其他失败一律回 Internal error，不带 data。"""
        from miagent.protocol import AgentError

        client, t = registered

        def boom(p):
            raise AgentError(ErrorCode.AG_INVALID_STATE, "内部状态不对")

        client.on_request("miclaw/x", boom)
        t.deliver(incoming("s1", "miclaw/x"))
        assert t.next_sent()["error"] == {"code": -32603, "message": "Agent 内部错误"}

    def test_notification_goes_to_its_handler(self, registered):
        client, t = registered
        got: queue.Queue = queue.Queue()
        client.on_notification("miclaw/n", got.put)
        t.deliver({"jsonrpc": "2.0", "method": "miclaw/n", "params": {"k": 1}})
        t.deliver({"jsonrpc": "2.0", "method": "miclaw/ignored"})   # 无处理函数：忽略
        assert got.get(timeout=WAIT) == {"k": 1}
        assert t.sent.empty()                                         # 通知不回复


# ============================================================
# 真实子进程管道
# ============================================================


SCRIPTED_SERVER = r"""
import json, sys
def send(m):
    sys.stdout.write((m if isinstance(m, str) else json.dumps(m)) + "\n"); sys.stdout.flush()
reqs = [json.loads(sys.stdin.readline()) for _ in range(2)]
send("这一行不是 JSON")
for r in reversed(reqs):
    send({"jsonrpc": "2.0", "id": r["id"], "result": {"n": r["params"]["n"]}})
send({"jsonrpc": "2.0", "id": "s1", "method": "miclaw/task.dispatch",
      "params": {"conversationId": "c", "request": "hi"}})
reply = json.loads(sys.stdin.readline())
send({"jsonrpc": "2.0", "method": "test/echo", "params": reply})
sys.stdin.read()
"""


def test_both_directions_share_one_real_pipe():
    client = MiClawClient(command=[sys.executable, "-c", SCRIPTED_SERVER], timeout=WAIT)
    client._registered = True
    echoed: queue.Queue = queue.Queue()
    client.on_request("miclaw/task.dispatch", lambda p: {"got": p["request"]})
    client.on_notification("test/echo", echoed.put)
    try:
        calls = [in_thread(client._request, "test/n", {"n": i}) for i in (1, 2)]
        assert [c.result(WAIT) for c in calls] == [{"n": 1}, {"n": 2}]
        assert echoed.get(timeout=WAIT) == {"jsonrpc": "2.0", "id": "s1",
                                            "result": {"got": "hi"}}
    finally:
        client._registered = False         # 脚本服务端不懂注销，跳过
        client.close()


# ============================================================
# 运行时接到客户端：MiClaw 派来的请求从管道一路到回复
# ============================================================


PERMS = ["calendar.read"]


@pytest.fixture
def wired():
    server = MiClawMockServer()
    client = MiClawClient(timeout=WAIT, transport=LoopbackTransport(server))
    client.connect("test", PERMS)
    yield server, client
    client.close()


class TestServe:
    def test_dispatch_is_answered_with_result(self, wired):
        server, client = wired
        with Runtime(StubAgent(), max_inflight=2) as rt:
            serve(rt, client)
            rid = server.dispatch_task("c1", "今晚有空吗")
            reply = server.reply_to(rid, timeout=WAIT)
        assert reply["result"] == {"answer": "答:今晚有空吗", "completed": True}

    def test_priority_on_the_wire_orders_admission(self, wired):
        server, client = wired
        agent = StubAgent()
        gate = agent.gate("first")
        with Runtime(agent, max_inflight=1) as rt:
            serve(rt, client)
            ids = [server.dispatch_task("c0", "first")]
            wait_until(lambda: agent.started == ["first"])
            ids.append(server.dispatch_task("c1", "bg", priority="background"))
            ids.append(server.dispatch_task("c2", "fg"))
            wait_until(lambda: rt._queues.keys() == {"c1", "c2"})
            gate.set()
            for rid in ids:
                server.reply_to(rid, timeout=WAIT)
        assert agent.started == ["first", "fg", "bg"]

    def test_malformed_params_get_mc_2003(self, wired):
        server, client = wired
        with Runtime(StubAgent(), max_inflight=1) as rt:
            serve(rt, client)
            rid = server.dispatch_task("c1", "x", priority="urgent")
            reply = server.reply_to(rid, timeout=WAIT)
        assert reply["error"]["data"]["code"] == "MC-2003"

    def test_conversation_end_reclaims_the_session(self, wired):
        server, client = wired
        with Runtime(StubAgent(), max_inflight=1) as rt:
            serve(rt, client)
            server.reply_to(server.dispatch_task("c1", "一"), timeout=WAIT)
            assert rt.session("c1") is not None
            server.end_conversation("c1")
            wait_until(lambda: rt.session("c1") is None)

    def test_agent_calls_miclaw_tools_while_serving_a_miclaw_request(self, wired):
        """最能说明分流的一例：处理 MiClaw 派来的请求期间，Agent 在同一条管道上
        反过来调用 MiClaw 的工具。两个方向的报文交错，各自配对。"""
        server, client = wired
        registry = ToolRegistry()
        registry.load_from_miclaw(client)
        weather = "system.query_weather"
        llm = llm_for(plan(task("task_1", weather, when="今晚"),
                           task("task_2", weather, when="明早")), answer="今晚晴，明早晴")
        rt = build_runtime(llm, registry, client.budget, retry_delay_ms=0,
                           config={"recursion_limit": 80})
        with rt:
            serve(rt, client)
            ids = [server.dispatch_task(c, "查天气") for c in ("c1", "c2")]
            replies = [server.reply_to(rid, timeout=WAIT) for rid in ids]
        assert all(r["result"]["completed"] for r in replies)
        assert all(r["result"]["answer"] == "今晚晴，明早晴" for r in replies)


def test_server_refuses_to_push_before_registration():
    server = MiClawMockServer()
    LoopbackTransport(server)
    with pytest.raises(RuntimeError):
        server.dispatch_task("c", "x")

