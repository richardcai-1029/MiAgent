"""多请求运行时：资源槽的仲裁、请求的准入与会话内串行、接入图之后的配额。"""

import threading
import time

import pytest

from miagent.graph import build_agent
from miagent.llm import FakeLLM
from miagent.runtime import Runtime, SlotPool, request_key
from miagent.tools import ToolRegistry, ToolSource
from miagent.tools.base import Tool

from .test_graph import llm_for, plan, task

WAIT = 5  # 测试里任何阻塞等待的上限，只为防止用例挂死


def wait_until(cond, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("等待超时")
        time.sleep(0.005)


class Peak:
    """记录同时进入临界区的最大数目。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.now = self.peak = 0

    def __enter__(self):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc):
        with self._lock:
            self.now -= 1


# ============================================================
# 资源槽
# ============================================================


class TestSlotPool:
    def test_capacity_is_never_exceeded(self):
        pool, peak = SlotPool(2), Peak()

        def work():
            with pool.hold(), peak:
                time.sleep(0.02)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT)
        assert peak.peak == 2

    def test_waiters_are_served_by_key_not_arrival(self):
        pool, order = SlotPool(1), []
        release = threading.Event()

        def holder():
            with pool.hold():
                release.wait(WAIT)

        def waiter(key, name):
            request_key.set(key)
            with pool.hold():
                order.append(name)

        threading.Thread(target=holder).start()
        wait_until(lambda: pool.waiting == 0 and pool._free == 0)
        # 按 低 → 高 → 中 的顺序排进队
        threads = []
        for key, name in [((5,), "low"), ((1,), "high"), ((3,), "mid")]:
            threads.append(threading.Thread(target=waiter, args=(key, name)))
            threads[-1].start()
            wait_until(lambda n=len(threads): pool.waiting == n)
        release.set()
        for t in threads:
            t.join(WAIT)
        assert order == ["high", "mid", "low"]

    def test_equal_keys_are_served_in_arrival_order(self):
        pool, order = SlotPool(1), []
        release = threading.Event()

        def holder():
            with pool.hold():
                release.wait(WAIT)

        def waiter(name):
            with pool.hold():
                order.append(name)

        threading.Thread(target=holder).start()
        wait_until(lambda: pool._free == 0)
        threads = []
        for name in "abc":
            threads.append(threading.Thread(target=waiter, args=(name,)))
            threads[-1].start()
            wait_until(lambda n=len(threads): pool.waiting == n)
        release.set()
        for t in threads:
            t.join(WAIT)
        assert order == ["a", "b", "c"]

    def test_slot_is_returned_when_the_body_raises(self):
        pool = SlotPool(1)
        with pytest.raises(RuntimeError):
            with pool.hold():
                raise RuntimeError
        with pool.hold():          # 若没归还，这里会永远阻塞
            pass

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            SlotPool(0)


# ============================================================
# 运行时：准入、会话内串行、优先级
# ============================================================


class StubAgent:
    """只实现 invoke 的替身。gate 为某个请求设一道闸，放行前它一直在飞。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.started: list[str] = []
        self.gates: dict[str, threading.Event] = {}
        self.seen_history: dict[str, list[str]] = {}
        self.peak = Peak()

    def gate(self, request):
        self.gates[request] = threading.Event()
        return self.gates[request]

    def invoke(self, state, config=None):
        req = state["user_request"]
        with self.lock:
            self.started.append(req)
            self.seen_history[req] = [t["request"] for t in state["history"]]
        with self.peak:
            if req in self.gates:
                assert self.gates[req].wait(WAIT)
            if req.startswith("boom"):
                raise RuntimeError(req)
        return {**state, "final_answer": f"答:{req}", "turn_summary": req,
                "failure": None, "episodes": []}


class TestRuntime:
    def test_same_session_runs_in_order_and_later_turns_see_earlier_ones(self):
        agent = StubAgent()
        with Runtime(agent, max_inflight=4) as rt:
            futures = [rt.submit(r, session="u1") for r in ("一", "二", "三")]
            assert [f.result(WAIT)["final_answer"] for f in futures] == ["答:一", "答:二", "答:三"]
        assert agent.peak.peak == 1
        assert agent.seen_history["三"] == ["一", "二"]
        assert [t["request"] for t in rt.session("u1").turns] == ["一", "二", "三"]

    def test_different_sessions_run_concurrently(self):
        agent = StubAgent()
        gate_a, gate_b = agent.gate("a"), agent.gate("b")
        with Runtime(agent, max_inflight=2) as rt:
            fa, fb = rt.submit("a", session="u1"), rt.submit("b", session="u2")
            wait_until(lambda: agent.peak.now == 2)      # 两个同时在飞
            gate_a.set(), gate_b.set()
            fa.result(WAIT), fb.result(WAIT)
        assert agent.seen_history["b"] == []             # 会话之间互不可见

    def test_inflight_is_capped_and_admission_follows_priority(self):
        agent = StubAgent()
        gate = agent.gate("first")
        with Runtime(agent, max_inflight=1) as rt:
            rt.submit("first", session="s0")
            wait_until(lambda: agent.started == ["first"])
            low = rt.submit("low", session="s1", priority=0)
            high = rt.submit("high", session="s2", priority=5)
            also_low = rt.submit("also_low", session="s3", priority=0)
            gate.set()
            for f in (low, high, also_low):
                f.result(WAIT)
        assert agent.peak.peak == 1
        assert agent.started == ["first", "high", "low", "also_low"]

    def test_a_busy_session_does_not_block_other_sessions(self):
        agent = StubAgent()
        gate = agent.gate("u1-a")
        with Runtime(agent, max_inflight=2) as rt:
            rt.submit("u1-a", session="u1")
            # u1 的第二个请求优先级更高，但它所在的会话正忙，u2 的请求先行
            rt.submit("u1-b", session="u1", priority=9)
            other = rt.submit("u2-a", session="u2")
            other.result(WAIT)
            assert "u1-b" not in agent.started
            gate.set()
        assert agent.started == ["u1-a", "u2-a", "u1-b"]

    def test_failure_goes_to_the_future_and_frees_the_session(self):
        agent = StubAgent()
        with Runtime(agent, max_inflight=1) as rt:
            bad = rt.submit("boom", session="u1")
            good = rt.submit("fine", session="u1")
            with pytest.raises(RuntimeError, match="boom"):
                bad.result(WAIT)
            assert good.result(WAIT)["final_answer"] == "答:fine"

    def test_queued_request_can_be_cancelled(self):
        agent = StubAgent()
        gate = agent.gate("first")
        with Runtime(agent, max_inflight=1) as rt:
            rt.submit("first", session="s0")
            wait_until(lambda: agent.started == ["first"])
            queued = rt.submit("never", session="s1")
            assert queued.cancel()
            gate.set()
        assert agent.started == ["first"]

    def test_close_waits_for_everything_and_rejects_new_requests(self):
        agent = StubAgent()
        rt = Runtime(agent, max_inflight=1)
        futures = [rt.submit(str(i), session=f"s{i}") for i in range(3)]
        rt.close()
        assert all(f.done() for f in futures)
        with pytest.raises(RuntimeError):
            rt.submit("late")

    def test_inflight_must_be_positive(self):
        with pytest.raises(ValueError):
            Runtime(StubAgent(), max_inflight=0)


# ============================================================
# 接入图：配额由所有请求共用
# ============================================================


class Probe(Tool):
    """一个 MiClaw 来源的工具：记录并发峰值与调用时看到的请求键。"""

    source = ToolSource.MICLAW
    name = "probe"
    description = "探针"
    input_schema = {"type": "object", "properties": {"tag": {"type": "string"}},
                    "required": ["tag"]}

    def __init__(self):
        self.peak = Peak()
        self.keys: dict[str, tuple] = {}

    def _run(self, args):
        with self.peak:
            self.keys[args["tag"]] = request_key.get()
            time.sleep(0.05)
        return args["tag"]


def two_probes(prefix):
    return plan(task("task_1", "probe", tag=f"{prefix}1"),
                task("task_2", "probe", tag=f"{prefix}2"))


class TestGraphUnderRuntime:
    def _runtime(self, probe, llm, slots):
        registry = ToolRegistry([probe])
        agent = build_agent(llm, registry, retry_delay_ms=0, miclaw_slots=slots)
        return Runtime(agent, max_inflight=2, config={"recursion_limit": 80})

    def _llm(self):
        """按提示词里的请求原文给出各自的计划：请求 A 调 a1、a2，请求 B 调 b1、b2。"""
        by_request = {"A": llm_for(two_probes("a")), "B": llm_for(two_probes("b"))}

        def responder(msgs):
            which = "B" if any("请求B" in m.content for m in msgs) else "A"
            return by_request[which]._responder(msgs)
        return FakeLLM(responder=responder)

    def test_miclaw_quota_is_shared_across_requests(self):
        probe = Probe()
        with self._runtime(probe, self._llm(), SlotPool(2)) as rt:
            fa, fb = rt.submit("请求A", session="u1"), rt.submit("请求B", session="u2")
            assert fa.result(WAIT)["failure"] is None
            assert fb.result(WAIT)["failure"] is None
        # 每个请求一轮派发两个 MiClaw 调用；不共用配额时同时在飞的会是四个
        assert len(probe.keys) == 4
        assert probe.peak.peak <= 2

    def test_request_key_reaches_parallel_branches(self):
        probe = Probe()
        with self._runtime(probe, self._llm(), SlotPool(2)) as rt:
            rt.submit("请求A", session="u1", priority=1).result(WAIT)
            rt.submit("请求B", session="u2", priority=7).result(WAIT)
        assert probe.keys["a1"] == probe.keys["a2"] and probe.keys["a1"][0] == -1
        assert probe.keys["b1"] == probe.keys["b2"] and probe.keys["b1"][0] == -7

    def test_model_calls_are_arbitrated_across_requests(self):
        peak, inner = Peak(), self._llm()

        def responder(msgs):
            with peak:
                time.sleep(0.01)
                return inner._responder(msgs)

        llm = FakeLLM(responder=responder)
        llm.slots = SlotPool(1)
        with self._runtime(Probe(), llm, SlotPool(2)) as rt:
            futures = [rt.submit(f"请求{c}", session=c) for c in "AB"]
            for f in futures:
                assert f.result(WAIT)["failure"] is None
        assert peak.peak == 1
