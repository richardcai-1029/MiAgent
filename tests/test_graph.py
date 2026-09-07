"""Agent 图测试：节点逻辑、路由、以及三条完整路径。"""

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent, initial_state, parse_plan
from miagent.graph.nodes import Deps, evaluator, scheduler
from miagent.graph.routers import route_after_evaluator, route_after_scheduler
from miagent.graph.state import MAX_REPLANS, Step, StepResult
from miagent.llm import FakeLLM, plan_json, step
from miagent.mock_server import MiClawMockServer
from miagent.mock_server.tools import reset_flaky
from miagent.protocol import AgentError, ErrorCode
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport


@tool()
def echo(text: str) -> str:
    """原样返回输入。

    Args:
        text: 任意文本
    """
    return text


@pytest.fixture
def registry():
    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("test", ["alarm.write", "screen.capture"])
    reset_flaky()
    reg = ToolRegistry([echo])
    reg.load_from_miclaw(client)
    yield reg
    client.close()


def llm_for(first, replan=None, answer="完成"):
    def responder(msgs):
        role = msgs[0].content
        if "重规划器" in role:
            return replan or plan_json()
        if "规划器" in role:
            return first
        return answer
    return FakeLLM(responder=responder)


def run(registry, task, llm):
    return build_agent(llm, registry).invoke(initial_state(task),
                                             {"recursion_limit": 60})


class TestParsePlan:
    def test_plain_json(self):
        p = parse_plan('{"steps":[{"tool":"echo","arguments":{"text":"hi"}}]}')
        assert p[0]["tool"] == "echo" and p[0]["id"] == 0

    def test_code_fence_is_stripped(self):
        """真实模型常把 JSON 包在 ```json 里。"""
        assert parse_plan('```json\n{"steps":[{"tool":"echo"}]}\n```')[0]["tool"] == "echo"

    def test_garbage_raises_ag_1001(self):
        """解析失败就失败，不猜 —— 猜出来的计划会让 Agent 做用户没要求的事。"""
        with pytest.raises(AgentError) as ei:
            parse_plan("我觉得应该先查一下电量")
        assert ei.value.code is ErrorCode.AG_PLAN_PARSE_FAILED

    def test_step_without_tool_raises(self):
        with pytest.raises(AgentError):
            parse_plan('{"steps":[{"reason":"忘了写 tool"}]}')


class TestScheduler:
    def _deps(self, registry):
        return Deps(llm=FakeLLM(script=[]), registry=registry)

    def test_routes_local_and_miclaw(self, registry):
        d = self._deps(registry)
        base = initial_state("t")
        base["plan"] = [Step(id=0, tool="echo", arguments={}, reason=""),
                        Step(id=1, tool="system.get_battery", arguments={}, reason="")]
        assert scheduler({**base, "cursor": 0}, d)["route"] == "local"
        assert scheduler({**base, "cursor": 1}, d)["route"] == "miclaw"

    def test_unknown_tool_routes_local(self, registry):
        """模型幻觉出的工具走本地路径，由注册表统一报 AG-2001。"""
        base = initial_state("t")
        base["plan"] = [Step(id=0, tool="system.nonexistent", arguments={}, reason="")]
        assert scheduler(base, self._deps(registry))["route"] == "local"

    def test_success_advances_cursor(self, registry):
        base = initial_state("t")
        base["plan"] = [Step(id=0, tool="echo", arguments={}, reason=""),
                        Step(id=1, tool="echo", arguments={}, reason="")]
        assert scheduler({**base, "verdict": "success"}, self._deps(registry))["cursor"] == 1

    def test_retry_keeps_cursor(self, registry):
        base = initial_state("t")
        base["plan"] = [Step(id=0, tool="echo", arguments={}, reason="")]
        assert scheduler({**base, "verdict": "retry"}, self._deps(registry))["cursor"] == 0

    def test_done_when_cursor_past_plan(self, registry):
        base = initial_state("t")
        base["plan"] = [Step(id=0, tool="echo", arguments={}, reason="")]
        assert scheduler({**base, "cursor": 1}, self._deps(registry))["current"] is None


class TestEvaluator:
    """判定不依赖大模型 —— 结论由错误码的段位决定。"""

    def _run(self, ok, policy, code=None, attempt=1, replans=0, done=0):
        s = initial_state("t")
        s["results"] = [None] * done
        s["replan_count"] = replans
        s["last"] = StepResult(step_id=0, tool="x", ok=ok, content="",
                               error_code=code, retry_policy=policy, attempt=attempt)
        return evaluator(s, Deps(llm=FakeLLM(script=[]), registry=ToolRegistry()))

    def test_success(self):
        assert self._run(True, "none")["verdict"] == "success"

    def test_band_1_retries(self):
        """MC-1xxx 传输抖动 -> BACKOFF -> 重试。"""
        assert self._run(False, "backoff", "MC-1003")["verdict"] == "retry"

    def test_retry_exhausted_becomes_replan(self):
        assert self._run(False, "backoff", "MC-1003", attempt=2)["verdict"] == "replan"

    def test_band_4_replans_without_retrying(self):
        """MC-4xxx 资源不足 -> DEGRADE -> 直接换方案，重试没意义。"""
        assert self._run(False, "degrade", "MC-4001")["verdict"] == "replan"

    def test_replan_limit_aborts(self):
        r = self._run(False, "degrade", "MC-4001", replans=MAX_REPLANS)
        assert r["verdict"] == "abort"
        assert r["failure"] == ErrorCode.AG_PLAN_NO_PROGRESS.value

    def test_total_step_limit_aborts(self):
        r = self._run(False, "none", "AG-2001", done=50)
        assert r["verdict"] == "abort"
        assert r["failure"] == ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value


class TestRouters:
    """路由只分派、不决策 —— 依据都由上游节点算好写进 State。"""

    def test_after_scheduler(self):
        assert route_after_scheduler({"current": None}) == "finalizer"
        assert route_after_scheduler({"current": {}, "route": "local"}) == "local_tool"
        assert route_after_scheduler({"current": {}, "route": "miclaw"}) == "mcp_executor"

    def test_after_evaluator(self):
        assert route_after_evaluator({"verdict": "success"}) == "scheduler"
        assert route_after_evaluator({"verdict": "retry"}) == "scheduler"
        assert route_after_evaluator({"verdict": "replan"}) == "replanner"
        assert route_after_evaluator({"verdict": "abort"}) == "finalizer"


class TestEndToEnd:
    def test_all_success(self, registry):
        out = run(registry, "查电量并设闹钟", llm_for(plan_json(
            step("system.get_battery"),
            step("system.set_alarm", time="08:00"))))
        assert out["failure"] is None
        assert [r["ok"] for r in out["results"]] == [True, True]

    def test_transient_failure_retries(self, registry):
        """MC-1003 -> 段位 1 -> 重试 -> 第二次成功。"""
        out = run(registry, "同步设置",
                  llm_for(plan_json(step("system.sync_settings"))))
        assert [r["error_code"] for r in out["results"]] == ["MC-1003", None]
        assert out["failure"] is None

    def test_resource_failure_replans(self, registry):
        """MC-4001 -> 段位 4 -> 不重试，换轻量方案。"""
        out = run(registry, "截屏",
                  llm_for(plan_json(step("system.capture_screen")),
                          replan=plan_json(step("system.get_battery"))))
        assert out["results"][0]["error_code"] == "MC-4001"
        assert out["results"][-1]["ok"] is True

    def test_attempts_reset_after_replan(self, registry):
        """回归测试：attempts 以 step id 为键，重规划后 id 从 0 重来，
        不清空的话新步骤会继承旧计数，凭空少一次重试机会。"""
        out = run(registry, "截屏",
                  llm_for(plan_json(step("system.capture_screen")),
                          replan=plan_json(step("system.sync_settings"))))
        after_replan = [r for r in out["results"] if r["tool"] == "system.sync_settings"]
        assert after_replan[0]["attempt"] == 1

    def test_hallucinated_tool_rejected_at_planning(self, registry):
        """工具名被收进 schema 的 enum，幻觉在【规划阶段】就被拒 ——
        不必白跑一步再报 AG-2001。"""
        out = run(registry, "做一件做不到的事",
                  llm_for(plan_json(step("system.does_not_exist"))))
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert out["results"] == []   # 一步都没执行，省下了一次工具调用
        assert out["answer"]          # 仍然给用户一个交代，而不是抛异常

    def test_repeated_execution_failure_aborts(self, registry):
        """工具存在但一直失败 -> 重规划次数耗尽 -> 中止并给出错误码。"""
        always_fail = plan_json(step("system.capture_screen"))
        out = run(registry, "反复截屏", llm_for(always_fail, replan=always_fail))
        assert out["failure"] == ErrorCode.AG_PLAN_NO_PROGRESS.value
        assert out["answer"]

    def test_planner_failure_does_not_crash_the_graph(self, registry):
        """模型输出完全不合 schema 时，用户仍应收到解释而非异常。"""
        out = run(registry, "随便问问", llm_for("我觉得你应该自己看一下"))
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert out["answer"]

    def test_empty_plan_goes_straight_to_finalizer(self, registry):
        out = run(registry, "你好", llm_for(plan_json(), answer="你好，有什么可以帮你？"))
        assert out["results"] == [] and out["answer"]
