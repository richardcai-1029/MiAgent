"""图节点与端到端测试（任务 DAG 版）。"""

import json

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent, dag, initial_state
from miagent.graph.nodes import Deps, evaluator, scheduler
from miagent.graph.routers import route_after_evaluator, route_after_scheduler
from miagent.graph.state import MAX_REPLANS, TaskOutcome, TaskStatus, new_task
from miagent.llm import FakeLLM
from miagent.mock_server import MiClawMockServer
from miagent.protocol import ErrorCode
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "location.fine", "screen.capture"]


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
    client.connect("test", PERMS)
    reg = ToolRegistry([echo])
    reg.load_from_miclaw(client)
    yield reg
    client.close()


def task(tid, tool_name, deps=None, **args):
    return {"id": tid, "description": f"任务 {tid}", "required_tool": tool_name,
            "dependencies": deps or [], "arguments": args}


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def llm_for(first, replan=None, answer="完成"):
    def responder(msgs):
        role = msgs[0].content
        if "重规划器" in role:
            return replan or plan()
        if "规划器" in role:
            return first
        return answer
    return FakeLLM(responder=responder)


def run(registry, request, llm):
    return build_agent(llm, registry).invoke(initial_state(request),
                                             {"recursion_limit": 80})


class TestSchedulerIsDeterministic:
    """Scheduler 全程不调用模型 —— 依赖解析有确定答案。"""

    def _deps(self, registry):
        return Deps(llm=FakeLLM(script=[]), registry=registry)

    def _state(self, tasks):
        s = initial_state("t")
        s["tasks"] = tasks
        return s

    def test_dispatches_only_dependency_free_task(self, registry):
        tasks = {"A": new_task("A", "", "echo", {"text": "a"}),
                 "B": new_task("B", "", "echo", {"text": "b"}, ["A"])}
        out = scheduler(self._state(tasks), self._deps(registry))
        assert out["current"]["id"] == "A"
        assert out["tasks"]["A"]["status"] is TaskStatus.RUNNING

    def test_routes_by_tool_source(self, registry):
        local = {"A": new_task("A", "", "echo", {"text": "x"})}
        remote = {"A": new_task("A", "", "system.query_weather", {"when": "今晚"})}
        d = self._deps(registry)
        assert scheduler(self._state(local), d)["route"] == "local"
        assert scheduler(self._state(remote), d)["route"] == "miclaw"

    def test_completion_detected(self, registry):
        tasks = {"A": new_task("A", "", "echo")}
        tasks["A"]["status"] = TaskStatus.DONE
        out = scheduler(self._state(tasks), self._deps(registry))
        # 节点返回的是【部分更新】：没返回 failure 键 = 不改动该字段
        assert out["current"] is None and out.get("failure") is None

    def test_cascades_failure_instead_of_deadlocking(self, registry):
        """依赖失败的任务会被级联标记，调度得以收敛而非死等。"""
        tasks = {"A": new_task("A", "", "echo"),
                 "B": new_task("B", "", "echo", dependencies=["A"])}
        tasks["A"]["status"] = TaskStatus.FAILED
        out = scheduler(self._state(tasks), self._deps(registry))
        assert out["tasks"]["B"]["status"] is TaskStatus.FAILED
        assert out["current"] is None
        assert out.get("failure") is None   # 级联后是"已完成"，不是死锁


class TestEvaluator:
    """判定不依赖模型 —— 结论由错误码的段位决定。"""

    def _run(self, ok, policy, code=None, attempt=1, replans=0, executions=0):
        s = initial_state("t")
        s["tasks"] = {"A": new_task("A", "", "echo")}
        s["replan_count"] = replans
        s["execution_count"] = executions
        s["last"] = TaskOutcome(task_id="A", tool="echo", ok=ok, content="c",
                                error_code=code, retry_policy=policy, attempt=attempt)
        return evaluator(s, Deps(llm=FakeLLM(script=[]), registry=ToolRegistry()))

    def test_success_marks_done(self):
        out = self._run(True, "none")
        assert out["verdict"] == "success"
        assert out["tasks"]["A"]["status"] is TaskStatus.DONE

    def test_band_1_retries_and_returns_task_to_pending(self):
        """重试要把任务退回 pending，Scheduler 下一轮才会重新派发它。"""
        out = self._run(False, "backoff", "MC-1003")
        assert out["verdict"] == "retry"
        assert out["tasks"]["A"]["status"] is TaskStatus.PENDING

    def test_band_4_replans_without_retrying(self):
        out = self._run(False, "degrade", "MC-4001")
        assert out["verdict"] == "replan"
        assert out["tasks"]["A"]["status"] is TaskStatus.FAILED

    def test_replan_limit_aborts(self):
        out = self._run(False, "degrade", "MC-4001", replans=MAX_REPLANS)
        assert out["verdict"] == "abort"
        assert out["failure"] == ErrorCode.AG_PLAN_NO_PROGRESS.value

    def test_execution_budget_aborts(self):
        out = self._run(False, "none", "AG-2001", executions=999)
        assert out["failure"] == ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value


class TestRouters:
    def test_after_scheduler(self):
        assert route_after_scheduler({"current": None}) == "finalizer"
        assert route_after_scheduler({"current": {}, "route": "local"}) == "local_tool"
        assert route_after_scheduler({"current": {}, "route": "miclaw"}) == "mcp_executor"

    def test_after_evaluator(self):
        for v, node in [("success", "scheduler"), ("retry", "scheduler"),
                        ("replan", "replanner"), ("abort", "finalizer")]:
            assert route_after_evaluator({"verdict": v}) == node


class TestEndToEnd:
    def test_linear_dependency_chain(self, registry):
        """A → B → C：严格按依赖顺序执行。"""
        out = run(registry, "查日历后订餐并建日程", llm_for(plan(
            task("t1", "system.query_calendar", when="今晚"),
            task("t2", "system.search_nearby", ["t1"], category="餐厅"),
            task("t3", "system.create_event", ["t2"], title="晚餐", when="19:00"))))
        assert out["failure"] is None
        order = [l for l in out["trace"] if l.startswith("scheduler: 派发")]
        assert [x.split()[2] for x in order] == ["t1", "t2", "t3"]

    def test_diamond_exposes_parallel_layer(self, registry):
        """菱形依赖：第二层两个任务同时就绪（当前串行执行，但机会被识别）。"""
        out = run(registry, "并行查天气和餐厅", llm_for(plan(
            task("t1", "system.query_calendar", when="今晚"),
            task("t2", "system.query_weather", ["t1"], when="今晚"),
            task("t3", "system.search_nearby", ["t1"], category="餐厅"),
            task("t4", "system.create_event", ["t2", "t3"], title="晚餐", when="19:00"))))
        assert out["execution_summary"]["parallel_layers"] == \
            [["t1"], ["t2", "t3"], ["t4"]]
        assert any("2 个任务就绪" in l for l in out["trace"])

    def test_replan_preserves_completed_work(self, registry):
        """重规划不能让已成功的任务重做 —— 端侧每次重做都是真实系统调用。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        assert out["failure"] is None
        assert out["tasks"]["t1"]["status"] is TaskStatus.DONE
        assert out["tasks"]["t1"]["retry_count"] == 1        # 只跑过一次
        assert "r1_t2b" in out["tasks"]

    def test_replan_keeps_failure_history(self, registry):
        """失败的任务保留在图里，否则用户看不到试过什么。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.book_restaurant", name="小馆 A")),
            replan=plan(task("t1b", "system.book_restaurant", name="小馆 B"))))
        assert out["tasks"]["t1"]["status"] is TaskStatus.FAILED
        assert out["tasks"]["t1"]["error"] == "MC-4002"
        assert out["execution_summary"]["failed"] == ["t1"]

    def test_cyclic_plan_rejected_before_execution(self, registry):
        """带环的任务图在规划阶段即被拒，一步都不执行。"""
        out = run(registry, "绕圈", llm_for(plan(
            task("a", "system.query_weather", ["b"], when="今晚"),
            task("b", "system.query_calendar", ["a"], when="今晚"))))
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value
        assert out["execution_count"] == 0
        assert out["final_answer"]

    def test_dangling_dependency_rejected(self, registry):
        out = run(registry, "依赖不存在的任务", llm_for(plan(
            task("a", "system.query_weather", ["nope"], when="今晚"))))
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value

    def test_hallucinated_tool_rejected_at_planning(self, registry):
        """工具名进了 schema 的 enum，幻觉在解析阶段就被拒。"""
        out = run(registry, "打开微信", llm_for(plan(task("a", "system.open_wechat"))))
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert out["execution_count"] == 0

    def test_transient_failure_retries_same_task(self, registry):
        """MC-1003 → 段位 1 → 重试同一个任务，不重规划。"""
        from miagent.mock_server.tools import reset_flaky
        reset_flaky()
        out = run(registry, "同步设置", llm_for(plan(task("t1", "system.sync_settings"))))
        assert out["failure"] is None
        assert out["tasks"]["t1"]["retry_count"] == 2
        assert out["replan_count"] == 0

    def test_unplannable_request_still_answers(self, registry):
        out = run(registry, "随便说说", llm_for("我不知道该怎么办"))
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert out["final_answer"]

    def test_empty_plan_goes_straight_to_finalizer(self, registry):
        out = run(registry, "你好", llm_for(plan(), answer="你好，有什么可以帮你？"))
        assert out["tasks"] == {} and out["final_answer"]
