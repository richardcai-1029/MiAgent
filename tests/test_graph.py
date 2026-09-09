"""图节点与端到端测试（任务 DAG 版）。"""

import json
import re

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent, dag, initial_state
from miagent.graph.nodes import Deps, evaluator, scheduler
from miagent.graph.routers import route_after_evaluator, route_after_scheduler
from langgraph.types import Send

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
        assert [d["task"]["id"] for d in out["dispatch"]] == ["A"]
        assert out["tasks"]["A"]["status"] is TaskStatus.RUNNING

    def test_routes_by_tool_source(self, registry):
        local = {"A": new_task("A", "", "echo", {"text": "x"})}
        remote = {"A": new_task("A", "", "system.query_weather", {"when": "今晚"})}
        d = self._deps(registry)
        assert scheduler(self._state(local), d)["dispatch"][0]["route"] == "local"
        assert scheduler(self._state(remote), d)["dispatch"][0]["route"] == "miclaw"

    def test_completion_detected(self, registry):
        tasks = {"A": new_task("A", "", "echo")}
        tasks["A"]["status"] = TaskStatus.DONE
        out = scheduler(self._state(tasks), self._deps(registry))
        # 节点返回的是【部分更新】：没返回 failure 键 = 不改动该字段
        assert out["dispatch"] == [] and out.get("failure") is None

    def test_cascades_failure_instead_of_deadlocking(self, registry):
        """依赖失败的任务会被级联标记，调度得以收敛而非死等。"""
        tasks = {"A": new_task("A", "", "echo"),
                 "B": new_task("B", "", "echo", dependencies=["A"])}
        tasks["A"]["status"] = TaskStatus.FAILED
        out = scheduler(self._state(tasks), self._deps(registry))
        assert out["tasks"]["B"]["status"] is TaskStatus.FAILED
        assert out["dispatch"] == []
        assert out.get("failure") is None   # 级联后是"已完成"，不是死锁


    def test_dispatches_all_independent_tasks_at_once(self, registry):
        """互不依赖的任务同一轮全部派发 —— 这是并行的前提。"""
        tasks = {x: new_task(x, "", "echo", {"text": x}) for x in "ABC"}
        out = scheduler(self._state(tasks), self._deps(registry))
        assert {d["task"]["id"] for d in out["dispatch"]} == {"A", "B", "C"}

    def test_miclaw_concurrency_is_capped(self, registry):
        """MiClaw 侧受握手下发的并发配额限制（清单 C-6）；超出的顺延到下一轮。"""
        tasks = {x: new_task(x, "", "system.query_weather", {"when": x}) for x in "ABCD"}
        deps = Deps(llm=FakeLLM(script=[]), registry=registry, max_concurrent_miclaw=2)
        out = scheduler(self._state(tasks), deps)
        assert len(out["dispatch"]) == 2
        assert "2 个因并发配额顺延" in out["trace"][0]

    def test_local_tools_are_not_capped(self, registry):
        """本地工具不占系统资源配额，也受 GIL 限制并发无收益，故不限流。"""
        tasks = {x: new_task(x, "", "echo", {"text": x}) for x in "ABCD"}
        deps = Deps(llm=FakeLLM(script=[]), registry=registry, max_concurrent_miclaw=1)
        assert len(scheduler(self._state(tasks), deps)["dispatch"]) == 4


class TestEvaluator:
    """判定不依赖模型 —— 结论由错误码的段位决定。"""

    def _run(self, ok, policy, code=None, attempt=1, replans=0, executions=0):
        s = initial_state("t")
        s["tasks"] = {"A": new_task("A", "", "echo")}
        s["replan_count"] = replans
        s["execution_count"] = executions
        s["outcomes"] = [TaskOutcome(task_id="A", tool="echo", ok=ok, content="c",
                                     error_code=code, retry_policy=policy, attempt=attempt)]
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


    def test_aggregates_multiple_outcomes(self):
        """并行时一轮可能有多条结果：任一需要重规划，整轮就走 Replanner。"""
        s = initial_state("t")
        s["tasks"] = {"A": new_task("A", "", "echo"), "B": new_task("B", "", "echo")}
        s["outcomes"] = [
            TaskOutcome(task_id="A", tool="echo", ok=True, content="ok",
                        error_code=None, retry_policy="none", attempt=1),
            TaskOutcome(task_id="B", tool="echo", ok=False, content="bad",
                        error_code="MC-4001", retry_policy="degrade", attempt=1),
        ]
        out = evaluator(s, Deps(llm=FakeLLM(script=[]), registry=ToolRegistry()))
        assert out["verdict"] == "replan"                       # 优先级最高的那个
        assert out["tasks"]["A"]["status"] is TaskStatus.DONE   # 成功的照常落库
        assert out["tasks"]["B"]["status"] is TaskStatus.FAILED

    def test_clears_outcomes_after_consuming(self):
        """消费完必须清空，否则下一轮会重复处理同一批结果。"""
        s = initial_state("t")
        s["tasks"] = {"A": new_task("A", "", "echo")}
        s["outcomes"] = [TaskOutcome(task_id="A", tool="echo", ok=True, content="ok",
                                     error_code=None, retry_policy="none", attempt=1)]
        out = evaluator(s, Deps(llm=FakeLLM(script=[]), registry=ToolRegistry()))
        assert out["outcomes"] == []


class TestRouters:
    def test_after_scheduler_no_work(self):
        assert route_after_scheduler({"dispatch": []}) == "finalizer"

    def test_after_scheduler_fans_out(self):
        """返回 Send 列表即并行派发；本地与 MiClaw 可在同一轮扇出到不同节点。"""
        t = new_task("A", "", "echo")
        sends = route_after_scheduler({"dispatch": [
            {"task": t, "route": "local"}, {"task": t, "route": "miclaw"}]})
        assert [x.node for x in sends] == ["local_tool", "mcp_executor"]
        assert all(isinstance(x, Send) for x in sends)

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
        order = [l for l in out["trace"] if "→ t" in l]
        assert [re.search(r"→ (t\d)", l).group(1) for l in order] == ["t1", "t2", "t3"]

    def test_diamond_exposes_parallel_layer(self, registry):
        """菱形依赖：第二层两个任务同时就绪（当前串行执行，但机会被识别）。"""
        out = run(registry, "并行查天气和餐厅", llm_for(plan(
            task("t1", "system.query_calendar", when="今晚"),
            task("t2", "system.query_weather", ["t1"], when="今晚"),
            task("t3", "system.search_nearby", ["t1"], category="餐厅"),
            task("t4", "system.create_event", ["t2", "t3"], title="晚餐", when="19:00"))))
        assert out["execution_summary"]["parallel_layers"] == \
            [["t1"], ["t2", "t3"], ["t4"]]
        assert any("本轮就绪 2 个，派发 2 个" in l for l in out["trace"])

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


# ============================================================
# 约束守卫：模型的调用范围
# ============================================================


def _functions_touching(attr_owner: str, attr: str, path) -> set[str]:
    """AST 扫描：找出哪些顶层函数里出现了 `attr_owner.attr` 形式的访问。"""
    import ast

    hits = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and node.attr == attr
                    and isinstance(node.value, ast.Name) and node.value.id == attr_owner):
                hits.add(fn.name)
    return hits


def test_model_is_confined_to_planning_nodes():
    """模型只允许在规划与收尾环节被调用。

    调度、执行、评估三个环节要回答的问题都有确定答案——依赖是否满足、
    是否超出重试上限、是否全部完成、该重试还是该换方案。用模型去猜一个
    我们确定知道答案的问题，既慢又不可复现，端侧尤其付不起这个代价。

    这条约束此前只写在文档里。文档约束不会在被破坏时报警，故在此固化：
    新加的节点若持有模型引用，本用例立即失败。
    """
    from pathlib import Path

    nodes_py = Path(__file__).resolve().parent.parent / "miagent" / "graph" / "nodes.py"
    touching = _functions_touching("deps", "llm", nodes_py)

    # _plan 是 Planner 与 Replanner 共用的规划实现；finalizer 生成给用户的回答。
    assert touching == {"_plan", "finalizer"}, (
        f"模型调用范围发生变化，当前出现在 {sorted(touching)}。"
        "调度、执行、评估环节的判断均有确定答案，必须由纯函数承担。"
    )
