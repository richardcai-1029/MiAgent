"""图节点与端到端测试（任务 DAG 版）。"""

import json
import re

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent, dag, initial_state
from miagent.graph.nodes import (Deps, evaluator, execute, finalizer,
                                 planner, scheduler, _tools_section)
from miagent.graph.routers import route_after_evaluator, route_after_scheduler
from langgraph.types import Send

from miagent.graph.state import (MAX_REPLANS, MAX_TOTAL_EXECUTIONS,
                                 TaskOutcome, TaskStatus, new_task)
from miagent.llm import FakeLLM
from miagent.memory import episodic
from miagent.mock_server import MiClawMockServer
from miagent.protocol import ErrorCode
from miagent.tools import ToolRegistry, ToolSource, tool
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "location.fine", "screen.capture"]


@tool()
def echo(text: str) -> str:
    """原样返回输入。

    Args:
        text: 任意文本
    """
    return text


@tool()
def join(left: str, right: str) -> str:
    """把两段文本拼起来。

    Args:
        left: 左边一段
        right: 右边一段
    """
    return f"{left}|{right}"


@pytest.fixture
def registry():
    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("test", PERMS)
    reg = ToolRegistry([echo, join])
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
    # 退避设为 0：重试行为由 TestRetryBackoff 单独验证，
    # 端到端用例不必为此真的等待。
    return build_agent(llm, registry, retry_delay_ms=0).invoke(
        initial_state(request), {"recursion_limit": 80})


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
        assert "t1" in out["execution_summary"]["completed"]
        assert sum("t1 ✓" in l for l in out["trace"]) == 1     # 只跑过一次
        assert "r1_t2b" in out["tasks"]

    def test_replan_keeps_failure_history(self, registry):
        """失败的任务进入情景记忆，否则用户看不到试过什么。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.book_restaurant", name="小馆 A")),
            replan=plan(task("t1b", "system.book_restaurant", name="小馆 B"))))
        failed = [e for e in out["episodes"] if not e["ok"]]
        assert [e["task_id"] for e in failed] == ["t1"]
        assert failed[0]["error"] == "MC-4002"
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


# ============================================================
# 多工具联动：任务间的数据流
# ============================================================

REF = "$from"


class TestToolChaining:
    """上游工具的结果作为下游工具的参数 —— 缺了它，多个工具只是多次
    互不相干的调用，构不成联动。"""

    def test_downstream_receives_upstream_result(self, registry):
        """t1 查电量 → t2 拿到 t1 的结果原文。"""
        out = run(registry, "看看电量并复述", llm_for(plan(
            task("t1", "system.get_battery"),
            task("t2", "echo", text={REF: "t1"}),
        )))
        assert out["failure"] is None
        assert out["tasks"]["t1"]["result"] == "电量 63%，未在充电"
        assert out["tasks"]["t2"]["result"] == "电量 63%，未在充电"

    def test_reference_alone_serializes_execution(self, registry):
        """只写引用、不写 dependencies，两个任务也必须分两层执行。

        引用即依赖 —— 否则 t2 会与 t1 同轮派发，取到还不存在的结果。
        """
        out = run(registry, "看看电量并复述", llm_for(plan(
            task("t1", "system.get_battery"),
            task("t2", "echo", text={REF: "t1"}),
        )))
        assert out["tasks"]["t2"]["dependencies"] == ["t1"]
        layers = dag.parallel_layers(out["tasks"])
        assert layers == [["t1"], ["t2"]]

    def test_original_arguments_keep_the_reference(self, registry):
        """求值只作用于派发出去的副本，任务图里存的仍是引用本身，
        重试时会重新求值。"""
        out = run(registry, "看看电量并复述", llm_for(plan(
            task("t1", "system.get_battery"),
            task("t2", "echo", text={REF: "t1"}),
        )))
        assert out["tasks"]["t2"]["arguments"] == {"text": {REF: "t1"}}

    def test_reference_to_missing_task_is_rejected_before_execution(self, registry):
        """引用不存在的任务 = 依赖缺失，在执行任何一步之前拦下。"""
        out = run(registry, "复述", llm_for(plan(
            task("t2", "echo", text={REF: "nope"}),
        )))
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value
        assert out["execution_count"] == 0

    def test_reference_cycle_is_rejected_before_execution(self, registry):
        out = run(registry, "复述", llm_for(plan(
            task("t1", "echo", text={REF: "t2"}),
            task("t2", "echo", text={REF: "t1"}),
        )))
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value
        assert out["execution_count"] == 0

    def test_parallel_branches_feed_one_downstream_task(self, registry):
        """扇入：两个并行任务的结果同时喂给下游一个任务。"""
        out = run(registry, "汇总", llm_for(plan(
            task("t1", "echo", text="A"),
            task("t2", "echo", text="B"),
            task("t3", "join", left={REF: "t1"}, right={REF: "t2"}),
        )))
        assert out["failure"] is None
        assert out["tasks"]["t3"]["status"] is TaskStatus.DONE
        assert out["tasks"]["t3"]["result"] == "A|B"
        assert dag.parallel_layers(out["tasks"]) == [["t1", "t2"], ["t3"]]


class TestRetryBackoff:
    """重试前先退避。RetryPolicy.BACKOFF 要求「退避后重试」，
    此前是下一轮立即重发 —— 策略名存实亡。"""

    def _deps(self, registry, waits):
        return Deps(llm=FakeLLM(script=[]), registry=registry,
                    retry_delay_ms=250, sleep=waits.append)

    def test_first_attempt_does_not_wait(self, registry):
        waits = []
        payload = {"task": new_task("t1", "任务 t1", "echo", {"text": "x"})}
        out = execute(payload, self._deps(registry, waits), ToolSource.LOCAL)
        assert waits == []
        assert out["outcomes"][0]["ok"]

    def test_retry_waits_before_reissuing(self, registry):
        waits = []
        task = new_task("t1", "任务 t1", "echo", {"text": "x"})
        task["retry_count"] = 1                 # 已失败过一次，本次是重试
        out = execute({"task": task}, self._deps(registry, waits), ToolSource.LOCAL)
        assert waits == [0.25], "重试没有退避，或退避时长换算错误"
        assert out["outcomes"][0]["attempt"] == 2

    def test_wait_is_recorded_in_the_trace(self, registry):
        waits = []
        task = new_task("t1", "任务 t1", "echo", {"text": "x"})
        task["retry_count"] = 1
        out = execute({"task": task}, self._deps(registry, waits), ToolSource.LOCAL)
        assert "退避 250ms" in out["trace"][0]


class TestContextBudget:
    """端侧窗口小，超限时削减低优先级片段而不是直接拒绝。"""

    def test_tool_descriptions_compact_to_names(self, registry):
        """完整 schema 放不下时保留工具名：模型仍能选对工具，
        整段丢掉则连选都无从选起。"""
        from miagent.llm.context import Section, fit

        body, notes = fit([_tools_section(registry), Section("目标", "目标：查电量")],
                          limit=300, estimate=len)
        assert notes == ["工具描述→紧凑形式"]
        assert "system.get_battery" in body, "工具名应当保留"
        assert "inputSchema" not in body, "完整 schema 应当已被削减掉"

    def test_planner_reports_overflow_instead_of_crashing(self, registry):
        """削减到不可裁片段仍放不下时，作为一次可解释的失败返回，
        而不是让异常炸掉整张图。"""
        llm = llm_for(plan())
        llm.context_limit = 10
        out = planner(initial_state("查一下电量"), Deps(llm=llm, registry=registry))
        assert out["failure"] == ErrorCode.AG_CONTEXT_OVERFLOW.value
        assert out["tasks"] == {}

    def test_finalizer_falls_back_to_deterministic_answer(self, registry):
        """回答不该因为窗口放不下而缺席。"""
        def must_not_be_called(_messages):
            raise AssertionError("窗口放不下时不应该再去调模型")

        llm = FakeLLM(responder=must_not_be_called, context_limit=10)
        state = initial_state("查一下电量")
        t = new_task("t1", "查电量", "system.get_battery")
        t["status"], t["result"] = TaskStatus.DONE, "电量 63%"
        state["tasks"] = {"t1": t}

        out = finalizer(state, Deps(llm=llm, registry=registry))
        assert out["final_answer"].startswith("已完成 1 项")
        assert "上下文放不下" in out["trace"][0]

    def test_no_trimming_note_when_everything_fits(self, registry):
        deps = Deps(llm=llm_for(plan(task("t1", "echo", text="hi"))), registry=registry)
        out = planner(initial_state("原样返回 hi"), deps)
        assert "上下文削减" not in out["trace"][0]


class TestPlanSizeIsChecked:
    """拆得过细会白白消耗端侧算力，每一步都是一次真实调用。
    上限直接取累计执行预算，不另立数字：待执行任务数超过预算的计划
    在预算内必然跑不完。"""

    def _oversized(self):
        return plan(*[task(f"t{i}", "echo", text=str(i))
                      for i in range(MAX_TOTAL_EXECUTIONS + 1)])

    def test_oversized_plan_is_rejected_before_any_execution(self, registry):
        out = run(registry, "做很多事", llm_for(self._oversized()))
        assert out["failure"] == ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value
        assert out["execution_count"] == 0, "被拒的计划不应该已经执行掉几步"
        assert out["tasks"] == {}

    def test_plan_at_the_budget_is_accepted(self, registry):
        """恰好等于预算的计划仍然可执行 —— 判据是「超过」而非「接近」。"""
        out = run(registry, "做很多事", llm_for(
            plan(*[task(f"t{i}", "echo", text=str(i))
                   for i in range(MAX_TOTAL_EXECUTIONS)])))
        assert out["failure"] is None
        assert len(out["tasks"]) == MAX_TOTAL_EXECUTIONS

    def test_rejected_plan_still_answers_the_user(self, registry):
        out = run(registry, "做很多事", llm_for(self._oversized()))
        assert out["final_answer"]


class TestExecutionBudgetBoundsSuccessToo:
    """累计执行预算此前只在失败分支里检查，因而完全约束不住顺利执行的流程。
    预算要能兜住的恰恰是「模型拆出多少就执行多少」这种情况。"""

    def _deps(self, registry):
        return Deps(llm=FakeLLM(script=[]), registry=registry)

    def test_scheduler_stops_dispatching_at_the_budget(self, registry):
        state = initial_state("x")
        state["tasks"] = {"t1": new_task("t1", "任务 t1", "echo", {"text": "a"})}
        state["execution_count"] = MAX_TOTAL_EXECUTIONS

        out = scheduler(state, self._deps(registry))
        assert out["dispatch"] == []
        assert out["failure"] == ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value

    def test_below_the_budget_still_dispatches(self, registry):
        state = initial_state("x")
        state["tasks"] = {"t1": new_task("t1", "任务 t1", "echo", {"text": "a"})}
        state["execution_count"] = MAX_TOTAL_EXECUTIONS - 1

        out = scheduler(state, self._deps(registry))
        assert len(out["dispatch"]) == 1


class TestCompletionIsVerified:
    """「所有任务都到了终态」不等于「目标达成」。"""

    def test_replan_that_produces_nothing_is_reported_as_unmet(self, registry):
        """重规划是恢复机制，它跑完却一个新任务都没产出，说明恢复没发生，
        失败任务不会再有人接手 —— 照常收尾会给出声称完成实则漏做的回答。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "join", left="只给了一个参数")),
            replan=plan()))
        assert out["execution_summary"]["failed"] == ["t2"]
        assert out["failure"] == ErrorCode.AG_TOOL_SCHEMA_INVALID.value

    def test_successful_replan_is_not_reported_as_failure(self, registry):
        """保留的失败任务是执行历史。重规划成功接手时，前一条路失败是正常剧情，
        不能据此判定目标未达成。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        assert out["failure"] is None

    def _failed(self, tid, code, generation=0):
        t = new_task(tid, f"任务 {tid}", "echo")
        t["status"], t["error"] = TaskStatus.FAILED, code
        return episodic.settle({tid: t}, [], generation)[0]

    def test_root_cause_is_preferred_over_cascaded_code(self):
        """级联失败的错误码统一是 AG-1005（前置任务失败），只说明被牵连，
        对用户没有信息量 —— 收尾时要给出根因。"""
        from miagent.graph.nodes import _root_failure

        episodes = [
            self._failed("t1", ErrorCode.AG_DEPENDENCY_UNRESOLVED.value),
            self._failed("t2", ErrorCode.MC_PERMISSION_DENIED.value),
        ]
        assert _root_failure(episodes) == ErrorCode.MC_PERMISSION_DENIED.value

    def test_latest_generation_wins_among_own_failures(self):
        from miagent.graph.nodes import _root_failure

        episodes = [
            self._failed("t1", ErrorCode.MC_PERMISSION_DENIED.value, generation=0),
            self._failed("r1_t1", ErrorCode.MC_RESOURCE_MEMORY_LIMIT.value, generation=1),
        ]
        assert _root_failure(episodes) == ErrorCode.MC_RESOURCE_MEMORY_LIMIT.value

    def test_all_cascaded_falls_back_to_the_first(self):
        from miagent.graph.nodes import _root_failure

        episodes = [self._failed("t1", ErrorCode.AG_DEPENDENCY_UNRESOLVED.value)]
        assert _root_failure(episodes) == ErrorCode.AG_DEPENDENCY_UNRESOLVED.value

    def test_no_failures_means_no_failure_code(self):
        from miagent.graph.nodes import _root_failure

        assert _root_failure([]) is None


# ============================================================
# 执行上下文清理：终态任务离开任务图，进入情景记忆
# ============================================================


class TestWorkingMemoryIsPruned:
    """任务图只装还要调度的东西；历史交给 episodes。"""

    def test_unreferenced_terminal_tasks_leave_the_graph(self, registry):
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        assert set(out["tasks"]) == {"r1_t2b"}
        assert [e["task_id"] for e in out["episodes"]] == ["t1", "t2", "r1_t2b"]
        assert any("结算 2 条记录，保留 0 个已完成" in l for l in out["trace"])

    def test_referenced_done_task_stays_for_its_result(self, registry):
        """新任务用 $from 引用已完成任务时，被引用者必须还在图里，派发前要取值。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "join", left="只给了一个参数")),
            replan=plan(task("t3", "join", left={REF: "t1"}, right="B"))))
        assert out["failure"] is None
        assert set(out["tasks"]) == {"t1", "r1_t3"}
        assert out["tasks"]["r1_t3"]["result"] == "A|B"

    def test_ancestors_of_referenced_tasks_stay_too(self, registry):
        """依赖是传递的：留下 t2 就得留下 t2 的上游，否则图校验会报依赖缺失。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "join", left={REF: "t1"}, right="B"),
                 task("t3", "join", ["t2"], left="只给了一个参数")),
            replan=plan(task("t4", "echo", text={REF: "t2"}))))
        assert out["failure"] is None
        assert set(out["tasks"]) == {"t1", "t2", "r1_t4"}
        assert out["tasks"]["r1_t4"]["result"] == "A|B"

    def test_superseded_pending_tasks_are_dropped(self, registry):
        """尚未执行、被新计划取代的任务不进历史 —— 它们没有发生过。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.book_restaurant", name="小馆 A"),
                 task("t2", "system.create_event", ["t1"], title="晚餐", when="19:00")),
            replan=plan(task("t1b", "system.book_restaurant", name="小馆 B"))))
        assert out["failure"] is None
        assert [e["task_id"] for e in out["episodes"]] == ["t1", "r1_t1b"]

    def test_cascaded_failures_are_settled_at_finalize(self, registry):
        """中止路径上被级联标记失败的任务没经过 Evaluator，收尾时仍要进入历史。"""
        s = initial_state("订餐")
        s["tasks"] = {"t1": new_task("t1", "订餐", "system.book_restaurant", {"name": "A"}),
                      "t2": new_task("t2", "建日程", "system.create_event", {}, ["t1"])}
        s["tasks"]["t1"]["status"], s["tasks"]["t1"]["error"] = TaskStatus.FAILED, "MC-4002"
        s["failure"] = "MC-4002"
        out = finalizer(s, Deps(llm=FakeLLM(["未完成"]), registry=registry))
        assert out["execution_summary"]["failed"] == ["t1", "t2"]
        codes = {e["task_id"]: e["error"] for e in out["episodes"]}
        assert codes["t2"] == ErrorCode.AG_DEPENDENCY_UNRESOLVED.value

    def test_summary_survives_pruning(self, registry):
        """执行概况从情景记忆与任务图合并得出，清理不影响用户看到的结果。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        s = out["execution_summary"]
        assert s["completed"] == ["t1", "r1_t2b"]
        assert s["failed"] == ["t2"]
        assert "t1" in s["results"]
        assert s["total"] == 3

    def test_failed_replan_is_not_masked_by_later_successes(self, registry):
        """重规划失败后，剩余任务照常执行；它们成功不代表恢复发生了。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "echo", ["t1"], text="B"),
                 task("t3", "join", left="只给了一个参数")),
            replan=plan(task("t4", "echo", ["nope"], text="C"))))    # 依赖不存在
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value
        assert "t2" in out["execution_summary"]["completed"]
