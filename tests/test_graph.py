"""图节点与端到端测试（任务 DAG 版）。"""

import json
import re

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent, dag, initial_state
from miagent.graph.nodes import (Deps, evaluator, execute, finalizer,
                                 planner, scheduler, _tools_section)
from miagent.graph.routers import (route_after_evaluator,
                                   route_after_goal_verifier,
                                   route_after_scheduler)
from langgraph.types import Send

from miagent.graph.state import (MAX_REPLANS, MAX_TOTAL_EXECUTIONS,
                                 TaskOutcome, TaskStatus, new_task)
from miagent.llm import FakeLLM
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


def final(answer="完成", summary="本轮已完成"):
    return json.dumps({"answer": answer, "summary": summary}, ensure_ascii=False)


def reviews(*items):
    """结果校验的判定。不给判定即全部通过 —— 漏判按通过处理。"""
    return json.dumps({"reviews": list(items)}, ensure_ascii=False)


def reject(task_id, reason="结果与任务不符", corrected=None):
    return {"task_id": task_id, "passed": False, "reason": reason,
            "corrected_arguments": corrected}


def goal(achieved=True, gap=""):
    return json.dumps({"achieved": achieved, "gap": gap}, ensure_ascii=False)


def _review_or_final(msgs):
    """默认的校验回复：全部通过。给只关心规划与收尾的用例用。"""
    role = msgs[0].content
    if "结果校验器" in role:
        return reviews()
    if "完成校验器" in role:
        return goal()
    return final()


def llm_for(first, replan=None, answer="完成", checked=None, done=None):
    def responder(msgs):
        role = msgs[0].content
        if "结果校验器" in role:
            return checked or reviews()
        if "完成校验器" in role:
            return done or goal()
        if "重规划器" in role:
            return replan or plan()
        if "规划器" in role:
            return first
        return final(answer)
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
        """没有可派发的任务 → 先过完成校验，再由它决定收尾还是补做。"""
        assert route_after_scheduler({"dispatch": []}) == "goal_verifier"

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

    def test_after_goal_verifier(self):
        assert route_after_goal_verifier({"verdict": "replan"}) == "replanner"
        assert route_after_goal_verifier({"verdict": None}) == "finalizer"
        assert route_after_goal_verifier({}) == "finalizer"


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

    def test_stringified_reference_is_repaired_before_execution(self, registry):
        """引用写成字符串 → schema 校验拒绝 → 错误喂回模型 → 改成对象后执行。
        乱码参数一次都不能传给工具。"""
        outs = iter([
            plan(task("a", "system.query_calendar", when="今晚"),
                 task("b", "system.create_event", title="晚餐", when='{"$from": "a"}')),
            plan(task("a", "system.query_calendar", when="今晚"),
                 task("b", "system.create_event", title="晚餐", when={"$from": "a"})),
        ])
        llm = FakeLLM(responder=lambda m: next(outs) if "规划器" in m[0].content
                      else _review_or_final(m))
        out = run(registry, "安排晚餐", llm)
        assert out["failure"] is None
        assert llm.repair_count == 1
        assert "写成了字符串" in llm.seen[1][-1].content        # 修复提示说明了错在哪
        assert out["tasks"]["b"]["dependencies"] == ["a"]      # 改成对象后依赖被派生
        assert out["tasks"]["b"]["result"] == "已创建日程「晚餐」于 今晚 19:00-22:00 空闲"

    def test_stringified_reference_unrepaired_fails_planning(self, registry):
        out = run(registry, "安排晚餐", llm_for(plan(
            task("b", "system.create_event", title="晚餐", when='{"$from": "nope"}'))))
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
    """模型只允许在规划、校验与收尾三类环节被调用。

    调度、执行、评估三个环节要回答的问题都有确定答案——依赖是否满足、
    是否超出重试上限、是否全部完成、该重试还是该换方案。用模型去猜一个
    我们确定知道答案的问题，既慢又不可复现，端侧尤其付不起这个代价。

    校验问的是「拿到的结果算不算达成了要做的事」，没有确定答案，因此交给
    模型；但它给出的只是判定，能改动什么由 verify.py 的纯函数决定。

    这条约束此前只写在文档里。文档约束不会在被破坏时报警，故在此固化：
    新加的节点若持有模型引用，本用例立即失败。
    """
    from pathlib import Path

    nodes_py = Path(__file__).resolve().parent.parent / "miagent" / "graph" / "nodes.py"
    touching = _functions_touching("deps", "llm", nodes_py)

    # _plan 是 Planner 与 Replanner 共用的规划实现；_review 是两个校验节点
    # 共用的调用实现；finalizer 生成给用户的回答。
    assert touching == {"_plan", "_review", "finalizer"}, (
        f"模型调用范围发生变化，当前出现在 {sorted(touching)}。"
        "调度、执行、评估环节的判断均有确定答案，必须由纯函数承担。"
    )


# ============================================================
# 多工具联动：任务间的数据流
# ============================================================

REF = "$from"


class TestDispatchOrder:
    """并发配额放不下全部就绪任务时，先派发关键路径上的任务。"""

    def _rounds(self, registry, first):
        out = build_agent(llm_for(first), registry, max_concurrent_miclaw=2,
                          retry_delay_ms=0, verify=False).invoke(
            initial_state("查五个时段的天气"), {"recursion_limit": 80})
        assert out["failure"] is None
        return [line for line in out["trace"]
                if line.startswith("scheduler: 本轮就绪")]

    def test_critical_path_is_not_deferred(self, registry):
        # task_1、task_2 是叶子；task_3 → task_4 → task_5 是一条链。
        # 先派两个叶子要跑四轮，先派链头只要三轮。
        weather = "system.query_weather"
        first = plan(task("task_1", weather, when="t1"),
                     task("task_2", weather, when="t2"),
                     task("task_3", weather, when="t3"),
                     task("task_4", weather, ["task_3"], when="t4"),
                     task("task_5", weather, ["task_4"], when="t5"))
        rounds = self._rounds(registry, first)
        assert len(rounds) == 3
        assert "task_3(miclaw), task_1(miclaw)" in rounds[0]

    def test_plan_order_survives_ten_or_more_tasks(self, registry):
        tasks = [task(f"task_{i}", "echo", text=str(i)) for i in range(1, 12)]
        out = build_agent(llm_for(plan(*tasks)), registry, retry_delay_ms=0,
                          verify=False).invoke(initial_state("回显"),
                                               {"recursion_limit": 80})
        dispatched = next(line for line in out["trace"]
                          if line.startswith("scheduler: 本轮就绪"))
        names = dispatched.split("→ ")[1].split(", ")
        assert names == [f"task_{i}(local)" for i in range(1, 12)]
        assert out["anchor"]["intent"][:3] == ["任务 task_1", "任务 task_2", "任务 task_3"]


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
        assert out["turn_summary"] == out["final_answer"]     # 摘要同源，不缺席
        assert "上下文放不下" in out["trace"][0]

    def test_replanner_trims_only_the_oversized_result(self, registry):
        """一条结果超长只让它自己换成简要形式：其余结论、失败记录与 id 都还在。"""
        from miagent.graph.nodes import replanner
        from miagent.memory import anchor as anchor_mod

        huge = new_task("t1", "截屏识别", "echo")
        huge["status"], huge["result"] = TaskStatus.DONE, "X" * 20000
        short = new_task("t2", "查电量", "system.get_battery")
        short["status"], short["result"] = TaskStatus.DONE, "电量 63%"
        bad = new_task("t3", "订餐", "system.book_restaurant", {"name": "小馆 A"})
        bad["status"], bad["error"], bad["result"] = TaskStatus.FAILED, "MC-4002", "已满座"
        state = initial_state("订餐")
        state["tasks"] = {"t1": huge, "t2": short, "t3": bad}
        state["anchor"] = anchor_mod.build("订餐", state["tasks"])
        llm = llm_for(plan(), replan=plan(task("t4", "system.book_restaurant", name="小馆 B")))

        out = replanner(state, Deps(llm=llm, registry=registry))
        body = llm.seen[-1][1].content
        assert "X" * 100 not in body
        assert "t1（已完成）: 截屏识别（结果从略）" in body      # id 仍可供 $from 引用
        assert "电量 63%" in body and "已满座" in body
        assert "已完成·t1→紧凑形式" in out["trace"][0]
        assert "工具描述" not in out["trace"][0]              # 腾出的空间已经够了

    def test_finalizer_keeps_the_other_conclusions(self, registry):
        """收尾时一条结果超长，其余结论照样交给模型 —— 回答与本轮摘要才有具体内容。"""
        huge = new_task("t1", "截屏识别", "echo")
        huge["status"], huge["result"] = TaskStatus.DONE, "X" * 20000
        short = new_task("t2", "查电量", "system.get_battery")
        short["status"], short["result"] = TaskStatus.DONE, "电量 63%"
        state = initial_state("查一下电量")
        state["tasks"] = {"t1": huge, "t2": short}
        llm = FakeLLM([final("电量 63%")])

        out = finalizer(state, Deps(llm=llm, registry=registry))
        body = llm.seen[-1][1].content
        assert "X" * 100 not in body
        assert "电量 63%" in body and "t1: 截屏识别 → 成功（结果从略）" in body
        assert out["final_answer"] == "电量 63%"
        assert "执行情况·t1→紧凑形式" in out["trace"][0]

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

# ============================================================
# 执行上下文清理：终态任务离开任务图，进入情景记忆
# ============================================================


class TestWorkingMemoryIsPruned:
    """任务图只装还要调度的东西；历史交给 episodes。

    边界在 test_ledger.py 穷举；这里只证明图把账本接上了。
    """

    def test_unreferenced_terminal_tasks_leave_the_graph(self, registry):
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        assert set(out["tasks"]) == {"r1_t2b"}
        assert [e["task_id"] for e in out["episodes"]] == ["t1", "t2", "r1_t2b"]
        s = out["execution_summary"]
        assert (s["completed"], s["failed"], s["total"]) == (["t1", "r1_t2b"], ["t2"], 3)

    def test_referenced_done_task_stays_for_its_result(self, registry):
        """新任务用 $from 引用已完成任务时，被引用者必须还在图里，派发前要取值。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "join", left="只给了一个参数")),
            replan=plan(task("t3", "join", left={REF: "t1"}, right="B"))))
        assert out["failure"] is None
        assert set(out["tasks"]) == {"t1", "r1_t3"}
        assert out["tasks"]["r1_t3"]["result"] == "A|B"


class TestFailureIsClearedOnlyByRecovery:
    """failure 的规则：成功的重规划清除它，单个任务成功不清除。"""

    def test_failed_replan_is_not_masked_by_later_successes(self, registry):
        """重规划失败后，剩余任务照常执行；它们成功不代表恢复发生了。"""
        out = run(registry, "汇总", llm_for(
            plan(task("t1", "echo", text="A"),
                 task("t2", "echo", ["t1"], text="B"),
                 task("t3", "join", left="只给了一个参数")),
            replan=plan(task("t4", "echo", ["nope"], text="C"))))    # 依赖不存在
        assert out["failure"] == ErrorCode.AG_INVALID_PLAN.value
        assert "t2" in out["execution_summary"]["completed"]

    def test_later_successful_replan_clears_an_earlier_failed_one(self, registry):
        """第二次重规划看到了全部失败记录并覆盖了剩余工作，恢复发生了。"""
        replans = [plan(task("t4", "echo", ["nope"], text="C")),      # 第一次：依赖不存在
                   plan(task("t5", "echo", text="D"))]               # 第二次：合法
        first = plan(task("t1", "echo", text="A"),
                     task("t2", "join", left="只给了一个参数"),
                     task("t3", "join", ["t1"], left="也只给了一个参数"))

        def responder(msgs):
            role = msgs[0].content
            if "重规划器" in role:
                return replans.pop(0)
            if "规划器" in role:
                return first
            return final()

        out = run(registry, "汇总", FakeLLM(responder=responder))
        assert out["replan_count"] == 2
        assert out["failure"] is None
        assert "r2_t5" in out["execution_summary"]["completed"]


# ============================================================
# 目标锚：重规划始终有「原本要做什么」可以对照
# ============================================================


class TestGoalAnchor:
    def test_planner_writes_the_anchor_once(self, registry):
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.query_calendar", when="今晚"),
                 task("t2", "system.book_restaurant", ["t1"], name="小馆 A")),
            replan=plan(task("t2b", "system.book_restaurant", name="小馆 B"))))
        assert out["anchor"] == {"goal": "订餐", "intent": ["任务 t1", "任务 t2"]}

    def test_failed_planning_leaves_no_anchor(self, registry):
        out = run(registry, "随便说说", llm_for("我不知道该怎么办"))
        assert out["anchor"] == {}

    def test_replanner_sees_the_anchor_first(self, registry):
        llm = llm_for(
            plan(task("t1", "system.book_restaurant", name="小馆 A")),
            replan=plan(task("t1b", "system.book_restaurant", name="小馆 B")))
        run(registry, "订餐", llm)
        replan_prompt = next(m[1].content for m in llm.seen if "重规划器" in m[0].content)
        assert replan_prompt.startswith("用户目标：订餐\n最初的拆解：\n  1. 任务 t1")

    def test_anchor_survives_trimming_that_drops_everything_else(self, registry):
        """窗口紧张时先削工具描述与历史，锚一个字都不能少。"""
        from miagent.llm.context import SEPARATOR, Section, fit
        from miagent.memory import anchor as anchor_mod

        a = anchor_mod.render(anchor_mod.build("订餐", {"t1": new_task("t1", "查日历", "echo")}))
        tools = _tools_section(registry)
        failed = Section("失败", "失败：" + "x" * 500, priority=2, compact="有 1 个失败")
        # 预算恰好只够放下锚加上另外两段的紧凑形式
        limit = len(SEPARATOR.join([a.text, tools.compact, failed.compact]))
        body, notes = fit([a, tools, failed], limit=limit, estimate=len)
        assert notes == ["失败→紧凑形式", "工具描述→紧凑形式"]
        assert body.startswith(a.text)


# ============================================================
# 原地打转：新计划全是失败过的调用
# ============================================================


class TestSpinningIsDetected:
    def test_identical_replan_is_not_executed(self, registry):
        """把失败过的调用原样再拆一遍，执行只会得到同样的失败。
        判定规则在 test_ledger.py；这里证明判定之后确实没有派发。"""
        out = run(registry, "订餐", llm_for(
            plan(task("t1", "system.book_restaurant", name="小馆 A")),
            replan=plan(task("t1_again", "system.book_restaurant", name="小馆 A"))))
        assert out["failure"] == ErrorCode.AG_PLAN_NO_PROGRESS.value
        assert out["execution_count"] == 1                    # 重复的那次没有派发
        assert out["tasks"] == {}
        assert out["errors"][-1]["repeated"] == ["r1_t1_again"]


# ============================================================
# 收尾的结构化输出：一次调用同时给出回答与本轮摘要
# ============================================================


class TestFinalizerStructuredOutput:
    def _state(self):
        state = initial_state("查一下电量")
        t = new_task("t1", "查电量", "system.get_battery")
        t["status"], t["result"] = TaskStatus.DONE, "电量 63%"
        state["tasks"] = {"t1": t}
        return state

    def test_failure_prompt_carries_the_codes_meaning(self, registry):
        """模型只看到错误码会自行猜原因；说明来自协议层的错误码表。"""
        state = self._state()
        state["failure"] = ErrorCode.AG_PLAN_PARSE_FAILED.value
        llm = FakeLLM([final()])
        finalizer(state, Deps(llm=llm, registry=registry))
        body = llm.seen[-1][1].content
        assert "AG-1001（模型输出无法解析为可执行计划）" in body
        assert "不要编造" in body

    def test_deterministic_fallback_also_explains_the_code(self, registry):
        """窗口放不下时的兜底回答同样带说明，用户不该拿到一个裸码。"""
        state = self._state()
        state["failure"] = ErrorCode.AG_PLAN_PARSE_FAILED.value
        llm = FakeLLM(script=[], context_limit=10)      # 必超窗 -> 走兜底
        out = finalizer(state, Deps(llm=llm, registry=registry))
        assert "AG-1001（模型输出无法解析为可执行计划）" in out["final_answer"]

    def test_answer_and_summary_come_from_one_call(self, registry):
        llm = FakeLLM([final("电量 63%", "查过电量，63%")])
        out = finalizer(self._state(), Deps(llm=llm, registry=registry))
        assert out["final_answer"] == "电量 63%"
        assert out["turn_summary"] == "查过电量，63%"
        assert llm.call_count == 1

    def test_schema_is_injected_into_the_prompt(self, registry):
        llm = FakeLLM([final()])
        finalizer(self._state(), Deps(llm=llm, registry=registry))
        assert "summary" in llm.seen[-1][-1].content
        assert "供下一轮规划参考" in llm.seen[-1][-1].content

    def test_unparsable_output_falls_back_after_repair(self, registry):
        """自修复后仍不合 schema：回答不缺席，摘要同源。"""
        llm = FakeLLM(responder=lambda _m: "电量还行吧")
        out = finalizer(self._state(), Deps(llm=llm, registry=registry))
        assert out["final_answer"].startswith("已完成 1 项")
        assert out["turn_summary"] == out["final_answer"]
        assert "模型输出不合 schema" in out["trace"][0]
        assert llm.repair_count == 1                           # 修过一次才放弃

    def test_repair_feeds_the_error_back(self, registry):
        """第一次给纯文本，收到错误提示后给出合法 JSON。"""
        calls = []

        def responder(msgs):
            calls.append(msgs)
            return "电量还行吧" if len(calls) == 1 else final("电量 63%", "查过电量")

        llm = FakeLLM(responder=responder)
        out = finalizer(self._state(), Deps(llm=llm, registry=registry))
        assert out["final_answer"] == "电量 63%"
        assert "不符合要求" in calls[1][-1].content

    def test_end_to_end_exposes_turn_summary(self, registry):
        out = run(registry, "原样返回 hi", llm_for(plan(task("t1", "echo", text="hi"))))
        assert out["turn_summary"] == "本轮已完成"

    def test_failed_planning_still_yields_a_summary(self, registry):
        out = run(registry, "随便说说", llm_for("我不知道该怎么办"))
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert out["turn_summary"]
