"""语义校验：结果判定、参数修正、完成判定。

分三段：
  · 纯函数 —— 什么值得校验、什么样的修正可以派发、判定能改动什么
  · 节点   —— 判定怎么产生、怎么落库、校验缺席时怎么退回
  · 端到端 —— 参数偏差就地修正、计划漏做被补上
"""

import json

import pytest

from miagent.client import MiClawClient
from miagent import build_agent, initial_state
from miagent.core import verify
from miagent.agent import Deps, Limits, evaluator, goal_verifier, result_verifier
from miagent.core.state import Review, TaskOutcome, TaskStatus, new_task
from miagent.llm import FakeLLM
from miagent.mock_server import MiClawMockServer
from miagent.protocol import ErrorCode
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "location.fine"]


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


# ---------- 构造 ----------


def outcome(task_id="t1", tool_name="echo", ok=True, content="ok",
            code=None, policy="none", attempt=1):
    return TaskOutcome(task_id=task_id, tool=tool_name, ok=ok, content=content,
                       error_code=code, retry_policy=policy, attempt=attempt)


def review(task_id="t1", ok=False, reason="结果与任务不符", correction=None):
    return Review(task_id=task_id, ok=ok, reason=reason, correction=correction)


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def spec(tid, tool_name, deps=None, **args):
    return {"id": tid, "description": f"任务 {tid}", "required_tool": tool_name,
            "dependencies": deps or [], "arguments": args}


def final(answer="完成", summary="本轮已完成"):
    return json.dumps({"answer": answer, "summary": summary}, ensure_ascii=False)


def reviews(*items):
    return json.dumps({"reviews": list(items)}, ensure_ascii=False)


def reject(task_id, reason="结果与任务不符", corrected=None):
    return {"task_id": task_id, "passed": False, "reason": reason,
            "corrected_arguments": corrected}


def goal(achieved=True, gap=""):
    return json.dumps({"achieved": achieved, "gap": gap}, ensure_ascii=False)


def llm_for(first, replan=None, answer="完成", checked=None, done=None):
    """按角色回复。校验的默认回复是「全部通过 / 目标已达成」。"""
    def responder(msgs):
        role = msgs[0].content
        if "结果校验器" in role:
            return checked.pop(0) if isinstance(checked, list) else (checked or reviews())
        if "完成校验器" in role:
            return done.pop(0) if isinstance(done, list) else (done or goal())
        if "重规划器" in role:
            return replan or plan()
        if "规划器" in role:
            return first
        return final(answer)
    return FakeLLM(responder=responder)


def run(registry, request, llm, **kw):
    return build_agent(llm, registry, retry_delay_ms=0, **kw).invoke(
        initial_state(request), {"recursion_limit": 80})


def state_with(task, last, review_=None, **fields):
    s = initial_state("回显 A")
    s["tasks"] = {task["id"]: task}
    s["dispatch"] = [{"task": task, "route": "local"}]
    s["outcomes"] = [last]
    if review_ is not None:
        s["reviews"] = [review_]
    s.update(fields)
    return s


# ============================================================
# 一、纯函数：什么值得校验
# ============================================================


class TestNeedsReview:
    """校验要多烧一轮端侧推理，只在它能给出新信息时才做。"""

    def test_success_is_reviewed(self):
        """工具说成功了，但成功的是不是要的东西无从确定。"""
        assert verify.needs_review(outcome(ok=True))

    def test_tool_layer_failure_is_reviewed(self):
        """段位 3 错在这次调用本身，可能只是参数写偏了。"""
        assert verify.needs_review(
            outcome(ok=False, code=ErrorCode.MC_TOOL_EXECUTION_FAILED.value))
        assert verify.needs_review(
            outcome(ok=False, code=ErrorCode.MC_TOOL_INVALID_PARAMS.value))

    def test_schema_failure_is_reviewed(self):
        assert verify.needs_review(
            outcome(ok=False, code=ErrorCode.AG_TOOL_SCHEMA_INVALID.value))

    def test_determined_failures_are_not_reviewed(self):
        """传输、资源、权限类失败的原因已经确定，校验给不出新信息。"""
        for code in (ErrorCode.MC_REQUEST_TIMEOUT, ErrorCode.MC_NOT_INITIALIZED,
                     ErrorCode.MC_RESOURCE_BUSY, ErrorCode.MC_PERMISSION_DENIED,
                     ErrorCode.AG_TOOL_EXECUTION_FAILED):
            assert not verify.needs_review(outcome(ok=False, code=code.value))


# ============================================================
# 二、纯函数：判定能改动什么
# ============================================================


class TestVerdictOnlyTightens:
    """校验只能把成功改判为失败，不能把失败改判为成功。"""

    def test_rejected_success_gets_the_rejection_code(self):
        assert (verify.outcome_code(outcome(ok=True), review(ok=False))
                == ErrorCode.AG_RESULT_REJECTED.value)

    def test_passed_success_has_no_code(self):
        assert verify.outcome_code(outcome(ok=True), review(ok=True)) is None
        assert verify.outcome_code(outcome(ok=True), None) is None

    def test_failure_keeps_its_own_code_even_if_passed(self):
        """模型说「其实成功了」不会让失败的调用变成完成。"""
        failed = outcome(ok=False, code=ErrorCode.MC_TOOL_EXECUTION_FAILED.value)
        assert (verify.outcome_code(failed, review(ok=True))
                == ErrorCode.MC_TOOL_EXECUTION_FAILED.value)

    def test_rejection_text_keeps_both_the_reason_and_the_result(self):
        text = verify.rejection_text(outcome(content="订单 SN002：已签收"),
                                     review(reason="查的不是 SN001"))
        assert "查的不是 SN001" in text and "订单 SN002：已签收" in text

    def test_passing_review_leaves_the_content_alone(self):
        assert verify.rejection_text(outcome(content="ok"), review(ok=True)) == "ok"


# ============================================================
# 三、纯函数：修正参数的三道闸
# ============================================================


SCHEMA = {"type": "object",
          "properties": {"text": {"type": "string"}, "n": {"type": "integer"}},
          "required": ["text"]}


class TestAcceptCorrection:
    def test_valid_correction_is_accepted(self):
        task = new_task("t1", "", "echo", {"text": "B"})
        args, why = verify.accept_correction(task, {"text": "A"}, SCHEMA)
        assert args == {"text": "A"} and why == ""

    def test_correction_is_normalized_like_a_real_dispatch(self):
        """与真实派发走同一个 normalize，修正不可能比原参数更宽松。"""
        task = new_task("t1", "", "echo", {"text": "B"})
        args, _ = verify.accept_correction(task, {"text": "A", "n": "3"}, SCHEMA)
        assert args == {"text": "A", "n": 3}

    def test_correction_failing_the_schema_is_refused(self):
        task = new_task("t1", "", "echo", {"text": "B"})
        args, why = verify.accept_correction(task, {"text": "A", "zzz": 1}, SCHEMA)
        assert args is None and "schema" in why

    def test_identical_correction_is_refused(self):
        """等价的「修正」重试一次只会得到同样的结果。"""
        task = new_task("t1", "", "echo", {"text": "A"})
        args, why = verify.accept_correction(task, {"text": "A"}, SCHEMA)
        assert args is None and "相同" in why

    def test_referenced_arguments_are_never_corrected(self):
        """引用的值来自上游结果，改写它等于用模型的记忆替换真实结果。"""
        task = new_task("t1", "", "echo", {"text": {"$from": "t0"}})
        args, why = verify.accept_correction(task, {"text": "A"}, SCHEMA)
        assert args is None and "引用" in why

    def test_missing_correction_is_refused(self):
        task = new_task("t1", "", "echo", {"text": "B"})
        assert verify.accept_correction(task, None, SCHEMA)[0] is None
        assert verify.accept_correction(task, {}, SCHEMA)[0] is None

    def test_unregistered_tool_cannot_be_corrected(self):
        task = new_task("t1", "", "nope", {"text": "B"})
        args, why = verify.accept_correction(task, {"text": "A"}, None)
        assert args is None and "未注册" in why


# ============================================================
# 四、校验节点：判定怎么产生
# ============================================================


class TestResultVerifier:
    def _deps(self, registry, llm, **kw):
        return Deps(llm=llm, registry=registry, **kw)

    def test_no_model_call_when_nothing_is_reviewable(self, registry):
        """本轮全是原因已确定的失败：一次推理都不烧。"""
        llm = FakeLLM(script=[])
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        s = state_with(task, outcome(ok=False, code=ErrorCode.MC_REQUEST_TIMEOUT.value))
        out = result_verifier(s, self._deps(registry, llm))
        assert llm.call_count == 0 and "reviews" not in out
        assert "跳过结果校验" in out["trace"][0]

    def test_disabled_verification_calls_nothing(self, registry):
        llm = FakeLLM(script=[])
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        out = result_verifier(state_with(task, outcome()), self._deps(registry, llm, verify=False))
        assert llm.call_count == 0 and "已关闭" in out["trace"][0]

    def test_passing_verdicts_produce_no_reviews(self, registry):
        llm = FakeLLM([reviews()])
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        out = result_verifier(state_with(task, outcome()), self._deps(registry, llm))
        assert out["reviews"] == []
        assert "全部通过" in out["trace"][0]

    def test_rejection_is_recorded_with_its_reason(self, registry):
        llm = FakeLLM([reviews(reject("t1", "回显的是 B，要的是 A"))])
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        out = result_verifier(state_with(task, outcome(content="B")), self._deps(registry, llm))
        assert out["reviews"] == [review("t1", False, "回显的是 B，要的是 A", None)]

    def test_correction_passes_through_the_gate(self, registry):
        llm = FakeLLM([reviews(reject("t1", "回显的是 B", {"text": "A"}))])
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        out = result_verifier(state_with(task, outcome(content="B")), self._deps(registry, llm))
        assert out["reviews"][0]["correction"] == {"text": "A"}
        assert "参数已修正" in out["trace"][0]

    def test_unusable_correction_is_dropped_not_the_verdict(self, registry):
        """修正过不了闸就退回重规划，但「未通过」这个判定仍然成立。"""
        llm = FakeLLM([reviews(reject("t1", "回显的是 B", {"zzz": 1}))])
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        out = result_verifier(state_with(task, outcome(content="B")), self._deps(registry, llm))
        assert out["reviews"][0]["ok"] is False
        assert out["reviews"][0]["correction"] is None
        assert "不就地修正" in out["trace"][0]

    def test_missing_verdict_counts_as_passed(self, registry):
        """两项待校验只判了一项：漏判的那项按通过处理，不因判定缺席变成失败。"""
        llm = FakeLLM([reviews(reject("t2", "回显的是 B"))])
        t1 = new_task("t1", "回显 A", "echo", {"text": "A"})
        t2 = new_task("t2", "回显 A", "echo", {"text": "B"})
        s = initial_state("回显 A")
        s["tasks"] = {"t1": t1, "t2": t2}
        s["dispatch"] = [{"task": t1, "route": "local"}, {"task": t2, "route": "local"}]
        s["outcomes"] = [outcome("t1", content="A"), outcome("t2", content="B")]
        out = result_verifier(s, self._deps(registry, llm))
        assert [r["task_id"] for r in out["reviews"]] == ["t2"]

    def _two(self, first, second):
        t1 = new_task("t1", "回显 A", "echo", {"text": "A"})
        t2 = new_task("t2", "回显 A", "echo", {"text": "B"})
        s = initial_state("回显 A")
        s["tasks"] = {"t1": t1, "t2": t2}
        s["dispatch"] = [{"task": t1, "route": "local"}, {"task": t2, "route": "local"}]
        s["outcomes"] = [outcome("t1", content=first), outcome("t2", content=second)]
        return s

    def test_oversized_item_is_left_out_the_rest_are_reviewed(self, registry):
        """放不下的那项整项不交给模型、按原判定处理；其余项照常校验。
        模型从不看到被截短或去掉结果的记录。"""
        llm = FakeLLM([reviews(reject("t2", "回显的是 B"))])
        out = result_verifier(self._two("X" * 20000, "B"), self._deps(registry, llm))
        body = llm.seen[-1][1].content
        assert "任务 t1" not in body and "X" * 100 not in body
        assert "任务 t2" in body
        assert [r["task_id"] for r in out["reviews"]] == ["t2"]
        assert "1 项放不下" in out["trace"][0]

    def test_left_out_item_cannot_be_judged(self, registry):
        """schema 只收留下的项：模型对没看到的任务下判定，解析阶段就拒掉。"""
        llm = FakeLLM(responder=lambda _m: reviews(reject("t1", "没看到也判")))
        out = result_verifier(self._two("X" * 20000, "B"), self._deps(registry, llm))
        assert "reviews" not in out and "校验未完成" in out["trace"][0]

    def test_nothing_fits_means_no_model_call(self, registry):
        def must_not_be_called(_messages):
            raise AssertionError("没有可交给模型的待校验项时不应该调模型")

        llm = FakeLLM(responder=must_not_be_called)
        out = result_verifier(self._two("X" * 20000, "Y" * 20000), self._deps(registry, llm))
        assert "reviews" not in out
        assert "待校验项都放不下" in out["trace"][0]

    def test_a_verdict_for_another_task_is_rejected_at_parsing(self, registry):
        """task_id 收进 enum：判定挂错任务比判错更难察觉，在解析阶段就拒掉。"""
        llm = FakeLLM(responder=lambda _m: reviews(reject("t9", "不存在的任务")))
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        out = result_verifier(state_with(task, outcome()), self._deps(registry, llm))
        assert "reviews" not in out and "校验未完成" in out["trace"][0]

    def test_unavailable_verification_falls_back_to_the_original_verdict(self, registry):
        """模型给不出合法判定：按没有语义校验处理，不误伤。"""
        llm = FakeLLM(responder=lambda _m: "我觉得挺好的")
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        out = result_verifier(state_with(task, outcome()), self._deps(registry, llm))
        assert "reviews" not in out
        assert "校验未完成" in out["trace"][0]

    def test_only_the_used_tools_are_described(self, registry):
        """校验提示词只带本轮用到的工具，其余 schema 进去也只是占窗口。"""
        llm = FakeLLM([reviews()])
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        result_verifier(state_with(task, outcome()), self._deps(registry, llm))
        body = llm.seen[-1][1].content
        assert "echo" in body and "system.query_calendar" not in body

    def test_the_prompt_shows_what_the_tool_actually_received(self, registry):
        """参数取派发时求值过的那一份，模型看到的不是 $from。"""
        llm = FakeLLM([reviews()])
        task = new_task("t1", "回显上游结果", "echo", {"text": {"$from": "t0"}})
        s = state_with(task, outcome(content="电量 63%"))
        s["dispatch"] = [{"task": {**task, "arguments": {"text": "电量 63%"}},
                          "route": "local"}]
        result_verifier(s, self._deps(registry, llm))
        assert '"text": "电量 63%"' in llm.seen[-1][1].content


# ============================================================
# 五、落库：判定与规则分在两处
# ============================================================


class TestEvaluatorAppliesReviews:
    def _deps(self, registry):
        return Deps(llm=FakeLLM(script=[]), registry=registry)

    def test_corrected_task_retries_with_the_new_arguments(self, registry):
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        s = state_with(task, outcome(content="B"),
                       review("t1", False, "回显的是 B", {"text": "A"}))
        out = evaluator(s, self._deps(registry))
        assert out["verdict"] == "retry"
        assert out["tasks"]["t1"]["status"] is TaskStatus.PENDING
        assert out["tasks"]["t1"]["arguments"] == {"text": "A"}

    def test_rejection_without_correction_goes_to_replanning(self, registry):
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        s = state_with(task, outcome(content="B"), review("t1", False, "回显的是 B"))
        out = evaluator(s, self._deps(registry))
        assert out["verdict"] == "replan"
        assert out["tasks"]["t1"]["status"] is TaskStatus.FAILED
        assert out["tasks"]["t1"]["error"] == ErrorCode.AG_RESULT_REJECTED.value
        assert "回显的是 B" in out["tasks"]["t1"]["result"]

    def test_passed_review_lands_as_success(self, registry):
        task = new_task("t1", "回显 A", "echo", {"text": "A"})
        out = evaluator(state_with(task, outcome(content="A")), self._deps(registry))
        assert out["verdict"] == "success"
        assert out["tasks"]["t1"]["status"] is TaskStatus.DONE

    def test_reviews_are_cleared_after_being_consumed(self, registry):
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        s = state_with(task, outcome(content="B"), review("t1", False))
        assert evaluator(s, self._deps(registry))["reviews"] == []

    def test_correction_does_not_get_an_extra_attempt_budget(self, registry):
        """修正后的重试与普通重试共用单任务重试上限，不另设次数。"""
        task = new_task("t1", "回显 A", "echo", {"text": "B"})
        s = state_with(task, outcome(content="B", attempt=2),
                       review("t1", False, "还是不对", {"text": "A"}))
        out = evaluator(s, self._deps(registry))
        assert out["verdict"] == "replan"
        assert out["tasks"]["t1"]["arguments"] == {"text": "B"}


# ============================================================
# 六、完成校验
# ============================================================


class TestGoalVerifier:
    def _state(self, registry, **fields):
        s = initial_state("订餐")
        s["anchor"] = {"goal": "订餐", "intent": ["查日历", "订餐厅"]}
        t = new_task("t1", "查日历", "system.query_calendar", {"when": "今晚"})
        t["status"], t["result"] = TaskStatus.DONE, "今晚 19:00-22:00 空闲"
        s["tasks"] = {"t1": t}
        s["execution_count"] = 1
        s.update(fields)
        return s

    def test_achieved_changes_nothing(self, registry):
        llm = FakeLLM([goal(True)])
        out = goal_verifier(self._state(registry), Deps(llm=llm, registry=registry))
        assert "failure" not in out and "verdict" not in out

    def test_unachieved_turns_to_replanning(self, registry):
        llm = FakeLLM([goal(False, "还没订餐厅")])
        out = goal_verifier(self._state(registry), Deps(llm=llm, registry=registry))
        assert out["failure"] == ErrorCode.AG_GOAL_NOT_ACHIEVED.value
        assert out["verdict"] == "replan" and out["gap"] == "还没订餐厅"

    def test_unachieved_without_replan_budget_goes_to_finalizer(self, registry):
        llm = FakeLLM([goal(False, "还没订餐厅")])
        out = goal_verifier(self._state(registry, replan_count=Limits().max_replans),
                          Deps(llm=llm, registry=registry))
        assert out["failure"] == ErrorCode.AG_GOAL_NOT_ACHIEVED.value
        assert "verdict" not in out

    def test_nothing_executed_means_nothing_to_verify(self, registry):
        llm = FakeLLM(script=[])
        out = goal_verifier(self._state(registry, execution_count=0, tasks={}),
                          Deps(llm=llm, registry=registry))
        assert out == {} and llm.call_count == 0

    def test_an_existing_failure_code_skips_the_call(self, registry):
        """已经带着错误码就不会声称成功，再判一次没有新信息。"""
        llm = FakeLLM(script=[])
        out = goal_verifier(self._state(registry, failure=ErrorCode.AG_INVALID_PLAN.value),
                          Deps(llm=llm, registry=registry))
        assert out == {} and llm.call_count == 0

    def test_disabled_verification_calls_nothing(self, registry):
        llm = FakeLLM(script=[])
        out = goal_verifier(self._state(registry), Deps(llm=llm, registry=registry, verify=False))
        assert out == {} and llm.call_count == 0

    def test_the_prompt_carries_the_anchor_and_the_record(self, registry):
        llm = FakeLLM([goal(True)])
        goal_verifier(self._state(registry), Deps(llm=llm, registry=registry))
        body = llm.seen[-1][1].content
        assert "用户目标：订餐" in body and "今晚 19:00-22:00 空闲" in body

    def test_unavailable_verification_finalizes_as_completed(self, registry):
        llm = FakeLLM(responder=lambda _m: "差不多吧")
        out = goal_verifier(self._state(registry), Deps(llm=llm, registry=registry))
        assert "failure" not in out and "完成校验未完成" in out["trace"][0]


# ============================================================
# 七、端到端
# ============================================================


class TestCorrectionEndToEnd:
    def test_parameter_deviation_is_fixed_in_place(self, registry):
        """查错了单号 → 校验给出正确单号 → 同一个任务带新参数重试成功。
        整张图不必重规划，也没有多出一个任务。"""
        llm = llm_for(
            plan(spec("t1", "system.query_order", order_id="SN002")),
            checked=[reviews(reject("t1", "查的不是 SN001", {"order_id": "SN001"})),
                     reviews()])
        out = run(registry, "查订单 SN001 的状态", llm)
        assert out["failure"] is None
        assert out["replan_count"] == 0
        assert out["tasks"]["t1"]["arguments"] == {"order_id": "SN001"}
        assert out["tasks"]["t1"]["result"] == "订单 SN001：已发货"
        assert out["execution_count"] == 2

    def test_successful_but_wrong_result_is_not_reported_as_done(self, registry):
        """工具成功、结果不对、修不出参数：如实转重规划，不宣称完成。"""
        llm = llm_for(
            plan(spec("t1", "echo", text="B")),
            replan=plan(),
            checked=reviews(reject("t1", "回显的是 B，用户要的是 A")))
        out = run(registry, "回显 A", llm)
        assert out["failure"] == ErrorCode.AG_RESULT_REJECTED.value
        assert out["execution_summary"]["failed"] == ["t1"]

    def test_rejection_reason_reaches_the_replanner(self, registry):
        llm = llm_for(plan(spec("t1", "echo", text="B")),
                      replan=plan(spec("t2", "echo", text="A")),
                      checked=[reviews(reject("t1", "回显的是 B，用户要的是 A")),
                               reviews()])
        run(registry, "回显 A", llm)
        body = next(m[1].content for m in llm.seen if "重规划器" in m[0].content)
        assert "回显的是 B，用户要的是 A" in body


class TestGoalVerificationEndToEnd:
    def test_missing_step_is_filled_in(self, registry):
        """计划漏了订餐厅这一步：任务全部成功，完成校验判未达成 → 补做。"""
        llm = llm_for(
            plan(spec("t1", "system.query_calendar", when="今晚")),
            replan=plan(spec("t2", "system.book_restaurant", name="小馆 B")),
            done=[goal(False, "查了日历，还没订餐厅"), goal(True)])
        out = run(registry, "查下今晚有空吗，有就订个餐厅", llm)
        assert out["failure"] is None
        assert out["replan_count"] == 1
        assert "r1_t2" in out["execution_summary"]["completed"]

    def test_the_gap_reaches_the_replanner(self, registry):
        llm = llm_for(
            plan(spec("t1", "system.query_calendar", when="今晚")),
            replan=plan(spec("t2", "system.book_restaurant", name="小馆 B")),
            done=[goal(False, "查了日历，还没订餐厅"), goal(True)])
        run(registry, "查下今晚有空吗，有就订个餐厅", llm)
        body = next(m[1].content for m in llm.seen if "重规划器" in m[0].content)
        role = next(m[0].content for m in llm.seen if "重规划器" in m[0].content)
        assert "还没订餐厅" in role
        assert "失败：\n  （无）" in body        # 没有失败，交代的是缺口

    def test_an_empty_replan_keeps_the_unachieved_code(self, registry):
        """重规划交了白卷：没有失败记录可取根因，未达成的码必须留住。"""
        llm = llm_for(plan(spec("t1", "system.query_calendar", when="今晚")),
                      replan=plan(),
                      done=goal(False, "还没订餐厅"))
        out = run(registry, "查下今晚有空吗，有就订个餐厅", llm)
        assert out["failure"] == ErrorCode.AG_GOAL_NOT_ACHIEVED.value

    def test_redoing_done_work_is_judged_as_no_progress(self, registry):
        """补做时把已经做成的调用原样再拆一遍：判无进展，不让它再执行一次。"""
        first = plan(spec("t1", "system.query_calendar", when="今晚"))
        llm = llm_for(first,
                      replan=plan(spec("t1", "system.query_calendar", when="今晚")),
                      done=goal(False, "还没订餐厅"))
        out = run(registry, "查下今晚有空吗，有就订个餐厅", llm)
        assert out["failure"] == ErrorCode.AG_PLAN_NO_PROGRESS.value
        assert out["execution_count"] == 1


class TestVerificationCanBeTurnedOff:
    def test_no_verification_calls_at_all(self, registry):
        """关掉之后两个校验节点直接透传，一次校验推理都不发生。"""
        llm = llm_for(plan(spec("t1", "echo", text="A")))
        out = run(registry, "回显 A", llm, verify=False)
        assert out["failure"] is None
        assert not [m for m in llm.seen if "校验器" in m[0].content]
