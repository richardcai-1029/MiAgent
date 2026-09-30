"""评测框架：打分能区分对错，标注答案驱动的模型能驱动整张图跑通。"""

import pytest

from bench.eval.env import Call, FaultPlan
from bench.eval.plan_score import score_plan
from bench.eval.score import score_turn
from bench.suite.catalog import CATALOG
from bench.suite.generate import generate


def _turn(effects, calls, deps=None, achievable=True, forbidden=()):
    tasks = [{"id": c["id"], "description": "", "required_tool": c["tool"],
              "dependencies": (deps or {}).get(c["id"], []), "arguments": {}} for c in calls]
    return {"gold": {"tasks": tasks}, "fallback": None, "forbidden": list(forbidden),
            "expected": {"achievable": achievable, "effects": effects, "calls": calls}}


SMS = {"to": "13800000000", "text": "到了"}
PHONE = {"name": "张伟"}


def _rec(tid, tool, args):
    return {"id": tid, "tool": tool, "arguments": args, "status": "done", "result": "x"}


def _call(seq, tool, args, t0, t1, ok=True):
    return Call(seq, tool, args, ok, None if ok else "MC-1003", t0, t1, 0)


@pytest.fixture
def chain():
    calls = [_rec("t1", "system.lookup_phone", PHONE), _rec("t2", "system.send_sms", SMS)]
    return _turn([{"tool": "system.send_sms", "arguments": SMS}], calls, {"t2": ["t1"]})


STATE = {"failure": None, "final_answer": "好的"}


def test_correct_execution_scores_success(chain):
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 1), _call(1, "system.send_sms", SMS, 2, 3)]
    assert score_turn(chain, calls, STATE, 2, 1)["success"]


def test_duplicate_side_effect_fails(chain):
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 1), _call(1, "system.send_sms", SMS, 2, 3),
             _call(2, "system.send_sms", SMS, 4, 5)]
    r = score_turn(chain, calls, STATE, 2, 1)
    assert not r["effects"] and r["duplicate_effects"] == 1


def test_downstream_before_upstream_fails_order(chain):
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 3), _call(1, "system.send_sms", SMS, 2, 4)]
    assert not score_turn(chain, calls, STATE, 2, 1)["order"]


def test_over_quota_fails(chain):
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 1), _call(1, "system.send_sms", SMS, 2, 3)]
    assert not score_turn(chain, calls, STATE, 2, 3)["quota"]


def test_claiming_success_on_unachievable_fails(chain):
    chain["expected"]["achievable"] = False
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 1), _call(1, "system.send_sms", SMS, 2, 3)]
    assert not score_turn(chain, calls, STATE, 2, 1)["status"]


def test_forbidden_call_fails(chain):
    chain["forbidden"] = [{"tool": "system.lookup_phone", "arguments": None}]
    calls = [_call(0, "system.lookup_phone", PHONE, 0, 1), _call(1, "system.send_sms", SMS, 2, 3)]
    assert not score_turn(chain, calls, STATE, 2, 1)["forbidden"]


# ---------------- 规划打分 ----------------

GOLD = [
    {"id": "t1", "required_tool": "system.lookup_phone", "arguments": {"name": "张伟"}, "dependencies": []},
    {"id": "t2", "required_tool": "system.send_sms",
     "arguments": {"to": {"$from": "t1"}, "text": "到了"}, "dependencies": ["t1"]},
]


def _model(text="到了", ref="a", extra=None):
    m = [{"id": "a", "required_tool": "system.lookup_phone", "arguments": {"name": "张伟"}, "dependencies": []},
         {"id": "b", "required_tool": "system.send_sms",
          "arguments": {"to": {"$from": ref}, "text": text}, "dependencies": [ref]}]
    return m + (extra or [])


def test_plan_exact_match_with_different_ids():
    s = score_plan(_model(), GOLD, {}, [])
    assert s["exact"] and s["calls_ok"] == 2 and s["edges_hit"] == 1


def test_plan_quotes_and_spaces_are_ignored():
    assert score_plan(_model(text="「到 了」"), GOLD, {}, [])["args_ok"] == 3


def test_plan_wrong_reference_is_wrong_argument():
    extra = [{"id": "c", "required_tool": "system.get_battery", "arguments": {}, "dependencies": []}]
    s = score_plan(_model(ref="c", extra=extra), GOLD, {}, [])
    assert s["args_ok"] == 2 and not s["exact"] and s["n_model"] == 3


def test_plan_accept_list():
    gold = [{"id": "t1", "required_tool": "system.set_volume", "arguments": {"level": 30}, "dependencies": []}]
    model = [{"id": "x", "required_tool": "system.set_volume", "arguments": {"level": "30"}, "dependencies": []}]
    assert score_plan(model, gold, {"t1": {"level": [30, "30"]}}, [])["exact"]


def test_plan_forbidden_repeat_is_counted():
    forbidden = [{"tool": "system.lookup_phone", "arguments": {"name": "张伟"}}]
    assert score_plan(_model(), GOLD, {}, forbidden)["forbidden_hits"] == 1


# ---------------- 故障注入 ----------------


def test_transient_first_fails_only_once():
    from miagent.protocol import MiClawError
    tool = FaultPlan([{"tool": "system.get_battery", "arguments": {},
                       "kind": "transient_first"}]).wrap(CATALOG["system.get_battery"].tool)
    with pytest.raises(MiClawError):
        tool.handler({})
    assert tool.handler({}) == CATALOG["system.get_battery"].run({})


# ---------------- 标注答案驱动整张图 ----------------


def test_oracle_drives_generated_cases_to_success():
    pytest.importorskip("langgraph")
    from bench.eval.framework import quiet, run_case

    quiet()
    rows = [row for case in generate(60, seed=3) for row in run_case(case)]
    assert rows and all(r["success"] for r in rows), [r for r in rows if not r["success"]][:1]


def test_ablation_without_verification_is_detected():
    """关掉语义校验后，参数写偏的用例应被判失败 —— 打分确实依赖框架做对。"""
    pytest.importorskip("langgraph")
    from bench.eval.framework import quiet, run_case

    quiet()
    drift = [c for c in generate(400, seed=3) if "sim:arg_drift" in c["tags"]][:5]
    assert drift
    rows = [row for case in drift for row in run_case(case, verify=False)]
    assert not all(r["success"] for r in rows)


def test_plan_unsupported_flag():
    assert score_plan([], [], {}, [], infeasible=True, flagged=["订机票"])["exact"]
    assert not score_plan([], [], {}, [], infeasible=False, flagged=["聊天"])["exact"]
    assert not score_plan([], [], {}, [], infeasible=True, flagged=[])["exact"]
    assert score_plan([], [], {}, [], infeasible=True)["unsupported_ok"] is None
