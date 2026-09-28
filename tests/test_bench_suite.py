"""评测数据集：生成确定、标注自洽、自检能抓出错误。"""

import copy
import json
import random

import pytest

from bench.suite import execute
from bench.suite.catalog import ALL_PERMISSIONS, CATALOG, DEFAULT_BUDGET
from bench.suite.check import CaseError, check, check_case
from bench.suite.compose import Graph, Ref, classify, literal_args, shape
from bench.suite.generate import MIX, generate, level, plan_counts
from bench.suite.phrases import BY_PARAM, BY_TOOL_PARAM, PHRASES, REF_ONLY, REFERENCES, Value


@pytest.fixture(scope="module")
def cases():
    return generate(1500, seed=11)


def _dump(cs):
    return [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cs]


def test_same_seed_same_bytes():
    assert _dump(generate(300, seed=5)) == _dump(generate(300, seed=5))


def test_different_seed_different_cases():
    assert _dump(generate(300, seed=5)) != _dump(generate(300, seed=6))


def test_generated_cases_pass_self_check(cases):
    assert check(cases) == {"cases": 1500, "turns": sum(len(c["turns"]) for c in cases)}


def test_counts_follow_mix():
    counts = plan_counts(10000)
    assert sum(counts.values()) == 10000
    assert counts == {c: round(10000 * w) for c, w in MIX.items()}


def test_every_category_present(cases):
    assert {c["category"] for c in cases} == set(MIX)


# ---------------- 自检能抓出错误 ----------------


def _first_with(cases, pred):
    return copy.deepcopy(next(c for c in cases if pred(c)))


def test_check_rejects_argument_not_in_request(cases):
    case = _first_with(cases, lambda c: c["category"] == "single"
                       and "name" in c["turns"][0]["gold"]["tasks"][0]["arguments"])
    case["turns"][0]["gold"]["tasks"][0]["arguments"]["name"] = "不存在的人"
    with pytest.raises(CaseError):
        check_case(case)


def test_check_rejects_tampered_expectation(cases):
    case = _first_with(cases, lambda c: c["turns"][0]["expected"]["effects"])
    case["turns"][0]["expected"]["effects"].pop()
    with pytest.raises(CaseError, match="参考执行"):
        check_case(case)


def test_check_rejects_unknown_tool(cases):
    case = _first_with(cases, lambda c: c["category"] == "single")
    case["turns"][0]["gold"]["tasks"][0]["required_tool"] = "system.nope"
    with pytest.raises(Exception):
        check_case(case)


def test_check_rejects_duplicate_request(cases):
    dup = copy.deepcopy(cases[0])
    dup["id"] = "mb-dup"
    with pytest.raises(CaseError, match="重复"):
        check([cases[0], dup])


# ---------------- 参考执行 ----------------


def _task(tid, tool, deps=(), **args):
    return {"id": tid, "description": tid, "required_tool": tool,
            "dependencies": list(deps), "arguments": args}


PERMS = list(ALL_PERMISSIONS)


def test_reference_passes_upstream_result_downstream():
    plan = [_task("t1", "system.lookup_phone", name="张伟"),
            _task("t2", "system.send_sms", ["t1"], to={"$from": "t1"}, text="到了")]
    out = execute.outcome(plan, None, faults=[], budget=DEFAULT_BUDGET, permissions=PERMS)
    phone = CATALOG["system.lookup_phone"].run({"name": "张伟"})
    assert out["achievable"]
    assert out["effects"] == [{"tool": "system.send_sms", "arguments": {"to": phone, "text": "到了"}}]


def test_transient_first_still_succeeds():
    plan = [_task("t1", "system.get_battery")]
    fault = {"tool": "system.get_battery", "arguments": {}, "kind": "transient_first"}
    rec = execute.run(plan, faults=[fault], budget=DEFAULT_BUDGET, permissions=PERMS)["t1"]
    assert rec["status"] == "done" and rec["flaky"]


def test_failure_blocks_downstream_but_not_siblings():
    plan = [_task("t1", "system.lookup_phone", name="张伟"),
            _task("t2", "system.send_sms", ["t1"], to={"$from": "t1"}, text="到了"),
            _task("t3", "system.set_volume", level=30)]
    fault = {"tool": "system.lookup_phone", "arguments": {"name": "张伟"}, "kind": "transient_all"}
    out = execute.outcome(plan, None, faults=[fault], budget=DEFAULT_BUDGET, permissions=PERMS)
    status = {r["id"]: r["status"] for r in out["calls"]}
    assert status == {"t1": "failed", "t2": "blocked", "t3": "done"}
    assert not out["achievable"]
    assert [e["tool"] for e in out["effects"]] == ["system.set_volume"]


def test_fallback_covering_failure_makes_goal_achievable():
    plan = [_task("t1", "system.book_restaurant", name="小馆 A", when="今晚 19:00")]
    fb = [{**_task("f1", "system.book_restaurant", name="小馆 B", when="今晚 19:00"),
           "replaces": "t1"}]
    fault = {"tool": "system.book_restaurant",
             "arguments": {"name": "小馆 A", "when": "今晚 19:00"}, "kind": "busy"}
    out = execute.outcome(plan, fb, faults=[fault], budget=DEFAULT_BUDGET, permissions=PERMS)
    assert out["achievable"]
    assert out["effects"] == [{"tool": "system.book_restaurant",
                               "arguments": {"name": "小馆 B", "when": "今晚 19:00"}}]


def test_memory_over_budget_fails_without_declared_fault():
    plan = [_task("t1", "system.record_screen", seconds=30)]
    rec = execute.run(plan, faults=[], budget=DEFAULT_BUDGET, permissions=PERMS)["t1"]
    assert rec["error"] == "MC-4001"


def test_unknown_order_is_business_failure():
    plan = [_task("t1", "system.track_package", order_id="SN123456")]
    rec = execute.run(plan, faults=[], budget=DEFAULT_BUDGET, permissions=PERMS)["t1"]
    assert (rec["status"], rec["error"]) == ("failed", "empty")


# ---------------- 目录与话术 ----------------


def test_every_tool_has_templates_and_samplers():
    for name, spec in CATALOG.items():
        params = spec.tool.input_schema["properties"]
        literal = [p for p in params if (name, p) not in REF_ONLY]
        if literal == list(params):
            assert frozenset() in PHRASES[name], name
        for p in literal:
            assert (name, p) in BY_TOOL_PARAM or p in BY_PARAM, (name, p)
        if spec.produces:
            assert name in REFERENCES, name


def test_handlers_are_pure():
    rng = random.Random(0)
    for name in CATALOG:
        if any(name == t for t, _ in REF_ONLY):
            continue
        args = {p: v.value for p, v in literal_args(name, rng).items()}
        assert CATALOG[name].run(args) == CATALOG[name].run(dict(args))


# ---------------- 结构判定 ----------------


def _graph(*edges_spec):
    g = Graph()
    for tool, refs in edges_spec:
        g.add(tool, {p: Ref(t) for p, t in refs.items()})
    return g


def test_classify_shapes():
    assert classify(_graph(("system.get_battery", {}))) == "single"
    assert classify(_graph(("system.get_battery", {}), ("system.get_city", {}))) == "parallel"
    chain = _graph(("system.read_clipboard", {}), ("system.translate", {"text": "t1"}))
    assert classify(chain) == "chain"
    fan_out = _graph(("system.read_clipboard", {}), ("system.translate", {"text": "t1"}),
                     ("system.create_note", {"content": "t1"}))
    assert classify(fan_out) == "fan_out"
    fan_in = _graph(("system.lookup_phone", {}), ("system.read_clipboard", {}),
                    ("system.send_sms", {"to": "t1", "text": "t2"}))
    assert classify(fan_in) == "fan_in"
    g = _graph(("system.recommend_place", {}), ("system.book_restaurant", {"name": "t1"}),
               ("system.query_traffic", {"destination": "t1"}))
    g.add("system.send_sms", {"text": Ref("t3"), "to": Value("13800000000", "")}, after=["t2"])
    assert classify(g) == "diamond"
    assert shape(g)["depth"] == 3


def test_level_rule():
    one = {"tasks": 1, "depth": 1}
    assert level([one], []) == 1
    assert level([{"tasks": 4, "depth": 4}], []) == 4
    assert level([one, one], []) == 2
    assert level([one], ["fault:busy"]) == 2
    assert level([{"tasks": 7, "depth": 4}], ["fault:x", "noise:fence"]) == 5
