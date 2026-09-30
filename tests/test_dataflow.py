"""任务间数据流：引用的识别、改写与求值。

与 test_dag.py 同属调度正确性的边界覆盖 —— 这些问题都有确定答案，
必须能被穷举测试，而不是靠跑整张图间接验证。
"""

import pytest

from miagent.core import dataflow
from miagent.agent.planning import _build_tasks
from miagent.core.schema import TaskSpec
from miagent.core.state import TaskStatus, new_task
from miagent.protocol import AgentError, ErrorCode

REF = dataflow.REFERENCE_KEY


def done(tid: str, result: str):
    t = new_task(tid, f"任务 {tid}", "some_tool")
    t["status"], t["result"] = TaskStatus.DONE, result
    return t


class TestReferenceRecognition:
    """引用的识别刻意严格：形似而不合规的一律不认，交由上层报错。"""

    def test_plain_reference(self):
        assert dataflow.as_reference({REF: "t1"}) == "t1"

    @pytest.mark.parametrize("value", [
        {REF: "t1", "path": "a"},   # 多一个键：想表达的不止是引用
        {REF: ""},                  # 空 id
        {REF: 3},                   # 非字符串
        {"from": "t1"},             # 键名不对
        {},
        "t1",
        None,
        ["t1"],
    ])
    def test_non_reference_values_are_not_guessed(self, value):
        assert dataflow.as_reference(value) is None

    def test_finds_references_in_nested_structures(self):
        args = {"a": {REF: "t1"}, "b": ["x", {REF: "t2"}], "c": 1,
                "d": {"deep": {REF: "t1"}}}
        assert dataflow.referenced_ids(args) == {"t1", "t2"}

    def test_no_reference_yields_empty(self):
        assert dataflow.referenced_ids({"a": 1, "b": ["x"], "c": {}}) == set()


class TestRemap:
    """重规划时新任务 id 会加前缀，引用必须同步改写。"""

    def test_mapped_ids_are_rewritten(self):
        out = dataflow.remap_references({"a": {REF: "t1"}}, {"t1": "r1_t1"})
        assert out == {"a": {REF: "r1_t1"}}

    def test_unmapped_ids_are_left_alone(self):
        """指向已完成任务的引用不加前缀 —— 那些任务的 id 没有变。"""
        out = dataflow.remap_references({"a": {REF: "t0"}}, {"t1": "r1_t1"})
        assert out == {"a": {REF: "t0"}}

    def test_non_reference_content_is_untouched(self):
        args = {"a": 1, "b": ["x", {"k": "v"}]}
        assert dataflow.remap_references(args, {"t1": "r1_t1"}) == args


class TestResolve:
    def test_reference_becomes_upstream_result(self):
        tasks = {"t1": done("t1", "63")}
        assert dataflow.resolve({"level": {REF: "t1"}}, tasks) == {"level": "63"}

    def test_resolves_inside_lists_and_nested_objects(self):
        tasks = {"t1": done("t1", "A"), "t2": done("t2", "B")}
        args = {"xs": [{REF: "t1"}, "lit"], "o": {"y": {REF: "t2"}}}
        assert dataflow.resolve(args, tasks) == {"xs": ["A", "lit"], "o": {"y": "B"}}

    def test_literals_pass_through(self):
        assert dataflow.resolve({"a": 1, "b": None}, {}) == {"a": 1, "b": None}

    def test_missing_target_is_rejected(self):
        with pytest.raises(AgentError) as e:
            dataflow.resolve({"a": {REF: "nope"}}, {"t1": done("t1", "x")})
        assert e.value.code is ErrorCode.AG_INVALID_PLAN

    def test_unfinished_target_is_rejected(self):
        """正常调度下不会发生（引用即依赖，只派发依赖全 done 的任务），
        但静默取到空值比报错难查得多。"""
        pending = new_task("t1", "任务 t1", "some_tool")
        with pytest.raises(AgentError) as e:
            dataflow.resolve({"a": {REF: "t1"}}, {"t1": pending})
        assert e.value.code is ErrorCode.AG_DEPENDENCY_UNRESOLVED


class TestStringifiedReferences:
    """写成字符串的引用不是引用：不派生依赖、不求值，会原样传给工具。
    必须在校验阶段识别出来。"""

    def test_serialized_reference_is_found_with_its_path(self):
        assert dataflow.stringified_references({"name": '{"$from": "t1"}'}) == ["name"]

    @pytest.mark.parametrize("value", [
        "{'$from': 't1'}",          # 单引号
        "{$from: t1}",              # 少引号
        "  {\"$from\":\"t1\"} ",  # 多空格
        "$from t1",                 # 只剩保留字
    ])
    def test_malformed_variants_are_all_caught(self, value):
        assert dataflow.stringified_references({"x": value}) == ["x"]

    def test_nested_paths(self):
        args = {"a": {"b": '{"$from": "t1"}'}, "c": ["ok", '{"$from": "t2"}']}
        assert dataflow.stringified_references(args) == ["a.b", "c[1]"]

    def test_real_references_and_plain_values_pass(self):
        args = {"a": {REF: "t1"}, "b": "今晚 19:30", "c": 3, "d": None, "e": [{REF: "t2"}]}
        assert dataflow.stringified_references(args) == []


class TestReferenceImpliesDependency:
    """引用即依赖：先后关系由引用派生，不指望模型再声明一遍。"""

    def _spec(self, tid, deps=None, **args):
        return TaskSpec(id=tid, description=f"任务 {tid}", dependencies=deps or [],
                        required_tool="echo", arguments=args)

    def test_reference_creates_the_edge(self):
        tasks = _build_tasks([self._spec("t1"), self._spec("t2", text={REF: "t1"})])
        assert tasks["t2"]["dependencies"] == ["t1"]

    def test_declared_and_derived_are_merged_without_duplicates(self):
        tasks = _build_tasks([
            self._spec("t1"), self._spec("t2"),
            self._spec("t3", deps=["t1"], text={REF: "t1"}, other={REF: "t2"}),
        ])
        assert sorted(tasks["t3"]["dependencies"]) == ["t1", "t2"]
        assert tasks["t3"]["dependencies"].count("t1") == 1

    def test_prefix_rewrites_both_ids_and_references(self):
        tasks = _build_tasks(
            [self._spec("t1"), self._spec("t2", text={REF: "t1"})], prefix="r1_")
        assert set(tasks) == {"r1_t1", "r1_t2"}
        assert tasks["r1_t2"]["dependencies"] == ["r1_t1"]
        assert tasks["r1_t2"]["arguments"]["text"] == {REF: "r1_t1"}

    def test_reference_to_kept_task_keeps_its_id(self):
        """重规划时引用已完成的任务：那些 id 没变，不该加前缀。"""
        tasks = _build_tasks([self._spec("t9", text={REF: "t0"})], prefix="r1_")
        assert tasks["r1_t9"]["dependencies"] == ["t0"]
        assert tasks["r1_t9"]["arguments"]["text"] == {REF: "t0"}
