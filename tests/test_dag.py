"""依赖解析测试。

这些是纯函数，不需要 LangGraph、不需要模型，所以可以把边界情况测穷 ——
调度的正确性正是最该被严格测试的部分。
"""

import pytest

from miagent.graph import dag
from miagent.graph.state import TaskStatus, new_task
from miagent.protocol import AgentError, ErrorCode


def mk(*specs):
    """specs: (id, deps) —— 描述与工具名对依赖解析无影响。"""
    return {i: new_task(i, f"任务 {i}", "t", dependencies=list(d)) for i, d in specs}


def mark(tasks, **statuses):
    for tid, st in statuses.items():
        tasks[tid]["status"] = st
    return tasks


class TestValidate:
    """非法图必须在执行任何一步之前被拦下。"""

    def test_valid_graph_passes(self):
        dag.validate(mk(("A", []), ("B", ["A"]), ("C", ["A", "B"])))

    def test_missing_dependency(self):
        with pytest.raises(AgentError) as ei:
            dag.validate(mk(("A", ["Z"])))
        assert ei.value.code is ErrorCode.AG_INVALID_PLAN

    def test_self_dependency(self):
        with pytest.raises(AgentError) as ei:
            dag.validate(mk(("A", ["A"])))
        assert ei.value.code is ErrorCode.AG_INVALID_PLAN

    def test_two_node_cycle(self):
        with pytest.raises(AgentError) as ei:
            dag.validate(mk(("A", ["B"]), ("B", ["A"])))
        assert "环" in ei.value.message

    def test_longer_cycle(self):
        with pytest.raises(AgentError):
            dag.validate(mk(("A", ["C"]), ("B", ["A"]), ("C", ["B"])))

    def test_cycle_detection_ignores_valid_diamond(self):
        """菱形不是环 —— 常见误判点。"""
        assert dag.find_cycle(mk(("A", []), ("B", ["A"]), ("C", ["A"]),
                                 ("D", ["B", "C"]))) is None

    def test_empty_graph_is_valid(self):
        dag.validate({})


class TestReady:
    """就绪判定：pending 且所有依赖都 done。"""

    def test_no_dependency_is_ready(self):
        assert dag.ready(mk(("A", []), ("B", []))) == ["A", "B"]

    def test_blocked_until_dependency_done(self):
        t = mk(("A", []), ("B", ["A"]))
        assert dag.ready(t) == ["A"]
        assert dag.ready(mark(t, A=TaskStatus.DONE)) == ["B"]

    def test_needs_all_dependencies(self):
        t = mark(mk(("A", []), ("B", []), ("C", ["A", "B"])), A=TaskStatus.DONE)
        assert "C" not in dag.ready(t)
        assert "C" in dag.ready(mark(t, B=TaskStatus.DONE))

    def test_running_task_is_not_ready_again(self):
        assert dag.ready(mark(mk(("A", [])), A=TaskStatus.RUNNING)) == []

    def test_diamond_exposes_parallel_pair(self):
        """并行机会的判定依据：同一轮就绪的任务不止一个。"""
        t = mark(mk(("A", []), ("B", ["A"]), ("C", ["A"]), ("D", ["B", "C"])),
                 A=TaskStatus.DONE)
        assert dag.ready(t) == ["B", "C"]


class TestCascadeFailures:
    def test_direct_dependent_fails(self):
        t = dag.cascade_failures(mark(mk(("A", []), ("B", ["A"])), A=TaskStatus.FAILED))
        assert t["B"]["status"] is TaskStatus.FAILED
        assert t["B"]["error"] == ErrorCode.AG_DEPENDENCY_UNRESOLVED.value

    def test_propagates_transitively(self):
        """失败要一路传播到下游的下游，否则末端任务会永远挂在 pending。"""
        t = dag.cascade_failures(
            mark(mk(("A", []), ("B", ["A"]), ("C", ["B"])), A=TaskStatus.FAILED))
        assert all(t[x]["status"] is TaskStatus.FAILED for x in "ABC")

    def test_does_not_touch_independent_branches(self):
        t = dag.cascade_failures(
            mark(mk(("A", []), ("B", ["A"]), ("X", [])), A=TaskStatus.FAILED))
        assert t["X"]["status"] is TaskStatus.PENDING

    def test_does_not_overwrite_completed(self):
        t = dag.cascade_failures(
            mark(mk(("A", []), ("B", [])), A=TaskStatus.FAILED, B=TaskStatus.DONE))
        assert t["B"]["status"] is TaskStatus.DONE


class TestCompletionAndDeadlock:
    def test_complete_when_all_terminal(self):
        assert dag.is_complete(mark(mk(("A", []), ("B", [])),
                                    A=TaskStatus.DONE, B=TaskStatus.FAILED))

    def test_not_complete_with_pending(self):
        assert not dag.is_complete(mark(mk(("A", []), ("B", [])), A=TaskStatus.DONE))

    def test_deadlock_when_dependency_failed_and_not_cascaded(self):
        """级联之前会呈现死锁，级联之后应当收敛。"""
        t = mark(mk(("A", []), ("B", ["A"])), A=TaskStatus.FAILED)
        assert dag.is_deadlocked(t)
        assert not dag.is_deadlocked(dag.cascade_failures(t))

    def test_running_task_is_not_deadlock(self):
        assert not dag.is_deadlocked(mark(mk(("A", []), ("B", ["A"])),
                                          A=TaskStatus.RUNNING))

    def test_empty_graph_is_complete(self):
        assert dag.is_complete({}) and not dag.is_deadlocked({})


class TestDerivedViews:
    """这些是从 tasks 算出来的，不单独存 —— 存两份必然漂移。"""

    def test_status_groups(self):
        t = mark(mk(("A", []), ("B", []), ("C", []), ("D", [])),
                 A=TaskStatus.DONE, B=TaskStatus.FAILED, C=TaskStatus.RUNNING)
        assert dag.completed_tasks(t) == ["A"]
        assert dag.failed_tasks(t) == ["B"]
        assert dag.running_tasks(t) == ["C"]
        assert dag.pending_tasks(t) == ["D"]

    def test_parallel_layers_linear(self):
        assert dag.parallel_layers(mk(("A", []), ("B", ["A"]), ("C", ["B"]))) == \
            [["A"], ["B"], ["C"]]

    def test_parallel_layers_diamond(self):
        assert dag.parallel_layers(mk(("A", []), ("B", ["A"]), ("C", ["A"]),
                                      ("D", ["B", "C"]))) == \
            [["A"], ["B", "C"], ["D"]]
