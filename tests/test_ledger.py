"""账本：终态任务离开任务图时的结算，通过 settle / accept / close 三个操作验证。

这里覆盖的是结算的边界 —— 哪些留、哪些走、何时判无进展、根因码怎么选、
failure 何时清除。全部字典进、字典出，不需要图引擎与协议栈；
整张图是否把账本接上了，由 test_graph.py 里少量端到端用例证明。
"""

from miagent.graph.state import TaskStatus, new_task
from miagent.memory import ledger
from miagent.protocol import ErrorCode

REF = "$from"


def done(tid, result="r", tool="echo", deps=None, **args):
    t = new_task(tid, f"任务 {tid}", tool, args, deps)
    t["status"], t["result"] = TaskStatus.DONE, result
    return t


def failed(tid, code="MC-4002", tool="echo", deps=None, **args):
    t = new_task(tid, f"任务 {tid}", tool, args, deps)
    t["status"], t["error"], t["result"] = TaskStatus.FAILED, code, "失败说明"
    return t


def pending(tid, tool="echo", deps=None, **args):
    return new_task(tid, f"任务 {tid}", tool, args, deps)


def merged_with(settlement, *new_tasks):
    """模拟 _plan 的产出：保留的已完成任务 + 模型给出的新任务。"""
    return {**settlement.keep, **{t["id"]: t for t in new_tasks}}


# ============================================================
# settle：重规划前
# ============================================================


class TestSettle:
    def test_generation_is_the_next_round(self):
        s = ledger.settle({"t1": done("t1")}, [], replan_count=0)
        assert s.generation == 1
        assert s.new_episodes[0]["generation"] == 0        # 记录属于刚结束的那一轮

    def test_only_unrecorded_terminal_tasks_are_settled(self):
        tasks = {"t1": done("t1"), "t2": failed("t2"), "t3": pending("t3")}
        first = ledger.settle(tasks, [], 0)
        assert [e["task_id"] for e in first.new_episodes] == ["t1", "t2"]

        again = ledger.settle(tasks, first.episodes, 1)
        assert again.new_episodes == []                    # 已记录的不重复
        assert [e["task_id"] for e in again.episodes] == ["t1", "t2"]

    def test_keep_is_the_done_tasks(self):
        s = ledger.settle({"t1": done("t1"), "t2": failed("t2"), "t3": pending("t3")}, [], 0)
        assert set(s.keep) == {"t1"}

    def test_current_round_renders_results_older_rounds_only_ids(self):
        s0 = ledger.settle({"t1": done("t1", "R1", text="a")}, [], 0)
        s1 = ledger.settle({"t2": done("t2", "R2", text="b")}, s0.episodes, 1)
        assert "R2" in s1.done_text
        assert "R1" not in s1.done_text and "t1" in s1.done_text

    def test_counts_are_deduped_by_call(self):
        s0 = ledger.settle({"t1": failed("t1", name="A")}, [], 0)
        s1 = ledger.settle({"r1_t1": done("r1_t1", name="A")}, s0.episodes, 1)
        assert (s1.n_done, s1.n_failed) == (1, 0)          # 同一调用先败后成，只算成功


# ============================================================
# accept：重规划后
# ============================================================


class TestAcceptPrunesWorkingMemory:
    """任务图只装还要调度的东西；历史交给 episodes。"""

    def test_unreferenced_terminal_tasks_leave_the_graph(self):
        s = ledger.settle({"t1": done("t1"), "t2": failed("t2", deps=["t1"])}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t2b", tool="book")))
        assert set(a.tasks) == {"r1_t2b"}
        assert (a.added, a.kept, a.failure) == (1, 0, None)

    def test_referenced_done_task_stays_for_its_result(self):
        s = ledger.settle({"t1": done("t1", "A"), "t2": failed("t2")}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t3", tool="join", deps=["t1"],
                                                    left={REF: "t1"}, right="B")))
        assert set(a.tasks) == {"t1", "r1_t3"}
        assert a.kept == 1

    def test_ancestors_of_referenced_tasks_stay_too(self):
        """依赖是传递的：留下 t2 就得留下 t2 的上游，否则图校验会报依赖缺失。"""
        tasks = {"t1": done("t1", "A"), "t2": done("t2", "A|B", deps=["t1"]),
                 "t3": failed("t3", deps=["t2"])}
        s = ledger.settle(tasks, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t4", deps=["t2"], text={REF: "t2"})))
        assert set(a.tasks) == {"t1", "t2", "r1_t4"}
        assert a.kept == 2

    def test_superseded_pending_tasks_are_dropped(self):
        """尚未执行、被新计划取代的任务不进历史 —— 它们没有发生过。"""
        s = ledger.settle({"t1": failed("t1", name="A"), "t2": pending("t2", deps=["t1"])}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t1b", name="B")))
        assert set(a.tasks) == {"r1_t1b"}
        assert [e["task_id"] for e in s.new_episodes] == ["t1"]


class TestAcceptDetectsSpinning:
    def test_identical_replan_is_not_accepted(self):
        """把失败过的调用原样再拆一遍，执行只会得到同样的失败。"""
        s = ledger.settle({"t1": failed("t1", tool="book", name="A")}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t1_again", tool="book", name="A")))
        assert a.tasks == {}
        assert a.failure == ErrorCode.AG_PLAN_NO_PROGRESS.value
        assert a.repeated == ["r1_t1_again"]

    def test_changed_arguments_are_a_real_retry(self):
        s = ledger.settle({"t1": failed("t1", tool="book", name="A")}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t1b", tool="book", name="B")))
        assert set(a.tasks) == {"r1_t1b"} and a.failure is None and a.repeated == []

    def test_partially_new_plan_is_allowed(self):
        """换方案的同时保留了某一步：那一步是否再次失败由执行回答，不猜。"""
        s = ledger.settle({"t1": failed("t1", tool="book", name="A")}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t1_again", tool="book", name="A"),
                                         pending("r1_t2", tool="calendar", when="今晚")))
        assert set(a.tasks) == {"r1_t1_again", "r1_t2"} and a.failure is None

    def test_redoing_a_successful_call_is_also_spinning(self):
        """新任务只是把已经做成的调用再拆一遍：没有进展，且副作用会发生第二次。"""
        s = ledger.settle({"t1": done("t1", tool="book", name="A")}, [], 0)
        a = ledger.accept(s, merged_with(s, pending("r1_t1_again", tool="book", name="A")))
        assert a.failure == ErrorCode.AG_PLAN_NO_PROGRESS.value
        assert a.tasks == {}


class TestAcceptDecidesFailure:
    """failure 的规则：成功的重规划清除它；未产出新任务则取根因码。"""

    def test_new_tasks_clear_failure(self):
        s = ledger.settle({"t1": failed("t1", ErrorCode.MC_PERMISSION_DENIED.value, name="A")}, [], 0)
        assert ledger.accept(s, merged_with(s, pending("r1_t2", name="B"))).failure is None

    def test_no_new_tasks_is_reported_as_unmet(self):
        """重规划是恢复机制，跑完却一个新任务都没产出，说明恢复没发生。"""
        s = ledger.settle({"t1": done("t1"),
                           "t2": failed("t2", ErrorCode.AG_TOOL_SCHEMA_INVALID.value)}, [], 0)
        a = ledger.accept(s, merged_with(s))
        assert a.failure == ErrorCode.AG_TOOL_SCHEMA_INVALID.value
        assert a.tasks == {} and a.added == 0

    def test_root_cause_is_preferred_over_cascaded_code(self):
        """级联失败的错误码统一是 AG-1005（前置任务失败），只说明被牵连，
        对用户没有信息量 —— 要给出根因。"""
        s = ledger.settle({"t1": failed("t1", ErrorCode.AG_DEPENDENCY_UNRESOLVED.value),
                           "t2": failed("t2", ErrorCode.MC_PERMISSION_DENIED.value)}, [], 0)
        assert ledger.accept(s, merged_with(s)).failure == ErrorCode.MC_PERMISSION_DENIED.value

    def test_latest_generation_wins_among_own_failures(self):
        s0 = ledger.settle({"t1": failed("t1", ErrorCode.MC_PERMISSION_DENIED.value)}, [], 0)
        s1 = ledger.settle({"r1_t1": failed("r1_t1", ErrorCode.MC_RESOURCE_MEMORY_LIMIT.value)},
                           s0.episodes, 1)
        assert ledger.accept(s1, merged_with(s1)).failure == \
            ErrorCode.MC_RESOURCE_MEMORY_LIMIT.value

    def test_all_cascaded_falls_back_to_the_cascade_code(self):
        s = ledger.settle({"t1": failed("t1", ErrorCode.AG_DEPENDENCY_UNRESOLVED.value)}, [], 0)
        assert ledger.accept(s, merged_with(s)).failure == \
            ErrorCode.AG_DEPENDENCY_UNRESOLVED.value

    def test_no_failures_means_no_failure_code(self):
        s = ledger.settle({"t1": done("t1")}, [], 0)
        assert ledger.accept(s, merged_with(s)).failure is None


# ============================================================
# close：收尾
# ============================================================


class TestClose:
    def test_cascaded_failures_are_settled(self):
        """中止路径上被级联标记失败的任务没经过 Evaluator，收尾时仍要进入历史。"""
        tasks = {"t1": failed("t1", "MC-4002"), "t2": pending("t2", deps=["t1"])}
        c = ledger.close(tasks, [], 0)
        assert c.summary["failed"] == ["t1", "t2"]
        assert c.tasks["t2"]["status"] is TaskStatus.FAILED
        codes = {e["task_id"]: e["error"] for e in c.new_episodes}
        assert codes["t2"] == ErrorCode.AG_DEPENDENCY_UNRESOLVED.value

    def test_summary_merges_episodes_with_remaining_graph(self):
        """执行概况从情景记忆与任务图合并得出，清理不影响用户看到的结果。"""
        s = ledger.settle({"t1": done("t1", "R1"), "t2": failed("t2")}, [], 0)
        remaining = {"t1": done("t1", "R1"),                 # 仍留在图里供引用
                     "r1_t3": done("r1_t3", "R3", deps=["t1"])}
        c = ledger.close(remaining, s.episodes, 1)
        assert c.summary["completed"] == ["t1", "r1_t3"]     # t1 不重复
        assert c.summary["failed"] == ["t2"]
        assert c.summary["results"] == {"t1": "R1", "r1_t3": "R3"}
        assert c.summary["total"] == 3
        assert [e["task_id"] for e in c.new_episodes] == ["r1_t3"]

    def test_history_lists_every_task_once(self):
        s = ledger.settle({"t1": done("t1", "R1")}, [], 0)
        c = ledger.close({"t1": done("t1", "R1"), "t2": pending("t2")}, s.episodes, 0)
        assert len(c.history) == 2
        assert "成功：R1" in c.history[0] and "未执行" in c.history[1]

    def test_nothing_executed(self):
        c = ledger.close({}, [], 0)
        assert c.history == [] and c.new_episodes == []
        assert c.summary["total"] == 0
