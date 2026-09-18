"""情景记忆：终态任务离开任务图之后的记录、去重、降级与合并视图。

与 test_dag.py 同属调度正确性的边界覆盖 —— 这些问题都有确定答案，
必须能被穷举测试，而不是靠跑整张图间接验证。
"""

from miagent.graph import dag
from miagent.graph.state import TaskStatus, new_task
from miagent.memory import anchor, episodic
from miagent.protocol import ErrorCode


def done(tid, result="r", tool="echo", deps=None, **args):
    t = new_task(tid, f"任务 {tid}", tool, args, deps)
    t["status"], t["result"] = TaskStatus.DONE, result
    return t


def failed(tid, code="MC-4002", tool="echo", **args):
    t = new_task(tid, f"任务 {tid}", tool, args)
    t["status"], t["error"], t["result"] = TaskStatus.FAILED, code, "失败说明"
    return t


def pending(tid, tool="echo", deps=None, **args):
    return new_task(tid, f"任务 {tid}", tool, args, deps)


class TestArgsDigest:
    def test_key_order_does_not_matter(self):
        assert episodic.args_digest({"a": 1, "b": 2}) == episodic.args_digest({"b": 2, "a": 1})

    def test_different_arguments_differ(self):
        assert episodic.args_digest({"a": 1}) != episodic.args_digest({"a": 2})

    def test_reference_form_is_what_is_fingerprinted(self):
        """指纹取模型写下的原始形式，引用不会被求值后再算。"""
        assert episodic.args_digest({"x": {"$from": "t1"}}) != episodic.args_digest({"x": "结果"})


class TestSettle:
    def test_only_terminal_tasks_are_recorded(self):
        tasks = {"a": done("a"), "b": failed("b"), "c": pending("c")}
        got = episodic.settle(tasks, [], generation=0)
        assert [e["task_id"] for e in got] == ["a", "b"]
        assert got[0]["ok"] and got[0]["result"] == "r"
        assert not got[1]["ok"] and got[1]["error"] == "MC-4002"

    def test_already_recorded_tasks_are_skipped(self):
        """留在图里供引用的 done 任务，下一次结算不重复记录。"""
        tasks = {"a": done("a")}
        first = episodic.settle(tasks, [], generation=0)
        assert episodic.settle(tasks, first, generation=1) == []

    def test_generation_is_stamped(self):
        got = episodic.settle({"a": done("a")}, [], generation=2)
        assert got[0]["generation"] == 2


class TestDedupe:
    def test_same_call_keeps_latest(self):
        eps = episodic.settle({"t1": failed("t1", name="A")}, [], 0)
        eps += episodic.settle({"r1_t1": done("r1_t1", name="A")}, eps, 1)
        assert [e["task_id"] for e in episodic.dedupe(eps)] == ["r1_t1"]

    def test_different_arguments_are_both_kept(self):
        eps = episodic.settle({"t1": failed("t1", name="A"), "t2": done("t2", name="B")}, [], 0)
        assert len(episodic.dedupe(eps)) == 2

    def test_order_is_preserved(self):
        eps = episodic.settle({"a": done("a", x=1), "b": done("b", x=2), "c": done("c", x=3)}, [], 0)
        assert [e["task_id"] for e in episodic.dedupe(eps)] == ["a", "b", "c"]


class TestRender:
    def test_current_generation_shows_results(self):
        eps = episodic.settle({"a": done("a", "结果A", x=1), "b": failed("b", x=2)}, [], 0)
        done_text, failed_text = episodic.render(eps, generation=0)
        assert "a（已完成）" in done_text and "结果A" in done_text
        assert "b:" in failed_text and "MC-4002" in failed_text and "失败说明" in failed_text

    def test_older_generations_drop_results_but_keep_ids(self):
        """更早几轮的结果已经看过；此后只需知道 id 存在以便引用。"""
        eps = episodic.settle({"a": done("a", "结果A", x=1), "b": failed("b", x=2)}, [], 0)
        done_text, failed_text = episodic.render(eps, generation=1)
        assert "a（已完成）" in done_text and "结果A" not in done_text
        assert "第 0 轮" in done_text
        assert "MC-4002" in failed_text and "失败说明" not in failed_text

    def test_empty_sections_are_explicit(self):
        assert episodic.render([], 0) == ("  （无）", "  （无）")

    def test_same_call_failed_then_succeeded_shows_only_the_success(self):
        eps = episodic.settle({"t1": failed("t1", x=1)}, [], 0)
        eps += episodic.settle({"r1_t1": done("r1_t1", "R", x=1)}, eps, 1)
        done_text, failed_text = episodic.render(eps, generation=1)
        assert "r1_t1" in done_text and failed_text == "  （无）"


class TestHistoryAndSummary:
    def test_summary_merges_episodes_with_remaining_graph(self):
        eps = episodic.settle({"t1": done("t1", "R1"), "t2": failed("t2")}, [], 0)
        tasks = {"t1": done("t1", "R1"),                      # 仍留在图里供引用
                 "r1_t3": done("r1_t3", "R3", deps=["t1"]),
                 "r1_t4": pending("r1_t4", deps=["r1_t3"])}
        s = episodic.summarize(tasks, eps)
        assert s["completed"] == ["t1", "r1_t3"]              # t1 不重复
        assert s["failed"] == ["t2"]
        assert s["pending"] == ["r1_t4"]
        assert s["results"] == {"t1": "R1", "r1_t3": "R3"}
        assert s["total"] == 4

    def test_history_lists_unsettled_tasks_after_episodes(self):
        eps = episodic.settle({"t1": done("t1", "R1")}, [], 0)
        tasks = {"t2": failed("t2", ErrorCode.AG_DEPENDENCY_UNRESOLVED.value),
                 "t3": pending("t3")}
        lines = episodic.history(eps, tasks)
        assert lines[0].startswith("  t1:") and "成功：R1" in lines[0]
        assert "AG-1005" in lines[1]
        assert "未执行" in lines[2]

    def test_history_does_not_repeat_recorded_tasks(self):
        tasks = {"t1": done("t1")}
        eps = episodic.settle(tasks, [], 0)
        assert len(episodic.history(eps, tasks)) == 1


class TestAncestors:
    def test_transitive_dependencies(self):
        tasks = {"a": done("a"), "b": done("b", deps=["a"]),
                 "c": pending("c", deps=["b"]), "d": done("d")}
        assert dag.ancestors(tasks, {"c"}) == {"a", "b"}

    def test_roots_themselves_are_excluded(self):
        tasks = {"a": done("a"), "b": pending("b", deps=["a"]), "c": pending("c", deps=["b"])}
        assert dag.ancestors(tasks, {"b", "c"}) == {"a"}

    def test_no_dependencies(self):
        assert dag.ancestors({"a": pending("a")}, {"a"}) == set()


class TestAnchor:
    def test_intent_follows_task_id_order(self):
        tasks = {"t2": pending("t2"), "t1": pending("t1")}
        a = anchor.build("订餐", tasks)
        assert a == {"goal": "订餐", "intent": ["任务 t1", "任务 t2"]}

    def test_render_is_not_reducible(self):
        section = anchor.render(anchor.build("订餐", {"t1": pending("t1")}))
        assert section.priority == 0 and section.compact is None
        assert "用户目标：订餐" in section.text and "1. 任务 t1" in section.text

    def test_empty_plan_still_renders_the_goal(self):
        assert "（无）" in anchor.render(anchor.build("你好", {})).text


class TestRepeatsFailures:
    def _failed_episodes(self):
        return episodic.settle({"t1": failed("t1", tool="book", name="A"),
                                "t2": done("t2", tool="echo", text="x")}, [], 0)

    def test_all_new_tasks_are_failed_calls(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="A")}
        assert episodic.repeats_failures(new, self._failed_episodes())

    def test_a_single_new_call_breaks_the_loop(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="A"),
               "r1_t3": pending("r1_t3", tool="book", name="B")}
        assert not episodic.repeats_failures(new, self._failed_episodes())

    def test_repeating_a_success_is_not_spinning(self):
        """成功过的调用再做一次不是打转 —— 判据只看失败记录。"""
        new = {"r1_t2": pending("r1_t2", tool="echo", text="x")}
        assert not episodic.repeats_failures(new, self._failed_episodes())

    def test_different_arguments_are_a_new_attempt(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="B")}
        assert not episodic.repeats_failures(new, self._failed_episodes())

    def test_empty_plan_is_not_spinning(self):
        assert not episodic.repeats_failures({}, self._failed_episodes())
