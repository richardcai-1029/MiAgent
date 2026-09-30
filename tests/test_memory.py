"""情景记忆：终态任务离开任务图之后的记录、去重、降级与合并视图。

与 test_dag.py 同属调度正确性的边界覆盖 —— 这些问题都有确定答案，
必须能被穷举测试，而不是靠跑整张图间接验证。
"""

from miagent.core import dag
from miagent.core.state import TaskStatus, new_task
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


def text(lines):
    return "\n".join(line.full for line in lines)


class TestRender:
    def test_current_generation_shows_results(self):
        eps = episodic.settle({"a": done("a", "结果A", x=1), "b": failed("b", x=2)}, [], 0)
        done_lines, failed_lines = episodic.render(eps, generation=0)
        assert "a（已完成）" in text(done_lines) and "结果A" in text(done_lines)
        assert ("b:" in text(failed_lines) and "MC-4002" in text(failed_lines)
                and "失败说明" in text(failed_lines))

    def test_brief_form_drops_the_result_but_keeps_id_and_outcome(self):
        """简要形式是逐条削减的落点：id 留着供引用，成败与错误码留着供判断。"""
        eps = episodic.settle({"a": done("a", "结果A", x=1), "b": failed("b", x=2)}, [], 0)
        (d,), (f,) = episodic.render(eps, generation=0)
        assert d.task_id == "a" and "a（已完成）" in d.brief and "结果A" not in d.brief
        assert f.task_id == "b" and "MC-4002" in f.brief and "失败说明" not in f.brief

    def test_older_generations_drop_results_but_keep_ids(self):
        """更早几轮的结果已经看过；此后只需知道 id 存在以便引用。"""
        eps = episodic.settle({"a": done("a", "结果A", x=1), "b": failed("b", x=2)}, [], 0)
        done_lines, failed_lines = episodic.render(eps, generation=1)
        assert "a（已完成）" in text(done_lines) and "结果A" not in text(done_lines)
        assert "第 0 轮" in text(done_lines)
        assert "MC-4002" in text(failed_lines) and "失败说明" not in text(failed_lines)
        assert all(line.full == line.brief for line in done_lines + failed_lines)

    def test_empty(self):
        assert episodic.render([], 0) == ([], [])

    def test_same_call_failed_then_succeeded_shows_only_the_success(self):
        eps = episodic.settle({"t1": failed("t1", x=1)}, [], 0)
        eps += episodic.settle({"r1_t1": done("r1_t1", "R", x=1)}, eps, 1)
        done_lines, failed_lines = episodic.render(eps, generation=1)
        assert [line.task_id for line in done_lines] == ["r1_t1"] and failed_lines == []


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
        assert lines[0].full.startswith("  t1:") and "成功：R1" in lines[0].full
        assert "AG-1005" in lines[1].full
        assert "未执行" in lines[2].full

    def test_history_brief_drops_results(self):
        eps = episodic.settle({"t1": done("t1", "R1"), "t2": failed("t2")}, [], 0)
        ok, bad = episodic.history(eps, {})
        assert "成功" in ok.brief and "R1" not in ok.brief
        assert "MC-4002" in bad.brief and "失败说明" not in bad.brief

    def test_unexecuted_tasks_have_nothing_to_drop(self):
        (line,) = episodic.history([], {"t3": pending("t3")})
        assert line.full == line.brief

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
    def test_intent_follows_plan_order_not_id_order(self):
        """按字符串排 id，task_10 会排在 task_2 之前。"""
        tasks = {f"task_{i}": new_task(f"task_{i}", f"第 {i} 步", "echo", seq=i - 1)
                 for i in (10, 2, 1, 3, 4, 5, 6, 7, 8, 9)}
        a = anchor.build("订餐", tasks)
        assert a["intent"] == [f"第 {i} 步" for i in range(1, 11)]

    def test_render_is_not_reducible(self):
        section = anchor.render(anchor.build("订餐", {"t1": pending("t1")}))
        assert section.priority == 0 and section.compact is None
        assert "用户目标：订餐" in section.text and "1. 任务 t1" in section.text

    def test_empty_plan_still_renders_the_goal(self):
        assert "（无）" in anchor.render(anchor.build("你好", {})).text


class TestRepeatsCalls:
    def _episodes(self):
        return episodic.settle({"t1": failed("t1", tool="book", name="A"),
                                "t2": done("t2", tool="echo", text="x")}, [], 0)

    def test_all_new_tasks_are_failed_calls(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="A")}
        assert episodic.repeats_calls(new, self._episodes())

    def test_a_single_new_call_breaks_the_loop(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="A"),
               "r1_t3": pending("r1_t3", tool="book", name="B")}
        assert not episodic.repeats_calls(new, self._episodes())

    def test_redoing_a_successful_call_is_also_spinning(self):
        """已经做成的事再做一遍同样不是进展，而且会让副作用发生第二次。"""
        new = {"r1_t2": pending("r1_t2", tool="echo", text="x")}
        assert episodic.repeats_calls(new, self._episodes())

    def test_different_arguments_are_a_new_attempt(self):
        new = {"r1_t1": pending("r1_t1", tool="book", name="B")}
        assert not episodic.repeats_calls(new, self._episodes())

    def test_empty_plan_is_not_spinning(self):
        assert not episodic.repeats_calls({}, self._episodes())
