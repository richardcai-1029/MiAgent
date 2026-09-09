"""上下文预算与裁剪。

端侧窗口小，正常任务也会触顶。此前超限直接抛 AG-3001 等于任务失败；
现在按优先级削减，削减不动了才拒绝。削减是纯逻辑，边界在此穷举。
"""

import pytest

from miagent.llm.context import Section, fit
from miagent.protocol import AgentError, ErrorCode


def size(text: str) -> int:
    return len(text)


class TestFitsWithoutTrimming:
    def test_everything_kept_when_within_budget(self):
        body, notes = fit([Section("a", "AAA"), Section("b", "BBB", priority=1)],
                          limit=100, estimate=size)
        assert body == "AAA\n\nBBB"
        assert notes == []

    def test_section_order_is_preserved(self):
        body, _ = fit([Section("a", "1"), Section("b", "2"), Section("c", "3")],
                      limit=100, estimate=size)
        assert body == "1\n\n2\n\n3"


class TestReductionOrder:
    def test_highest_priority_number_is_reduced_first(self):
        _, notes = fit([
            Section("keep", "K"),
            Section("low", "L" * 40, priority=1, compact="l"),
            Section("high", "H" * 40, priority=3, compact="h"),
        ], limit=50, estimate=size)
        assert notes == ["high→紧凑形式"]

    def test_longer_section_goes_first_within_the_same_priority(self):
        """同样是削减一个片段，先削减长的更快腾出空间。"""
        _, notes = fit([
            Section("short", "S" * 10, priority=1, compact="s"),
            Section("long", "L" * 60, priority=1, compact="l"),
        ], limit=30, estimate=size)
        assert notes == ["long→紧凑形式"]

    def test_stops_as_soon_as_it_fits(self):
        """够了就停，不多裁 —— 每多裁一段模型就少一份依据。"""
        body, notes = fit([
            Section("keep", "K"),
            Section("a", "A" * 40, priority=1, compact="a"),
            Section("b", "B" * 40, priority=2, compact="b"),
        ], limit=50, estimate=size)
        assert notes == ["b→紧凑形式"]
        assert "A" * 40 in body

    def test_reduces_further_when_one_pass_is_not_enough(self):
        _, notes = fit([
            Section("a", "A" * 40, priority=1, compact="a"),
            Section("b", "B" * 40, priority=2, compact="b"),
        ], limit=10, estimate=size)
        assert notes == ["b→紧凑形式", "a→紧凑形式"]


class TestReductionForms:
    def test_compact_form_replaces_the_full_text(self):
        body, _ = fit([Section("t", "X" * 50, priority=1, compact="摘要")],
                      limit=10, estimate=size)
        assert body == "摘要"

    def test_section_without_compact_form_is_dropped(self):
        body, notes = fit([Section("keep", "K"), Section("t", "X" * 50, priority=1)],
                          limit=10, estimate=size)
        assert body == "K"
        assert notes == ["t→已丢弃"]

    def test_no_partial_truncation(self):
        """不按字符截断：切开的 JSON 或历史是语法损坏的文本，比丢掉更糟。"""
        body, _ = fit([Section("t", '{"a": 1, "b": 2}', priority=1, compact="{}")],
                      limit=5, estimate=size)
        assert body in ("{}", "")


class TestOverflowIsStillPossible:
    def test_unreducible_content_over_budget_raises(self):
        with pytest.raises(AgentError) as e:
            fit([Section("goal", "G" * 100)], limit=10, estimate=size)
        assert e.value.code is ErrorCode.AG_CONTEXT_OVERFLOW

    def test_error_detail_reports_required_and_limit(self):
        with pytest.raises(AgentError) as e:
            fit([Section("goal", "G" * 100)], limit=10, estimate=size)
        assert e.value.detail["limit"] == 10
        assert e.value.detail["required"] >= 100
        assert e.value.detail["kept"] == ["goal"]

    def test_reduces_everything_it_can_before_giving_up(self):
        with pytest.raises(AgentError):
            fit([Section("goal", "G" * 100), Section("x", "X" * 100, priority=1)],
                limit=10, estimate=size)


class TestEstimatorIsInjected:
    def test_budget_is_measured_with_the_given_estimator(self):
        """真实 tokenizer 接入后只需换掉 estimate，削减逻辑不动。"""
        halved = lambda text: len(text) // 2          # noqa: E731
        body, notes = fit([Section("a", "A" * 30, priority=1, compact="a")],
                          limit=20, estimate=halved)
        assert notes == [] and body == "A" * 30
