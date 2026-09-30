"""对话上下文：进程内的多轮会话。

图的一次 invoke 处理一个请求。多轮对话里第二个请求常常承接第一个
（「再帮我订个位」承接「今晚有空吗」），规划时需要知道上一轮得出了什么。

Session 持有历次轮次的记录，把它们作为对话历史传给下一轮的规划器。
进入提示词的不是上一轮的全部执行历史，而是收尾时产出的本轮摘要 ——
一两句话，说明做了什么、结论是什么。

轮次只在进程内保留，不落盘。进入提示词的部分由上下文预算决定，放不下时
从最旧的一轮开始削（见 render_history）。保留多少轮也由它推出，不另立数字：
从最新一轮往回数，连同更新的各轮已经放不进模型窗口的那一轮，削减时必然先于
所有更新的轮次被丢弃，此后永远进不了提示词 —— 留着它只占内存（见 reachable）。

★ 本模块不 import langgraph：Session 只要求 agent 有 invoke 方法。
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.state import AgentState, Turn, initial_state
from ..llm.context import SEPARATOR, Estimator, Section

__all__ = ["Agent", "Session", "Turn", "reachable", "render_history"]


class Agent(Protocol):
    """Session 与 Runtime 对 Agent 的全部要求：一个 invoke 方法。"""

    def invoke(self, state: AgentState, config: dict[str, Any] | None = None) -> AgentState: ...


class Session:
    """一段对话。每次 run 是一轮，轮次记录在 turns 里。

    history_limit 给定时，每轮结束后只留还可能进入提示词的轮次（见 reachable），
    取模型的上下文上限，estimate 取同一个模型的计量口径。不给则全部保留。
    """

    def __init__(self, agent: Agent, config: dict[str, Any] | None = None,
                 history_limit: int | None = None, estimate: Estimator = len) -> None:
        self._agent = agent
        self._config = config
        self._history_limit = history_limit
        self._estimate = estimate
        self._next_number = 1
        self.turns: list[Turn] = []

    def run(self, request: str) -> AgentState:
        state = initial_state(request)
        state["history"] = list(self.turns)
        out = self._agent.invoke(state, self._config)
        self.turns.append(Turn(
            number=self._next_number, request=request, answer=out["final_answer"],
            summary=out["turn_summary"], failure=out.get("failure"),
            episodes=list(out.get("episodes", [])),
        ))
        self._next_number += 1
        if self._history_limit is not None:
            keep = reachable(self.turns, self._history_limit, self._estimate)
            del self.turns[:len(self.turns) - keep]
        return out

    def clear(self) -> None:
        """忘掉全部轮次。下一轮规划将看不到任何对话历史，轮次从 1 重新编号。"""
        self.turns.clear()
        self._next_number = 1


def reachable(turns: list[Turn], limit: int, estimate: Estimator = len) -> int:
    """在预算 limit 之内，最近几轮还可能进入提示词。

    削减按优先级进行，越旧越先（见 render_history 与 llm.context.fit）。
    某一轮留在提示词里时，比它新的各轮都还是完整形式 —— 它们要等它被丢弃之后
    才轮到。因此若它连同更新的各轮的完整形式已超出预算，它必然被丢弃，
    更早的轮次同样如此。最新一轮总是可能留下：它有紧凑形式。

    limit 取模型的上下文上限即可：规划器的实际预算只会更小，留下的只多不少，
    削与不削，规划器拿到的提示词相同。前提是 estimate 随文本增长而不减。
    """
    if not turns:
        return 0
    texts = [s.text for s in render_history(turns)]
    kept = 1
    while kept < len(texts):
        if estimate(SEPARATOR.join(texts[-(kept + 1):])) > limit:
            break
        kept += 1
    return kept


def render_history(turns: list[Turn]) -> list[Section]:
    """把轮次渲染成规划提示词的片段，每轮一段。

    越旧的一轮优先级数字越大、越先被削减：上一轮先降为只剩结论，再往前的
    整段丢弃。规划器真正需要的是紧接着的上文；更早的轮次若放不下，
    丢掉比把上一轮挤掉要好。

    优先级从 2 起：工具描述是 1，对话历史在它之前被削减 —— 没有工具
    描述就无从规划，没有历史只是少了上文。
    """
    sections: list[Section] = []
    n = len(turns)
    for i, turn in enumerate(turns):
        age = n - 1 - i                       # 0 是上一轮
        label = "上一轮" if age == 0 else f"第 {turn['number']} 轮"
        status = f"（未完成，{turn['failure']}）" if turn["failure"] else ""
        text = f"{label}{status}\n  用户：{turn['request']}\n  结论：{turn['summary']}"
        sections.append(Section(
            f"对话历史·{label}", text, priority=2 + age,
            compact=f"{label}结论：{turn['summary']}" if age == 0 else None,
        ))
    return sections
