"""对话上下文：进程内的多轮会话。

图的一次 invoke 处理一个请求。多轮对话里第二个请求常常承接第一个
（「再帮我订个位」承接「今晚有空吗」），规划时需要知道上一轮得出了什么。

Session 持有历次轮次的记录，把它们作为对话历史传给下一轮的规划器。
进入提示词的不是上一轮的全部执行历史，而是收尾时产出的本轮摘要 ——
一两句话，说明做了什么、结论是什么。

轮次只在进程内保留，不落盘。保留多少轮不设上限；进入提示词的部分
由上下文预算决定，放不下时从最旧的一轮开始削（见 render_history）。

★ 本模块不 import langgraph：Session 只要求 agent 有 invoke 方法。
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from ..graph.state import AgentState, initial_state
from ..llm.context import Section
from .episodic import Episode


class Turn(TypedDict):
    request: str
    answer: str
    summary: str                # 收尾产出的本轮摘要，下一轮规划看的就是它
    failure: str | None         # 非空表示这一轮没完成，值为错误码
    episodes: list[Episode]     # 这一轮的执行历史，供调用方查看，不进提示词


class Agent(Protocol):
    def invoke(self, state: AgentState, config: dict[str, Any] | None = None) -> AgentState: ...


class Session:
    """一段对话。每次 run 是一轮，轮次记录在 turns 里。"""

    def __init__(self, agent: Agent, config: dict[str, Any] | None = None) -> None:
        self._agent = agent
        self._config = config
        self.turns: list[Turn] = []

    def run(self, request: str) -> AgentState:
        state = initial_state(request)
        state["history"] = list(self.turns)
        out = self._agent.invoke(state, self._config)
        self.turns.append(Turn(
            request=request, answer=out["final_answer"], summary=out["turn_summary"],
            failure=out.get("failure"), episodes=list(out.get("episodes", [])),
        ))
        return out

    def clear(self) -> None:
        """忘掉全部轮次。下一轮规划将看不到任何对话历史。"""
        self.turns.clear()


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
        label = "上一轮" if age == 0 else f"第 {i + 1} 轮"
        status = f"（未完成，{turn['failure']}）" if turn["failure"] else ""
        text = f"{label}{status}\n  用户：{turn['request']}\n  结论：{turn['summary']}"
        sections.append(Section(
            f"对话历史·{label}", text, priority=2 + age,
            compact=f"{label}结论：{turn['summary']}" if age == 0 else None,
        ))
    return sections
