"""目标锚：一次会话里不变的那部分上下文。

重规划只看到「已完成什么、失败了什么、请给出剩余任务」。几轮之后，
提示词里全是局部的成败记录，最初要达成什么反而只剩一句用户原话 ——
新计划很容易越走越偏，而且没有任何东西能指出它偏了。

锚在首次规划成功后写入，此后只读：用户目标原文，加上首次拆解给出的
步骤描述。后者是模型自己对「怎样才算完成」的最初理解，之后每一轮
重规划都以它为参照。

锚是提示词里不可裁的片段（priority 0）。窗口放不下锚时抛 AG-3001，
表示的是「这个任务在这个模型上放不下」，而不是把目标裁掉再让模型猜。

★ 本模块是纯函数：不依赖 LangGraph、不调用模型、没有副作用。
"""

from __future__ import annotations

from ..graph.state import Anchor, Task
from ..llm.context import Section

__all__ = ["Anchor", "build", "render"]


def build(goal: str, tasks: dict[str, Task]) -> Anchor:
    return Anchor(goal=goal, intent=[tasks[tid]["description"] for tid in sorted(tasks)])


def render(anchor: Anchor) -> Section:
    """锚的提示词形式。priority 0：不可裁。"""
    steps = "\n".join(f"  {i}. {d}" for i, d in enumerate(anchor["intent"], 1)) or "  （无）"
    return Section("目标锚",
                   f"用户目标：{anchor['goal']}\n最初的拆解：\n{steps}")
