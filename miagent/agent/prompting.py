"""各节点共用的提示词片段。

片段按优先级参与预算裁剪（见 llm.context）：放不下时低优先级的先削减，
削减了什么写进 trace。
"""

from __future__ import annotations

import json

from ..llm.context import Section
from ..memory.episodic import Line
from ..tools import ToolRegistry


def tools_section(registry: ToolRegistry, names: set[str] | None = None,
                  priority: int = 1) -> Section:
    """工具描述：预算不够时削减为只剩工具名。

    完整 schema 是规划质量的主要输入，但窗口触顶时保留工具名，
    模型仍有机会选对工具并配合自修复补齐参数；整段丢掉则连选都无从选起。

    names 给定时只描述这几个工具。校验时只有本轮调过的工具与修正有关，
    其余工具的 schema 进了提示词也只是占窗口。
    """
    tools = [t for t in registry if names is None or t.name in names]
    schemas = json.dumps([t.to_model_schema() for t in tools], ensure_ascii=False, indent=2)
    return Section(
        "工具描述",
        f"可用工具：\n{schemas}",
        priority=priority,
        compact="可用工具：" + "、".join(t.name for t in tools),
    )


def trim_note(notes: list[str]) -> str:
    """把削减说明拼进 trace。裁掉了什么必须能看见。"""
    return f"（上下文削减：{'，'.join(notes)}）" if notes else ""


def records(title: str, lines: list[Line], priority: int,
            empty: str = "  （无）") -> list[Section]:
    """执行记录按条成段：标题一段，每条记录一段。

    放不下时逐条换成去掉结果原文的简要形式，最长的先换（见 context.choose）——
    一条结果超长只让它自己降级，其余记录的结论原样留在提示词里。
    简要形式是 id 与描述，是引用它、交代它的最低限度，因此不再往下削；
    简要形式并不更短的记录（本身不带结果原文，或结果极短）保持原样。
    """
    return [Section(title, f"{title}：" + ("" if lines else f"\n{empty}")),
            *(Section(f"{title}·{line.task_id}", line.full, priority=priority,
                      compact=line.brief) for line in lines)]

