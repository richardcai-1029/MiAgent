"""上下文预算与裁剪。

端侧模型的窗口显著小于云端。工具一多、执行历史一长，正常任务也会触顶——
此时直接拒绝（AG-3001）等于任务失败，这对端侧不可接受。

做法是把提示词拆成带优先级的片段，超出预算时按优先级由高到低依次削减，
削减到放得下为止：

    优先级 0    不可裁 —— 用户目标、输出格式约束。裁掉它们请求就没有意义了
    优先级 >0   可裁 —— 数字越大越先被削减

削减有两档：片段若给了紧凑形式（compact）就换成紧凑形式，否则整段丢弃。
**不做按字符截断**：把一份 JSON Schema 或一段执行历史从中间切开，
得到的是语法损坏的文本，模型多半会被带偏，比直接丢掉更糟。

削减的结果会被返回给调用方写进 trace。裁掉了什么必须能看见，
否则模型基于残缺上下文给出的错误答案将无从解释。

★ 本模块是纯函数：给定同样的片段与预算，削减结果完全确定，可穷举测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..protocol import AgentError, ErrorCode

Estimator = Callable[[str], int]

SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Section:
    """提示词的一个片段。"""

    name: str                      # 出现在 trace 里，说明裁掉的是什么
    text: str                      # 完整形式
    priority: int = 0              # 0 表示不可裁；越大越先被削减
    compact: str | None = None     # 削减后的替代形式；None 表示整段丢弃


def fit(sections: list[Section], limit: int, estimate: Estimator) -> tuple[str, list[str]]:
    """把片段拼成不超过 limit 的提示词，返回文本与削减说明。

    削减顺序：优先级大的先削减；同优先级时先削减更长的那个 —— 同样是削减
    一个片段，先削减长的更快腾出空间，削减的总片段数也就更少。

    不可裁的片段本身就超出预算时抛 AG-3001。此时它表示的是「这个请求在这个
    模型上放不下」，而不是「上下文管理没做」。
    """
    chosen: dict[str, str | None] = {s.name: s.text for s in sections}
    notes: list[str] = []

    def total() -> int:
        return estimate(_join(chosen))

    reducible = sorted(
        (s for s in sections if s.priority > 0),
        key=lambda s: (s.priority, estimate(s.text)),
        reverse=True,
    )

    for section in reducible:
        if total() <= limit:
            break
        chosen[section.name] = section.compact
        notes.append(f"{section.name}→" + ("紧凑形式" if section.compact else "已丢弃"))

    if total() > limit:
        raise AgentError(
            ErrorCode.AG_CONTEXT_OVERFLOW,
            f"不可裁减的内容已占 {total()}，超出预算 {limit}",
            detail={"required": total(), "limit": limit,
                    "kept": [n for n, v in chosen.items() if v is not None]},
        )
    return _join(chosen), notes


def _join(chosen: dict[str, str | None]) -> str:
    return SEPARATOR.join(v for v in chosen.values() if v)
