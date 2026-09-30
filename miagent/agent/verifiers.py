"""ResultVerifier 与 GoalVerifier：结果与完成状态的语义判定。

判定的效力边界写在 core.verify，这里只负责把要校验的东西拼成提示词、
把模型的回答取回来。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel

from ..core import verify
from ..core.schema import GoalReview, result_review_model_for
from ..core.state import AgentState, Review, TaskOutcome
from ..llm import system, user
from ..llm.context import Section, choose, join
from ..memory import anchor as anchor_mod, ledger
from ..protocol import AgentError, ErrorCode
from .deps import Deps
from .prompting import tools_section, trim_note


# ★ 校验不看残缺的记录：基于残缺记录做出的校验结论，错在「把做成了的事
#   判成没做成」这一侧，比不校验更糟。因此记录只有「整条给模型看」与
#   「整条不看」两种状态，从不换成去掉结果原文的形式：
#     · 结果校验逐项独立，放不下的那一项整项不交给模型，按原判定处理，
#       其余项照常校验；
#     · 完成校验要看全部记录才能下结论，执行记录不可裁，放不下就整个不校验。
#   不可裁的部分本身放不下时，由调用方按 AG-3001 退回原判定。工具描述先于待校验项削减 ——
#   少了它至多是给不出修正参数，判定本身不受影响。


Chosen = dict[str, str | None]    # 片段名 → 采用的文本，None 表示已丢弃


def _review(deps: Deps, role: str, sections: list[Section], model: type[BaseModel],
            narrow: Callable[[Chosen], type[BaseModel] | None] | None = None,
            ) -> tuple[BaseModel | None, Chosen, list[str]]:
    """拼提示词、调模型，返回判定、各片段的取舍（见 context.choose）与削减说明。

    narrow 给定时按留下的片段收紧 schema：返回 None 表示没有可交给模型的内容，
    此时不调模型，判定为 None。预算按 model 的 schema 留 —— 收紧后的 schema
    只会更短。
    """
    reserve = deps.llm.structured_reserve(model) + deps.llm.estimate(role)
    chosen, notes = choose(sections, deps.llm.context_limit - reserve, deps.llm.estimate)
    final = narrow(chosen) if narrow is not None else model
    if final is None:
        return None, chosen, notes
    return deps.llm.complete_structured([system(role), user(join(chosen))], final), chosen, notes


_RESULT_VERIFIER_ROLE = (
    "你是端侧智能助理的结果校验器。逐个判断任务的执行结果是否达成了该任务要做的事，"
    "只依据给出的信息判断，不要设想没有写出来的内容。"
    "结果确实是任务要的东西时 passed 为 true；"
    "结果与任务不符（查到的不是要查的对象、创建的内容与要求不一致、结果为空或答非所问）时 passed 为 false。"
    "只有当失败或不符的原因是参数写错、且用户目标里已经给出了正确取值时，"
    "才在 corrected_arguments 里给出修正后的完整参数；推断不出正确取值就填 null，不要猜。"
    "每个待校验任务给且只给一条判定。"
)


def _review_items(state: AgentState, pending: list[TaskOutcome]) -> list[Section]:
    """待校验清单，每项一段，放不下的项整项丢弃（见上方 ★）。

    参数取实际传入工具的那一份（引用已求值），否则模型看到的是 `$from`
    而不是工具真正收到的东西。
    """
    dispatched = {d["task"]["id"]: d["task"] for d in state.get("dispatch") or []}
    sections = [Section("待校验", "待校验的执行结果：")]
    for o in pending:
        task = dispatched.get(o["task_id"]) or state["tasks"][o["task_id"]]
        args = json.dumps(task["arguments"], ensure_ascii=False)
        outcome = (f"  执行结果：{o['content']}" if o["ok"]
                   else f"  调用失败：{o['content']}")
        sections.append(Section(_review_item(o["task_id"]),
                                f"任务 {o['task_id']}：{task['description']}\n"
                                f"  调用：{o['tool']}，参数 {args}\n{outcome}",
                                priority=1))
    return sections


def _review_item(task_id: str) -> str:
    return f"待校验·{task_id}"


def _reviewed(pending: list[TaskOutcome], chosen: Chosen) -> list[TaskOutcome]:
    """放得下、交给了模型的待校验项。"""
    return [o for o in pending if chosen[_review_item(o["task_id"])] is not None]


def result_verifier(state: AgentState, deps: Deps) -> dict[str, Any]:
    """对本轮的执行结果做语义校验，产出判定，不改任务状态。

    落库仍归 Evaluator：判定与规则分在两处，才能一边换判定来源、
    一边保证「该重试还是该重规划」始终由同一套确定性规则推出。
    """
    outcomes = state.get("outcomes") or []
    pending = [o for o in outcomes if verify.needs_review(o)]
    if not deps.verify or not pending:
        why = "已关闭" if not deps.verify else "本轮无可校验项（失败原因已确定）"
        return {"trace": [f"result_verifier: 跳过结果校验（{why}）"]}

    tools = {o["tool"] for o in pending}
    sections = [
        Section("用户目标", f"用户目标：{state['user_request']}"),
        tools_section(deps.registry, tools, priority=2),
        *_review_items(state, pending),
        Section("要求", "请逐条判断结果是否达成了该任务要做的事。"),
    ]

    def narrow(chosen: Chosen) -> type[BaseModel] | None:
        kept = _reviewed(pending, chosen)
        return result_review_model_for([o["task_id"] for o in kept]) if kept else None

    try:
        batch, chosen, notes = _review(
            deps, _RESULT_VERIFIER_ROLE, sections,
            result_review_model_for([o["task_id"] for o in pending]), narrow)
    except AgentError as e:
        # 校验不可用（窗口放不下、输出始终不合 schema、模型不可达）时退回原判定：
        # 校验只收紧不放宽，因此它缺席只是回到没有语义校验的判定，不会误伤。
        return {"trace": [f"result_verifier: 校验未完成（{e.code}），本轮按原判定处理"]}

    if batch is None:
        return {"trace": ["result_verifier: 待校验项都放不下，本轮按原判定处理" + trim_note(notes)]}

    kept = _reviewed(pending, chosen)
    reviews, lines = _accept_reviews(state, deps, batch, kept)
    skipped = len(pending) - len(kept)
    return {"reviews": reviews,
            "trace": [f"result_verifier: 校验 {len(kept)} 项 → {'；'.join(lines)}"
                      + (f"；{skipped} 项放不下，按原判定处理" if skipped else "")
                      + trim_note(notes)]}


def _accept_reviews(state: AgentState, deps: Deps, batch: Any,
                    pending: list[TaskOutcome]) -> tuple[list[Review], list[str]]:
    """把模型的判定过一遍确定性的闸，返回可落库的判定与 trace 说明。

    漏判的任务按通过处理：判定缺席不该让一个成功的调用变成失败。
    """
    judged = {r.task_id: r for r in batch.reviews}
    reviews: list[Review] = []
    lines: list[str] = []
    for o in pending:
        verdict = judged.get(o["task_id"])
        if verdict is None or verdict.passed:
            continue
        task = state["tasks"][o["task_id"]]
        tool = deps.registry.get(task["required_tool"])
        correction, refused = verify.accept_correction(
            task, verdict.corrected_arguments,
            tool.input_schema if tool is not None else None)
        reviews.append(Review(task_id=o["task_id"], ok=False,
                              reason=verdict.reason, correction=correction))
        lines.append(f"{o['task_id']} 未通过（{verdict.reason}）"
                     + (f"，参数已修正为 {json.dumps(correction, ensure_ascii=False)}"
                        if correction else f"，不就地修正：{refused}"))
    if not lines:
        lines.append("全部通过")
    return reviews, lines


_GOAL_VERIFIER_ROLE = (
    "你是端侧智能助理的完成校验器。对照用户目标与最初的拆解，"
    "判断这一轮的执行是否已经把用户要的事做完。"
    "目标的每一点都有对应的执行结果时 achieved 为 true，gap 留空；"
    "还有要点没做、或执行结果没有覆盖它时 achieved 为 false，gap 用一句话说明还差什么。"
    "只依据执行记录判断，不要设想没有写出来的内容，也不要因为回答可以更完善就判未达成。"
)


def goal_verifier(state: AgentState, deps: Deps) -> dict[str, Any]:
    """全部任务都到了终态之后、收尾之前，校验目标是否真的达成。

    「所有任务都到了终态」只说明没有东西可调度了。计划本身漏掉了一步、
    或者每一步都成功却合不成用户要的结果，在任务图这一层没有任何迹象 ——
    这个判断没有确定答案，交给模型，判定未达成时转重规划去补。

    三种情况不花这次推理：校验已关闭、本轮一次工具都没调（没有执行可校验）、
    已经带着错误码（不会声称成功，再判一次没有新信息）。
    """
    if not deps.verify or state["execution_count"] == 0 or state.get("failure"):
        return {}

    s = ledger.survey(state["tasks"], state["episodes"])
    # 规划已认定做不成的部分收尾时自会报未达成；这里只判能做的部分做完没有，
    # 否则会为一件没有工具能做的事去重规划。
    unsupported = state.get("unsupported") or []
    known = ([Section("无法完成", "以下部分没有可用工具，已确定无法完成，判断时不计入：\n"
                      + "\n".join(f"  · {u}" for u in unsupported))] if unsupported else [])
    try:
        review, _, notes = _review(deps, _GOAL_VERIFIER_ROLE, [
            anchor_mod.render(state["anchor"]),
            Section("执行记录", "本轮执行记录：\n"
                    + ("\n".join(line.full for line in s.history) or "  （无）")),
            *known,
            Section("要求", "请判断用户目标是否已经达成。"),
        ], GoalReview)
    except AgentError as e:
        return {"trace": [f"goal_verifier: 完成校验未完成（{e.code}），按已完成收尾"]}

    if review.achieved:
        return {"trace": [f"goal_verifier: 完成校验通过{trim_note(notes)}"]}

    code = ErrorCode.AG_GOAL_NOT_ACHIEVED.value
    out: dict[str, Any] = {
        "failure": code, "gap": review.gap,
        "errors": [{"stage": "goal_verifier", "code": code, "gap": review.gap}]}
    if state["replan_count"] >= deps.limits.max_replans:
        out["trace"] = [f"goal_verifier: 目标未达成（{review.gap}），重规划已达上限 → finalizer"]
        return out
    # 判定归模型，去哪儿归路由：verdict 是节点与路由之间既有的那一个字段。
    out["verdict"] = "replan"
    out["trace"] = [f"goal_verifier: 目标未达成（{review.gap}）→ replanner" + trim_note(notes)]
    return out
