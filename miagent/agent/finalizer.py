"""Finalizer：汇总成给用户的回答与本轮摘要。"""

from __future__ import annotations

from typing import Any

from ..core.schema import FinalOutput
from ..core.state import AgentState
from ..llm import LLMMessage, system, user
from ..llm.context import Section, fit
from ..memory import ledger
from ..protocol import AgentError, ErrorCode, describe
from .deps import Deps
from .prompting import records, trim_note


def finalizer(state: AgentState, deps: Deps) -> dict[str, Any]:
    """收尾：结算剩余任务，由模型生成回答与本轮摘要；模型给不出时退回确定性摘要。"""
    # 收账：结算剩余的终态任务，收尾之后 episodes 就是这一轮完整的执行历史。
    c = ledger.close(state["tasks"], state["episodes"], state["replan_count"])
    summary = c.summary

    # 规划认定有做不成的部分：即使能做的都做完了，目标也没有达成。
    # 这一条不依赖执行 —— 请求里的事全都做不成时，本轮一次工具都不会调。
    unsupported = state.get("unsupported") or []
    failure = state.get("failure")
    if failure is None and unsupported:
        failure = ErrorCode.AG_GOAL_NOT_ACHIEVED.value
    state = {**state, "failure": failure}

    if unsupported:
        ask = (f"以下部分没有可用的工具，无法完成：{'；'.join(unsupported)}。"
               f"回答请说明做成了什么、哪些做不到，不要声称做到了做不到的部分。")
        if state["failure"] != ErrorCode.AG_GOAL_NOT_ACHIEVED.value:
            ask += f"另有失败：{_failure_text(state['failure'])}。"
    elif state.get("failure"):
        # 带上错误码的中文说明：模型只看到 AG-1001 这样的码时会自行猜测原因
        # （实测编出「请检查网络」），说明来自协议层的同一张表，不另写一份。
        ask = (f"任务未能完成，原因：{_failure_text(state['failure'])}。"
               f"回答请说明做到了哪一步、卡在哪里、建议用户怎么办；不要编造上述以外的原因。")
    else:
        ask = "请根据执行结果回答用户。"
    # 摘要是下一轮规划器唯一能看到的上文：要带具体结论（时间、数值、结果），
    # 只写「查了什么」的过程描述会让下一轮无从承接，只能重查。
    ask += (" 另给出本轮摘要，供下一轮规划参考：写明具体结论（如查到的时间段、"
            "创建的日程），不要只描述做了什么。")

    role = "你是端侧智能助理。回答要简洁，只说结论，不复述过程。"
    # 与 _plan 相同：给注入的 schema 说明与自修复反馈留出位置。
    reserve = deps.llm.structured_reserve(FinalOutput) + deps.llm.estimate(role)
    sections = [
        Section("用户目标", f"用户目标：{state['user_request']}"),
        # 执行明细可能很长。逐条削减，超长的结果先让位：其余记录的结论还在，
        # 回答与本轮摘要才写得出具体内容 —— 摘要是下一轮唯一能承接的上文。
        *records("执行情况", c.history, 2, empty="  （未执行任何任务）"),
        Section("要求", ask),
    ]

    def refit(extra: int) -> list[LLMMessage]:
        more, _ = fit(sections, deps.llm.context_limit - reserve - extra, deps.llm.estimate)
        return [system(role), user(more)]

    try:
        body, notes = fit(sections, deps.llm.context_limit - reserve, deps.llm.estimate)
        output = deps.llm.complete_structured([system(role), user(body)], FinalOutput, refit=refit)
        answer, turn_summary = output.answer, output.summary
        note = trim_note(notes)
    except AgentError as e:
        # 窗口放不下、自修复后输出仍不合 schema、模型不可达。回答不该因此缺席 ——
        # 工具已经执行过，请求以异常结束会让调用方不知道做到了哪一步。
        # 用确定性的执行概况兜底，用户至少知道结果；摘要同源，保证下一轮总有东西可参考。
        answer = turn_summary = _summary_answer(state, summary)
        note = f"（{_FINALIZER_FALLBACK.get(e.code, '模型调用失败')}，改用确定性摘要：{e.code}）"

    return {"tasks": c.tasks, "episodes": c.new_episodes, "final_answer": answer,
            "turn_summary": turn_summary, "execution_summary": summary, "failure": failure,
            "trace": ["finalizer: 已生成回答" + note]}


_FINALIZER_FALLBACK = {
    ErrorCode.AG_CONTEXT_OVERFLOW: "上下文放不下",
    ErrorCode.AG_LLM_INVALID_RESPONSE: "模型输出不合 schema",
    ErrorCode.AG_LLM_UNAVAILABLE: "模型不可达",
}


def _summary_answer(state: AgentState, summary: dict[str, Any]) -> str:
    """不经模型的回答。模型给不出回答时的兜底，内容完全由执行结果决定。"""
    done, failed = len(summary["completed"]), len(summary["failed"])
    head = f"已完成 {done} 项、失败 {failed} 项。"
    if state.get("failure"):
        return head + f"任务未能完成：{_failure_text(state['failure'])}。"
    return head


def _failure_text(code: str) -> str:
    """错误码连同它的中文说明，如「AG-1001（模型输出无法解析为可执行计划）」。"""
    return f"{code}（{describe(ErrorCode(code))}）"
