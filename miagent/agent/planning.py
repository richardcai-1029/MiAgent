"""Planner 与 Replanner：用户目标 → 校验过的任务图。

两个节点共用同一条规划通道（_plan）：拼提示词、结构化输出与自修复、
并入保留任务后的结构复核、规模检查。区别只在交给模型的材料 ——
Replanner 多了目标锚与本轮的成败记录。
"""

from __future__ import annotations

from typing import Any

from ..core import dag, dataflow
from ..core.schema import task_plan_model_for
from ..core.state import AgentState, Task, new_task
from ..llm import LLMMessage, system, user
from ..llm.context import Section, fit
from ..memory import anchor as anchor_mod, ledger
from ..memory.session import render_history
from ..protocol import AgentError, ErrorCode
from .deps import Deps
from .prompting import records, tools_section, trim_note


def _build_tasks(specs: list[Any], prefix: str = "", start: int = 0) -> dict[str, Task]:
    """把模型产出的 TaskSpec 列表转成运行时任务图。

    prefix 用于重规划：新任务的 id 加前缀，避免与已完成任务撞名。
    依赖与参数引用里指向新任务的 id 同步加前缀，指向已完成任务的保持原样。

    规划序号按模型输出的顺序从 start 起编；重规划时接着保留任务的序号往后编。

    参数里的引用会派生出依赖边（见 dataflow 模块）：模型只要写了引用，
    先后关系就已经确定，不必再指望它在 dependencies 里重复声明一遍。
    """
    new_ids = {s.id: prefix + s.id for s in specs}
    tasks: dict[str, Task] = {}
    for i, s in enumerate(specs, start):
        arguments = dataflow.remap_references(s.arguments, new_ids)
        deps = [new_ids.get(d, d) for d in s.dependencies]
        derived = dataflow.referenced_ids(arguments) - set(deps)
        tasks[new_ids[s.id]] = new_task(
            new_ids[s.id], s.description, s.required_tool, arguments,
            deps + sorted(derived), seq=i)
    return tasks


def _plan(deps: Deps, role: str, sections: list[Section], prefix: str = "",
          keep: dict[str, Task] | None = None) -> tuple[dict[str, Task], list[str], list[str]]:
    """要求模型产出任务图，并做结构校验。返回任务图、上下文削减说明，
    以及模型认定没有工具能完成的部分（见 TaskPlan.unsupported_actions）。

    提示词按预算拼装：端侧窗口小，工具一多、历史一长就会触顶，
    此时削减低优先级片段，而不是直接拒绝（见 llm/context.py）。

    两道关卡：
      · schema 校验（AG-1001）—— 格式、字段、工具名，以及计划内部的结构：
        依赖是否存在、有无自依赖、有无环。结构错误与格式错误一样喂回给模型
        自修复，改不对才判失败
      · 图结构校验（AG-1004）—— 加上前缀、并入保留任务之后的整张图再验一遍
    格式对但结构错的图必须在执行前拦下，否则会表现为莫名其妙的死锁。
    """
    model = task_plan_model_for([t.name for t in deps.registry], known=list(keep or {}))

    # complete_structured 会另外注入 schema 说明、必要时再追加一条自修复反馈，
    # 预算里给两者留出位置，否则拼好的提示词在首次调用或自修复时超窗。
    reserve = deps.llm.structured_reserve(model) + deps.llm.estimate(role)
    body, notes = fit(sections, deps.llm.context_limit - reserve, deps.llm.estimate)

    def refit(extra: int) -> list[LLMMessage]:
        """自修复要回传原输出时，按同一套削减规则再腾出 extra 的位置。"""
        more, _ = fit(sections, deps.llm.context_limit - reserve - extra, deps.llm.estimate)
        return [system(role), user(more)]

    try:
        parsed = deps.llm.complete_structured([system(role), user(body)], model, refit=refit)
    except AgentError as e:
        if e.code is ErrorCode.AG_LLM_INVALID_RESPONSE:
            # 分层：模型层说「输出不合 schema」，规划层说「没能产出可执行计划」
            raise AgentError(ErrorCode.AG_PLAN_PARSE_FAILED,
                             "模型未能产出合法任务图", detail=e.detail) from e
        raise

    start = max((t["seq"] for t in (keep or {}).values()), default=-1) + 1
    tasks = {**(keep or {}), **_build_tasks(parsed.tasks, prefix, start)}
    dag.validate(tasks)          # 结构非法 -> AG-1004

    # 规模合理性：拆得过细会白白消耗端侧算力，每一步都是一次真实调用。
    # 上限直接取累计执行预算，不另立一个数字 —— 待执行任务数超过预算的计划
    # 在预算内必然跑不完，与其执行到一半才发现，不如现在就拒绝。
    planned = len(tasks) - len(keep or {})
    if planned > deps.limits.max_total_executions:
        raise AgentError(
            ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED,
            f"计划拆出 {planned} 个任务，超过累计执行预算 {deps.limits.max_total_executions}",
            detail={"planned": planned, "budget": deps.limits.max_total_executions})
    # 直接回答类请求没有「做不成的操作」，模型列出的也不算（见 schema 的说明）
    return tasks, notes, [] if parsed.direct_answer else list(parsed.unsupported_actions)


# 措辞针对小模型的三种常见偏差：凭空加任务、把 $from 引用写成字符串、
# 重查对话历史里已有的结论（qwen2.5-omni-7b 实测均出现过）。
# 只写抽象规则，不放带具体工具或参数的示例：小模型会把示例内容当成任务照抄。
_PLANNER_ROLE = (
    "你是端侧智能助理的任务规划器。把用户目标拆解为最少的可执行任务，"
    "只做用户明确要求的事，不要添加用户没有提出的任务。"
    "用 dependencies 显式表达任务之间的先后关系；"
    "没有先后关系的任务不要写依赖，它们会被并行执行。"
    "参数取值规则：用户已经说出的具体内容（时间、标题、地点等）原样填入参数，"
    "不要为了得到它们再去查询；只有参数值必须来自前一个任务的执行结果时，"
    '才把该参数的值写成 {"$from": "那个任务的 id"} 这样的 JSON 对象'
    "（不要把它加引号写成字符串），先后关系会由此自动确定。"
    "若给出了对话历史，当前目标承接其中的结论：已有的结论直接使用，"
    "不要再规划任务重复查询。"
    "闲聊、常识问答、计算这类直接回答即可的目标，任务列表与 unsupported_actions 都为空，"
    "由收尾直接回答。用户要求设备去执行、但没有任何可用工具能完成的操作，"
    "写进 unsupported_actions，不要用不相干的工具去顶替。"
)


def planner(state: AgentState, deps: Deps) -> dict[str, Any]:
    """首次规划：把用户目标拆成任务图，写入目标锚。规划失败时以空任务图直达收尾。"""
    try:
        tasks, notes, unsupported = _plan(deps, _PLANNER_ROLE, [
            tools_section(deps.registry),
            # 对话历史每轮一段，越旧越先削减；没有历史时这里为空。
            *render_history(state.get("history", [])),
            Section("用户目标", f"用户目标：{state['user_request']}"),
        ])
    except AgentError as e:
        # 规划失败不能炸掉整个调用 —— 空任务图会让 Scheduler 立刻判定完成
        # 并转向 Finalizer，用户至少收到一句解释。
        return {"tasks": {}, "failure": e.code.value,
                "errors": [{"stage": "planner", "code": e.code.value, "detail": e.detail}],
                "trace": [f"planner: 规划失败 {e.code}"]}

    layers = dag.parallel_layers(tasks)
    undo = f"；无法完成：{'、'.join(unsupported)}" if unsupported else ""
    return {
        "tasks": tasks, "verdict": None, "unsupported": unsupported,
        # 目标锚在这里写入，此后只读：之后每一轮重规划都以首次拆解为参照。
        "anchor": anchor_mod.build(state["user_request"], tasks),
        "trace": [f"planner: {len(tasks)} 个任务，{len(layers)} 层依赖 → {layers}"
                  + undo + trim_note(notes)],
    }


def replanner(state: AgentState, deps: Deps) -> dict[str, Any]:
    """重规划保留已完成的任务，只重建剩余部分。

    否则一次失败会让前面成功的工作全部重做 —— 端侧尤其浪费，
    每次重做都是一次真实的系统调用。

    终态任务在这里离开任务图：账本先结算为情景记忆，规划之后再验收 ——
    只把新计划引用到的已完成任务留在图里，判无进展，定 failure。
    图里只装还要调度的东西，历史交给 episodes —— 否则每一轮重规划都带着
    越来越长的历史，挤占端侧本就小的窗口。

    两种入口对应两段不同的交代：任务失败后是「换条路把剩下的做完」，
    完成校验判未达成后是「按指出的缺口把漏掉的补上」。后者没有失败记录，
    只写前一句会让模型去找一个并不存在的失败。
    """
    s = ledger.settle(state["tasks"], state["episodes"], state["replan_count"])
    gap = state.get("gap")
    why = (f" 任务都执行完了，但校验发现目标尚未达成：{gap}。"
           "请只规划补齐这个缺口所需的任务，已经做成的事不要重做。"
           if gap else
           " 原任务图中有任务失败了，请基于已完成的结果重新规划【剩余】工作，"
           "避开失败的做法。")
    try:
        merged, notes, unsupported = _plan(
            deps,
            _PLANNER_ROLE.replace("任务规划器", "任务重规划器")
            + why + "新任务可以依赖已完成任务的 id。",
            [
                # 目标锚放最前、不可裁：几轮重规划之后提示词里全是局部的成败记录，
                # 新计划要有一个「原本要做什么」可以对照，否则越走越偏且无从察觉。
                anchor_mod.render(state["anchor"]),
                tools_section(deps.registry),
                # 已完成的结果原文最先被削减：重规划真正需要的是「哪条路走不通」，
                # 已完成部分只要知道 id 与做过什么，具体结果下游任务可以再引用。
                *records("已完成", s.done, 3),
                # 失败信息是重规划的主要依据，比已完成结果后削减。
                *records("失败", s.failed, 2),
                Section("要求", "请给出剩余任务。"),
            ],
            prefix=f"r{s.generation}_",
            keep=s.keep,        # 新任务可以依赖已完成任务；失败任务只存在于情景记忆
        )
    except AgentError as e:
        return {"episodes": s.new_episodes, "replan_count": s.generation, "gap": None,
                "failure": e.code.value,
                "errors": [{"stage": "replanner", "code": e.code.value}],
                "trace": [f"replanner: 重规划失败 {e.code}"]}

    # standing：完成校验判定的未达成没有对应的失败任务，重规划交白卷时
    # 那个码必须留住，否则「没补上缺口」会被当成「没有问题」。
    a = ledger.accept(s, merged, standing=state.get("failure"), clear=deps.verify)
    known = state.get("unsupported") or []
    out: dict[str, Any] = {
        "tasks": a.tasks, "episodes": s.new_episodes, "dispatch": [], "verdict": None,
        "failure": a.failure, "replan_count": s.generation, "gap": None,
        "unsupported": known + [u for u in unsupported if u not in known],
    }
    if a.repeated:
        out["errors"] = [{"stage": "replanner", "code": a.failure, "repeated": a.repeated}]
        out["trace"] = [f"replanner: 第 {s.generation} 次重规划，新增 {a.added} 个任务"
                        f"全部是失败过的调用，判定无进展（{a.failure}）" + trim_note(notes)]
        return out

    note = f"，未产出新任务，判定未达成（{a.failure}）" if a.failure else ""
    out["trace"] = [f"replanner: 第 {s.generation} 次重规划，结算 {len(s.new_episodes)} 条记录，"
                    f"保留 {a.kept} 个已完成供引用，新增 {a.added} 个任务{note}"
                    + trim_note(notes)]
    return out
