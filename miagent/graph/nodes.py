"""图的六类节点。

每个节点都是「读 State，返回要更新的字段」的普通函数。

职责划分的一条硬线：

    交给模型的： 把目标拆成什么任务、任务之间有什么依赖、失败后换什么方案
    绝不交给的： 依赖是否满足、是否超出重试上限、是否全部完成、
                 哪些可以并行、是否死锁

  后者都有确定答案，写在 dag.py 里，是纯函数、可穷举测试。
  用模型去猜一个我们确定知道答案的问题，既慢又不可靠。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ..llm import system, user
from ..llm.context import Section, fit
from ..memory import Episode, anchor as anchor_mod, episodic
from ..memory.session import render_history
from ..protocol import AgentError, ErrorCode, RetryPolicy
from ..tools import ToolRegistry, ToolSource
from . import dag, dataflow
from .nodes_meta import Deps
from .schema import FinalOutput, task_plan_model_for
from .state import (
    MAX_ATTEMPTS_PER_TASK,
    DispatchItem,
    MAX_REPLANS,
    MAX_TOTAL_EXECUTIONS,
    AgentState,
    Task,
    TaskOutcome,
    TaskStatus,
    new_task,
)


# ============================================================
# 规划：模型输出 → 校验过的任务图
# ============================================================


def _build_tasks(specs: list[Any], prefix: str = "") -> dict[str, Task]:
    """把模型产出的 TaskSpec 列表转成运行时任务图。

    prefix 用于重规划：新任务的 id 加前缀，避免与已完成任务撞名。
    依赖与参数引用里指向新任务的 id 同步加前缀，指向已完成任务的保持原样。

    参数里的引用会派生出依赖边（见 dataflow 模块）：模型只要写了引用，
    先后关系就已经确定，不必再指望它在 dependencies 里重复声明一遍。
    """
    new_ids = {s.id: prefix + s.id for s in specs}
    tasks: dict[str, Task] = {}
    for s in specs:
        arguments = dataflow.remap_references(s.arguments, new_ids)
        deps = [new_ids.get(d, d) for d in s.dependencies]
        derived = dataflow.referenced_ids(arguments) - set(deps)
        tasks[new_ids[s.id]] = new_task(
            new_ids[s.id], s.description, s.required_tool, arguments,
            deps + sorted(derived))
    return tasks


def _plan(deps: Deps, role: str, sections: list[Section], prefix: str = "",
          keep: dict[str, Task] | None = None) -> tuple[dict[str, Task], list[str]]:
    """要求模型产出任务图，并做结构校验。返回任务图与上下文削减说明。

    提示词按预算拼装：端侧窗口小，工具一多、历史一长就会触顶，
    此时削减低优先级片段，而不是直接拒绝（见 llm/context.py）。

    两道关卡：
      · schema 校验（AG-1001）—— 格式、字段、工具名是否合法
      · 图结构校验（AG-1004）—— 依赖是否存在、有无自依赖、有无环
    格式对但结构错的图必须在执行前拦下，否则会表现为莫名其妙的死锁。
    """
    model = task_plan_model_for([t.name for t in deps.registry])

    # complete_structured 会另外注入一份 schema，预算里必须给它留出位置，
    # 否则拼好的提示词加上 schema 仍会超窗。
    schema_json = json.dumps(model.model_json_schema(), ensure_ascii=False)
    reserve = deps.llm.estimate(schema_json) + deps.llm.estimate(role)
    body, notes = fit(sections, deps.llm.context_limit - reserve, deps.llm.estimate)

    try:
        parsed = deps.llm.complete_structured([system(role), user(body)], model)
    except AgentError as e:
        if e.code is ErrorCode.AG_LLM_INVALID_RESPONSE:
            # 分层：模型层说「输出不合 schema」，规划层说「没能产出可执行计划」
            raise AgentError(ErrorCode.AG_PLAN_PARSE_FAILED,
                             "模型未能产出合法任务图", detail=e.detail) from e
        raise

    tasks = {**(keep or {}), **_build_tasks(parsed.tasks, prefix)}
    dag.validate(tasks)          # 结构非法 -> AG-1004

    # 规模合理性：拆得过细会白白消耗端侧算力，每一步都是一次真实调用。
    # 上限直接取累计执行预算，不另立一个数字 —— 待执行任务数超过预算的计划
    # 在预算内必然跑不完，与其执行到一半才发现，不如现在就拒绝。
    planned = len(tasks) - len(keep or {})
    if planned > MAX_TOTAL_EXECUTIONS:
        raise AgentError(
            ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED,
            f"计划拆出 {planned} 个任务，超过累计执行预算 {MAX_TOTAL_EXECUTIONS}",
            detail={"planned": planned, "budget": MAX_TOTAL_EXECUTIONS})
    return tasks, notes


def _describe_tools(registry: ToolRegistry) -> str:
    return json.dumps(registry.to_model_schemas(), ensure_ascii=False, indent=2)


def _tools_section(registry: ToolRegistry) -> Section:
    """工具描述：预算不够时削减为只剩工具名。

    完整 schema 是规划质量的主要输入，但窗口触顶时保留工具名，
    模型仍有机会选对工具并配合自修复补齐参数；整段丢掉则连选都无从选起。
    """
    return Section(
        "工具描述",
        f"可用工具：\n{_describe_tools(registry)}",
        priority=1,
        compact="可用工具：" + "、".join(t.name for t in registry),
    )


def _trace_trim(notes: list[str]) -> str:
    """把削减说明拼进 trace。裁掉了什么必须能看见。"""
    return f"（上下文削减：{'，'.join(notes)}）" if notes else ""


# ============================================================
# ① Planner —— 用户目标 → 结构化任务图
# ============================================================

_PLANNER_ROLE = (
    "你是端侧智能助理的任务规划器。把用户目标拆解为最少的可执行任务，"
    "并用 dependencies 显式表达任务之间的先后关系。"
    "没有先后关系的任务不要写依赖 —— 它们会被并行执行。"
    "某个参数要用到前一个任务的结果时，把该参数的值写成 "
    '{"$from": "那个任务的 id"}，先后关系会由此自动确定。'
    "若给出了对话历史，当前目标可能承接之前的结论，以历史中的结论为准。"
)


def planner(state: AgentState, deps: Deps) -> dict[str, Any]:
    try:
        tasks, notes = _plan(deps, _PLANNER_ROLE, [
            _tools_section(deps.registry),
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
    return {
        "tasks": tasks, "verdict": None,
        # 目标锚在这里写入，此后只读：之后每一轮重规划都以首次拆解为参照。
        "anchor": anchor_mod.build(state["user_request"], tasks),
        "trace": [f"planner: {len(tasks)} 个任务，{len(layers)} 层依赖 → {layers}"
                  + _trace_trim(notes)],
    }


# ============================================================
# ② Scheduler —— 确定性依赖解析
# ============================================================


def scheduler(state: AgentState, deps: Deps) -> dict[str, Any]:
    """全部逻辑都是确定性的，不调用模型。

        级联失败 → 判定完成 / 死锁 → 找出就绪任务 → 按并发配额派发
    """
    tasks = dag.cascade_failures(state["tasks"])

    if dag.is_complete(tasks):
        n_ok, n_bad = len(dag.completed_tasks(tasks)), len(dag.failed_tasks(tasks))
        # 这里不追加失败判定：保留下来的失败任务是执行历史，未必代表目标没达成
        # —— 重规划成功接手时，前一条路失败恰恰是正常剧情。
        # 「恢复机制交了白卷」由 Replanner 自己判定，见该节点。
        return {"tasks": tasks, "dispatch": [], "verdict": None,
                "execution_summary": episodic.summarize(tasks, state["episodes"]),
                "trace": [f"scheduler: 全部完成（成功 {n_ok} / 失败 {n_bad}）→ finalizer"]}

    # 累计执行预算。此前只在失败分支里检查，因而完全约束不住顺利执行的流程 ——
    # 模型拆出多少任务就执行多少次。预算要能兜住的恰恰是这种情况。
    if state["execution_count"] >= MAX_TOTAL_EXECUTIONS:
        return {"tasks": tasks, "dispatch": [], "verdict": None,
                "failure": ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value,
                "execution_summary": episodic.summarize(tasks, state["episodes"]),
                "trace": [f"scheduler: 已执行 {state['execution_count']} 次，"
                          f"达到累计预算 {MAX_TOTAL_EXECUTIONS} → finalizer"]}

    if dag.is_deadlocked(tasks):
        return {"tasks": tasks, "dispatch": [], "verdict": None,
                "failure": ErrorCode.AG_DEPENDENCY_UNRESOLVED.value,
                "execution_summary": episodic.summarize(tasks, state["episodes"]),
                "trace": ["scheduler: 依赖无法满足，调度死锁 → finalizer"]}

    ready_ids = dag.ready(tasks)

    # ★ 并行派发：同一轮就绪的任务彼此无依赖，可以同时执行。
    #   MiClaw 侧受握手时下发的并发配额限制（清单 C-6）；
    #   本地工具不设限 —— 它们受 GIL 约束，并发也不会更快。
    dispatch: list[DispatchItem] = []
    miclaw_used = 0
    deferred = 0
    for tid in ready_ids:
        tool = deps.registry.get(tasks[tid]["required_tool"])
        route = "miclaw" if tool is not None and tool.source is ToolSource.MICLAW else "local"
        if route == "miclaw":
            if miclaw_used >= deps.max_concurrent_miclaw:
                deferred += 1        # 超出配额的留到下一轮
                continue
            miclaw_used += 1
        tasks[tid]["status"] = TaskStatus.RUNNING
        # 参数里的引用在这里落实成上游任务的结果。放在派发前做，
        # 执行节点因此拿到的是一份参数已经确定的任务，不需要知道数据流的存在。
        ready_task = dict(tasks[tid])
        ready_task["arguments"] = dataflow.resolve(tasks[tid]["arguments"], tasks)
        dispatch.append(DispatchItem(task=ready_task, route=route))

    note = f"，{deferred} 个因并发配额顺延" if deferred else ""
    names = ", ".join(f"{d['task']['id']}({d['route']})" for d in dispatch)
    return {
        "tasks": tasks, "dispatch": dispatch, "verdict": None,
        "trace": [f"scheduler: 本轮就绪 {len(ready_ids)} 个，派发 {len(dispatch)} 个"
                  f"{note} → {names}"],
    }


def _root_failure(episodes: list[Episode]) -> str | None:
    """从失败记录里挑出根因的错误码。

    级联失败的错误码统一是 AG-1005（前置任务失败），它只说明"被牵连"，
    对用户没有信息量。优先取自身失败的那一个；同为自身失败时取最近一轮的。
    """
    failed = [e for e in episodes if not e["ok"]]
    if not failed:
        return None
    own = [e for e in failed
           if e["error"] != ErrorCode.AG_DEPENDENCY_UNRESOLVED.value]
    return max(own or failed, key=lambda e: e["generation"])["error"]


# ============================================================
# ③④ 执行 —— 两个节点共用这一份实现
# ============================================================


def execute(payload: dict[str, Any], deps: Deps, source: ToolSource) -> dict[str, Any]:
    """本地工具与 MiClaw 工具的共用执行体。

    ★ 这个节点是被 Send 扇出调用的，payload 就是它看到的【全部 state】——
      里面只有 Send 携带的那一个任务，读不到 tasks、execution_count 等主状态字段。
      因此返回的是【增量】：outcomes 追加一条、execution_count 加一，
      由 reducer 汇总。若返回绝对值，多个分支会互相覆盖。

    现在两条路径一致。将来 MiClaw 侧要加批量合并请求、按工具粒度的超时、
    配额预筛时，只在这个函数里分叉，图的结构不用动。
    """
    task = payload["task"]
    attempt = task["retry_count"] + 1

    # 退避。能走到重试的只有段位 1（传输抖动）与段位 2（协议状态）两类失败，
    # 它们的共同点是「过一会儿可能就好了」；立即重发时状况还没来得及改变。
    if attempt > 1:
        deps.sleep(deps.retry_delay_ms / 1000)

    result = deps.registry.invoke(task["required_tool"], task["arguments"])

    outcome = TaskOutcome(
        task_id=task["id"], tool=task["required_tool"], ok=not result.is_error,
        # 存 for_model()：失败时它带着「下一步该怎么办」，
        # Replanner 与 Finalizer 拿到的就是可直接用的信息
        content=result.for_model(),
        error_code=result.error_code.value if result.error_code else None,
        retry_policy=result.retry_policy.value, attempt=attempt,
    )
    mark = "✓" if outcome["ok"] else f"✗ {outcome['error_code']}"
    waited = f"，退避 {deps.retry_delay_ms}ms 后" if attempt > 1 else ""
    return {"outcomes": [outcome], "execution_count": 1,
            "trace": [f"{source.value}: {task['id']} {mark}（{waited}第 {attempt} 次）"]}


# ============================================================
# ⑤ Evaluator —— 判定并落库任务状态
# ============================================================

_RETRIABLE = {RetryPolicy.BACKOFF.value, RetryPolicy.REHANDSHAKE.value}


# 判定优先级：任一任务需要中止就中止，其次重规划，其次重试
_VERDICT_RANK = {"success": 0, "retry": 1, "replan": 2, "abort": 3}


def evaluator(state: AgentState, deps: Deps) -> dict[str, Any]:
    """★ 不调用大模型。

    「该重试还是该重规划」由错误码的段位直接决定 —— 传输抖动重试，
    资源不足换方案。这是错误码分段设计的最终兑现。

    并行执行时本轮可能有多条结果，逐条落库后按优先级聚合成一个判定：
    只要有任务需要重规划，整轮就走 Replanner —— 它拿到的是完整任务图，
    能同时看到本轮成功与失败的部分。
    """
    tasks = {tid: dict(t) for tid, t in state["tasks"].items()}
    verdicts: list[str] = []
    failure = None
    errors: list[dict[str, Any]] = []
    lines: list[str] = []

    for last in state["outcomes"]:
        task = tasks[last["task_id"]]
        task["retry_count"] = last["attempt"]

        if last["ok"]:
            task["status"], task["result"] = TaskStatus.DONE, last["content"]
            verdict = "success"
        else:
            task["error"] = last["error_code"]
            errors.append({"task": task["id"], "code": last["error_code"],
                           "attempt": last["attempt"]})
            if state["execution_count"] >= MAX_TOTAL_EXECUTIONS:
                task["status"], task["result"] = TaskStatus.FAILED, last["content"]
                verdict = "abort"
                failure = ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value
            elif (last["retry_policy"] in _RETRIABLE
                  and last["attempt"] < MAX_ATTEMPTS_PER_TASK):
                # 退回 pending，让 Scheduler 下一轮重新派发同一个任务
                task["status"], verdict = TaskStatus.PENDING, "retry"
            elif state["replan_count"] < MAX_REPLANS:
                task["status"], task["result"] = TaskStatus.FAILED, last["content"]
                verdict = "replan"
            else:
                task["status"], task["result"] = TaskStatus.FAILED, last["content"]
                verdict = "abort"
                failure = ErrorCode.AG_PLAN_NO_PROGRESS.value

        verdicts.append(verdict)
        lines.append(f"{task['id']}→{verdict}")

    final = max(verdicts, key=lambda v: _VERDICT_RANK[v]) if verdicts else "success"
    out: dict[str, Any] = {
        "tasks": tasks, "verdict": final, "errors": errors,
        "outcomes": [],          # 空列表触发 reducer 重置，避免下一轮重复消费
        "trace": [f"evaluator: {' '.join(lines)} → {final}"
                  + (f"（{failure}）" if failure else "")]}
    # 只在判定中止时写 failure。Replanner 失败后剩余任务仍会继续执行，
    # 它们成功不代表恢复发生了 —— 无条件写回 None 会把那个失败抹掉。
    if failure:
        out["failure"] = failure
    return out


# ============================================================
# ⑥ Replanner —— 带着失败上下文重建剩余任务图
# ============================================================


def replanner(state: AgentState, deps: Deps) -> dict[str, Any]:
    """重规划保留已完成的任务，只重建剩余部分。

    否则一次失败会让前面成功的工作全部重做 —— 端侧尤其浪费，
    每次重做都是一次真实的系统调用。

    终态任务在这里离开任务图：先结算为情景记忆，再只把新计划引用到的
    已完成任务留在图里。图里只装还要调度的东西，历史交给 episodes ——
    否则每一轮重规划都带着越来越长的历史，挤占端侧本就小的窗口。
    """
    tasks = state["tasks"]
    ended = state["replan_count"]            # 刚结束的这一轮
    generation = ended + 1
    settled = episodic.settle(tasks, state["episodes"], ended)
    episodes = [*state["episodes"], *settled]
    done = {tid: t for tid, t in tasks.items() if t["status"] is TaskStatus.DONE}

    done_text, failed_text = episodic.render(episodes, ended)
    shown = episodic.dedupe(episodes)
    n_done = sum(1 for e in shown if e["ok"])
    n_failed = len(shown) - n_done
    try:
        merged, notes = _plan(
            deps,
            _PLANNER_ROLE.replace("任务规划器", "任务重规划器")
            + " 原任务图中有任务失败了，请基于已完成的结果重新规划【剩余】工作，"
              "避开失败的做法。新任务可以依赖已完成任务的 id。",
            [
                # 目标锚放最前、不可裁：几轮重规划之后提示词里全是局部的成败记录，
                # 新计划要有一个「原本要做什么」可以对照，否则越走越偏且无从察觉。
                anchor_mod.render(state["anchor"]),
                _tools_section(deps.registry),
                # 已完成的结果原文最先被削减：重规划真正需要的是「哪条路走不通」，
                # 已完成部分只要知道有多少即可，具体结果下游任务可以再引用。
                Section("已完成", f"已完成：\n{done_text}", priority=3,
                        compact=f"已完成 {n_done} 个任务（结果从略）"),
                # 失败信息是重规划的主要依据，比已完成结果后削减。
                Section("失败", f"失败：\n{failed_text}", priority=2,
                        compact=f"有 {n_failed} 个任务失败（详情从略）"),
                Section("要求", "请给出剩余任务。"),
            ],
            prefix=f"r{generation}_",
            keep=done,          # 新任务可以依赖已完成任务；失败任务只存在于情景记忆
        )
    except AgentError as e:
        return {"episodes": settled, "replan_count": generation, "failure": e.code.value,
                "errors": [{"stage": "replanner", "code": e.code.value}],
                "trace": [f"replanner: 重规划失败 {e.code}"]}

    new_ids = set(merged) - set(done)
    added = len(new_ids)

    # 原地打转：新计划里的每一个调用都失败过。执行它们只会得到同样的失败，
    # 白白消耗端侧算力与执行预算。据实判定为无进展循环（AG-1003），不派发。
    if episodic.repeats_failures({tid: merged[tid] for tid in new_ids}, episodes):
        code = ErrorCode.AG_PLAN_NO_PROGRESS
        return {"tasks": {}, "episodes": settled, "dispatch": [], "verdict": None,
                "failure": code.value, "replan_count": generation,
                "errors": [{"stage": "replanner", "code": code.value,
                            "repeated": sorted(new_ids)}],
                "trace": [f"replanner: 第 {generation} 次重规划，新增 {added} 个任务"
                          f"全部是失败过的调用，判定无进展（{code}）" + _trace_trim(notes)]}

    # 只留新任务引用到的已完成任务（含其上游）：validate 要求依赖边指向图里
    # 存在的任务，且派发前要取它们的结果。其余终态任务已在情景记忆里。
    needed = dag.ancestors(merged, new_ids)
    pruned = {tid: t for tid, t in merged.items() if tid in new_ids or tid in needed}

    # 重规划是失败之后的恢复机制。它跑完却一个新任务都没产出，说明恢复没有
    # 发生，而失败的部分不会再有人接手 —— 若照常收尾，用户会拿到一个声称完成、
    # 实则漏做了事情的回答。据实标记为未达成，错误码取根因。
    failure = _root_failure(episodes) if added == 0 else None
    note = f"，未产出新任务，判定未达成（{failure}）" if failure else ""
    return {"tasks": pruned, "episodes": settled, "dispatch": [], "verdict": None,
            "failure": failure, "replan_count": generation,
            "trace": [f"replanner: 第 {generation} 次重规划，结算 {len(settled)} 条记录，"
                      f"保留 {len(needed)} 个已完成供引用，新增 {added} 个任务{note}"
                      + _trace_trim(notes)]}


# ============================================================
# ⑦ Finalizer —— 汇总成给用户的回答
# ============================================================


def finalizer(state: AgentState, deps: Deps) -> dict[str, Any]:
    tasks = dag.cascade_failures(state["tasks"])
    # 结算剩余的终态任务：收尾之后 episodes 就是这一轮完整的执行历史。
    settled = episodic.settle(tasks, state["episodes"], state["replan_count"])
    episodes = [*state["episodes"], *settled]
    summary = episodic.summarize(tasks, episodes)

    detail = "\n".join(episodic.history(episodes, tasks)) or "  （未执行任何任务）"

    if state.get("failure"):
        ask = (f"任务未能完成（错误码 {state['failure']}）。回答请说明做到了哪一步、"
               f"卡在哪里、建议用户怎么办。")
    else:
        ask = "请根据执行结果回答用户。"
    ask += " 另给出本轮摘要，供下一轮规划参考。"

    role = "你是端侧智能助理。回答要简洁，只说结论，不复述过程。"
    # 与 _plan 相同：complete_structured 会另外注入一份 schema，预算里要留出位置。
    schema_json = json.dumps(FinalOutput.model_json_schema(), ensure_ascii=False)
    reserve = deps.llm.estimate(schema_json) + deps.llm.estimate(role)
    try:
        body, notes = fit([
            Section("用户目标", f"用户目标：{state['user_request']}"),
            # 执行明细可能很长；削减后只剩成败计数，仍足以给出一句可用的回答。
            Section("执行情况", f"执行情况：\n{detail}", priority=2,
                    compact=f"执行情况：成功 {len(summary['completed'])} 个、"
                            f"失败 {len(summary['failed'])} 个（明细从略）"),
            Section("要求", ask),
        ], deps.llm.context_limit - reserve, deps.llm.estimate)
        output = deps.llm.complete_structured([system(role), user(body)], FinalOutput)
        answer, turn_summary = output.answer, output.summary
        note = _trace_trim(notes)
    except AgentError as e:
        if e.code not in (ErrorCode.AG_CONTEXT_OVERFLOW, ErrorCode.AG_LLM_INVALID_RESPONSE):
            raise
        # 窗口放不下，或自修复后输出仍不合 schema。回答不该因此缺席 ——
        # 用确定性的执行概况兜底，用户至少知道做到了哪一步；摘要同源，
        # 保证下一轮总有东西可参考。
        answer = turn_summary = _summary_answer(state, summary)
        why = "上下文放不下" if e.code is ErrorCode.AG_CONTEXT_OVERFLOW else "模型输出不合 schema"
        note = f"（{why}，改用确定性摘要：{e.code}）"

    return {"tasks": tasks, "episodes": settled, "final_answer": answer,
            "turn_summary": turn_summary, "execution_summary": summary,
            "trace": ["finalizer: 已生成回答" + note]}


def _summary_answer(state: AgentState, summary: dict[str, Any]) -> str:
    """不经模型的回答。窗口放不下时的兜底，内容完全由执行结果决定。"""
    done, failed = len(summary["completed"]), len(summary["failed"])
    head = f"已完成 {done} 项、失败 {failed} 项。"
    if state.get("failure"):
        return head + f"任务未能完成，错误码 {state['failure']}。"
    return head
