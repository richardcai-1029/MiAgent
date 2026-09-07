"""图的六类节点。

每个节点都是「读 State，返回要更新的字段」的普通函数。

职责划分的一条硬线（设计稿里的「不交给 MiMo 的逻辑」）：

    交给模型的： 把目标拆成什么任务、任务之间有什么依赖、失败后换什么方案
    绝不交给的： 依赖是否满足、是否超出重试上限、是否全部完成、
                 哪些可以并行、是否死锁

  后者都有确定答案，写在 dag.py 里，是纯函数、可穷举测试。
  用模型去猜一个我们确定知道答案的问题，既慢又不可靠。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..llm import LLM, LLMMessage, system, user
from ..protocol import AgentError, ErrorCode, RetryPolicy
from ..tools import ToolRegistry, ToolSource
from . import dag
from .schema import TaskPlan, task_plan_model_for
from .state import (
    MAX_ATTEMPTS_PER_TASK,
    MAX_REPLANS,
    MAX_TOTAL_EXECUTIONS,
    AgentState,
    Task,
    TaskOutcome,
    TaskStatus,
    new_task,
)


@dataclass
class Deps:
    llm: LLM
    registry: ToolRegistry


# ============================================================
# 规划：模型输出 → 校验过的任务图
# ============================================================


def _build_tasks(specs: list[Any], prefix: str = "") -> dict[str, Task]:
    """把模型产出的 TaskSpec 列表转成运行时任务图。

    prefix 用于重规划：新任务的 id 加前缀，避免与已完成任务撞名。
    依赖里指向新任务的 id 同步加前缀，指向已完成任务的保持原样。
    """
    new_ids = {s.id: prefix + s.id for s in specs}
    tasks: dict[str, Task] = {}
    for s in specs:
        deps = [new_ids.get(d, d) for d in s.dependencies]
        tasks[new_ids[s.id]] = new_task(
            new_ids[s.id], s.description, s.required_tool, s.arguments, deps)
    return tasks


def _plan(deps: Deps, messages: list[LLMMessage], prefix: str = "",
          keep: dict[str, Task] | None = None) -> dict[str, Task]:
    """要求模型产出任务图，并做结构校验。

    两道关卡：
      · schema 校验（AG-1001）—— 格式、字段、工具名是否合法
      · 图结构校验（AG-1004）—— 依赖是否存在、有无自依赖、有无环
    格式对但结构错的图必须在执行前拦下，否则会表现为莫名其妙的死锁。
    """
    model = task_plan_model_for([t.name for t in deps.registry])
    try:
        parsed = deps.llm.complete_structured(messages, model)
    except AgentError as e:
        if e.code is ErrorCode.AG_LLM_INVALID_RESPONSE:
            # 分层：模型层说「输出不合 schema」，规划层说「没能产出可执行计划」
            raise AgentError(ErrorCode.AG_PLAN_PARSE_FAILED,
                             "模型未能产出合法任务图", detail=e.detail) from e
        raise

    tasks = {**(keep or {}), **_build_tasks(parsed.tasks, prefix)}
    dag.validate(tasks)          # 结构非法 -> AG-1004
    return tasks


def _describe_tools(registry: ToolRegistry) -> str:
    return json.dumps(registry.to_model_schemas(), ensure_ascii=False, indent=2)


# ============================================================
# ① Planner —— 用户目标 → 结构化任务图
# ============================================================

_PLANNER_ROLE = (
    "你是端侧智能助理的任务规划器。把用户目标拆解为最少的可执行任务，"
    "并用 dependencies 显式表达任务之间的先后关系。"
    "没有先后关系的任务不要写依赖 —— 它们会被并行执行。"
)


def planner(state: AgentState, deps: Deps) -> dict[str, Any]:
    try:
        tasks = _plan(deps, [
            system(_PLANNER_ROLE),
            user(f"可用工具：\n{_describe_tools(deps.registry)}\n\n"
                 f"用户目标：{state['user_request']}"),
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
        "trace": [f"planner: {len(tasks)} 个任务，{len(layers)} 层依赖 → {layers}"],
    }


# ============================================================
# ② Scheduler —— 确定性依赖解析
# ============================================================


def scheduler(state: AgentState, deps: Deps) -> dict[str, Any]:
    """全部逻辑都是确定性的，不调用模型。

        级联失败 → 判定完成 / 死锁 → 找出就绪任务 → 派发
    """
    tasks = dag.cascade_failures(state["tasks"])

    if dag.is_complete(tasks):
        n_ok, n_bad = len(dag.completed_tasks(tasks)), len(dag.failed_tasks(tasks))
        return {"tasks": tasks, "current": None, "route": None, "verdict": None,
                "execution_summary": dag.summary(tasks),
                "trace": [f"scheduler: 全部完成（成功 {n_ok} / 失败 {n_bad}）→ finalizer"]}

    if dag.is_deadlocked(tasks):
        # 还有没跑完的任务，却既无就绪也无在跑 —— 图本身有问题
        return {"tasks": tasks, "current": None, "route": None, "verdict": None,
                "failure": ErrorCode.AG_DEPENDENCY_UNRESOLVED.value,
                "execution_summary": dag.summary(tasks),
                "trace": ["scheduler: 依赖无法满足，调度死锁 → finalizer"]}

    ready_ids = dag.ready(tasks)
    # 串行实现取第一个；ready_ids 长度大于 1 说明这些任务彼此无依赖，
    # 接 LangGraph 的 Send 做并行派发时，直接用整个列表即可。
    task_id = ready_ids[0]
    tasks[task_id]["status"] = TaskStatus.RUNNING

    tool = deps.registry.get(tasks[task_id]["required_tool"])
    route = "miclaw" if tool is not None and tool.source is ToolSource.MICLAW else "local"

    parallel_note = f"（本轮 {len(ready_ids)} 个任务就绪，串行取首个）" if len(ready_ids) > 1 else ""
    return {
        "tasks": tasks, "current": tasks[task_id], "route": route, "verdict": None,
        "trace": [f"scheduler: 派发 {task_id} "
                  f"({tasks[task_id]['required_tool']}) → {route}{parallel_note}"],
    }


# ============================================================
# ③④ 执行 —— 两个节点共用这一份实现
# ============================================================


def execute(state: AgentState, deps: Deps, source: ToolSource) -> dict[str, Any]:
    """本地工具与 MiClaw 工具的共用执行体。

    现在两条路径一致。将来 MiClaw 侧要加并发批处理、按工具粒度的超时、
    配额预筛时，只在这个函数里分叉，图的结构不用动。
    """
    task = state["current"]
    attempt = task["retry_count"] + 1
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
    return {"last": outcome, "execution_count": state["execution_count"] + 1,
            "trace": [f"{source.value}: {task['id']} {mark}（第 {attempt} 次）"]}


# ============================================================
# ⑤ Evaluator —— 判定并落库任务状态
# ============================================================

_RETRIABLE = {RetryPolicy.BACKOFF.value, RetryPolicy.REHANDSHAKE.value}


def evaluator(state: AgentState, deps: Deps) -> dict[str, Any]:
    """★ 不调用大模型。

    「该重试还是该重规划」由错误码的段位直接决定 —— 传输抖动重试，
    资源不足换方案。这是错误码分段设计的最终兑现。
    """
    last = state["last"]
    tasks = {tid: dict(t) for tid, t in state["tasks"].items()}
    task = tasks[last["task_id"]]
    task["retry_count"] = last["attempt"]

    failure = None
    errors: list[dict[str, Any]] = []

    if last["ok"]:
        task["status"], task["result"], verdict = TaskStatus.DONE, last["content"], "success"
    else:
        task["error"] = last["error_code"]
        errors.append({"task": task["id"], "code": last["error_code"],
                       "attempt": last["attempt"]})

        if state["execution_count"] >= MAX_TOTAL_EXECUTIONS:
            task["status"], verdict = TaskStatus.FAILED, "abort"
            task["result"] = last["content"]
            failure = ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value
        elif last["retry_policy"] in _RETRIABLE and last["attempt"] < MAX_ATTEMPTS_PER_TASK:
            # 退回 pending，让 Scheduler 下一轮重新派发同一个任务
            task["status"], verdict = TaskStatus.PENDING, "retry"
        elif state["replan_count"] < MAX_REPLANS:
            task["status"], task["result"], verdict = TaskStatus.FAILED, last["content"], "replan"
        else:
            task["status"], task["result"], verdict = TaskStatus.FAILED, last["content"], "abort"
            failure = ErrorCode.AG_PLAN_NO_PROGRESS.value

    return {"tasks": tasks, "verdict": verdict, "failure": failure, "errors": errors,
            "trace": [f"evaluator: {task['id']} → {verdict}"
                      + (f"（{failure}）" if failure else "")]}


# ============================================================
# ⑥ Replanner —— 带着失败上下文重建剩余任务图
# ============================================================


def replanner(state: AgentState, deps: Deps) -> dict[str, Any]:
    """重规划保留已完成的任务，只重建剩余部分。

    否则一次失败会让前面成功的工作全部重做 —— 端侧尤其浪费，
    每次重做都是一次真实的系统调用。
    """
    tasks = state["tasks"]
    done = {tid: t for tid, t in tasks.items() if t["status"] is TaskStatus.DONE}
    # 已失败的任务也保留：它们是终态，不影响调度，但构成执行历史 ——
    # 丢掉的话用户就看不到「试过什么、为什么没成」。
    # 被丢弃的只有那些尚未执行、已被新计划取代的任务。
    terminal = {tid: t for tid, t in tasks.items()
                if t["status"] in (TaskStatus.DONE, TaskStatus.FAILED)}
    generation = state["replan_count"] + 1

    done_text = "\n".join(f"  {tid}（已完成）: {t['description']} → {t['result']}"
                          for tid, t in done.items()) or "  （无）"
    failed_text = "\n".join(f"  {tid}: {t['description']} → 失败 {t['error']}：{t['result']}"
                            for tid, t in tasks.items()
                            if t["status"] is TaskStatus.FAILED) or "  （无）"

    try:
        merged = _plan(
            deps,
            [system(_PLANNER_ROLE.replace("任务规划器", "任务重规划器")
                    + " 原任务图中有任务失败了，请基于已完成的结果重新规划【剩余】工作，"
                      "避开失败的做法。新任务可以依赖已完成任务的 id。"),
             user(f"可用工具：\n{_describe_tools(deps.registry)}\n\n"
                  f"用户目标：{state['user_request']}\n\n"
                  f"已完成：\n{done_text}\n\n失败：\n{failed_text}\n\n"
                  f"请给出剩余任务。")],
            prefix=f"r{generation}_",
            keep=terminal,      # 已完成/已失败的原样保留，不重做也不丢历史
        )
    except AgentError as e:
        return {"replan_count": generation, "failure": e.code.value,
                "errors": [{"stage": "replanner", "code": e.code.value}],
                "trace": [f"replanner: 重规划失败 {e.code}"]}

    added = len(merged) - len(terminal)
    return {"tasks": merged, "current": None, "verdict": None,
            "replan_count": generation,
            "trace": [f"replanner: 第 {generation} 次重规划，保留 {len(done)} 个已完成、"
                      f"{len(terminal) - len(done)} 个已失败，新增 {added} 个任务"]}


# ============================================================
# ⑦ Finalizer —— 汇总成给用户的回答
# ============================================================


def finalizer(state: AgentState, deps: Deps) -> dict[str, Any]:
    tasks = dag.cascade_failures(state["tasks"])
    summary = dag.summary(tasks)

    detail = "\n".join(
        f"  {tid}: {t['description']} → "
        + ("成功：" + str(t["result"]) if t["status"] is TaskStatus.DONE
           else f"失败（{t['error']}）：{t['result']}")
        for tid, t in tasks.items()
    ) or "  （未执行任何任务）"

    if state.get("failure"):
        ask = (f"任务未能完成（错误码 {state['failure']}）。请说明做到了哪一步、"
               f"卡在哪里、建议用户怎么办。")
    else:
        ask = "请根据执行结果，用一两句话回答用户。"

    answer = deps.llm.complete([
        system("你是端侧智能助理。回答要简洁，只说结论，不复述过程。"),
        user(f"用户目标：{state['user_request']}\n\n执行情况：\n{detail}\n\n{ask}"),
    ]).content

    return {"tasks": tasks, "final_answer": answer, "execution_summary": summary,
            "trace": ["finalizer: 已生成回答"]}
