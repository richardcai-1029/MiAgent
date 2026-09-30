"""Scheduler：确定性依赖解析与并发派发。不调用模型。"""

from __future__ import annotations

from typing import Any

from ..core import dag, dataflow
from ..core.state import AgentState, DispatchItem, TaskStatus
from ..protocol import ErrorCode
from ..tools import ToolSource
from .deps import Deps


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
                "trace": [f"scheduler: 全部完成（成功 {n_ok} / 失败 {n_bad}）→ 完成校验"]}

    # 累计执行预算。此前只在失败分支里检查，因而完全约束不住顺利执行的流程 ——
    # 模型拆出多少任务就执行多少次。预算要能兜住的恰恰是这种情况。
    if state["execution_count"] >= deps.limits.max_total_executions:
        return {"tasks": tasks, "dispatch": [], "verdict": None,
                "failure": ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value,
                "trace": [f"scheduler: 已执行 {state['execution_count']} 次，"
                          f"达到累计预算 {deps.limits.max_total_executions} → finalizer"]}

    if dag.is_deadlocked(tasks):
        return {"tasks": tasks, "dispatch": [], "verdict": None,
                "failure": ErrorCode.AG_DEPENDENCY_UNRESOLVED.value,
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
