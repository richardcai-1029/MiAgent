"""任务图的依赖解析。

★ 本模块是纯函数：不依赖 LangGraph、不依赖大模型、没有副作用。

  这是刻意的。「哪些任务现在能跑」「是否死锁」「是否全部完成」
  属于调度的正确性核心，必须能被穷举测试。把它们和图节点耦在一起，
  就只能通过跑整张图来间接验证 —— 慢，而且测不全边界。

设计稿里强调「不交给 MiMo 的逻辑」，指的就是这里的全部内容：
依赖是否满足、是否超出重试上限、是否全部完成、哪些可以并行。
这些都有确定答案，用模型去猜既慢又不可靠。
"""

from __future__ import annotations

from typing import Any

from ..protocol import AgentError, ErrorCode
from .state import Task, TaskStatus

Tasks = dict[str, Task]


# ============================================================
# 一、结构校验：在执行任何一步之前就把非法图拦下
# ============================================================


def validate(tasks: Tasks) -> None:
    """校验任务图结构。任一项违反即抛 AG-1004。

    三类非法：依赖指向不存在的任务、自依赖、存在环。
    都必须在执行前拦下 —— 一个环会让调度器永远找不到可跑的任务，
    表现为莫名其妙的死锁，比当场报错难查得多。
    """
    for tid, task in tasks.items():
        for dep in task["dependencies"]:
            if dep == tid:
                raise AgentError(ErrorCode.AG_INVALID_PLAN,
                                 f"任务 {tid} 依赖自身", detail={"task": tid})
            if dep not in tasks:
                raise AgentError(ErrorCode.AG_INVALID_PLAN,
                                 f"任务 {tid} 依赖了不存在的任务 {dep}",
                                 detail={"task": tid, "missing": dep})

    cycle = find_cycle(tasks)
    if cycle:
        raise AgentError(ErrorCode.AG_INVALID_PLAN,
                         f"任务图存在环：{' → '.join(cycle)}",
                         detail={"cycle": cycle})


def find_cycle(tasks: Tasks) -> list[str] | None:
    """用 Kahn 拓扑排序检测环；有环则返回其中一个环上的节点。

    Kahn 的原理：反复摘掉「没有未满足依赖」的节点。能全部摘完就无环；
    摘不完，剩下的必定都在环上。
    """
    indegree = {tid: len(t["dependencies"]) for tid, t in tasks.items()}
    dependents: dict[str, list[str]] = {tid: [] for tid in tasks}
    for tid, task in tasks.items():
        for dep in task["dependencies"]:
            if dep in dependents:
                dependents[dep].append(tid)

    queue = [tid for tid, d in indegree.items() if d == 0]
    removed = 0
    while queue:
        tid = queue.pop()
        removed += 1
        for child in dependents[tid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if removed == len(tasks):
        return None
    return sorted(tid for tid, d in indegree.items() if d > 0)


# ============================================================
# 二、派生视图：全部从 tasks 算出来，不单独存
# ============================================================


def by_status(tasks: Tasks, *status: TaskStatus) -> list[str]:
    return [tid for tid, t in tasks.items() if t["status"] in status]


def completed_tasks(tasks: Tasks) -> list[str]:
    return by_status(tasks, TaskStatus.DONE)


def running_tasks(tasks: Tasks) -> list[str]:
    return by_status(tasks, TaskStatus.RUNNING)


def failed_tasks(tasks: Tasks) -> list[str]:
    return by_status(tasks, TaskStatus.FAILED)


def pending_tasks(tasks: Tasks) -> list[str]:
    return by_status(tasks, TaskStatus.PENDING, TaskStatus.READY)


def tool_results(tasks: Tasks) -> dict[str, str]:
    return {tid: t["result"] for tid, t in tasks.items() if t["result"] is not None}


# ============================================================
# 三、调度判定
# ============================================================


def ready(tasks: Tasks) -> list[str]:
    """返回现在就能执行的任务 id。

    判定条件（设计稿里的核心规则）：

        status 为 pending / ready  且  所有 dependencies 都已 done

    返回的是【列表】而不是单个 —— 长度大于 1 就意味着这些任务之间
    没有依赖关系，可以并行派发。串行实现取第一个即可，
    将来接 LangGraph 的 Send 做并行时，直接用整个列表。
    """
    out = []
    for tid, task in tasks.items():
        if task["status"] not in (TaskStatus.PENDING, TaskStatus.READY):
            continue
        if all(tasks[d]["status"] is TaskStatus.DONE for d in task["dependencies"]):
            out.append(tid)
    return sorted(out)


def cascade_failures(tasks: Tasks) -> Tasks:
    """把「依赖已经失败」的任务一并标记为失败。

    否则它们会永远停在 pending —— 依赖不可能变成 done，
    调度器却仍认为还有活没干完，表现为死锁。
    这类级联要一次传播到底（失败的下游的下游也要标记）。
    """
    result = {tid: dict(t) for tid, t in tasks.items()}
    changed = True
    while changed:
        changed = False
        for tid, task in result.items():
            if task["status"] in (TaskStatus.DONE, TaskStatus.FAILED):
                continue
            broken = [d for d in task["dependencies"]
                      if result[d]["status"] is TaskStatus.FAILED]
            if broken:
                task["status"] = TaskStatus.FAILED
                task["error"] = ErrorCode.AG_DEPENDENCY_UNRESOLVED.value
                task["result"] = f"前置任务 {', '.join(broken)} 失败，本任务无法执行"
                changed = True
    return result  # type: ignore[return-value]


def is_complete(tasks: Tasks) -> bool:
    """全部任务都已 done 或 failed。"""
    return all(t["status"] in (TaskStatus.DONE, TaskStatus.FAILED)
               for t in tasks.values())


def is_deadlocked(tasks: Tasks) -> bool:
    """还有没跑完的任务，却既没有可执行的、也没有正在执行的。

    在 cascade_failures 之后仍出现这种情况，说明图本身有问题
    （通常是环，但 validate 已经拦过；也可能是重规划产出的残缺图）。
    """
    if is_complete(tasks):
        return False
    return not ready(tasks) and not running_tasks(tasks)


# ============================================================
# 四、可观测性
# ============================================================


def parallel_layers(tasks: Tasks) -> list[list[str]]:
    """把任务图按依赖分层：同一层内的任务彼此无依赖，可并行。

    用于展示图的并行度，也是将来 Send 并行派发的依据。
    """
    remaining = {tid: set(t["dependencies"]) for tid, t in tasks.items()}
    layers: list[list[str]] = []
    done: set[str] = set()
    while remaining:
        layer = sorted(tid for tid, deps in remaining.items() if deps <= done)
        if not layer:                      # 有环，validate 应该已经拦过
            break
        layers.append(layer)
        done |= set(layer)
        remaining = {t: d for t, d in remaining.items() if t not in done}
    return layers


def summary(tasks: Tasks) -> dict[str, Any]:
    """执行概况，供 Finalizer 与调用方使用。"""
    return {
        "total": len(tasks),
        "completed": completed_tasks(tasks),
        "failed": failed_tasks(tasks),
        "pending": pending_tasks(tasks),
        "results": tool_results(tasks),
        "parallel_layers": parallel_layers(tasks),
    }
