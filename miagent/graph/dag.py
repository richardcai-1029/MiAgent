"""任务图的依赖解析。

★ 本模块是纯函数：不依赖 LangGraph、不依赖大模型、没有副作用。

  这是刻意的。「哪些任务现在能跑」「是否死锁」「是否全部完成」
  属于调度的正确性核心，必须能被穷举测试。把它们和图节点耦在一起，
  就只能通过跑整张图来间接验证 —— 慢，而且测不全边界。

依赖是否满足、是否超出重试上限、是否全部完成、哪些可以并行、
是否死锁 —— 这些都有确定答案，一律由本模块判定，不交给模型。
"""

from __future__ import annotations


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


def ancestors(tasks: Tasks, roots: set[str]) -> set[str]:
    """roots 直接或间接依赖的全部任务 id（不含 roots 自身）。

    重规划后只有这些已完成任务需要留在图里：新任务要引用它们的结果，
    而 validate 要求每条依赖边都指向图里存在的任务。其余终态任务已进入
    情景记忆，继续留在图里只是冗余。
    """
    out: set[str] = set()
    stack = [d for r in roots if r in tasks for d in tasks[r]["dependencies"]]
    while stack:
        tid = stack.pop()
        if tid in out or tid not in tasks:
            continue
        out.add(tid)
        stack.extend(tasks[tid]["dependencies"])
    return out - roots


# ============================================================
# 三、调度判定
# ============================================================


def downstream_depth(tasks: Tasks) -> dict[str, int]:
    """每个未终结任务之后最长还串着几个未终结任务（叶子为 0）。

    只沿未终结的任务计：已完成的不必再跑，已失败的不会再跑，
    它们都不在剩余的执行路径上。
    """
    live = {tid for tid, t in tasks.items()
            if t["status"] not in (TaskStatus.DONE, TaskStatus.FAILED)}
    dependents: dict[str, list[str]] = {tid: [] for tid in live}
    for tid in live:
        for dep in tasks[tid]["dependencies"]:
            if dep in live:
                dependents[dep].append(tid)

    depth: dict[str, int] = {}

    def visit(tid: str) -> int:           # validate 已保证无环
        if tid not in depth:
            depth[tid] = max((visit(c) + 1 for c in dependents[tid]), default=0)
        return depth[tid]

    for tid in live:
        visit(tid)
    return depth


def ready(tasks: Tasks) -> list[str]:
    """返回现在就能执行的任务 id，按派发优先级排好序。

    判定条件：

        status 为 pending / ready  且  所有 dependencies 都已 done

    返回的是【列表】而不是单个 —— 长度大于 1 就意味着这些任务之间
    没有依赖关系，可以并行派发。

    顺序只在并发配额放不下全部就绪任务时起作用：排在前面的这一轮派发，
    其余顺延。每一轮要等本轮全部结束才进入下一轮，顺延了关键路径上的任务，
    整张图就多跑一轮。因此按下游最长链从长到短排，同长按规划序号：

        key = (-下游最长链长度, 规划序号, id)
    """
    out = []
    for tid, task in tasks.items():
        if task["status"] not in (TaskStatus.PENDING, TaskStatus.READY):
            continue
        if all(tasks[d]["status"] is TaskStatus.DONE for d in task["dependencies"]):
            out.append(tid)
    depth = downstream_depth(tasks)
    return sorted(out, key=lambda tid: (-depth[tid], tasks[tid]["seq"], tid))


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
