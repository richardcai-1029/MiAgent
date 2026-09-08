"""Agent 图的共享状态：任务 DAG 模型。

任务之间以 dependencies 显式表达先后关系，Scheduler 靠依赖解析决定
现在能执行哪些：

    task.status == pending 且所有 dependencies 都 done  →  可执行

tasks 是任务状态的唯一事实来源。「已完成/执行中/已失败的任务」
「各任务的结果」这类信息都从它派生（见 dag.py），不另存字段 ——
同一事实存两份，任一处漏更新就会让 Scheduler 看到自相矛盾的状态，
且不报错，只表现为行为异常。
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

# ---- 循环出口的兜底上限 ----
MAX_ATTEMPTS_PER_TASK = 2   # 单个任务最多重试几次
MAX_REPLANS = 2             # 最多重规划几次，超出 -> AG-1003
MAX_TOTAL_EXECUTIONS = 20   # 单次会话累计最多执行几次工具，超出 -> AG-1002

Verdict = Literal["success", "retry", "replan", "abort"]
Route = Literal["local", "miclaw"]


class TaskStatus(StrEnum):
    """任务生命周期。

        pending ──依赖全部 done──→ ready ──派发──→ running ──┬─→ done
           │                                                └─→ failed
           └──依赖中有 failed──────────────────────────────────→ failed（级联）
    """

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Task(TypedDict):
    id: str
    description: str            # 这一步要达成什么，给人和 Replanner 看
    dependencies: list[str]     # 必须先完成的任务 id
    required_tool: str
    arguments: dict[str, Any]
    status: TaskStatus
    result: str | None
    error: str | None           # 失败时的错误码
    retry_count: int


def append_or_reset(old: list[Any], new: list[Any]) -> list[Any]:
    """列表 reducer：空列表表示重置，否则追加。

    并行执行时多个分支各自写回一条结果，需要追加；Evaluator 消费完
    本轮结果后需要清空。两种语义都要支持，故不能直接用 operator.add。
    """
    return [] if not new else [*(old or []), *new]


class DispatchItem(TypedDict):
    """Scheduler 决定本轮要派发的一项。route 随任务走，
    使得同一轮里本地工具与 MiClaw 工具可以同时派发。"""

    task: Task
    route: Route


class TaskOutcome(TypedDict):
    """一次执行的产出，供 Evaluator 判定。"""

    task_id: str
    tool: str
    ok: bool
    content: str
    error_code: str | None
    retry_policy: str | None
    attempt: int


class AgentState(TypedDict, total=False):
    # ---------- 输入 ----------
    user_request: str

    # ---------- 任务图：唯一事实来源 ----------
    tasks: dict[str, Task]

    # ---------- 本轮调度 ----------
    # 列表而非单个：同一轮里彼此无依赖的任务会被一起派发。
    dispatch: list[DispatchItem]
    # 并行分支各自写回一条结果，用 reducer 汇总；Evaluator 消费后清空。
    outcomes: Annotated[list[TaskOutcome], append_or_reset]

    # ---------- 历史（只增不改，用 reducer）----------
    errors: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[str], operator.add]

    # ---------- 循环控制 ----------
    verdict: Verdict | None
    replan_count: int
    # 用 add reducer：并行分支各自返回 1，由框架累加。
    # 若用普通字段，多个分支同时写回会互相覆盖，计数偏低。
    execution_count: Annotated[int, operator.add]

    # ---------- 输出 ----------
    execution_summary: dict[str, Any]
    final_answer: str
    failure: str | None         # 非空表示任务未完成，值为错误码


def new_task(
    task_id: str,
    description: str,
    required_tool: str,
    arguments: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> Task:
    return Task(
        id=task_id,
        description=description,
        dependencies=list(dependencies or []),
        required_tool=required_tool,
        arguments=dict(arguments or {}),
        status=TaskStatus.PENDING,
        result=None,
        error=None,
        retry_count=0,
    )


def initial_state(user_request: str) -> AgentState:
    """带 reducer 的字段必须给初值，否则首次合并会失败。"""
    return AgentState(
        user_request=user_request,
        tasks={},
        dispatch=[],
        outcomes=[],
        errors=[],
        trace=[],
        verdict=None,
        replan_count=0,
        execution_count=0,
        execution_summary={},
        final_answer="",
        failure=None,
    )
