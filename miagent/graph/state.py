"""Agent 图的共享状态：任务 DAG 模型。

与线性计划的区别：任务之间有显式依赖，Scheduler 靠依赖解析决定
「现在能跑哪些」，而不是靠一个递增的游标。这让「并行、分支、汇合」
成为状态模型本身的能力，而不是事后打的补丁。

    task.status == pending 且所有 dependencies 都 done  →  可执行

★ 关于派生字段的说明

  设计稿里列了 completed_tasks / running_tasks / failed_tasks / tool_results。
  这些信息 tasks 里已经有了（每个 Task 自带 status 与 result），
  再单独存一份就是同一事实的两份拷贝 —— 一旦某处漏更新，Scheduler 会
  看到自相矛盾的状态，而且不报错，只是行为诡异。

  因此它们实现为【从 tasks 派生的函数】（见 dag.py），不进 State。
  可见性一点不少，但不可能不一致。
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
    current: Task | None        # Scheduler 选出的待执行任务
    route: Route | None
    last: TaskOutcome | None

    # ---------- 历史（只增不改，用 reducer）----------
    errors: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[str], operator.add]

    # ---------- 循环控制 ----------
    verdict: Verdict | None
    replan_count: int
    execution_count: int        # 累计工具调用次数，用于全局上限

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
        current=None,
        route=None,
        last=None,
        errors=[],
        trace=[],
        verdict=None,
        replan_count=0,
        execution_count=0,
        execution_summary={},
        final_answer="",
        failure=None,
    )
