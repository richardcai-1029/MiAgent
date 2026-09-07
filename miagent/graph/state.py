"""Agent 图的共享状态。

State 是所有节点之间唯一的通信手段 —— 节点不互相调用，只读写它。

字段用不用 reducer，判断标准只有一条：

    这个字段是「历史」还是「当前状态」？

    历史（只增不改）    → 用 reducer 追加：results、trace
    当前状态（会被覆写）→ 普通字段，替换：plan、cursor、verdict…

plan 特意【不用】reducer：Replanner 产出的是一份全新计划，
如果追加，新旧计划会混在一起，Scheduler 就会去执行已经被废弃的步骤。
这类错误不会报错，只会让 Agent 行为诡异 —— 属于最难查的一类 bug。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# ---- 循环出口的兜底上限。有循环的图必须有出口，否则会无限转 ----
MAX_ATTEMPTS_PER_STEP = 2   # 同一步最多重试几次，超出 -> AG-1002
MAX_REPLANS = 2             # 最多重规划几次，超出 -> AG-1003

Verdict = Literal["success", "retry", "replan"]
Route = Literal["local", "miclaw"]


class Step(TypedDict):
    """计划中的一步。"""

    id: int
    tool: str
    arguments: dict[str, Any]
    reason: str            # 为什么需要这一步。给人看，也给 Replanner 看


class StepResult(TypedDict):
    """一步的执行结果。"""

    step_id: int
    tool: str
    ok: bool
    content: str
    error_code: str | None
    retry_policy: str | None
    attempt: int


class AgentState(TypedDict, total=False):
    # ---------- 输入 ----------
    task: str                     # 用户的原始任务，全程只读

    # ---------- 计划 ----------
    plan: list[Step]              # 替换：Replanner 会整体换掉它
    cursor: int                   # 执行到第几步

    # ---------- 本轮调度 ----------
    current: Step | None          # Scheduler 选出的待执行步骤
    route: Route | None           # Scheduler 判定该走本地还是 MiClaw
    last: StepResult | None       # 刚执行完的结果，供 Evaluator 判定

    # ---------- 历史（只增不改，用 reducer）----------
    results: Annotated[list[StepResult], operator.add]
    trace: Annotated[list[str], operator.add]

    # ---------- 循环控制 ----------
    verdict: Verdict | None       # Evaluator 的判定
    attempts: dict[str, int]      # 步骤 id -> 已尝试次数
    replan_count: int

    # ---------- 输出 ----------
    answer: str
    failure: str | None           # 非空表示任务未能完成，值为错误码


def initial_state(task: str) -> AgentState:
    """构造初始状态。带 reducer 的字段必须给初值，否则首次合并会失败。"""
    return AgentState(
        task=task,
        plan=[],
        cursor=0,
        current=None,
        route=None,
        last=None,
        results=[],
        trace=[],
        verdict=None,
        attempts={},
        replan_count=0,
        answer="",
        failure=None,
    )
