"""Agent 图的共享状态：任务 DAG 模型。

任务之间以 dependencies 显式表达先后关系，Scheduler 靠依赖解析决定
现在能执行哪些：

    task.status == pending 且所有 dependencies 都 done  →  可执行

tasks 是调度状态的唯一事实来源：只装当前还要调度的任务。「哪些就绪」
「是否死锁」「是否完成」都从它派生（见 dag.py），不另存字段 ——
同一事实存两份，任一处漏更新就会让 Scheduler 看到自相矛盾的状态，
且不报错，只表现为行为异常。

到了终态的任务在重规划时离开 tasks，压缩为 episodes（见 miagent.memory）。
两者各管一段生命周期：tasks 是工作记忆，episodes 是情景记忆；
执行概况从两者合并得出。
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
    # 规划序号：模型给出这一步的先后。id 是模型起的名字，按字符串排序时
    # task_10 会排在 task_2 之前；需要「原本的顺序」时一律看它。
    seq: int


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


class Review(TypedDict):
    """一条执行结果的语义校验判定。由校验节点产出，Evaluator 落库。

    判定本身只回答「这次执行算不算达成了任务的目标」；该重试还是该重规划
    仍由 Evaluator 按既有规则推出。correction 非空表示模型给出的新参数已经
    通过确定性把关（见 graph.verify），可以直接拿去重试同一个任务。
    """

    task_id: str
    ok: bool
    reason: str
    correction: dict[str, Any] | None


class Episode(TypedDict):
    """情景记忆的一条记录：离开任务图的终态任务的压缩形式。
    只保留重规划与收尾用得到的部分；行为见 miagent.memory.episodic。"""

    task_id: str
    description: str
    tool: str
    args_digest: str        # 参数指纹：同 tool 同指纹即「同一个调用」
    ok: bool
    error: str | None       # 失败时的错误码
    result: str | None      # 成功时的结果原文；失败时是给模型看的失败说明
    generation: int         # 产生它的那一轮，取当时的 replan_count


class Anchor(TypedDict):
    """目标锚：一次请求里不变的部分。行为见 miagent.memory.anchor。"""

    goal: str               # 用户目标原文
    intent: list[str]       # 首次拆解的各步描述，按规划序号排列


class Turn(TypedDict):
    """对话的一轮。由 Session 记录并在下一轮填入 history，见 miagent.memory.session。"""

    number: int                 # 这是对话的第几轮，从 1 起；更早的轮次被削掉后不变
    request: str
    answer: str
    summary: str                # 收尾产出的本轮摘要，下一轮规划看的就是它
    failure: str | None         # 非空表示这一轮没完成，值为错误码
    episodes: list[Episode]     # 这一轮的执行历史，供调用方查看，不进提示词


class AgentState(TypedDict, total=False):
    # ---------- 输入 ----------
    user_request: str
    # 对话历史：之前各轮的记录，由 Session 在 run 时填入，见 miagent.memory.session。
    history: list[Turn]

    # ---------- 任务图：唯一事实来源 ----------
    tasks: dict[str, Task]

    # 目标锚：首次规划成功后写入，此后只读，见 miagent.memory.anchor。
    anchor: Anchor

    # ---------- 本轮调度 ----------
    # 列表而非单个：同一轮里彼此无依赖的任务会被一起派发。
    dispatch: list[DispatchItem]
    # 并行分支各自写回一条结果，用 reducer 汇总；Evaluator 消费后清空。
    outcomes: Annotated[list[TaskOutcome], append_or_reset]
    # 本轮结果的语义校验判定，与 outcomes 一同被 Evaluator 消费后清空。
    reviews: Annotated[list[Review], append_or_reset]

    # ---------- 历史（只增不改，用 reducer）----------
    errors: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[str], operator.add]
    # 情景记忆：离开任务图的终态任务压缩后的记录，见 miagent.memory。
    # 只增不改：Replanner 与 Finalizer 经账本结算后写回新增的部分。
    episodes: Annotated[list[Episode], operator.add]

    # ---------- 循环控制 ----------
    verdict: Verdict | None
    replan_count: int
    # 用 add reducer：并行分支各自返回 1，由框架累加。
    # 若用普通字段，多个分支同时写回会互相覆盖，计数偏低。
    execution_count: Annotated[int, operator.add]

    # ---------- 输出 ----------
    execution_summary: dict[str, Any]
    final_answer: str
    turn_summary: str           # 本轮摘要，供下一轮规划参考
    # 完成校验指出的缺口：非空表示执行跑完了但目标没达成，内容是还差什么。
    # 由重规划消费，规划完即清除。
    gap: str | None
    # 规划认定没有任何可用工具能完成的部分（能力不支持、所需工具因权限不可见）。
    # 这一轮的事实，不因重规划清除：非空时收尾以「目标未达成」结束。
    unfulfilled: list[str]

    # 非空表示任务未完成，值为错误码。
    # 清除规则只有一条：成功的重规划清除它（新计划按构造覆盖了全部失败记录），
    # 单个任务成功不清除 —— 重规划失败后剩余任务照常执行，它们成功不代表恢复发生了。
    # 重规划侧的取值由 miagent.memory.ledger 给出；其余节点只在中止时写入。
    failure: str | None


def new_task(
    task_id: str,
    description: str,
    required_tool: str,
    arguments: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
    seq: int = 0,
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
        seq=seq,
    )


def initial_state(user_request: str) -> AgentState:
    """带 reducer 的字段必须给初值，否则首次合并会失败。"""
    return AgentState(
        user_request=user_request,
        history=[],
        tasks={},
        anchor={},
        dispatch=[],
        outcomes=[],
        reviews=[],
        errors=[],
        trace=[],
        episodes=[],
        verdict=None,
        replan_count=0,
        execution_count=0,
        execution_summary={},
        final_answer="",
        turn_summary="",
        gap=None,
        unfulfilled=[],
        failure=None,
    )
