"""Agent 的图拓扑：有哪些节点、节点之间怎么连、分叉处怎么走。

拓扑本身与编排框架无关，写成声明；适配层（miagent.adapters）照此构图：

    START → Planner → Scheduler ─┬→ LocalExecutor  ─┐
                                 ├→ MiClawExecutor ─┴→ ResultVerifier → Evaluator ─┬→ Scheduler
                                 └→ GoalVerifier ─┬→ Finalizer → END               ├→ Replanner → Scheduler
                                                  └→ Replanner → Scheduler         └→ Finalizer

原则：路由函数【只分派，不决策】。判断依据（dispatch、verdict）都由上游
节点算好并写进 State，路由只把它翻译成下一步。这样决策逻辑集中在节点里、
可单独测试，而整张图的走向一眼能看完。
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ..core.state import AgentState
from ..tools import ToolSource
from .evaluator import evaluator
from .executor import executor
from .finalizer import finalizer
from .planning import planner, replanner
from .scheduler import scheduler
from .verifiers import goal_verifier, result_verifier

START = "__start__"
END = "__end__"


class Node(NamedTuple):
    """一个图节点：实现函数，以及除 state、deps 之外要绑定的参数。"""

    fn: Callable[..., dict[str, Any]]
    bind: dict[str, Any]


class Fanout(NamedTuple):
    """扇出的一个分支：以 payload 作为全部 state 调用 node 一次。

    同一次路由返回的所有分支并行执行，各自返回的增量按 AgentState 上声明的
    reducer 合并；全部分支结束后才进入下游节点。
    """

    node: str
    payload: dict[str, Any]


class Branch(NamedTuple):
    """条件边：source 之后调用 route 决定去向，targets 是全部可能的去向。"""

    source: str
    route: Callable[[AgentState], str | list[Fanout]]
    targets: tuple[str, ...]


def route_after_scheduler(state: AgentState) -> str | list[Fanout]:
    """Scheduler 之后：没有可派发的任务就去完成校验，否则按任务扇出。

    ★ 每个分支只携带自己的任务，执行节点因此读不到主状态，只能返回增量。
      同一轮里本地工具与 MiClaw 工具可以同时扇出到不同节点，
      因为 route 是随任务走的，不是全局的。
    """
    dispatch = state.get("dispatch") or []
    if not dispatch:
        return "goal_verifier"
    return [Fanout("miclaw_executor" if item["route"] == "miclaw" else "local_executor",
                   {"task": item["task"]})
            for item in dispatch]


def route_after_goal_verifier(state: AgentState) -> str:
    """完成校验之后：判未达成且还能重规划就去补，否则收尾。

    判定写在节点里（见 verifiers.goal_verifier），这里只翻译 verdict。
    """
    return "replanner" if state.get("verdict") == "replan" else "finalizer"


def route_after_evaluator(state: AgentState) -> str:
    """Evaluator 之后：成功/重试回调度，重规划去 Replanner，中止去收尾。"""
    verdict = state.get("verdict")
    if verdict == "replan":
        return "replanner"
    if verdict == "abort":
        return "finalizer"
    return "scheduler"          # success 与 retry 都回到调度


NODES: dict[str, Node] = {
    "planner": Node(planner, {}),
    "scheduler": Node(scheduler, {}),
    # ★ 两个执行节点共用一份实现，只有 source 不同：分列两个节点，
    #   使 MiClaw 侧的批量、超时、配额处理能在执行体内分叉而不改拓扑。
    "local_executor": Node(executor, {"source": ToolSource.LOCAL}),
    "miclaw_executor": Node(executor, {"source": ToolSource.MICLAW}),
    # 校验：执行与落库之间插结果校验，调度判完成与收尾之间插完成校验。
    "result_verifier": Node(result_verifier, {}),
    "goal_verifier": Node(goal_verifier, {}),
    "evaluator": Node(evaluator, {}),
    "replanner": Node(replanner, {}),
    "finalizer": Node(finalizer, {}),
}

EDGES: tuple[tuple[str, str], ...] = (
    (START, "planner"),
    ("planner", "scheduler"),
    # 两条执行路径汇合到结果校验，再落库
    ("local_executor", "result_verifier"),
    ("miclaw_executor", "result_verifier"),
    ("result_verifier", "evaluator"),
    ("replanner", "scheduler"),
    ("finalizer", END),
)

BRANCHES: tuple[Branch, ...] = (
    # 分叉一：调度之后并行扇出到执行节点，或转入完成校验
    Branch("scheduler", route_after_scheduler,
           ("local_executor", "miclaw_executor", "goal_verifier")),
    # 分叉二：评估之后回调度（成功/重试）、去重规划、或中止收尾
    Branch("evaluator", route_after_evaluator, ("scheduler", "replanner", "finalizer")),
    # 分叉三：完成校验之后补做缺口，或收尾
    Branch("goal_verifier", route_after_goal_verifier, ("replanner", "finalizer")),
)
