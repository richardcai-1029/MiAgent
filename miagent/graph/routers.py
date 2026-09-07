"""条件边的路由函数。

原则：路由函数【只分派，不决策】。

判断依据（route、verdict）都由上游节点算好并写进 State，
路由函数只是把它翻译成下一个节点名。这样做的好处是：
决策逻辑集中在节点里、可单独测试；而路由本身一眼能看完，
整张图的走向不会藏在难懂的条件表达式里。
"""

from __future__ import annotations

from .state import AgentState


def route_after_scheduler(state: AgentState) -> str:
    """Scheduler 之后：还有步骤就去执行，没有就收尾。"""
    if state.get("current") is None:
        return "finalizer"
    return "mcp_executor" if state.get("route") == "miclaw" else "local_tool"


def route_after_evaluator(state: AgentState) -> str:
    """Evaluator 之后：成功/重试回调度，重规划去 Replanner，中止去收尾。"""
    verdict = state.get("verdict")
    if verdict == "replan":
        return "replanner"
    if verdict == "abort":
        return "finalizer"
    return "scheduler"          # success 与 retry 都回到调度
