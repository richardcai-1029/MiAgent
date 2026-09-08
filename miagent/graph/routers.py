"""条件边的路由函数。

原则：路由函数【只分派，不决策】。判断依据（dispatch、verdict）都由上游
节点算好并写进 State，路由只把它翻译成下一步。这样决策逻辑集中在节点里、
可单独测试，而整张图的走向一眼能看完。
"""

from __future__ import annotations

from langgraph.types import Send

from .state import AgentState


def route_after_scheduler(state: AgentState) -> str | list[Send]:
    """Scheduler 之后：没有可派发的任务就收尾，否则并行扇出。

    ★ 返回 Send 列表即为并行派发。每个 Send 携带自己的任务，
      Send 的 payload 就是那次节点调用看到的全部 state ——
      因此执行节点读不到主状态，只能返回增量，由 reducer 汇总。

      同一轮里本地工具与 MiClaw 工具可以同时扇出到不同节点，
      因为 route 是随任务走的，不是全局的。
    """
    dispatch = state.get("dispatch") or []
    if not dispatch:
        return "finalizer"
    return [
        Send("mcp_executor" if item["route"] == "miclaw" else "local_tool",
             {"task": item["task"]})
        for item in dispatch
    ]


def route_after_evaluator(state: AgentState) -> str:
    """Evaluator 之后：成功/重试回调度，重规划去 Replanner，中止去收尾。"""
    verdict = state.get("verdict")
    if verdict == "replan":
        return "replanner"
    if verdict == "abort":
        return "finalizer"
    return "scheduler"          # success 与 retry 都回到调度
