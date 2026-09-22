"""把节点和边组装成可执行的图。"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from ..llm import LLM
from ..tools import ToolRegistry, ToolSource
from . import nodes
from .routers import (route_after_evaluator, route_after_goal_verifier,
                      route_after_scheduler)
from .state import AgentState


def build_agent(llm: LLM, registry: ToolRegistry,
                max_concurrent_miclaw: int = 2, retry_delay_ms: int = 200,
                verify: bool = True, **compile_kwargs):
    """构建 Agent 图。

    max_concurrent_miclaw 应取自握手时下发的 ResourceBudget.max_concurrent_calls，
    限制同一轮并行派发的 MiClaw 调用数（清单 C-6）。本地工具不受此限。

    retry_delay_ms 是重试前的退避时长，默认值无外部依据，见 Deps 的说明。

    verify 打开工具结果与完成状态的语义校验（见 nodes 的校验节点）。
    它的代价是每轮执行至多多一次模型调用、每轮会话至多多一次，
    收益是「调用成功了但做的不是要做的事」这类偏差能被发现并就地修正。
    关掉之后两个校验节点直接透传，图的拓扑不变。

    compile_kwargs 透传给 LangGraph 的 compile()，
    例如 checkpointer=... 与 interrupt_before=["mcp_executor"]
    可以在调用系统能力前暂停，交由用户确认。
    """
    deps = nodes.Deps(llm=llm, registry=registry,
                      max_concurrent_miclaw=max_concurrent_miclaw,
                      retry_delay_ms=retry_delay_ms, verify=verify)
    bind = lambda fn, **kw: partial(fn, deps=deps, **kw)  # noqa: E731

    g = StateGraph(AgentState)

    g.add_node("planner", bind(nodes.planner))
    g.add_node("scheduler", bind(nodes.scheduler))
    # ★ 两个节点共用一份实现，只有 source 参数不同。
    #   将来 MiClaw 侧要加并发与超时时，在 nodes.execute 里分叉即可。
    g.add_node("local_tool", bind(nodes.execute, source=ToolSource.LOCAL))
    g.add_node("mcp_executor", bind(nodes.execute, source=ToolSource.MICLAW))
    # 校验：结果判定与完成判定。执行与落库之间插结果校验，
    # 调度判完成与收尾之间插完成校验。
    g.add_node("result_verifier", bind(nodes.verify_results))
    g.add_node("goal_verifier", bind(nodes.verify_goal))
    g.add_node("evaluator", bind(nodes.evaluator))
    g.add_node("replanner", bind(nodes.replanner))
    g.add_node("finalizer", bind(nodes.finalizer))

    g.add_edge(START, "planner")
    g.add_edge("planner", "scheduler")

    # 分叉一：调度之后并行扇出到执行节点，或转入完成校验。
    # route_after_scheduler 返回 Send 列表时即为并行派发。
    g.add_conditional_edges("scheduler", route_after_scheduler,
                            ["local_tool", "mcp_executor", "goal_verifier"])

    # 两条执行路径汇合到结果校验，再落库
    g.add_edge("local_tool", "result_verifier")
    g.add_edge("mcp_executor", "result_verifier")
    g.add_edge("result_verifier", "evaluator")

    # 分叉二：评估之后回调度（成功/重试）、去重规划、或中止收尾
    g.add_conditional_edges("evaluator", route_after_evaluator,
                            ["scheduler", "replanner", "finalizer"])

    # 分叉三：完成校验之后补做缺口，或收尾
    g.add_conditional_edges("goal_verifier", route_after_goal_verifier,
                            ["replanner", "finalizer"])

    g.add_edge("replanner", "scheduler")
    g.add_edge("finalizer", END)

    return g.compile(**compile_kwargs)
