"""把节点和边组装成可执行的图。"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from ..llm import LLM
from ..tools import ToolRegistry, ToolSource
from . import nodes
from .routers import route_after_evaluator, route_after_scheduler
from .state import AgentState


def build_agent(llm: LLM, registry: ToolRegistry,
                max_concurrent_miclaw: int = 2, **compile_kwargs):
    """构建 Agent 图。

    max_concurrent_miclaw 应取自握手时下发的 ResourceBudget.max_concurrent_calls，
    限制同一轮并行派发的 MiClaw 调用数（清单 C-6）。本地工具不受此限。

    compile_kwargs 透传给 LangGraph 的 compile()，
    例如 checkpointer=... 与 interrupt_before=["mcp_executor"]
    可以在调用系统能力前暂停，交由用户确认。
    """
    deps = nodes.Deps(llm=llm, registry=registry,
                      max_concurrent_miclaw=max_concurrent_miclaw)
    bind = lambda fn, **kw: partial(fn, deps=deps, **kw)  # noqa: E731

    g = StateGraph(AgentState)

    g.add_node("planner", bind(nodes.planner))
    g.add_node("scheduler", bind(nodes.scheduler))
    # ★ 两个节点共用一份实现，只有 source 参数不同。
    #   将来 MiClaw 侧要加并发与超时时，在 nodes.execute 里分叉即可。
    g.add_node("local_tool", bind(nodes.execute, source=ToolSource.LOCAL))
    g.add_node("mcp_executor", bind(nodes.execute, source=ToolSource.MICLAW))
    g.add_node("evaluator", bind(nodes.evaluator))
    g.add_node("replanner", bind(nodes.replanner))
    g.add_node("finalizer", bind(nodes.finalizer))

    g.add_edge(START, "planner")
    g.add_edge("planner", "scheduler")

    # 分叉一：调度之后并行扇出到执行节点，或直接收尾。
    # route_after_scheduler 返回 Send 列表时即为并行派发。
    g.add_conditional_edges("scheduler", route_after_scheduler,
                            ["local_tool", "mcp_executor", "finalizer"])

    # 两条执行路径汇合到评估
    g.add_edge("local_tool", "evaluator")
    g.add_edge("mcp_executor", "evaluator")

    # 分叉二：评估之后回调度（成功/重试）、去重规划、或中止收尾
    g.add_conditional_edges("evaluator", route_after_evaluator,
                            ["scheduler", "replanner", "finalizer"])

    g.add_edge("replanner", "scheduler")
    g.add_edge("finalizer", END)

    return g.compile(**compile_kwargs)
