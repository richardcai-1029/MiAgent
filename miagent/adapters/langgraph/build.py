"""照 agent.topology 的声明构建 LangGraph 图。

本模块是整个项目里唯一 import langgraph 的地方。框架语义的对应关系：

    topology 声明            LangGraph
    Node(fn, bind)           add_node(name, partial(fn, deps=deps, **bind))
    EDGES                    add_edge，START / END 换成框架的哨兵
    Branch                   add_conditional_edges，targets 作为路径表
    Fanout(node, payload)    Send(node, payload)
    AgentState 的 reducer    StateGraph 按 Annotated 元数据合并

这些语义约定由 tests/test_framework_contract.py 固化。
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ...agent import topology
from ...agent.deps import Deps, Limits
from ...core.state import AgentState
from ...llm import LLM
from ...tools import ToolRegistry

if TYPE_CHECKING:
    from ...runtime.slots import SlotPool

_SENTINELS = {topology.START: START, topology.END: END}


def _router(route: Callable[[AgentState], Any]) -> Callable[[AgentState], Any]:
    """把拓扑的路由结果翻译成 LangGraph 的：Fanout 列表 → Send 列表。"""
    def translated(state: AgentState) -> Any:
        nxt = route(state)
        return nxt if isinstance(nxt, str) else [Send(b.node, b.payload) for b in nxt]
    translated.__name__ = route.__name__
    return translated


def build_agent(llm: LLM, registry: ToolRegistry,
                max_concurrent_miclaw: int = 2, retry_delay_ms: int = 200,
                verify: bool = True, miclaw_slots: SlotPool | None = None,
                limits: Limits | None = None, **compile_kwargs):
    """构建 Agent 图。

    max_concurrent_miclaw 应取自握手时下发的 ResourceBudget.max_concurrent_calls，
    限制同一轮并行派发的 MiClaw 调用数（清单 C-6）。本地工具不受此限。

    miclaw_slots 是多个请求共用的 MiClaw 调用槽，容量取 max_concurrent_calls；
    同一个编译好的图被多个请求同时 invoke 时必须给（见 miagent.runtime）。

    retry_delay_ms 是重试前的退避时长，默认值无外部依据，见 Deps 的说明。

    verify 打开工具结果与完成状态的语义校验（见 agent.verifiers）。
    它的代价是每轮执行至多多一次模型调用、每轮会话至多多一次，
    收益是「调用成功了但做的不是要做的事」这类偏差能被发现并就地修正。
    关掉之后两个校验节点直接透传，图的拓扑不变。

    limits 是循环出口的上限，缺省取 Limits 的默认值。

    compile_kwargs 透传给 LangGraph 的 compile()，
    例如 checkpointer=... 与 interrupt_before=["miclaw_executor"]
    可以在调用系统能力前暂停，交由用户确认。
    """
    deps = Deps(llm=llm, registry=registry,
                max_concurrent_miclaw=max_concurrent_miclaw,
                retry_delay_ms=retry_delay_ms, verify=verify,
                miclaw_slots=miclaw_slots, limits=limits or Limits())

    g = StateGraph(AgentState)
    for name, node in topology.NODES.items():
        g.add_node(name, partial(node.fn, deps=deps, **node.bind))
    for a, b in topology.EDGES:
        g.add_edge(_SENTINELS.get(a, a), _SENTINELS.get(b, b))
    for branch in topology.BRANCHES:
        g.add_conditional_edges(branch.source, _router(branch.route), list(branch.targets))
    return g.compile(**compile_kwargs)
