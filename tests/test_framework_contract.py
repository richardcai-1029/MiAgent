"""LangGraph 框架契约测试。

本文件与业务逻辑无关，只固化一件事：**我们依赖了 LangGraph 的哪些语义**。

为什么单独成文件——这些语义写在框架文档里，不在类型签名里。签名变了会报错，
语义变了不会。以并行派发为例：若 Send 的 payload 改为与主 state 合并，
执行节点会突然读得到 tasks 与 execution_count，而它返回的仍是增量，
表现为计数偏低、结果互相覆盖，全程不抛任何异常。

升级 LangGraph 时先跑这一组：失败能直接指出是哪条约定变了，
而不必从一堆业务测试的失败里反推。每条契约标注了依赖它的改造项，
以及该契约失效后的**静默**表现。

测试用的都是最小图，不引用 miagent 的任何模块 —— 断言的对象是框架本身。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph", reason="图引擎为可选依赖（清单 E-1）")

from langgraph.checkpoint.memory import InMemorySaver          # noqa: E402
from langgraph.graph import END, START, StateGraph             # noqa: E402
from langgraph.types import Send                               # noqa: E402


def _append_or_reset(old: list[Any], new: list[Any]) -> list[Any]:
    """与 state.append_or_reset 同构的 reducer，此处独立复制一份。

    契约测试断言的是框架行为，不应因业务代码改动而连带失败。
    """
    return [] if not new else [*(old or []), *new]


class _State(TypedDict, total=False):
    marker: str
    dispatch: list[dict[str, Any]]
    outcomes: Annotated[list[str], _append_or_reset]
    counter: Annotated[int, operator.add]


def _fanout_graph(worker, sink=None):
    """构造 扇出 → 汇合 的最小图，与 Agent 图的 scheduler → 执行 → evaluator 同形。"""

    def scheduler(state: _State) -> dict[str, Any]:
        return {"dispatch": [{"id": "a"}, {"id": "b"}], "marker": "MAIN"}

    def route(state: _State) -> list[Send]:
        return [Send("worker", {"task": item}) for item in state["dispatch"]]

    g = StateGraph(_State)
    g.add_node("scheduler", scheduler)
    g.add_node("worker", worker)
    g.add_node("sink", sink or (lambda state: {}))
    g.add_edge(START, "scheduler")
    g.add_conditional_edges("scheduler", route, ["worker"])
    g.add_edge("worker", "sink")
    g.add_edge("sink", END)
    return g.compile()


def _initial() -> _State:
    return _State(marker="", dispatch=[], outcomes=[], counter=0)


# ============================================================
# 契约一：Send 的 payload 就是节点看到的全部 state
# ============================================================


def test_send_payload_is_the_entire_node_state():
    """依赖方：G-5 并行派发（routers.route_after_scheduler、nodes.execute）。

    执行节点因此读不到 tasks / execution_count，只能返回增量交由 reducer 汇总。
    契约失效（payload 与主 state 合并）后的静默表现：节点若改回返回绝对值，
    多个并行分支会互相覆盖，且不报错。
    """
    seen: list[dict[str, Any]] = []

    def worker(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(dict(payload))
        return {"outcomes": [payload["task"]["id"]], "counter": 1}

    _fanout_graph(worker).invoke(_initial())

    assert len(seen) == 2
    for payload in seen:
        assert set(payload) == {"task"}, (
            f"Send 的 payload 不再是节点看到的全部 state，多出：{set(payload) - {'task'}}。"
            "nodes.execute 返回增量的前提已不成立。"
        )


# ============================================================
# 契约二：并行分支的返回值经 reducer 汇总，不互相覆盖
# ============================================================


def test_parallel_branches_merge_through_reducer():
    """依赖方：G-5（outcomes 用 append_or_reset、execution_count 用 operator.add）。

    契约失效后的静默表现：execution_count 计数偏低，
    使 MAX_TOTAL_EXECUTIONS 这道循环出口失效（清单 G-2）。
    """
    def worker(payload: dict[str, Any]) -> dict[str, Any]:
        return {"outcomes": [payload["task"]["id"]], "counter": 1}

    final = _fanout_graph(worker).invoke(_initial())

    assert sorted(final["outcomes"]) == ["a", "b"], "并行分支的列表返回值未被追加合并"
    assert final["counter"] == 2, (
        f"两个分支各返回 1，累加后应为 2，实际 {final['counter']}——"
        "整数 reducer 的累加语义已变，循环出口的计数不再可信"
    )


# ============================================================
# 契约三：节点返回空列表时 reducer 仍被调用
# ============================================================


def test_empty_list_return_reaches_the_reducer():
    """依赖方：state.append_or_reset 的「空列表即重置」约定（Evaluator 消费后清空）。

    契约失效（空值被视为「无更新」而跳过合并）后的静默表现：
    outcomes 不再被清空，Evaluator 下一轮重复消费上一轮的执行结果。
    """
    def worker(payload: dict[str, Any]) -> dict[str, Any]:
        return {"outcomes": [payload["task"]["id"]], "counter": 1}

    def sink(state: _State) -> dict[str, Any]:
        assert sorted(state["outcomes"]) == ["a", "b"], "汇合节点未读到两个分支的结果"
        return {"outcomes": []}          # 空列表 -> 期望触发 reducer 的重置分支

    final = _fanout_graph(worker, sink).invoke(_initial())

    assert final["outcomes"] == [], (
        f"返回空列表未能重置该字段，实际残留 {final['outcomes']}——"
        "Evaluator 将重复消费上一轮结果"
    )


# ============================================================
# 契约四：条件边接受 Send 列表作为返回值
# ============================================================


def test_conditional_edge_accepts_send_list():
    """依赖方：G-5（route_after_scheduler 同时返回节点名与 Send 列表两种类型）。

    契约失效后的表现：构图或调用时报错——这一条是显式失败，
    列在此处是为了让「并行派发依赖哪些 API」一处可见。
    """
    calls: list[str] = []

    def scheduler(state: _State) -> dict[str, Any]:
        return {"dispatch": state["dispatch"]}

    def route(state: _State) -> str | list[Send]:
        if not state["dispatch"]:
            return "sink"                # 无可派发时退化为普通字符串路由
        return [Send("worker", {"task": item}) for item in state["dispatch"]]

    def worker(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload["task"]["id"])
        return {"counter": 1}

    def sink(state: _State) -> dict[str, Any]:
        calls.append("sink")
        return {}

    g = StateGraph(_State)
    g.add_node("scheduler", scheduler)
    g.add_node("worker", worker)
    g.add_node("sink", sink)
    g.add_edge(START, "scheduler")
    g.add_conditional_edges("scheduler", route, ["worker", "sink"])
    g.add_edge("worker", "sink")
    g.add_edge("sink", END)
    app = g.compile()

    app.invoke({**_initial(), "dispatch": [{"id": "x"}]})
    assert calls == ["x", "sink"]

    calls.clear()
    app.invoke(_initial())
    assert calls == ["sink"], "空 dispatch 时未走字符串路由分支"


# ============================================================
# 契约五：interrupt_before 可暂停在指定节点之前，并能恢复
# ============================================================


def test_interrupt_before_pauses_and_resumes():
    """依赖方：build_agent 的 compile_kwargs 透传。

    端侧据此在调用系统能力之前暂停、交由用户确认后继续，
    是权限受限场景「中断 → 授权 → 恢复」路径的框架前提。
    契约失效后的静默表现：系统调用不再暂停，直接执行。
    """
    def before(state: _State) -> dict[str, Any]:
        return {"marker": state.get("marker", "") + "before;"}

    def risky(state: _State) -> dict[str, Any]:
        return {"marker": state["marker"] + "risky;"}

    g = StateGraph(_State)
    g.add_node("before", before)
    g.add_node("risky", risky)
    g.add_edge(START, "before")
    g.add_edge("before", "risky")
    g.add_edge("risky", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_before=["risky"])

    config = {"configurable": {"thread_id": "contract"}}
    paused = app.invoke(_initial(), config)
    assert paused["marker"] == "before;", "interrupt_before 未能在目标节点之前暂停"
    assert app.get_state(config).next == ("risky",), "暂停点的 next 不是被中断的节点"

    resumed = app.invoke(None, config)
    assert resumed["marker"] == "before;risky;", "以 None 恢复执行失败"
