"""图的七个节点。

每个节点都是「读 State，返回要更新的字段」的普通函数 —— 没有魔法。

节点需要 llm 和 registry，但 LangGraph 规定节点签名是 f(state) -> dict，
没有位置传依赖。解决办法是用 functools.partial 把依赖预先绑进去
（见 build.py）—— 与我们让两个执行节点共用一份实现是同一个技巧。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..llm import LLM, LLMMessage, system, user
from ..llm.base import extract_json
from ..protocol import AgentError, ErrorCode, RetryPolicy
from ..tools import ToolRegistry, ToolSource
from .schema import Plan, plan_model_for
from .state import (
    MAX_ATTEMPTS_PER_STEP,
    MAX_REPLANS,
    MAX_TOTAL_STEPS,
    AgentState,
    Step,
    StepResult,
)


@dataclass
class Deps:
    """节点运行所需的外部依赖。"""

    llm: LLM
    registry: ToolRegistry


# ============================================================
# 计划的解析：模型返回文本，我们要把它变成结构化的步骤
# ============================================================

def _to_steps(parsed: BaseModel) -> list[Step]:
    """把校验通过的 Pydantic 模型转成图内部用的 Step。"""
    return [
        Step(id=i, tool=s.tool, arguments=s.arguments, reason=s.reason)
        for i, s in enumerate(parsed.steps)
    ]


def parse_plan(text: str, model: type[BaseModel] = Plan) -> list[Step]:
    """把模型输出解析并校验成步骤列表。不合 schema 即 AG-1001。

    这里不做「容错兜底猜一个计划」—— 模型没按格式输出就是没输出，
    猜出来的计划会让 Agent 做用户没要求的事，比直接失败危险得多。
    """
    try:
        return _to_steps(model.model_validate_json(extract_json(text)))
    except Exception as e:
        raise AgentError(
            ErrorCode.AG_PLAN_PARSE_FAILED,
            "模型输出无法解析为计划",
            detail={"raw": text[:200], "reason": str(e)[:200]},
        ) from e


def _plan_with_schema(deps: Deps, messages: list[LLMMessage]) -> list[Step]:
    """要求模型产出结构化计划。

    工具名会被收进 schema 的 enum —— 幻觉出的工具在【解析阶段】就被拒，
    不必白跑一步再报 AG-2001。
    """
    model = plan_model_for([t.name for t in deps.registry])
    try:
        return _to_steps(deps.llm.complete_structured(messages, model))
    except AgentError as e:
        if e.code is ErrorCode.AG_LLM_INVALID_RESPONSE:
            # 分层：模型层说「输出不合 schema」，规划层说「没能产出可执行计划」
            raise AgentError(ErrorCode.AG_PLAN_PARSE_FAILED,
                             "模型未能产出合法计划", detail=e.detail) from e
        raise


# ============================================================
# ① Planner —— 任务 + 工具清单 → 计划
# ============================================================


def planner(state: AgentState, deps: Deps) -> dict[str, Any]:
    tools = json.dumps(deps.registry.to_model_schemas(), ensure_ascii=False, indent=2)
    try:
        plan = _plan_with_schema(deps, [
            system("你是端侧智能助理的规划器。根据用户任务和可用工具，拆解出最少的执行步骤。"),
            user(f"可用工具：\n{tools}\n\n任务：{state['task']}"),
        ])
    except AgentError as e:
        # 规划失败不能把整个调用炸掉 —— 用户至少要收到一句解释。
        # 空计划会让 Scheduler 立刻判定"已完成"并转向 Finalizer，
        # Finalizer 看到 failure 就会生成失败说明。
        return {"plan": [], "cursor": 0, "verdict": None, "failure": e.code.value,
                "trace": [f"planner: 规划失败 {e.code}"]}
    return {
        "plan": plan,
        "cursor": 0,
        "verdict": None,
        "trace": [f"planner: 产出 {len(plan)} 步计划 → "
                  f"{[s['tool'] for s in plan]}"],
    }


# ============================================================
# ② Scheduler —— 推进游标、选出下一步、判定路由
# ============================================================


def scheduler(state: AgentState, deps: Deps) -> dict[str, Any]:
    plan, cursor = state["plan"], state["cursor"]

    # 上一轮判定为 success 才前进；retry 保持原位重做同一步
    if state.get("verdict") == "success":
        cursor += 1

    if cursor >= len(plan):
        return {"cursor": cursor, "current": None, "route": None, "verdict": None,
                "trace": ["scheduler: 计划已全部完成 → finalizer"]}

    step = plan[cursor]
    tool = deps.registry.get(step["tool"])
    # 工具不存在（模型幻觉）时走本地路径，由注册表统一报 AG-2001
    route = "miclaw" if tool is not None and tool.source is ToolSource.MICLAW else "local"

    return {
        "cursor": cursor, "current": step, "route": route, "verdict": None,
        "trace": [f"scheduler: 第 {cursor + 1}/{len(plan)} 步 "
                  f"{step['tool']} → {route}"],
    }


# ============================================================
# ③④ 执行 —— 两个节点共用这一份实现
# ============================================================


def execute(state: AgentState, deps: Deps, source: ToolSource) -> dict[str, Any]:
    """本地工具与 MiClaw 工具的共用执行体。

    现在两条路径完全一样。将来 MiClaw 侧要加并发批处理、按工具粒度的
    超时、配额预筛时，只在这个函数里分叉，图的结构不用动。
    """
    step = state["current"]
    attempt = state["attempts"].get(str(step["id"]), 0) + 1

    result = deps.registry.invoke(step["tool"], step["arguments"])

    outcome = StepResult(
        step_id=step["id"],
        tool=step["tool"],
        ok=not result.is_error,
        # 存 for_model() 而非 content：失败时它带着「下一步该怎么办」，
        # Replanner 和 Finalizer 拿到的就是可直接用的信息
        content=result.for_model(),
        error_code=result.error_code.value if result.error_code else None,
        retry_policy=result.retry_policy.value,
        attempt=attempt,
    )
    mark = "✓" if outcome["ok"] else f"✗ {outcome['error_code']}"
    return {"last": outcome,
            "trace": [f"{source.value}: {step['tool']} {mark}（第 {attempt} 次）"]}


# ============================================================
# ⑤ Evaluator —— 判定 success / retry / replan / abort
# ============================================================

# 这两类错误重试有意义：临时故障、会话状态问题
_RETRIABLE = {RetryPolicy.BACKOFF.value, RetryPolicy.REHANDSHAKE.value}


def evaluator(state: AgentState, deps: Deps) -> dict[str, Any]:
    """★ 这个节点不调用大模型。

    「该重试还是该重规划」已经由错误码的段位决定了 —— 用模型去判断
    一个我们确定知道答案的问题，既慢又不可靠。这是错误码分段设计的
    最后一次兑现。
    """
    last = state["last"]
    attempts = {**state["attempts"], str(last["step_id"]): last["attempt"]}
    total_done = len(state["results"]) + 1

    if last["ok"]:
        verdict, failure = "success", None
    elif total_done >= MAX_TOTAL_STEPS:
        verdict, failure = "abort", ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value
    elif last["retry_policy"] in _RETRIABLE and last["attempt"] < MAX_ATTEMPTS_PER_STEP:
        verdict, failure = "retry", None
    elif state["replan_count"] < MAX_REPLANS:
        verdict, failure = "replan", None
    else:
        verdict, failure = "abort", ErrorCode.AG_PLAN_NO_PROGRESS.value

    return {"verdict": verdict, "results": [last], "attempts": attempts,
            "failure": failure,
            "trace": [f"evaluator: {verdict}" + (f"（{failure}）" if failure else "")]}


# ============================================================
# ⑥ Replanner —— 带着失败信息重新规划
# ============================================================


def replanner(state: AgentState, deps: Deps) -> dict[str, Any]:
    done = "\n".join(
        f"  第{r['step_id'] + 1}步 {r['tool']}: {'成功' if r['ok'] else '失败'} — {r['content']}"
        for r in state["results"]
    )
    tools = json.dumps(deps.registry.to_model_schemas(), ensure_ascii=False)
    try:
        plan = _plan_with_schema(deps, [
            system("你是端侧智能助理的重规划器。原计划中有步骤失败了，"
                   "请基于已有结果重新规划【剩余】要做的事，避开失败的做法。"),
            user(f"可用工具：{tools}\n\n任务：{state['task']}\n\n"
                 f"已执行情况：\n{done}\n\n请给出新的步骤。"),
        ])
    except AgentError as e:
        return {"plan": [], "cursor": 0, "current": None, "verdict": None,
                "replan_count": state["replan_count"] + 1, "failure": e.code.value,
                "trace": [f"replanner: 重规划失败 {e.code}"]}
    return {
        "plan": plan,
        "cursor": 0,            # 新计划从头开始
        "current": None,
        "verdict": None,
        # ★ 必须清空重试计数。attempts 以 step id 为键，而新计划的 id
        #   又从 0 开始 —— 不清空的话，全新的第一步会继承旧步骤的次数，
        #   凭空少一次重试机会。这类 bug 不报错，只是行为悄悄变差。
        "attempts": {},
        "replan_count": state["replan_count"] + 1,
        "trace": [f"replanner: 第 {state['replan_count'] + 1} 次重规划 → "
                  f"{[s['tool'] for s in plan]}"],
    }


# ============================================================
# ⑦ Finalizer —— 汇总成给用户的回答
# ============================================================


def finalizer(state: AgentState, deps: Deps) -> dict[str, Any]:
    results = "\n".join(
        f"  {r['tool']}: {'成功' if r['ok'] else '失败'} — {r['content']}"
        for r in state["results"]
    ) or "  （未执行任何步骤）"

    if state.get("failure"):
        prompt = (f"任务未能完成（错误码 {state['failure']}）。请向用户简要说明"
                  f"做到了哪一步、卡在哪里、建议怎么办。")
    else:
        prompt = "请根据执行结果，用一两句话回答用户。"

    answer = deps.llm.complete([
        system("你是端侧智能助理。回答要简洁，只说结论，不要复述过程。"),
        user(f"任务：{state['task']}\n\n执行结果：\n{results}\n\n{prompt}"),
    ]).content
    return {"answer": answer, "trace": ["finalizer: 已生成回答"]}
