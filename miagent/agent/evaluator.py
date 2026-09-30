"""Evaluator：合并执行结果与校验判定，落库任务状态。不调用模型。"""

from __future__ import annotations

from typing import Any

from ..core import verify
from ..core.state import AgentState, TaskStatus
from ..protocol import ErrorCode, RetryPolicy
from .deps import Deps


_RETRIABLE = {RetryPolicy.BACKOFF.value, RetryPolicy.REHANDSHAKE.value}


# 判定优先级：任一任务需要中止就中止，其次重规划，其次重试
_VERDICT_RANK = {"success": 0, "retry": 1, "replan": 2, "abort": 3}


def evaluator(state: AgentState, deps: Deps) -> dict[str, Any]:
    """★ 不调用大模型。

    「该重试还是该重规划」由错误码的段位直接决定 —— 传输抖动重试，
    资源不足换方案。这是错误码分段设计的最终兑现。

    并行执行时本轮可能有多条结果，逐条落库后按优先级聚合成一个判定：
    只要有任务需要重规划，整轮就走 Replanner —— 它拿到的是完整任务图，
    能同时看到本轮成功与失败的部分。

    语义校验的判定在这里与执行结果合并：未通过校验的成功按失败落库
    （错误码 AG-2005），带着已通过把关的修正参数的则换上新参数重试。
    合并规则确定，判定本身来自校验节点。
    """
    tasks = {tid: dict(t) for tid, t in state["tasks"].items()}
    reviews = {r["task_id"]: r for r in state.get("reviews") or []}
    verdicts: list[str] = []
    failure = None
    errors: list[dict[str, Any]] = []
    lines: list[str] = []

    for last in state["outcomes"]:
        task = tasks[last["task_id"]]
        task["retry_count"] = last["attempt"]
        review = reviews.get(last["task_id"])
        code = verify.outcome_code(last, review)
        content = verify.rejection_text(last, review)
        correction = verify.correction_of(review)

        if code is None:
            task["status"], task["result"] = TaskStatus.DONE, content
            verdict = "success"
        else:
            task["error"] = code
            errors.append({"task": task["id"], "code": code,
                           "attempt": last["attempt"]})
            if state["execution_count"] >= deps.limits.max_total_executions:
                task["status"], task["result"] = TaskStatus.FAILED, content
                verdict = "abort"
                failure = ErrorCode.AG_PLAN_MAX_STEPS_EXCEEDED.value
            elif correction is not None and last["attempt"] < deps.limits.max_attempts_per_task:
                # 参数偏差已被修正：换上新参数重试同一个任务，不必重规划整张图。
                task["arguments"] = correction
                task["status"], verdict = TaskStatus.PENDING, "retry"
            elif (last["retry_policy"] in _RETRIABLE
                  and last["attempt"] < deps.limits.max_attempts_per_task):
                # 退回 pending，让 Scheduler 下一轮重新派发同一个任务
                task["status"], verdict = TaskStatus.PENDING, "retry"
            elif state["replan_count"] < deps.limits.max_replans:
                task["status"], task["result"] = TaskStatus.FAILED, content
                verdict = "replan"
            else:
                task["status"], task["result"] = TaskStatus.FAILED, content
                verdict = "abort"
                failure = ErrorCode.AG_PLAN_NO_PROGRESS.value

        verdicts.append(verdict)
        lines.append(f"{task['id']}→{verdict}")

    final = max(verdicts, key=lambda v: _VERDICT_RANK[v]) if verdicts else "success"
    out: dict[str, Any] = {
        "tasks": tasks, "verdict": final, "errors": errors,
        # 空列表触发 reducer 重置，避免下一轮重复消费
        "outcomes": [], "reviews": [],
        "trace": [f"evaluator: {' '.join(lines)} → {final}"
                  + (f"（{failure}）" if failure else "")]}
    # 只在判定中止时写 failure。Replanner 失败后剩余任务仍会继续执行，
    # 它们成功不代表恢复发生了 —— 无条件写回 None 会把那个失败抹掉。
    if failure:
        out["failure"] = failure
    return out
