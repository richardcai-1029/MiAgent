"""执行节点：LocalExecutor 与 MiClawExecutor 共用的执行体。不调用模型。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..core.state import TaskOutcome
from ..tools import ToolSource
from .deps import Deps


def executor(payload: dict[str, Any], deps: Deps, source: ToolSource) -> dict[str, Any]:
    """本地工具与 MiClaw 工具的共用执行体。

    ★ 这个节点按任务扇出调用，payload 就是它看到的【全部 state】——
      里面只有派发的那一个任务（{"task": ...}），读不到 tasks、execution_count
      等主状态字段。因此返回的是【增量】：outcomes 追加一条、execution_count 加一，
      由 AgentState 上声明的 reducer 汇总。若返回绝对值，多个分支会互相覆盖。

    现在两条路径一致。将来 MiClaw 侧要加批量合并请求、按工具粒度的超时、
    配额预筛时，只在这个函数里分叉，图的结构不用动。
    """
    task = payload["task"]
    attempt = task["retry_count"] + 1

    # 退避。能走到重试的只有段位 1（传输抖动）与段位 2（协议状态）两类失败，
    # 它们的共同点是「过一会儿可能就好了」；立即重发时状况还没来得及改变。
    if attempt > 1:
        deps.sleep(deps.retry_delay_ms / 1000)

    slots = deps.miclaw_slots if source is ToolSource.MICLAW else None
    with slots.hold() if slots else nullcontext():
        result = deps.registry.invoke(task["required_tool"], task["arguments"])

    outcome = TaskOutcome(
        task_id=task["id"], tool=task["required_tool"], ok=not result.is_error,
        # 存 for_model()：失败时它带着「下一步该怎么办」，
        # Replanner 与 Finalizer 拿到的就是可直接用的信息
        content=result.for_model(),
        error_code=result.error_code.value if result.error_code else None,
        retry_policy=result.retry_policy.value, attempt=attempt,
    )
    mark = "✓" if outcome["ok"] else f"✗ {outcome['error_code']}"
    waited = f"退避 {deps.retry_delay_ms}ms 后，" if attempt > 1 else ""
    return {"outcomes": [outcome], "execution_count": 1,
            "trace": [f"{source.value}_executor: {task['id']} {mark}（{waited}第 {attempt} 次）"]}
