"""节点运行所需的外部依赖与循环出口的上限。

每个节点都是 `(state, deps) -> 增量` 的普通函数，节点之外的一切 —— 模型、
工具、并发配额、退避、上限 —— 都经 Deps 注入。编排框架只负责把 deps 绑定到
节点上（见 miagent.adapters），换框架时这里不变。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..llm import LLM
from ..tools import ToolRegistry

if TYPE_CHECKING:
    from ..runtime.slots import SlotPool


@dataclass(frozen=True)
class Limits:
    """循环出口的兜底上限。模型输出再离谱，一次请求也会在这些上限内结束。"""

    max_attempts_per_task: int = 2    # 单个任务最多执行几次（含首次）
    max_replans: int = 2              # 最多重规划几次，超出 -> AG-1003
    # 单次请求累计最多执行几次工具，超出 -> AG-1002。
    # 一份计划拆出的任务数也以它为上限：超出预算的计划在预算内必然跑不完。
    max_total_executions: int = 20


@dataclass
class Deps:
    """节点运行所需的全部外部依赖。由适配层构造一次，绑定到每个节点上。"""

    llm: LLM
    registry: ToolRegistry
    # MiClaw 侧并发上限，取自握手时下发的 ResourceBudget.max_concurrent_calls。
    # 本地工具不受此限：它们是 CPU 密集的，受 GIL 限制并发无收益，
    # 也不占用系统侧资源配额。
    max_concurrent_miclaw: int = 2

    # MiClaw 调用槽，多个请求同时在飞时共用（见 miagent.runtime）。
    # max_concurrent_miclaw 只约束单个请求一轮派发多少，几个请求合起来
    # 仍可能超出配额；配额是按会话下发的，要由所有请求共用的这个池兜住。
    # 为 None 时不仲裁，适用于同一时刻只有一个请求的用法。
    miclaw_slots: SlotPool | None = None

    # 重试前的等待时长。RetryPolicy.BACKOFF 要求"退避后重试"，立即重发
    # 对传输抖动与资源占用这两类失败几乎没有意义 —— 状况还没来得及改变。
    #
    # ⚠️ 默认值无外部依据，仅为占位：MiClaw 的资源配额未定义退避时长。
    #    真实数值应由 MiClaw 规范或项目决策给定后替换。
    #
    # 单任务执行次数上限见 Limits.max_attempts_per_task，默认下一次任务
    # 至多退避一次，退避时长是单一固定值。
    retry_delay_ms: int = 200

    # 等待的实现可注入，使退避行为能在测试中被观察，不必真的等。
    sleep: Callable[[float], None] = field(default=time.sleep)

    # 是否做语义校验（工具结果是否达成任务、整轮目标是否达成）。
    # 关掉则两个校验节点直接透传，且重规划不再清除失败（见 ledger.accept）：
    # 没有完成校验确认新计划真的接手了失败的部分，就按错误码如实报告未达成。
    # 因此关掉之后执行失败不会被报成完成，代价是恢复成功的轮次也报为未达成。
    # 模型漏做一步、参数写偏这类没有失败记录的偏差，关掉之后就发现不了。
    verify: bool = True

    # 循环出口的上限。不同部署形态可以给不同的值，默认值见 Limits。
    limits: Limits = field(default_factory=Limits)
