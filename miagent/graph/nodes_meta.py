"""节点运行所需的外部依赖。

单独成模块，使 `miagent.graph` 的导出不必触及 nodes 与 build ——
后者会引入图引擎。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..llm import LLM
from ..tools import ToolRegistry

if TYPE_CHECKING:
    from ..runtime.slots import SlotPool


@dataclass
class Deps:
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
    # 单任务重试上限见 MAX_ATTEMPTS_PER_TASK，故一次任务至多退避一次，
    # 退避时长是单一固定值。
    retry_delay_ms: int = 200

    # 等待的实现可注入，使退避行为能在测试中被观察，不必真的等。
    sleep: Callable[[float], None] = field(default=time.sleep)

    # 是否做语义校验（工具结果是否达成任务、整轮目标是否达成）。
    # 关掉则两个校验节点直接透传，且重规划不再清除失败（见 ledger.accept）：
    # 没有完成校验确认新计划真的接手了失败的部分，就按错误码如实报告未达成。
    # 因此关掉之后执行失败不会被报成完成，代价是恢复成功的轮次也报为未达成。
    # 模型漏做一步、参数写偏这类没有失败记录的偏差，关掉之后就发现不了。
    verify: bool = True
