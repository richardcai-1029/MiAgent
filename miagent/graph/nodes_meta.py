"""节点运行所需的外部依赖。

单独成模块，使 `miagent.graph` 的导出不必触及 nodes 与 build ——
后者会引入图引擎。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..llm import LLM
from ..tools import ToolRegistry


@dataclass
class Deps:
    llm: LLM
    registry: ToolRegistry
    # MiClaw 侧并发上限，取自握手时下发的 ResourceBudget.max_concurrent_calls。
    # 本地工具不受此限：它们是 CPU 密集的，受 GIL 限制并发无收益，
    # 也不占用系统侧资源配额。
    max_concurrent_miclaw: int = 2

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
