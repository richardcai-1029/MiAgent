"""节点运行所需的外部依赖。

单独成模块，使 `miagent.graph` 的导出不必触及 nodes 与 build ——
后者会引入图引擎。
"""

from __future__ import annotations

from dataclasses import dataclass

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
