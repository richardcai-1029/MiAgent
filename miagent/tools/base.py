"""工具统一抽象。

Agent 面对两类工具，差异是实打实的：

              本地工具              MiClaw 系统工具
    开销      微秒（函数调用）        毫秒（跨进程 IPC）
    失败      Python 异常            MC- 错误码
    权限      无                     需授权，可能被拒
    副作用    通常无                 通常有（发短信、开 App）
    重试      幂等，可随意重试        未必幂等

处理原则是把这两件事分开：

  · 对模型统一 —— 只暴露 name / description / inputSchema 三样。
    模型不需要知道调用走不走 IPC，多给它一个维度就多一份出错机会。
  · 对框架分层 —— source / required_permission / 重试策略保留差异，
    超时长度、权限过滤、日志分层全靠它。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..protocol import ErrorCode, MiAgentError, RetryPolicy, retry_policy
from .validation import normalize


class ToolSource(StrEnum):
    """工具来源。只给框架看，不进模型的工具列表。"""

    LOCAL = "local"      # 进程内 Python 函数
    MICLAW = "miclaw"    # 经协议调用的系统能力


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结果。

    关键设计：**工具失败不抛异常，而是返回带 is_error 的结果。**

    因为在 Agent 循环里，工具失败是"正常剧情"而不是"程序崩了" ——
    模型需要看到失败原因，然后换个方案继续。如果任由异常往上抛，
    整张图就断了，模型连补救的机会都没有。
    """

    content: str
    is_error: bool = False
    error_code: ErrorCode | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def retry_policy(self) -> RetryPolicy:
        """该怎么处理这次失败。由错误码段位推导，不需要逐条登记。"""
        return retry_policy(self.error_code) if self.error_code else RetryPolicy.NONE

    def for_model(self) -> str:
        """转成给模型看的文本。

        这段文字是 **prompt，不是日志**。所以它必须回答模型的下一个问题：
        「那我现在该怎么办？」—— 只说"失败了"，模型多半会原样重试一次。
        """
        if not self.is_error:
            return self.content

        advice = {
            RetryPolicy.BACKOFF:     "这是临时故障，可以稍后重试同样的调用。",
            RetryPolicy.REHANDSHAKE: "会话状态异常，需要重新建立连接后再试。",
            RetryPolicy.DEGRADE:     "设备资源不足，请换一个更轻量的方案，不要重试。",
            RetryPolicy.ASK_USER:    "权限不足，请告知用户需要授权，不要重试。",
            RetryPolicy.NONE:        "原样重试不会成功，请修正参数或换一种方式。",
        }[self.retry_policy]
        code = f"[{self.error_code}] " if self.error_code else ""
        return f"调用失败：{code}{self.content}\n{advice}"


class Tool(ABC):
    """所有工具的基类。

    用「模板方法」模式：invoke() 负责所有工具共通的事（参数校验、异常兜底），
    子类只实现 _run() 这一个真正干活的方法。这样新增工具时不可能忘记做校验。
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    source: ToolSource
    required_permission: str | None = None
    estimated_memory_mb: int = 0

    def invoke(self, arguments: dict[str, Any] | None = None) -> ToolResult:
        """调用工具。任何情况下都返回 ToolResult，不向上抛异常。"""
        args = arguments or {}
        try:
            return ToolResult(content=self._run(self._validate(args)))
        except MiAgentError as e:
            # 我们自己体系内的错误，错误码原样保留
            return ToolResult(content=e.message, is_error=True,
                              error_code=e.code, detail=e.detail)
        except Exception as e:
            # 工具实现里的意外异常。不能让它炸掉整张图。
            return ToolResult(
                content=f"工具内部错误: {e}",
                is_error=True,
                error_code=ErrorCode.AG_TOOL_EXECUTION_FAILED,
                detail={"exception": type(e).__name__},
            )

    def _validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """按 schema 校验并归一化参数，返回可直接传给 _run 的新字典。

        不合法的调用在本地就被挡下，省掉一次无谓的 IPC；无歧义的格式漂移
        （数字写成字符串、单值漏了数组包装）在这里被吸收，不必多烧一轮
        端侧推理去让模型重写。边界见 validation 模块。
        """
        try:
            return normalize(args, self.input_schema)
        except MiAgentError as e:
            e.detail["tool"] = self.name
            raise

    @abstractmethod
    def _run(self, args: dict[str, Any]) -> str:
        """子类实现：真正干活，返回文本结果。失败就抛 MiAgentError。"""

    def to_model_schema(self) -> dict[str, Any]:
        """转成喂给大模型的格式。

        ★ 只有这三个字段。source、权限、内存开销一律不给模型 ——
        它不需要，给了反而增加它出错的可能。
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} ({self.source})>"
