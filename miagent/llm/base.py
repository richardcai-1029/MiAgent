"""大模型接口抽象。

设计目标是让上层（图节点）完全不知道底下是哪个模型：
Fake（测试）、MiMo 端侧、云端 API 三种实现共享同一契约，可热切换。

接口刻意保持最小 —— 只有「给一段对话，返回一段文本」。
把文本解析成计划、判断解析是否成功，都是上层的事，不属于模型层。
接口越小，替换实现的成本越低。

端侧相对云端多出的两项约束体现在这里：
  · context_limit  端侧模型窗口显著小于云端，超限要在发出前拦住（AG-3001）
  · elapsed_ms     推理耗时是端侧的核心指标，每次调用都要可测量
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from ..protocol import AgentError, ErrorCode

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str

    @property
    def size(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    elapsed_ms: float = 0.0
    prompt_chars: int = 0
    model: str = ""

    def __str__(self) -> str:
        return self.content


def system(content: str) -> LLMMessage:
    return LLMMessage("system", content)


def user(content: str) -> LLMMessage:
    return LLMMessage("user", content)


def assistant(content: str) -> LLMMessage:
    return LLMMessage("assistant", content)


class LLM(ABC):
    """模型接口。

    与 Tool 一样采用模板方法：complete() 负责所有实现共通的事
    （上下文长度检查、计时、统计），子类只实现 _complete()。
    这样新接一个模型时不可能漏掉上下文检查。
    """

    name: str = "llm"
    context_limit: int = 8000   # 以字符数近似衡量；端侧模型这个值会明显更小

    def __init__(self) -> None:
        self.call_count = 0
        self.total_elapsed_ms = 0.0

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        prompt_chars = sum(m.size for m in messages)
        if prompt_chars > self.context_limit:
            # 端侧窗口小，宁可在发出前拦住，也不要让模型截断输入后
            # 返回一个看起来合理、实则基于残缺上下文的答案。
            raise AgentError(
                ErrorCode.AG_CONTEXT_OVERFLOW,
                f"上下文 {prompt_chars} 字符，超出 {self.name} 的 {self.context_limit} 上限",
                detail={"prompt_chars": prompt_chars, "limit": self.context_limit},
            )

        started = time.perf_counter()
        content = self._complete(messages)
        elapsed = (time.perf_counter() - started) * 1000

        self.call_count += 1
        self.total_elapsed_ms += elapsed
        return LLMResponse(content=content, elapsed_ms=round(elapsed, 2),
                           prompt_chars=prompt_chars, model=self.name)

    @abstractmethod
    def _complete(self, messages: list[LLMMessage]) -> str:
        """子类实现：真正调用模型，返回文本。"""

    def stats(self) -> dict[str, float]:
        """累计统计，供 bench 使用（任务书要求的「推理耗时」指标）。"""
        return {
            "calls": self.call_count,
            "total_ms": round(self.total_elapsed_ms, 2),
            "avg_ms": round(self.total_elapsed_ms / self.call_count, 2) if self.call_count else 0.0,
        }
