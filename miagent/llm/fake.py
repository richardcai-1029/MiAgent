"""假模型：不联网、确定性、可编排。

它承担两个职责：
  · 测试时提供确定性输出 —— 否则断言无从写起
  · 不依赖任何推理服务，让整张图能完整跑通并验证每条分支

两种用法：
    FakeLLM(script=["第一次的回答", "第二次的回答"])   按序返回
    FakeLLM(responder=lambda msgs: ...)               自定义逻辑
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from ..protocol import AgentError, ErrorCode
from .base import LLM, LLMMessage

Responder = Callable[[list[LLMMessage]], str]


class FakeLLM(LLM):
    """不联网、确定性的模型实现：按脚本依次返回，或由 responder 按对话内容作答。"""

    name = "fake"

    def __init__(
        self,
        script: Sequence[str] | None = None,
        responder: Responder | None = None,
        context_limit: int = 8000,
    ) -> None:
        super().__init__()
        self._script = list(script or [])
        self._responder = responder
        self.context_limit = context_limit
        self.seen: list[list[LLMMessage]] = []   # 记录每次收到的对话，便于断言

    def _complete(self, messages: list[LLMMessage]) -> str:
        self.seen.append(messages)
        if self._responder is not None:
            return self._responder(messages)
        if not self._script:
            # 脚本用完还被调用，通常意味着图的循环次数超出预期 —— 这是个信号，
            # 不该静默返回空串把问题掩盖过去。
            raise AgentError(
                ErrorCode.AG_LLM_UNAVAILABLE,
                "FakeLLM 脚本已耗尽（图的调用次数超出预期？）",
                detail={"calls": self.call_count},
            )
        return self._script.pop(0)

    @property
    def last_prompt(self) -> str:
        """最后一次收到的完整对话，调试时用。"""
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in self.seen[-1]) if self.seen else ""


def plan_json(*steps: dict[str, Any]) -> str:
    """构造一份计划的 JSON 文本，供 FakeLLM 脚本使用。

    真实模型返回的是文本，所以这里也返回文本 —— 让 Planner 的解析逻辑
    在测试中走的是与生产完全相同的路径。
    """
    return json.dumps({"steps": list(steps)}, ensure_ascii=False)


def step(tool: str, reason: str = "", **arguments: Any) -> dict[str, Any]:
    """构造计划中的一步。"""
    return {"tool": tool, "arguments": arguments, "reason": reason}
