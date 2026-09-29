"""记录每次推理的模型包装：耗时、提示词与输出长度、token 用量、调用它的节点。

走的仍是 OllamaLLM 的 OpenAI 兼容端点与 /no_think 软开关，与生产用法一致；
只在发请求处多取一份 usage（服务端回报的 prompt / completion token 数）。
keep_prompts > 0 时另存前若干条完整提示词，供推理速度回放。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from miagent.llm import LLMMessage
from miagent.llm.ollama import NO_THINK, OllamaLLM
from miagent.llm.base import system

ROLES = (("结果校验器", "result_verifier"), ("完成校验器", "goal_verifier"),
         ("重规划器", "replanner"), ("规划器", "planner"))


def role_of(messages: list[LLMMessage]) -> str:
    head = messages[0].content if messages else ""
    return next((name for key, name in ROLES if key in head), "finalizer")


class RecordingOllama(OllamaLLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []
        self.keep_prompts = 0
        self._lock = threading.Lock()

    def _request(self, messages: list[LLMMessage], **kwargs: Any) -> str:
        if not self._think:
            messages = [*messages, system(NO_THINK)]
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        }
        t0 = time.perf_counter()
        try:
            resp = self._ensure_client().chat.completions.create(**request)
        except Exception as e:
            from miagent.llm.openai_compatible import _as_agent_error
            raise _as_agent_error(e, self._model) from e
        elapsed = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        with self._lock:
            self.records.append({
                "role": role_of(messages),
                "ms": round(elapsed, 1),
                "prompt_chars": sum(len(m.content) for m in messages),
                "output_chars": len(text),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "structured": "response_format" in kwargs,
            })
            if len(self.prompts) < self.keep_prompts:
                fmt = kwargs.get("response_format", {}).get("json_schema", {}).get("schema")
                self.prompts.append({"role": role_of(messages), "format": fmt,
                                     "messages": request["messages"]})
        return text
