"""OpenAI 兼容接口的模型实现（MiMo 云端 / DeepSeek 等）。

⚠️ 尚未针对真实 endpoint 验证 —— MiMo 接口未开放。
   保留此文件是为了固定接入形态：接上真实服务后，上层图代码一行不用改。

★ openai 包在 _complete 内部才导入，不在模块顶层。
  这样端侧只用 FakeLLM 或本地模型时，不会把整个 openai 包及其
  HTTP 依赖栈加载进内存 —— 与清单 E-2/E-3 的裁剪目标一致。
"""

from __future__ import annotations

from .base import LLM, LLMMessage


class OpenAICompatibleLLM(LLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        context_limit: int = 32000,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.name = model
        self.context_limit = context_limit
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI   # 延迟导入，见模块文档

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=self._timeout)
        return self._client

    def _complete(self, messages: list[LLMMessage]) -> str:
        resp = self._ensure_client().chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content or ""
