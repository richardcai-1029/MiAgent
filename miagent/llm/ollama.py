"""Ollama 本地部署（OpenAI 兼容端点 /v1）的接入预设。

OllamaLLM 不新增任何调用逻辑，只把 OpenAICompatibleLLM 的开关按 Ollama
的行为预置好：

  · 鉴权          Ollama 不校验 API Key，填占位符即可，不读环境变量
  · 结构化输出    /v1 支持 response_format=json_schema（Ollama 0.34.2 实测），
                  保留原生模式，不走 prompt 注入
  · 思考模式      Qwen3 这类思考型模型默认先输出大段推理再作答，一次规划
                  要 15~30 s；/v1 端点不认 think 字段（实测无效），只认
                  Qwen 约定的 `/no_think` 软开关。这里在发出前追加一条
                  system("/no_think")，图代码不必知道底下是思考型模型。
                  追加在末尾而非改写首条 system：自修复轮次会在对话尾部
                  续接 assistant / user，软开关以最近一次出现为准。

context_limit 仍按字符数近似（见 LLM.estimate）。Ollama 真正的上限是部署侧
Modelfile 的 num_ctx（token 数，`/api/show` 可查），超出时服务端**静默截断**
输入而不报错；0.34.2 没有 tokenize 端点，字符到 token 只能靠实测比例换算，
所以这个值要按部署侧配置由调用方给出，预设不替它猜。

换算依据（2026-09-22 实测，qwen3-8b Q4_K_M，本框架四轮会话共 8 次调用）：
planner prompt 含工具 schema、ASCII 多，3.2~3.4 字符/token；finalizer 中文为主，
2.0~2.1 字符/token。按最保守的 2.0 折算，num_ctx=8192 对应约 16000 字符。

用法：
    llm = OllamaLLM(base_url="http://192.168.3.121:11434/v1", model="qwen3-8b:latest")
    llm = OllamaLLM(..., think=True)     # 复杂推理场景放开思考
"""

from __future__ import annotations

from typing import Any

from .base import LLMMessage, system
from .openai_compatible import OpenAICompatibleLLM

DEFAULT_MODEL = "qwen3-8b:latest"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
NO_THINK = "/no_think"


class OllamaLLM(OpenAICompatibleLLM):
    """Ollama 本地部署的预设：占位鉴权、保留原生结构化输出、追加 /no_think 关闭思考。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        context_limit: int = 32000,
        timeout: float = 120.0,
        think: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key="ollama",
            context_limit=context_limit,
            timeout=timeout,
        )
        self._think = think

    def _request(self, messages: list[LLMMessage], **kwargs: Any) -> str:
        if not self._think:
            messages = [*messages, system(NO_THINK)]
        return super()._request(messages, **kwargs)
