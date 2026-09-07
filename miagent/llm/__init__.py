"""模型层：Fake / MiMo 端侧 / 云端三种实现共享同一契约。

OpenAICompatibleLLM 不在此处导出，需显式从子模块导入 ——
避免端侧场景无谓地加载 openai 及其 HTTP 依赖栈。
"""

from .base import LLM, LLMMessage, LLMResponse, assistant, system, user
from .fake import FakeLLM, plan_json, step

__all__ = [
    "LLM", "FakeLLM", "LLMMessage", "LLMResponse",
    "assistant", "plan_json", "step", "system", "user",
]
