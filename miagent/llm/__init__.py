"""模型层：所有实现共享 LLM 这一个契约，上层不感知底下是哪个模型。

    base               LLM 契约、结构化输出与自修复、上下文超限拦截
    context            提示词片段与按优先级的预算裁剪
    fake               FakeLLM：确定性、可编排，测试与离线演示用
    openai_compatible  OpenAI 兼容端点的通用实现
    qwen / ollama      两个部署预设：通义千问（DashScope）、Ollama 本地部署

OpenAI 兼容实现及其预设不在此处导出，需显式从子模块导入 ——
避免端侧场景无谓地加载 openai 及其 HTTP 依赖栈。接入新的模型后端
（包括 MiMo）只需实现 LLM 契约，图与节点不变。
"""

from .base import LLM, LLMMessage, LLMResponse, assistant, system, user
from .fake import FakeLLM, plan_json, step

__all__ = [
    "LLM", "FakeLLM", "LLMMessage", "LLMResponse",
    "assistant", "plan_json", "step", "system", "user",
]
