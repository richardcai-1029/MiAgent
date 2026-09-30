"""通义千问（DashScope OpenAI 兼容模式）的接入预设。

QwenLLM 不新增任何调用逻辑，只是把 OpenAICompatibleLLM 的几个开关按
DashScope 的要求预置好：

  · 流式调用      qwen2.5-omni 系列不接受非流式请求，会直接报错
  · modalities    omni 是多模态模型，这里只要文本：["text"]
  · 结构化输出    走基类的「schema 注入 + 校验 + 自修复」，不用服务端 JSON 模式
  · API Key       依次取：构造参数 api_key → 环境变量 DASHSCOPE_API_KEY
                  → 当前目录 .env 文件里的同名项（.env 已被 .gitignore 排除）

用法：
    export DASHSCOPE_API_KEY=sk-...          # 或写进项目根目录的 .env
    llm = QwenLLM()                          # 默认 qwen2.5-omni-7b
    llm = QwenLLM(model="qwen-plus")         # 换模型只改这一处
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..protocol import AgentError, ErrorCode
from .openai_compatible import OpenAICompatibleLLM

DEFAULT_MODEL = "qwen2.5-omni-7b"
DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"
DOTENV = Path(".env")


def _read_dotenv(path: Path, key: str) -> str | None:
    """从 .env 里取一项。只认 `KEY=VALUE` 行，跳过注释与空行，
    去掉值两侧的引号；文件不存在时返回 None。不引入 python-dotenv。"""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().removeprefix("export ").strip() == key:
            return v.strip().strip("'\"") or None
    return None


class QwenLLM(OpenAICompatibleLLM):
    """通义千问（DashScope）的预设：流式调用、只取文本模态、API Key 依次从参数、环境变量、.env 读取。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        context_limit: int = 32000,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV) or _read_dotenv(DOTENV, API_KEY_ENV)
        if not key:
            raise AgentError(
                ErrorCode.AG_LLM_UNAVAILABLE,
                f"未提供 API Key：请传入 api_key，设置环境变量 {API_KEY_ENV}，"
                f"或写入 {DOTENV.resolve()}",
                detail={"model": model, "env": API_KEY_ENV},
            )

        extra_body: dict[str, Any] = {}
        if "omni" in model:
            # omni 模型的输出模态是必填项；其他 Qwen 模型不认这个字段，不能带。
            extra_body["modalities"] = ["text"]

        super().__init__(
            model=model,
            base_url=base_url,
            api_key=key,
            context_limit=context_limit,
            timeout=timeout,
            stream=True,
            native_structured_output=False,
            extra_body=extra_body,
        )
