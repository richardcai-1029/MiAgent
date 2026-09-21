"""OpenAI 兼容接口的模型实现（Qwen / MiMo 云端 / DeepSeek 等）。

这一层只负责「把对话发出去、把文本收回来」，两处接入形态上的差异
通过构造参数吸收，上层图代码对此无感：

  · stream       部分模型（如 qwen2.5-omni）只接受流式调用；
                 流式时把增量拼成完整文本再返回，对上层仍是一次同步补全。
  · extra_body   服务商私有的请求字段（如 Qwen omni 的 modalities），
                 原样透传，不在这里做任何解释。
  · native_structured_output
                 服务端是否支持 JSON Schema 约束输出。不支持的模型走基类的
                 「prompt 注入 schema + 校验 + 自修复」路径。

★ openai 包在 _ensure_client 内部才导入，不在模块顶层。
  这样端侧只用 FakeLLM 或本地模型时，不会把整个 openai 包及其
  HTTP 依赖栈加载进内存。
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ..protocol import AgentError, ErrorCode
from .base import LLM, LLMMessage

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleLLM(LLM):
    # 类级默认：服务端用 JSON Schema 模式保证输出格式，无需自修复重试。
    # 具体服务不支持时，通过构造参数 native_structured_output=False 在实例上关掉。
    supports_native_structured_output = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        context_limit: int = 32000,
        timeout: float = 30.0,
        stream: bool = False,
        native_structured_output: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.name = model
        self.context_limit = context_limit
        self.supports_native_structured_output = native_structured_output
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._stream = stream
        self._extra_body = dict(extra_body or {})
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI   # 延迟导入，见模块文档

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=self._timeout)
        return self._client

    def _request(self, messages: list[LLMMessage], **kwargs: Any) -> str:
        """发一次 chat.completions 请求，返回拼好的完整文本。

        流式与非流式的差异只在这里收口：流式把每个 chunk 的 delta.content
        累加；带 include_usage 时最后一个 chunk 没有 choices，直接跳过。
        """
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        }
        if self._extra_body:
            request["extra_body"] = self._extra_body

        try:
            if not self._stream:
                resp = self._ensure_client().chat.completions.create(**request)
                return resp.choices[0].message.content or ""

            parts: list[str] = []
            for chunk in self._ensure_client().chat.completions.create(
                    stream=True, **request):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    parts.append(delta.content)
            return "".join(parts)
        except AgentError:
            raise
        except Exception as e:                      # 网络、鉴权、限流、服务端 5xx
            raise _as_agent_error(e, self._model) from e

    def _complete(self, messages: list[LLMMessage]) -> str:
        return self._request(messages)

    def _complete_structured(self, messages: list[LLMMessage], schema: type[T]) -> str:
        """服务端支持时用 JSON Schema 模式保证格式；否则退回基类的校验 + 自修复。"""
        if not self.supports_native_structured_output:
            return self.complete(messages).content
        return self._request(messages, response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True,
                            "schema": schema.model_json_schema()},
        })


def _as_agent_error(exc: Exception, model: str) -> AgentError:
    """把 openai SDK 的异常收成框架错误码，上层只需认 AG-5001。

    只在这里 import openai 的异常类型，同样是为了不让端侧提前加载它。
    """
    from openai import APIStatusError

    detail: dict[str, Any] = {"model": model, "cause": type(exc).__name__}
    if isinstance(exc, APIStatusError):
        detail["status"] = exc.status_code
    return AgentError(ErrorCode.AG_LLM_UNAVAILABLE,
                      f"调用 {model} 失败：{str(exc)[:200]}", detail=detail)
