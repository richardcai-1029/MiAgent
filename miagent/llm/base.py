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

import json
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from ..protocol import AgentError, ErrorCode

if TYPE_CHECKING:
    from ..runtime.slots import SlotPool

T = TypeVar("T", bound=BaseModel)

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


def first_json_object(text: str) -> str | None:
    """截出第一个括号配对完整的 JSON 对象，顺带去掉串外的尾随逗号。

    逐字符扫描而不是取「第一个 `{` 到最后一个 `}`」：模型输出里出现两个
    JSON 块时，后者会把两块连同中间的文字一起截进来，得到的东西既不是
    前一个对象也不是后一个。

    扫描过程区分字符串内外，因此去掉的一定是结构上的尾随逗号，
    不会动到字符串值里的逗号。

    括号未配平（输出被截断）时返回 None —— **不修补截断的 JSON**。
    补出来的括号位置只是猜测，得到的对象结构可能与模型的本意不同，
    而这个对象会被真实执行。让它失败并触发自修复是更安全的选择。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    trailing: list[int] = []      # 结构上的尾随逗号，其绝对下标
    last_comma = -1               # 最近一个串外逗号；遇到别的实义字符即作废

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string, last_comma = True, -1
        elif ch == ",":
            last_comma = i
        elif ch in "{[":
            depth, last_comma = depth + 1, -1
        elif ch in "}]":
            if last_comma != -1:      # 逗号与闭合符之间只有空白 -> 尾随逗号
                trailing.append(last_comma)
            depth, last_comma = depth - 1, -1
            if depth == 0:
                dropped = set(trailing)
                return "".join(c for j, c in enumerate(text[start:i + 1], start)
                               if j not in dropped)
        elif not ch.isspace():
            last_comma = -1

    return None


def extract_json(text: str) -> str:
    """从模型输出里截出 JSON 部分。

    端侧模型不保证每次都输出规整的 JSON。这一层吸收几种稳定出现且
    **无歧义**的脏输出：markdown 围栏、前后的客套话、结构上的尾随逗号、
    正文里出现多个 JSON 块。

    引号错用（单引号、全角引号）与截断不在此列 —— 那些的修复方式不唯一，
    交由 complete_structured 的自修复重试处理。

    截不出东西时原样返回：解析失败后的错误信息要连同原始输出一起喂回给
    模型，截过的文本会让它看不出自己错在哪。
    """
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.removeprefix("json").strip()
            if part.startswith("{"):
                return first_json_object(part) or part
    return first_json_object(text) or text


class LLM(ABC):
    """模型接口。

    与 Tool 一样采用模板方法：complete() 负责所有实现共通的事
    （上下文长度检查、计时、统计），子类只实现 _complete()。
    这样新接一个模型时不可能漏掉上下文检查。
    """

    name: str = "llm"
    context_limit: int = 8000   # 以字符数近似衡量；端侧模型这个值会明显更小

    # 实现是否原生保证结构化输出（JSON mode / 约束解码）。
    # 为 False 时，complete_structured 退化为「prompt 注入 schema + 校验 + 自修复」。
    supports_native_structured_output: bool = False

    # 推理资源槽。多个请求共用一个模型实例时，由它决定同时进行几次推理、
    # 谁先谁后（见 miagent.runtime）；为 None 时不仲裁。容量取部署侧
    # 实际能同时服务的推理数。
    slots: SlotPool | None = None

    def __init__(self) -> None:
        self.call_count = 0
        self.total_elapsed_ms = 0.0
        self.repair_count = 0

    def estimate(self, text: str) -> int:
        """估算一段文本占多少上下文预算。

        默认按字符数计。这是**近似值而非真实 token 数** —— MiMo 的 tokenizer
        当前不可获取，字符数是唯一可用的口径。接入真实 tokenizer 后覆写此方法，
        上层的预算与裁剪逻辑不受影响。
        """
        return len(text)

    def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        return self._guarded(messages, lambda: self._complete(messages))

    def _guarded(self, messages: list[LLMMessage], call: Callable[[], str]) -> LLMResponse:
        """所有实现共通的事都在这里：上下文长度检查、计时、统计。

        普通补全与结构化补全都经过这一层，子类钩子（_complete /
        _complete_structured）无论内部怎么发请求，都不可能绕开检查与计数。
        """
        prompt_chars = sum(self.estimate(m.content) for m in messages)
        if prompt_chars > self.context_limit:
            # 端侧窗口小，宁可在发出前拦住，也不要让模型截断输入后
            # 返回一个看起来合理、实则基于残缺上下文的答案。
            raise AgentError(
                ErrorCode.AG_CONTEXT_OVERFLOW,
                f"上下文 {prompt_chars} 字符，超出 {self.name} 的 {self.context_limit} 上限",
                detail={"prompt_chars": prompt_chars, "limit": self.context_limit},
            )

        # 计时从拿到槽之后开始：elapsed_ms 衡量的是推理耗时，不含排队。
        with self.slots.hold() if self.slots else nullcontext():
            started = time.perf_counter()
            content = call()
            elapsed = (time.perf_counter() - started) * 1000

            self.call_count += 1
            self.total_elapsed_ms += elapsed
        return LLMResponse(content=content, elapsed_ms=round(elapsed, 2),
                           prompt_chars=prompt_chars, model=self.name)

    # ------------------------------------------------------------
    # 结构化输出
    # ------------------------------------------------------------

    def complete_structured(self, messages: list[LLMMessage], schema: type[T],
                            max_repairs: int = 1) -> T:
        """要求模型返回符合 schema 的结构化结果。

        调用方只声明「我要这个形状的数据」，怎么保证由实现层决定：

            原生支持（云端 JSON mode / 端侧约束解码）
                -> _complete_structured 覆写，格式由推理栈保证
            不支持（当前的 FakeLLM）
                -> 把 schema 注入 prompt，输出后校验，不合格就带着
                   错误原因让模型重来（自修复）

        schema 由 Pydantic 模型自动导出，与校验用的是同一个定义，
        不存在「提示词写的格式」和「实际校验的格式」不一致的可能。
        """
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        convo = [*messages, system(
            f"你的回复必须是、且只能是符合下面 JSON Schema 的 JSON。"
            f"不要输出任何解释、前言或代码块标记。\n\n{schema_json}"
        )]

        last_error = ""
        for attempt in range(max_repairs + 1):
            raw = self._guarded(convo, lambda: self._complete_structured(convo, schema)).content
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_error = _summarize_errors(e)
                if attempt >= max_repairs:
                    raise AgentError(
                        ErrorCode.AG_LLM_INVALID_RESPONSE,
                        f"模型输出不符合 schema（已重试 {attempt} 次）",
                        detail={"schema": schema.__name__, "errors": last_error,
                                "raw": raw[:200]},
                    ) from e
                # ★ 自修复的关键：不是原样重试，而是把「你上次错在哪」喂回去
                self.repair_count += 1
                convo = [*convo, assistant(raw), user(
                    f"上面的输出不符合要求：{last_error}\n"
                    f"请重新输出，只给合法 JSON，不要有其他内容。")]

        raise AssertionError("unreachable")

    def _complete_structured(self, messages: list[LLMMessage], schema: type[T]) -> str:
        """默认走普通补全。原生支持结构化输出的实现应覆写此方法。

        调 _complete 而非 complete：检查与计数已由 complete_structured 做过。
        """
        return self._complete(messages)

    @abstractmethod
    def _complete(self, messages: list[LLMMessage]) -> str:
        """子类实现：真正调用模型，返回文本。"""

    def stats(self) -> dict[str, float]:
        """累计统计，供 bench 使用（任务书要求的「推理耗时」指标）。"""
        return {
            "calls": self.call_count,
            "total_ms": round(self.total_elapsed_ms, 2),
            "avg_ms": round(self.total_elapsed_ms / self.call_count, 2) if self.call_count else 0.0,
            "repairs": self.repair_count,
        }


def _summarize_errors(exc: Exception) -> str:
    """把校验错误压成一句能喂回给模型的话。

    直接把 pydantic 的完整报错塞回 prompt 太长，而且夹杂 URL 与类型术语，
    小模型容易被带偏。这里只保留「哪个字段、错在哪」。
    """
    if isinstance(exc, ValidationError):
        return "；".join(
            f"字段 {'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
            for e in exc.errors()[:5]
        )
    return str(exc)[:200]
