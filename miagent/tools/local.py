"""本地工具：进程内的 Python 函数。

写工具的人不应该手写 JSON Schema —— 手写的 schema 和函数签名是两份
拷贝，迟早会不同步（改了参数忘了改 schema，模型就会传错参数）。
这里从签名和 docstring 自动生成，函数签名成为唯一事实来源。
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

from .base import Tool, ToolSource

# Python 类型 -> JSON Schema 类型。只覆盖基础类型，
# 复杂结构请显式传 input_schema，不要指望自动推断猜对。
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_arg_docs(doc: str | None) -> dict[str, str]:
    """从 docstring 的 Args: 段落里提取每个参数的说明。

    支持这种常见写法：

        Args:
            to: 收件人号码
            text: 短信正文
    """
    if not doc or "Args:" not in doc:
        return {}
    body = doc.split("Args:", 1)[1]
    docs: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue                   # Args: 后面通常紧跟一个空行，跳过
        if line.endswith(":") and " " not in line:
            break                      # 遇到下一个段落标题（Returns: 等）才停
        m = re.match(r"^(\w+)\s*:\s*(.+)$", line)
        if m:
            docs[m.group(1)] = m.group(2).strip()
    return docs


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """由函数签名生成 JSON Schema。没有默认值的参数即为必填。"""
    sig = inspect.signature(fn)
    arg_docs = _parse_arg_docs(inspect.getdoc(fn))

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        json_type = _TYPE_MAP.get(param.annotation, "string")
        properties[name] = {"type": json_type,
                            "description": arg_docs.get(name, name)}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


class LocalTool(Tool):
    """把一个普通 Python 函数包装成工具。"""

    source = ToolSource.LOCAL

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        # 描述取 docstring 的第一段（Args: 之前的部分）—— 那才是给模型看的
        doc = inspect.getdoc(fn) or ""
        self.description = description or doc.split("Args:")[0].strip() or self.name
        self.input_schema = input_schema or schema_from_signature(fn)

    def _run(self, args: dict[str, Any]) -> str:
        return str(self._fn(**args))


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], LocalTool]:
    """装饰器：把函数变成 LocalTool。

        @tool()
        def add_days(date: str, days: int) -> str:
            '''在给定日期上加若干天。

            Args:
                date: 起始日期，格式 YYYY-MM-DD
                days: 要增加的天数
            '''
            ...

    装饰后 add_days 就是一个 LocalTool 实例，可以直接注册进 ToolRegistry。
    """

    def decorator(fn: Callable[..., Any]) -> LocalTool:
        return LocalTool(fn, name=name, description=description, input_schema=input_schema)

    return decorator
