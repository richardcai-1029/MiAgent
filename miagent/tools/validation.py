"""工具参数的校验与归一化。

模型输出的参数与工具声明的 schema 之间存在稳定的漂移：数字被写成字符串、
单个值被写成数组、多写一个字段。这一层负责在**调用发生之前**把无歧义的
漂移吸收掉，把有歧义的挡回去。

★ 权限边界写死在这里，不可放宽：

    允许        去掉包装（单值 → 单元素数组）、无损类型转换（"3" → 3）
    禁止        猜缺失的必填参数、丢掉 schema 不认识的字段后放行

  理由是失败与猜错的代价不对称。格式失败会走自修复或重规划，是可控的；
  猜错会产生一个形式合法、语义错误的调用，并被真实执行 —— 端侧的系统调用
  常带不可撤销的副作用（发短信、改设置），猜错比失败危险得多。

校验放在本地做的另一个收益：不合法的调用不必跨出进程，省掉一次 IPC 往返。
"""

from __future__ import annotations

import re

from typing import Any

from ..protocol import AgentError, ErrorCode

_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")


class _Reject(Exception):
    """单个参数无法归一化。仅在本模块内部使用。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _coerce(value: Any, json_type: str) -> Any:
    """把 value 转成 json_type 要求的形态，只做无损转换，否则拒绝。

    bool 在 Python 里是 int 的子类，必须显式排除 —— 否则 True 会被当作
    合法的 integer 放行，而工具拿到的是布尔值。
    """
    if json_type == "string":
        # 不做数字到字符串的转换：63 与 63.0 会得到不同的文本，
        # 该由模型明确写出它要哪一个。
        if isinstance(value, str):
            return value
        raise _Reject(f"应为字符串，实际是 {type(value).__name__}")

    if json_type == "integer":
        if isinstance(value, bool):
            raise _Reject("应为整数，实际是布尔值")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and _INTEGER_TEXT.match(value.strip()):
            return int(value.strip())
        raise _Reject(f"应为整数，实际是 {value!r}")

    if json_type == "number":
        if isinstance(value, bool):
            raise _Reject("应为数值，实际是布尔值")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise _Reject(f"应为数值，实际是 {value!r}") from None
        raise _Reject(f"应为数值，实际是 {type(value).__name__}")

    if json_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise _Reject(f"应为布尔值，实际是 {value!r}")

    if json_type == "array":
        # 只有一个元素时模型常常省掉数组这层包装，补回来是无歧义的。
        return value if isinstance(value, list) else [value]

    if json_type == "object":
        if isinstance(value, dict):
            return value
        raise _Reject(f"应为对象，实际是 {type(value).__name__}")

    return value          # schema 未声明类型或声明了我们不处理的类型，原样放行


def normalize(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """校验并归一化一次调用的参数，返回可直接传给工具的新字典。

    问题一次性全部收集后再报，而不是遇到第一个就返回：这段错误信息会被
    喂回给模型做自修复，一次说清所有问题，模型才可能一次改对。
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    problems: list[str] = []

    missing = [f for f in required if f not in arguments]
    problems += [f"缺少必填参数 {f}" for f in missing]

    # 不认识的字段一律拒绝，不静默丢弃：模型写了它，说明它认为这个字段有意义，
    # 丢掉之后调用会以一种它没预期的方式执行。
    unknown = [f for f in arguments if f not in properties]
    problems += [f"工具没有名为 {f} 的参数" for f in unknown]

    normalized: dict[str, Any] = {}
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue
        try:
            coerced = _coerce(value, spec.get("type", ""))
        except _Reject as e:
            problems.append(f"参数 {name}: {e.reason}")
            continue
        enum = spec.get("enum")
        if enum is not None and coerced not in enum:
            problems.append(f"参数 {name}: 取值必须是 {enum} 之一，实际是 {coerced!r}")
            continue
        normalized[name] = coerced

    if problems:
        raise AgentError(
            ErrorCode.AG_TOOL_SCHEMA_INVALID,
            "；".join(problems),
            detail={"problems": problems, "expected": properties,
                    "required": required},
        )
    return normalized
