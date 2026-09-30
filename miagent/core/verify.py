"""语义校验：模型判定「做到的是不是要做的」，纯函数决定这个判定算不算数。

工具返回成功，只说明调用形式合法：`system.query_order` 查了一个订单号、
`system.create_event` 建了一个日程，它们都会说自己成功了。至于查的是不是
用户要的那个单号、建的是不是用户要的那件事，工具不知道，错误码也不知道 ——
这个问题没有确定答案，`dag.py` 那套纯函数给不出来，只能由模型判断。

这与「有确定答案的问题不交给模型」不冲突，两者的分界是**问题有没有确定答案**：

    格式、类型、枚举、依赖、是否完成   有确定答案 → schema 与纯函数（已有）
    结果是否达成了任务描述说的目的     无确定答案 → 模型判定，本模块把关

模型的判定不是终局，本模块加两道确定性的闸：

  · **只收紧，不放宽。** 校验只能把「成功」改判为未通过，不能把失败改判为成功。
    模型幻觉出一句「其实成功了」不会让任何失败的任务变成完成。
  · **修正必须自证。** 模型给出的新参数要通过工具 schema 的校验、与原参数确有
    不同、且不触碰引用，才会被真正派发。改不出合法参数就退回重规划。

★ 本模块是纯函数：不调用模型、不 import langgraph、没有副作用。
"""

from __future__ import annotations

from typing import Any

from ..protocol import AgentError, ErrorCode
from ..tools.validation import normalize
from . import dataflow
from .state import Review, Task, TaskOutcome

__all__ = ["needs_review", "accept_correction", "correction_of",
           "outcome_code", "rejection_text"]


# 参数校验未过：调用还没出进程就被本地挡下，错在参数，正是可修正的一类。
_REVIEWABLE_AG = {ErrorCode.AG_TOOL_SCHEMA_INVALID.value}


def needs_review(outcome: TaskOutcome) -> bool:
    """这条执行结果值不值得送去语义校验。

    校验要多烧一轮端侧推理，因此只在校验能给出新信息时才做：

      · 成功 —— 工具说成功了，但成功的是不是任务要的东西无从确定，送校验
      · 工具执行层失败（段位 3）与参数校验失败 —— 错在这次调用本身，
        可能只是参数写偏了，送校验以求就地修正
      · 传输、协议、资源、权限类失败（段位 1/2/4/5）—— 原因已经确定，
        重试还是换方案由错误码段位直接推出，校验给不出新信息，不送

    最后一条是刻意的：失败多的轮次反而不额外烧推理。
    """
    if outcome["ok"]:
        return True
    code = outcome["error_code"]
    if code is None:
        return False
    if code in _REVIEWABLE_AG:
        return True
    return code.startswith("MC-") and code[3] == "3"


def correction_of(review: Review | None) -> dict[str, Any] | None:
    """取出这条判定里可派发的修正参数；判定通过或没有修正时为 None。"""
    if review is None or review["ok"]:
        return None
    return review["correction"]


def accept_correction(task: Task, corrected: Any,
                      schema: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    """决定模型给出的修正参数能不能直接拿去重试。返回 (参数, 不接受的原因)。

    三条闸，任一不过就退回 None，该任务按普通失败走重规划：

      1. **不碰引用。** 带 `$from` 的参数，值来自上游任务的执行结果；
         把它改写成字面量等于用模型的记忆替换真实结果。这类偏差的正确
         去处是重规划，不是就地改参数。
      2. **过工具 schema。** 与真实派发走同一个 `normalize`，因此修正过的
         参数不可能比模型原本写的那份更宽松。
      3. **确有不同。** 与原参数等价的「修正」重试一次只会得到同样的结果，
         白白消耗一次端侧调用。

    不另设修正次数上限：修正后的重试和普通重试走同一条路，
    同样受单任务重试上限与累计执行预算约束。
    """
    if not isinstance(corrected, dict) or not corrected:
        return None, "未给出修正参数"
    if dataflow.referenced_ids(task["arguments"]):
        return None, "参数含上游引用，不接受就地修正"
    if schema is None:
        return None, "工具未注册，无法校验修正参数"
    try:
        normalized = normalize(corrected, schema)
    except AgentError as e:
        return None, f"修正参数未通过 schema 校验：{e.message}"
    if normalized == task["arguments"]:
        return None, "修正参数与原参数相同"
    return normalized, ""


def rejection_text(outcome: TaskOutcome, review: Review | None) -> str:
    """任务落库时写进 result 的那段话，也是重规划与收尾看到的文本。

    成功却被判未通过时，模型下一步需要知道两件事：工具到底返回了什么、
    它为什么不算达成。只写其中一件都会让重规划无从下手。
    """
    if review is None or review["ok"] or not outcome["ok"]:
        return outcome["content"]
    return f"结果未达成任务目标：{review['reason']}（工具返回：{outcome['content']}）"


def outcome_code(outcome: TaskOutcome, review: Review | None) -> str | None:
    """这次执行最终记在账上的错误码。

    校验只收紧不放宽：原本就失败的保留自己的码，原本成功却未通过校验的
    记 AG-2005，通过校验的没有码。
    """
    if not outcome["ok"]:
        return outcome["error_code"]
    if review is not None and not review["ok"]:
        return ErrorCode.AG_RESULT_REJECTED.value
    return None
