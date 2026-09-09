"""任务间的数据流：把上游任务的结果作为下游任务的参数。

没有这一层，任务图里的多个工具只是多次互不相干的调用 —— 模型必须在规划时
就把所有参数写死，而「先查电量、再按电量决定提醒内容」这类组合根本无法表达。

引用写在 arguments 里，形式是一个只含 `$from` 的对象：

    {"tool": "system.set_alarm",
     "arguments": {"time": {"$from": "t1"}}}

★ 本模块是纯函数：不依赖 LangGraph、不调用模型、没有副作用。
  「这个参数引用了谁」「引用该取什么值」都有确定答案，与依赖解析同属
  调度的正确性核心，因此和 dag.py 一样必须能被穷举测试。

两条设计约定：

  · **引用即依赖。** 引用了 t1 就自动产生对 t1 的依赖边，无需模型另外在
    dependencies 里写一遍。模型很容易写了引用却漏填依赖，那会让下游任务在
    上游还没跑完时就被派发，取到空结果 —— 而且不报错。与其校验两处是否一致，
    不如让依赖从引用派生，使这种不一致在结构上不可能出现。

  · **只支持引用整个结果，不提供取字段的路径语法。** 需要取字段时用本地工具
    （如 json_field）串一步。多一种语法就多一处要解析、要校验、要向模型解释
    的东西，而这件事已有工具能做。
"""

from __future__ import annotations

from typing import Any

from ..protocol import AgentError, ErrorCode
from .state import Task, TaskStatus

# 引用对象的唯一键。取 `$` 前缀是沿用 JSON Schema 的惯例，
# 与工具的普通参数名不会撞。
REFERENCE_KEY = "$from"


def as_reference(value: Any) -> str | None:
    """若 value 是一个引用对象，返回被引用的任务 id，否则返回 None。

    只认「有且仅有 $from 一个键、值为非空字符串」的对象。要求严格是刻意的：
    多出别的键说明模型想表达的不止是引用（也许是取字段），这时应当报错让它重来，
    而不是猜它的意思后放行 —— 猜错会产生一个形式合法、语义错误的调用。
    """
    if (isinstance(value, dict) and set(value) == {REFERENCE_KEY}
            and isinstance(value[REFERENCE_KEY], str) and value[REFERENCE_KEY]):
        return value[REFERENCE_KEY]
    return None


def referenced_ids(arguments: Any) -> set[str]:
    """找出 arguments 里引用到的全部任务 id（递归，含嵌套的对象与数组）。"""
    ref = as_reference(arguments)
    if ref is not None:
        return {ref}
    if isinstance(arguments, dict):
        return set().union(*(referenced_ids(v) for v in arguments.values())) if arguments else set()
    if isinstance(arguments, list):
        return set().union(*(referenced_ids(v) for v in arguments)) if arguments else set()
    return set()


def remap_references(arguments: Any, mapping: dict[str, str]) -> Any:
    """按 mapping 重写引用中的任务 id，其余原样返回。

    重规划时新任务的 id 会加前缀，引用必须同步改写，否则会指向不存在的任务。
    """
    ref = as_reference(arguments)
    if ref is not None:
        return {REFERENCE_KEY: mapping.get(ref, ref)}
    if isinstance(arguments, dict):
        return {k: remap_references(v, mapping) for k, v in arguments.items()}
    if isinstance(arguments, list):
        return [remap_references(v, mapping) for v in arguments]
    return arguments


def resolve(arguments: Any, tasks: dict[str, Task]) -> Any:
    """把引用替换成被引用任务的结果。

    在派发之前做，因此执行节点拿到的是一份参数已经落实的任务，
    不需要知道数据流的存在。

    被引用的任务必须已经完成。正常调度下这一条恒成立 —— 引用即依赖，
    而 Scheduler 只派发依赖全部 done 的任务。这里仍然检查，是因为
    静默取到 None 比报错难查得多。
    """
    ref = as_reference(arguments)
    if ref is not None:
        task = tasks.get(ref)
        if task is None:
            raise AgentError(
                ErrorCode.AG_INVALID_PLAN,
                f"参数引用了不存在的任务 {ref}",
                detail={"missing": ref})
        if task["status"] is not TaskStatus.DONE:
            raise AgentError(
                ErrorCode.AG_DEPENDENCY_UNRESOLVED,
                f"参数引用的任务 {ref} 尚未完成（当前 {task['status']}）",
                detail={"reference": ref, "status": str(task["status"])})
        return task["result"]
    if isinstance(arguments, dict):
        return {k: resolve(v, tasks) for k, v in arguments.items()}
    if isinstance(arguments, list):
        return [resolve(v, tasks) for v in arguments]
    return arguments
