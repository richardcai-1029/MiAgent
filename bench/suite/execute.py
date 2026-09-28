"""参考执行：不经过框架，按标注的任务图直接算出每个调用的结果。

标注答案里「下游参数该收到什么值」「哪些副作用该发生」由这里算出。它只用
工具目录里的纯函数与用例声明的故障，不依赖 miagent 的调度实现 —— 评测时
拿框架的实际执行去对照它，二者不能出自同一份代码。

故障只描述「这个调用会怎样失败」，不涉及框架怎么应对（重试几次、何时重规划）：

    transient_first  第一次调用超时（MC-1003），再调即成功
    transient_all    每次调用都超时（MC-1003）
    busy             资源被占用（MC-4002），如餐厅订满、附近无车
    （内在的）       工具内存开销超出配额（MC-4001）、权限未授予（工具不可见）、
                     业务上查无结果（isError）—— 由配额、权限与参数本身决定，不必声明

执行按依赖先后进行；上游失败的任务不执行。备选计划（fallback）在主计划之后
执行，可以引用主计划里已完成的任务。
"""

from __future__ import annotations

from typing import Any

from .catalog import CATALOG

REF = "$from"
FAULT_CODES = {"transient_first": "MC-1003", "transient_all": "MC-1003", "busy": "MC-4002"}


def _resolve(value: Any, results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, dict) and set(value) == {REF}:
        return results[value[REF]]["result"]
    return value


def fault_for(faults: list[dict[str, Any]], tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """按 (工具, 实际参数) 找这次调用对应的故障。"""
    for f in faults:
        if f["tool"] == tool and f["arguments"] == arguments:
            return f
    return None


def run(tasks: list[dict[str, Any]], *, faults: list[dict[str, Any]], budget: dict[str, Any],
        permissions: list[str], done: dict[str, dict[str, Any]] | None = None
        ) -> dict[str, dict[str, Any]]:
    """执行一份计划，返回 {任务 id: 记录}。done 是已有的结果，可被引用。

    记录的 status：done / failed（调用了但失败）/ blocked（上游失败，未调用）。
    """
    results: dict[str, dict[str, Any]] = dict(done or {})
    pending = list(tasks)
    while pending:
        ready = [t for t in pending if all(d in results for d in t["dependencies"])]
        if not ready:
            raise ValueError(f"计划无法执行完：{[t['id'] for t in pending]}")
        for task in ready:
            pending.remove(task)
            results[task["id"]] = _call(task, results, faults, budget, permissions)
    return {t["id"]: results[t["id"]] for t in tasks}


def _call(task: dict[str, Any], results: dict[str, dict[str, Any]],
          faults: list[dict[str, Any]], budget: dict[str, Any],
          permissions: list[str]) -> dict[str, Any]:
    tool = task["required_tool"]
    base = {"id": task["id"], "tool": tool}
    if any(results[d]["status"] != "done" for d in task["dependencies"]):
        return {**base, "status": "blocked", "arguments": None, "result": None, "error": None}

    spec = CATALOG[tool]
    args = {k: _resolve(v, results) for k, v in task["arguments"].items()}
    base["arguments"] = args
    perm = spec.tool.required_permission
    if perm is not None and perm not in permissions:
        return {**base, "status": "failed", "result": None, "error": "MC-5001"}
    if spec.tool.estimated_memory_mb > budget["max_memory_mb"]:
        return {**base, "status": "failed", "result": None, "error": "MC-4001"}
    fault = fault_for(faults, tool, args)
    if fault is not None and fault["kind"] != "transient_first":
        return {**base, "status": "failed", "result": None, "error": FAULT_CODES[fault["kind"]]}
    output = spec.run(args)
    if output is None:
        return {**base, "status": "failed", "result": None, "error": "empty"}
    return {**base, "status": "done", "result": output, "error": None,
            "flaky": fault is not None}


def outcome(gold: list[dict[str, Any]], fallback: list[dict[str, Any]] | None, *,
            faults: list[dict[str, Any]], budget: dict[str, Any],
            permissions: list[str]) -> dict[str, Any]:
    """一轮的参考结果：各调用的记录、应发生的副作用、目标能否达成。

    达成的条件：主计划全部完成；或者失败的部分由备选计划接手且备选计划全部完成。
    备选计划覆盖的是失败任务及其下游，因此主计划里与之无关的任务仍须完成。
    """
    main = run(gold, faults=faults, budget=budget, permissions=permissions)
    records = list(main.values())
    achieved = all(r["status"] == "done" for r in records)
    if not achieved and fallback:
        done = {k: v for k, v in main.items() if v["status"] == "done"}
        extra = run(fallback, faults=faults, budget=budget, permissions=permissions, done=done)
        records += list(extra.values())
        replaced = {t["replaces"] for t in fallback if t.get("replaces")}
        failed = {k for k, v in main.items() if v["status"] != "done"}
        achieved = (failed <= replaced
                    and all(r["status"] == "done" for r in extra.values()))
    effects = [{"tool": r["tool"], "arguments": r["arguments"]} for r in records
               if r["status"] == "done" and CATALOG[r["tool"]].side_effect]
    return {"calls": records, "effects": effects, "achievable": achieved}
