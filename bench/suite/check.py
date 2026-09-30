"""数据集自检：逐条确认标注答案成立。

    python -m bench.suite.check                    # 检查 bench/data/cases.jsonl

检查项（任一不成立即报错并指出用例）：

  · 标准任务图能通过框架自己的计划 schema（工具名收进可见工具的枚举）与
    任务图结构校验（依赖存在、无自依赖、无环）
  · 参考执行里每个实际发出的调用，参数都已符合工具 schema（归一化不改变它）
  · 标注的参数值能从原话推出：字符串原值出现在这一轮的原话里，或是前几轮
    执行的结果（跨轮承接），或是引用；口语换算类参数（时间、数值、开关）除外
  · 引用只指向同一份计划里更早的任务；备选计划只引用主计划里的任务
  · 故障都会被触发：每条故障都对应参考执行里的某个调用
  · 重新做一遍参考执行，结果与文件里记录的一致
  · 结构类别与结构特征一致；首轮原话在全数据集中不重复
  · 不应发生的调用不在应发生的副作用里
  · 订餐厅引用的推荐结果，推荐的是能订座的类别
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from miagent.graph import dag
from miagent.graph.schema import task_plan_model_for
from miagent.graph.state import new_task
from miagent.tools.validation import normalize

from . import execute
from .catalog import CATALOG, DINING
from .compose import Graph, Node, Ref, classify, shape
from .phrases import Value

# 口语说法与标注值不同的参数：原话里出现的是换算前的说法
_CONVERTED = {"time", "level", "on", "seconds", "people"}


class CaseError(AssertionError):
    pass


def _visible(permissions: list[str]) -> list[str]:
    return [t for t, s in CATALOG.items()
            if s.tool.required_permission is None or s.tool.required_permission in permissions]


def _graph(tasks: list[dict[str, Any]]) -> Graph:
    g = Graph()
    for t in tasks:
        args = {p: Ref(v[execute.REF]) if isinstance(v, dict) and execute.REF in v else Value(v, "")
                for p, v in t["arguments"].items()}
        refs = {v.task for v in args.values() if isinstance(v, Ref)}
        g.nodes[t["id"]] = Node(t["id"], t["required_tool"], args,
                                [d for d in t["dependencies"] if d not in refs])
    return g


def check_turn(case: dict[str, Any], i: int, prior_results: list[Any]) -> None:
    turn = case["turns"][i]
    where = f"{case['id']} 第 {i + 1} 轮"
    gold = turn["gold"]["tasks"]
    fallback = (turn["fallback"] or {}).get("tasks") or []

    # 框架自己的 schema 与结构校验
    visible = _visible(case["permissions"])
    strip = lambda t: {k: v for k, v in t.items() if k != "replaces"}  # noqa: E731
    # 备选计划可以依赖主计划里的任务，与重规划可以依赖已完成任务同理
    for plan, known in ((gold, []), (fallback, [t["id"] for t in gold])):
        if plan:
            task_plan_model_for(visible, known).model_validate({"tasks": [strip(t) for t in plan]})
    tasks = {t["id"]: new_task(t["id"], t["description"], t["required_tool"], t["arguments"],
                               t["dependencies"]) for t in gold}
    dag.validate(tasks)

    # 引用只指向更早的任务
    seen: list[str] = []
    for t in gold:
        for d in t["dependencies"]:
            if d not in seen:
                raise CaseError(f"{where}：{t['id']} 依赖了 {d}，它不在更早的任务里")
        seen.append(t["id"])
    gold_ids = set(seen)
    for t in fallback:
        for d in t["dependencies"]:
            if d not in seen:
                raise CaseError(f"{where}：备选任务 {t['id']} 依赖了未知任务 {d}")
        if t["replaces"] not in gold_ids:
            raise CaseError(f"{where}：备选任务 {t['id']} 接替的 {t['replaces']} 不存在")
        seen.append(t["id"])

    # 参考执行可重现
    again = execute.outcome(gold, fallback or None, faults=turn["faults"],
                            budget=case["budget"], permissions=case["permissions"])
    if again["calls"] != turn["expected"]["calls"] or again["effects"] != turn["expected"]["effects"]:
        raise CaseError(f"{where}：参考执行结果与记录不一致")
    if again["achievable"] is False and turn["expected"]["achievable"]:
        raise CaseError(f"{where}：参考执行不可达成，记录却是可达成")

    # 实际发出的调用参数已符合 schema
    for rec in again["calls"]:
        if rec["arguments"] is None:
            continue
        schema = CATALOG[rec["tool"]].tool.input_schema
        if normalize(rec["arguments"], schema) != rec["arguments"]:
            raise CaseError(f"{where}：{rec['id']} 的参数不是 schema 的规范形态")

    # 标注值能从原话推出
    for t in gold + fallback:
        for p, v in t["arguments"].items():
            if isinstance(v, dict) or p in _CONVERTED:
                continue
            if str(v) not in turn["request"] and v not in prior_results:
                raise CaseError(f"{where}：{t['id']}.{p}={v!r} 在原话与上文里都找不到")

    # 故障都会被触发
    calls = [(r["tool"], r["arguments"]) for r in again["calls"]]
    for f in turn["faults"]:
        if (f["tool"], f["arguments"]) not in calls:
            raise CaseError(f"{where}：故障 {f['tool']} 不会被触发")

    by_id = {t["id"]: t for t in gold + fallback}
    for t in gold + fallback:
        ref = t["arguments"].get("name") if t["required_tool"] == "system.book_restaurant" else None
        up = by_id.get(ref[execute.REF]) if isinstance(ref, dict) else None
        if up and up["required_tool"] == "system.recommend_place" \
                and up["arguments"]["category"] not in DINING:
            raise CaseError(f"{where}：{t['id']} 要订座的是一家{up['arguments']['category']}")

    for f in turn["forbidden"]:
        for e in turn["expected"]["effects"]:
            if e["tool"] == f["tool"] and f["arguments"] in (None, e["arguments"]):
                raise CaseError(f"{where}：{f['tool']} 既被禁止又是应发生的副作用")


def check_case(case: dict[str, Any]) -> None:
    prior: list[Any] = []
    for i, turn in enumerate(case["turns"]):
        check_turn(case, i, prior)
        prior += [r["result"] for r in turn["expected"]["calls"] if r["result"] is not None]
    first = _graph(case["turns"][0]["gold"]["tasks"])
    if shape(first) != case["shape"]:
        raise CaseError(f"{case['id']}：结构特征与记录不一致")
    if case["category"] in ("single", "parallel", "chain", "fan_in", "fan_out", "diamond", "mixed") \
            and classify(first) != case["category"]:
        raise CaseError(f"{case['id']}：结构判定为 {classify(first)}，记录为 {case['category']}")


def check(cases: list[dict[str, Any]]) -> dict[str, int]:
    firsts: set[str] = set()
    for case in cases:
        check_case(case)
        request = case["turns"][0]["request"]
        if request in firsts:
            raise CaseError(f"{case['id']}：首轮原话重复：{request}")
        firsts.add(request)
    return {"cases": len(cases), "turns": sum(len(c["turns"]) for c in cases)}


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集自检")
    ap.add_argument("path", nargs="?", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "cases.jsonl")
    args = ap.parse_args()
    print(json.dumps(check(load(args.path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
