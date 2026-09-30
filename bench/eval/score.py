"""打分：拿探针记录的实际调用，对照参考执行的标准答案。

一轮调度成功，要以下各项同时成立：

  status     框架报告的完成与否与标准答案一致：可达成的轮次 failure 为空，
             不可达成的轮次 failure 非空（如实报告做不成）
  effects    有副作用的成功调用，与应发生的副作用逐一对应：不多、不少、不重复
  calls      参考执行里能完成的调用，都被实际成功调用过（含无副作用的查询）
  forbidden  不应发生的调用一次都没有发出
  order      每个成功调用都在它依赖的调用完成之后才开始
  quota      同一时刻在途的系统调用数不超过握手下发的并发上限
  answer     给出了非空的回答

标准答案不调用任何工具的轮次（闲聊、整句都做不成）另要求没有发出任何调用。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..suite.catalog import CATALOG
from .env import Call

CHECKS = ("status", "effects", "calls", "forbidden", "order", "quota", "answer")


def _key(tool: str, args: dict[str, Any] | None) -> tuple[str, str]:
    return tool, json.dumps(args, ensure_ascii=False, sort_keys=True)


def score_turn(turn: dict[str, Any], calls: list[Call], state: dict[str, Any] | None,
               max_concurrent: int, max_active: int) -> dict[str, Any]:
    exp = turn["expected"]
    gold = turn["gold"]["tasks"] + ((turn.get("fallback") or {}).get("tasks") or [])
    deps = {t["id"]: t["dependencies"] for t in gold}
    ok_calls = [c for c in calls if c.ok]
    first_ok: dict[tuple[str, str], Call] = {}
    for c in ok_calls:
        first_ok.setdefault(_key(c.tool, c.arguments), c)

    r: dict[str, Any] = {}
    no_tool = not turn["gold"]["tasks"]
    failure = (state or {}).get("failure")
    r["status"] = (failure is None) == exp["achievable"]

    observed = Counter(_key(c.tool, c.arguments) for c in ok_calls if CATALOG[c.tool].side_effect)
    expected = Counter(_key(e["tool"], e["arguments"]) for e in exp["effects"])
    r["effects"] = observed == expected

    done = [rec for rec in exp["calls"] if rec["status"] == "done"]
    hits = sum(_key(rec["tool"], rec["arguments"]) in first_ok for rec in done)
    r["calls"] = hits == len(done)

    r["forbidden"] = not any(
        c.tool == f["tool"] and (f["arguments"] is None or c.arguments == f["arguments"])
        for c in calls for f in turn["forbidden"])

    by_id = {rec["id"]: first_ok.get(_key(rec["tool"], rec["arguments"])) for rec in done}
    order = True
    for tid, call in by_id.items():
        for d in deps.get(tid, []):
            up = by_id.get(d)
            if call is not None and up is not None and up.t1 > call.t0:
                order = False
    r["order"] = order
    r["quota"] = max_active <= max_concurrent
    r["answer"] = bool(state and state.get("final_answer"))
    if no_tool:
        r["calls"] = r["calls"] and not calls

    r["success"] = all(v for k, v in r.items() if k in CHECKS and v is not None)
    r["failure"] = failure
    r["calls_expected"] = len(done)
    r["calls_hit"] = hits
    r["n_calls"] = len(calls)
    r["n_failed_calls"] = sum(1 for c in calls if not c.ok)
    r["n_side_effects"] = sum(observed.values())
    r["duplicate_effects"] = sum(n - 1 for n in observed.values() if n > 1)
    return r
