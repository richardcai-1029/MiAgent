"""规划打分：模型给出的任务图对照标准任务图。

先配对，再逐项比：

  配对    模型任务与标准任务按工具配对，同一工具出现多次时取参数吻合最多的配法
  工具    配上的是命中；模型多出来的是误报，标准里没配上的是漏报
  参数    标准任务的每个参数：字面值要与标准值（或 accept 列出的写法）一致，
          比较前去掉首尾空白、引号与内部空格；引用要指向与被引标准任务配对的
          那个模型任务
  依赖    依赖边经配对映射后与标准依赖边比较（含由引用派生的边）
  调用    一个标准任务算「调用正确」：配上了、且参数全部正确
  整图    工具一一对应、参数全对、依赖边相同、没有多余任务

模型多填了标准里没有的可选参数（如自作主张补上出行方式）单独计数，不计入参数准确率。
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from ..suite.execute import REF


def norm(v: Any) -> str:
    s = str(v).strip().strip("「」\"'“”").replace(" ", "")
    return s.lower()


def _is_ref(v: Any) -> bool:
    return isinstance(v, dict) and set(v) == {REF}


def _literal_hits(model: dict[str, Any], gold: dict[str, Any], accept: dict[str, list[Any]]) -> int:
    return sum(1 for p, g in gold["arguments"].items()
               if not _is_ref(g) and p in model["arguments"]
               and norm(model["arguments"][p]) in {norm(x) for x in accept.get(p, [g])})


def match(model: list[dict[str, Any]], gold: list[dict[str, Any]],
          accept: dict[str, dict[str, list[Any]]]) -> dict[str, str]:
    """标准任务 id → 模型任务 id。"""
    pairs: dict[str, str] = {}
    for tool in {g["required_tool"] for g in gold}:
        gs = [g for g in gold if g["required_tool"] == tool]
        ms = [m for m in model if m["required_tool"] == tool]
        if not ms:
            continue
        k = min(len(gs), len(ms))
        best, best_score = None, -1
        if len(gs) <= 5 and len(ms) <= 5:
            for gp in permutations(gs, k):
                for mp in permutations(ms, k):
                    s = sum(_literal_hits(m, g, accept.get(g["id"], {})) for g, m in zip(gp, mp))
                    if s > best_score:
                        best, best_score = list(zip(gp, mp)), s
        else:                                   # 规模超出时按顺序配对
            best = list(zip(gs, ms))
        for g, m in best or []:
            pairs[g["id"]] = m["id"]
    return pairs


def score_plan(model: list[dict[str, Any]], gold: list[dict[str, Any]],
               accept: dict[str, dict[str, list[Any]]],
               forbidden: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = match(model, gold, accept)
    by_model = {m["id"]: m for m in model}
    back = {mid: gid for gid, mid in pairs.items()}

    args_total = args_ok = calls_ok = 0
    extra_args = 0
    # 参数错误的分类：缺参数、字面值错、该引用却写了字面值、引用指错、不该引用却引用
    arg_errors = {"missing": 0, "literal_wrong": 0, "ref_as_literal": 0,
                  "ref_wrong": 0, "literal_as_ref": 0}
    for g in gold:
        mid = pairs.get(g["id"])
        m = by_model.get(mid) if mid else None
        ok_all = m is not None
        for p, gv in g["arguments"].items():
            args_total += 1
            if m is None:
                ok_all = False
                continue
            if p not in m["arguments"]:
                arg_errors["missing"] += 1
                ok_all = False
                continue
            mv = m["arguments"][p]
            if _is_ref(gv):
                good = _is_ref(mv) and back.get(mv[REF]) == gv[REF]
                if not good:
                    arg_errors["ref_wrong" if _is_ref(mv) else "ref_as_literal"] += 1
            else:
                good = not _is_ref(mv) and norm(mv) in {norm(x) for x in accept.get(g["id"], {}).get(p, [gv])}
                if not good:
                    arg_errors["literal_as_ref" if _is_ref(mv) else "literal_wrong"] += 1
            args_ok += good
            ok_all &= good
        if m is not None:
            extra_args += len(set(m["arguments"]) - set(g["arguments"]))
        calls_ok += ok_all

    # 依赖边分两种：由引用派生的数据边，与用户明说先后的顺序边
    data_edges = {(v[REF], g["id"]) for g in gold for v in g["arguments"].values() if _is_ref(v)}
    gold_edges = {(d, g["id"]) for g in gold for d in g["dependencies"]}
    model_edges = {(back.get(d, f"?{d}"), back.get(m["id"], f"?{m['id']}"))
                   for m in model for d in m["dependencies"]}
    edges_hit = len(gold_edges & model_edges)
    order_edges = gold_edges - data_edges

    forbidden_hits = sum(
        1 for m in model for f in forbidden
        if m["required_tool"] == f["tool"]
        and (f["arguments"] is None or m["arguments"] == f["arguments"]))

    tools_exact = sorted(m["required_tool"] for m in model) == sorted(g["required_tool"] for g in gold)
    exact = (tools_exact and len(pairs) == len(gold) == len(model)
             and calls_ok == len(gold) and model_edges == gold_edges)
    return {
        "n_gold": len(gold), "n_model": len(model), "tp": len(pairs),
        "args_total": args_total, "args_ok": args_ok, "calls_ok": calls_ok,
        "extra_args": extra_args,
        "edges_gold": len(gold_edges), "edges_model": len(model_edges), "edges_hit": edges_hit,
        "order_edges_gold": len(order_edges), "order_edges_hit": len(order_edges & model_edges),
        "arg_errors": arg_errors,
        "tools_exact": tools_exact, "exact": exact, "forbidden_hits": forbidden_hits,
    }
