"""汇总各项评测结果，输出指标（metrics.json）与报告用的表格（tables.md）。

    python -m bench.eval.report

读 bench/results/ 下这些目录（缺哪个就跳过哪一节）：

    framework-full      框架侧全量
    framework-noverify  框架侧全量，关掉语义校验（消融）
    memory              框架内存专项
    concurrency-<N>     多请求并发，同时在飞 N 个
    model-plan          模型侧规划准确率
    model-e2e           模型侧全链路
    model-speed         推理速度回放

比例一律附 Wilson 95% 置信区间。模型侧按类别分层抽样，每类条数相同；
「按数据集加权」一列把各类别的结果按类别在全量数据集中的占比加权，
反映的是在整份数据集分布下的期望值。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

RESULTS = Path(__file__).resolve().parents[1] / "results"
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "manifest.json"
CHECKS = ("status", "effects", "calls", "forbidden", "order", "quota", "answer")
CATEGORY_ORDER = ("single", "parallel", "chain", "fan_in", "fan_out", "diamond", "mixed",
                  "multi_turn", "no_tool", "unsupported", "permission_denied")
CATEGORY_CN = {"single": "单任务", "parallel": "并行", "chain": "数据流链", "fan_in": "扇入",
               "fan_out": "扇出", "diamond": "菱形", "mixed": "混合", "multi_turn": "多轮",
               "no_tool": "无需工具", "unsupported": "能力不支持", "permission_denied": "权限被拒"}


# ---------------- 统计 ----------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(k: int, n: int, ci: bool = True) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "—"
    return f"{p * 100:.2f}%" + (f"（{lo * 100:.2f}~{hi * 100:.2f}）" if ci else "")


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else math.nan


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def groups(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Iterable[str]]
           ) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for k in key(r):
            out[k].append(r)
    return out


def table(head: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(head) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------- 各节 ----------------


def framework(md: list[str], m: dict[str, Any]) -> None:
    rows = load_rows(RESULTS / "framework-full" / "turns.jsonl")
    if not rows:
        return
    summ = json.loads((RESULTS / "framework-full" / "summary.json").read_text())
    n, k = len(rows), sum(r["success"] for r in rows)
    m["framework"] = {"turns": n, "cases": summ["cases"], "success": k, "rate": wilson(k, n)}
    md.append("### 调度成功率（框架侧全量）\n")
    md.append(f"全量 {summ['cases']:,} 条用例、{n:,} 轮：成功 {k:,} 轮，**{pct(k, n)}**。\n")
    by = groups(rows, lambda r: [r["category"]])
    md.append(table(["类别", "轮数", "成功", "成功率（95% CI）"],
                    [[CATEGORY_CN[c], len(by[c]), sum(r["success"] for r in by[c]),
                      pct(sum(r["success"] for r in by[c]), len(by[c]))]
                     for c in CATEGORY_ORDER if c in by]))
    md.append("")
    tags = groups(rows, lambda r: [t for t in r["tags"] if t.startswith(("fault:", "noise:", "sim:"))])
    md.append(table(["叠加项", "轮数", "成功率", "自修复次数", "重规划次数", "失败的工具调用"],
                    [[t, len(v), pct(sum(r["success"] for r in v), len(v), ci=False),
                      sum(r["repairs"] for r in v), sum(r["replans"] for r in v),
                      sum(r["n_failed_calls"] for r in v)] for t, v in sorted(tags.items())]))
    md.append("")
    lv = groups(rows, lambda r: [str(r["level"])])
    md.append(table(["难度等级", "轮数", "成功率"],
                    [[k2, len(v), pct(sum(r["success"] for r in v), len(v))] for k2, v in sorted(lv.items())]))
    md.append("")
    fails = {c: sum(1 for r in rows if r[c] is False) for c in CHECKS}
    m["framework"]["check_failures"] = fails
    exp = sum(r.get("calls_expected", 0) for r in rows)
    hit = sum(r.get("calls_hit", 0) for r in rows)
    m["framework"]["calls"] = {"expected": exp, "hit": hit}
    dups = sum(r["duplicate_effects"] for r in rows)
    effects = sum(r["n_side_effects"] for r in rows)
    md.append(f"逐项检查未通过的轮数：{'、'.join(f'{c} {v}' for c, v in fails.items())}。"
              f"副作用调用共 {effects:,} 次，重复发生 {dups} 次。\n")
    if exp:
        md.append(f"调用执行正确率（参考执行中能完成的调用，被以完全相同的参数成功调用）："
                  f"{hit:,} / {exp:,}，**{pct(hit, exp)}**。\n")

    # 框架自身开销：去掉模型与工具耗时后剩下的部分。只取一次走通的轮次 ——
    # 重试含 200 ms 退避等待，那是框架的策略时长而不是计算开销
    clean = [r for r in rows if not any(t.startswith(("fault:", "noise:", "sim:")) for t in r["tags"])
             and r["n_failed_calls"] == 0 and r["replans"] == 0 and r["repairs"] == 0]
    over = [r["wall_ms"] - r["llm_ms"] - r["tool_ms"] for r in clean]
    by_exec = groups(clean, lambda r: [str(min(r["executions"], 7))])
    m["framework"]["overhead_ms"] = {"p50": q(over, .5), "p95": q(over, .95), "p99": q(over, .99)}
    md.append("### 框架自身开销\n")
    md.append(f"取一次走通（无故障、无重试、无重规划、无自修复）的 {len(clean):,} 轮，"
              f"单轮耗时扣除模型与工具耗时：P50 "
              f"{q(over, .5):.2f} ms，P95 {q(over, .95):.2f} ms，P99 {q(over, .99):.2f} ms。\n")
    md.append(table(["工具调用次数", "轮数", "P50 (ms)", "P95 (ms)", "P99 (ms)"],
                    [[k2, len(v), f"{q(o := [r['wall_ms'] - r['llm_ms'] - r['tool_ms'] for r in v], .5):.2f}",
                      f"{q(o, .95):.2f}", f"{q(o, .99):.2f}"]
                     for k2, v in sorted(by_exec.items(), key=lambda x: int(x[0]))]))
    md.append("")
    res = summ["resources"]
    md.append(f"全量运行 CPU 时间 {res['cpu_seconds']:.1f} s，折合每轮 "
              f"{res['cpu_seconds'] * 1000 / n:.2f} ms；每轮平均模型调用 "
              f"{sum(r['llm_calls'] for r in rows) / n:.2f} 次。\n")
    m["framework"]["cpu_ms_per_turn"] = res["cpu_seconds"] * 1000 / n

    ab = load_rows(RESULTS / "framework-noverify" / "turns.jsonl")
    if ab:
        ka = sum(r["success"] for r in ab)
        m["ablation"] = {"turns": len(ab), "success": ka}
        md.append("### 消融：关掉语义校验\n")
        md.append(f"同一份数据关掉语义校验（`verify=False`）：成功 {ka:,} / {len(ab):,}，"
                  f"{pct(ka, len(ab))}。按叠加项与类别列出成功率下降之处：\n")
        def key(r: dict[str, Any]) -> list[str]:
            ks = [t for t in r["tags"] if t.startswith(("fault:", "sim:"))]
            if r["category"] in ("permission_denied", "unsupported"):
                ks.append(r["category"])
            return ks

        at, ft = groups(ab, key), groups(rows, key)
        md.append(table(["分组", "轮数", "开校验", "关校验"],
                        [[t, len(v), pct(sum(r["success"] for r in ft[t]), len(ft[t]), ci=False),
                          pct(sum(r["success"] for r in v), len(v), ci=False)]
                         for t, v in sorted(at.items())]))
        md.append("")


def memory(md: list[str], m: dict[str, Any]) -> None:
    p = RESULTS / "memory" / "summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    m["memory"] = {k: v for k, v in s.items() if k != "series"}
    rp = s["request_peak_heap_mb"]
    md.append("### 框架内存\n")
    md.append(table(["项", "值"], [
        ["导入框架后常驻（RSS）", f"{s['rss_after_import_mb']:.1f} MB"],
        ["预热 100 条后常驻（RSS）", f"{s['rss_after_warmup_mb']:.1f} MB"],
        ["单请求 Python 堆峰值增量 P50 / P95 / 最大",
         f"{rp['p50'] * 1024:.0f} / {rp['p95'] * 1024:.0f} / {rp['max'] * 1024:.0f} KB"],
        [f"连续 {s['cases']:,} 条后 Python 堆的增长斜率", f"{s['heap_slope_mb_per_1k'] * 1024:.1f} KB / 千条"],
        ["存活对象数的增长斜率", f"{s['objects_slope_per_1k']:.0f} 个 / 千条"],
        ["每个空闲会话的 Python 堆占用",
         f"{s['idle_session']['heap_per_session_kb']:.1f} KB（{s['idle_session']['sessions']} 个会话、"
         f"{s['idle_session']['turns']} 轮）"],
    ], "lr"))
    md.append("")


def concurrency(md: list[str], m: dict[str, Any]) -> None:
    dirs = sorted(RESULTS.glob("concurrency-*"), key=lambda d: int(d.name.split("-")[1]))
    dirs = [d for d in dirs if (d / "summary.json").exists()]
    if not dirs:
        return
    out = []
    m["concurrency"] = []
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        rows = load_rows(d / "requests.jsonl")
        k = sum(r["success"] for r in rows)
        fg = [r["wait_ms"] for r in rows if r["priority"] == "foreground"]
        bg = [r["wait_ms"] for r in rows if r["priority"] == "background"]
        lat = [r["latency_ms"] for r in rows]
        m["concurrency"].append({**{x: s[x] for x in ("inflight", "cases", "max_active_calls", "capacity",
                                                       "makespan_s", "throughput_rps")},
                                 "success": k, "rss_peak_mb": s["resources"]["rss_peak_mb"]})
        out.append([s["inflight"], f"{k}/{len(rows)}", f"{s['max_active_calls']} / {s['capacity']}",
                    f"{s['throughput_rps']:.2f}", f"{q(lat, .5) / 1000:.2f}", f"{q(lat, .95) / 1000:.2f}",
                    f"{q(fg, .5) / 1000:.2f}", f"{q(bg, .5) / 1000:.2f}",
                    f"{s['resources']['rss_peak_mb']:.1f}", s["resources"]["threads_peak"]])
    s0 = json.loads((dirs[0] / "summary.json").read_text())
    md.append("### 多请求并发\n")
    md.append(f"每组 {s0['cases']:,} 个请求一次性提交，共用一个 MiClaw 会话（并发配额 {s0['capacity']}）"
              f"与一个推理槽；每次系统调用固定等待 {s0['tool_latency_ms']:.0f} ms（评测参数，用来让并行可观察，"
              f"不代表真实系统服务的时延）。\n")
    md.append(table(["同时在飞", "成功", "在途调用峰值 / 配额", "吞吐 (请求/s)", "完成时延 P50 (s)",
                     "完成时延 P95 (s)", "前台排队 P50 (s)", "后台排队 P50 (s)", "RSS 峰值 (MB)", "线程峰值"], out))
    md.append("")


ERROR_CN = {"plan_failed": "未产出可执行计划", "spurious_action": "无需工具却调用了工具",
            "substitute": "做不到的部分用无关工具顶替", "tool_missing": "漏了应调用的工具",
            "tool_extra": "多调了工具", "argument": "参数错误", "dependency": "依赖关系错误",
            "correct": "完全正确"}


def classify_plan_error(r: dict[str, Any], infeasible: bool) -> str:
    """一轮规划的主要错误，按列出的先后取第一个成立的。"""
    if r["plan_failure"]:
        return "plan_failed"
    if r["n_gold"] == 0 and r["n_model"] > 0:
        return "spurious_action"
    if infeasible and r["n_model"] > r["tp"]:
        return "substitute"
    if r["tp"] < r["n_gold"]:
        return "tool_missing"
    if r["n_model"] > r["tp"]:
        return "tool_extra"
    if r["calls_ok"] < r["n_gold"]:
        return "argument"
    if not r["exact"]:
        return "dependency"
    return "correct"


def _rescore(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按模型的原始输出与当前的打分规则重新打分，并标上错误类别。"""
    from ..suite.check import load
    from .plan_score import score_plan

    data = Path(__file__).resolve().parents[1] / "data" / "cases.jsonl"
    cases = {c["id"]: c for c in load(data)}
    out = []
    for r in rows:
        case = cases[r["case"]]
        turn = case["turns"][r["turn"]]
        s = score_plan(r["model_tasks"], turn["gold"]["tasks"], turn.get("accept") or {},
                       turn["forbidden"])
        row = {**r, **s}
        infeasible = case["category"] in ("unsupported", "permission_denied")
        row["error_class"] = classify_plan_error(row, infeasible)
        out.append(row)
    return out


def model_plan(md: list[str], m: dict[str, Any]) -> None:
    rows = load_rows(RESULTS / "model-plan" / "turns.jsonl")
    if not rows:
        return
    rows = _rescore(rows)
    summ = json.loads((RESULTS / "model-plan" / "summary.json").read_text())
    share = json.loads(MANIFEST.read_text())["by_category"] if MANIFEST.exists() else {}
    total = sum(share.values()) or 1

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        tp = sum(r["tp"] for r in rs); ng = sum(r["n_gold"] for r in rs); nm = sum(r["n_model"] for r in rs)
        eh = sum(r["edges_hit"] for r in rs); eg = sum(r["edges_gold"] for r in rs); em = sum(r["edges_model"] for r in rs)
        p, rc = (tp / nm if nm else 1.0), (tp / ng if ng else 1.0)
        ep, er = (eh / em if em else 1.0), (eh / eg if eg else 1.0)
        return {"turns": len(rs), "tool_p": p, "tool_r": rc, "tool_f1": f1(p, rc),
                "calls_ok": sum(r["calls_ok"] for r in rs), "n_gold": ng,
                "args_ok": sum(r["args_ok"] for r in rs), "args_total": sum(r["args_total"] for r in rs),
                "edge_p": ep, "edge_r": er, "edge_f1": f1(ep, er),
                "exact": sum(r["exact"] for r in rs),
                "failed": sum(1 for r in rs if r["plan_failure"]),
                "repaired": sum(1 for r in rs if r["repairs"]),
                "extra_args": sum(r["extra_args"] for r in rs),
                "forbidden_hits": sum(r["forbidden_hits"] for r in rs)}

    allm = agg(rows)
    by = groups(rows, lambda r: [r["category"]])
    cats = {c: agg(v) for c, v in by.items()}
    weighted = {key: sum(cats[c][key] * share.get(c, 0) for c in cats) / total
                for key in ("tool_f1", "edge_f1")}
    weighted["exact"] = sum(cats[c]["exact"] / cats[c]["turns"] * share.get(c, 0) for c in cats) / total
    weighted["calls"] = sum((cats[c]["calls_ok"] / cats[c]["n_gold"] if cats[c]["n_gold"] else 1.0)
                            * share.get(c, 0) for c in cats if cats[c]["n_gold"]) / \
        (sum(share.get(c, 0) for c in cats if cats[c]["n_gold"]) or 1)
    m["model_plan"] = {"all": allm, "by_category": cats, "weighted": weighted, "cases": summ.get("cases")}

    codes: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["plan_failure"]:
            codes[r["plan_failure"]] += 1
    md.append("### 工具调用准确率（模型侧，规划节点）\n")
    md.append(f"按类别分层抽样 {summ.get('cases')} 条用例（每类 {summ.get('per_category')} 条）、"
              f"{len(rows):,} 轮。\n")
    md.append(table(["指标", "抽样整体", "按数据集加权"], [
        ["工具选择 精确率 / 召回率 / F1",
         f"{allm['tool_p'] * 100:.2f}% / {allm['tool_r'] * 100:.2f}% / {allm['tool_f1'] * 100:.2f}%",
         f"F1 {weighted['tool_f1'] * 100:.2f}%"],
        ["调用完全正确率（工具与参数全对）", pct(allm["calls_ok"], allm["n_gold"]), f"{weighted['calls'] * 100:.2f}%"],
        ["参数准确率", pct(allm["args_ok"], allm["args_total"]), "—"],
        ["依赖边 精确率 / 召回率 / F1",
         f"{allm['edge_p'] * 100:.2f}% / {allm['edge_r'] * 100:.2f}% / {allm['edge_f1'] * 100:.2f}%",
         f"F1 {weighted['edge_f1'] * 100:.2f}%"],
        ["整图完全匹配率", pct(allm["exact"], allm["turns"]), f"{weighted['exact'] * 100:.2f}%"],
        ["规划失败（未产出可执行计划）", pct(allm["failed"], allm["turns"]), "—"],
        ["经自修复的轮次", pct(allm["repaired"], allm["turns"]), "—"],
    ], "lrr"))
    md.append("")
    if codes:
        md.append("规划失败的错误码：" + "、".join(f"{c} {v} 轮" for c, v in sorted(codes.items())) + "。\n")
    md.append(f"标准里没有、模型自行补上的参数 {allm['extra_args']} 个；"
              f"多轮里重做上一轮已做过的调用 {allm['forbidden_hits']} 次。\n")
    md.append(table(["类别", "轮数", "工具 F1", "调用完全正确率", "依赖边 F1", "整图匹配率", "规划失败"],
                    [[CATEGORY_CN[c], cats[c]["turns"], f"{cats[c]['tool_f1'] * 100:.1f}%",
                      pct(cats[c]["calls_ok"], cats[c]["n_gold"], ci=False) if cats[c]["n_gold"] else "—",
                      f"{cats[c]['edge_f1'] * 100:.1f}%" if cats[c]["edge_p"] != 1.0 or cats[c]["edge_r"] != 1.0
                      or any(r["edges_gold"] for r in by[c]) else "—",
                      pct(cats[c]["exact"], cats[c]["turns"], ci=False), cats[c]["failed"]]
                     for c in CATEGORY_ORDER if c in cats]))
    md.append("")

    classes = groups(rows, lambda r: [r["error_class"]])
    m["model_plan"]["error_classes"] = {k: len(v) for k, v in classes.items()}
    md.append("每轮按主要错误归类（按表中先后取第一个成立的）：\n")
    md.append(table(["类别", "轮数", "占比"],
                    [[ERROR_CN[k], len(classes[k]), pct(len(classes[k]), len(rows), ci=False)]
                     for k in ERROR_CN if k in classes], "lrr"))
    md.append("")
    errs: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in r["arg_errors"].items():
            errs[k] += v
    m["model_plan"]["arg_errors"] = dict(errs)
    og = sum(r["order_edges_gold"] for r in rows)
    oh = sum(r["order_edges_hit"] for r in rows)
    md.append(f"参数错误的构成（配上的任务中）：缺参数 {errs['missing']}、字面值不符 {errs['literal_wrong']}、"
              f"该引用上一步结果却写了字面值 {errs['ref_as_literal']}、引用指错任务 {errs['ref_wrong']}、"
              f"不该引用却写成引用 {errs['literal_as_ref']}。用户明说先后（「都办完之后」）的顺序依赖，"
              f"模型表达出来的 {oh} / {og}（{pct(oh, og, ci=False)}）。\n")
    m["model_plan"]["order_edges"] = {"gold": og, "hit": oh}


def model_e2e(md: list[str], m: dict[str, Any]) -> None:
    rows = load_rows(RESULTS / "model-e2e" / "turns.jsonl")
    if not rows:
        return
    summ = json.loads((RESULTS / "model-e2e" / "summary.json").read_text())
    n, k = len(rows), sum(r["success"] for r in rows)
    m["model_e2e"] = {"turns": n, "success": k, "rate": wilson(k, n)}
    md.append("### 调度成功率（模型侧全链路）\n")
    md.append(f"按类别分层抽样 {summ.get('cases')} 条用例（每类 {summ.get('per_category')} 条）、{n} 轮，"
              f"真实模型承担规划、校验、重规划与收尾，环境注入用例声明的故障：成功 {k} 轮，**{pct(k, n)}**。\n")
    by = groups(rows, lambda r: [r["category"]])
    md.append(table(["类别", "轮数", "成功率", "平均模型调用", "单轮耗时 P50 (s)"],
                    [[CATEGORY_CN[c], len(by[c]), pct(sum(r["success"] for r in by[c]), len(by[c]), ci=False),
                      f"{sum(len(r['llm']) for r in by[c]) / len(by[c]):.1f}",
                      f"{q([r['wall_ms'] for r in by[c]], .5) / 1000:.1f}"]
                     for c in CATEGORY_ORDER if c in by]))
    md.append("")
    fails = {c: sum(1 for r in rows if r[c] is False) for c in CHECKS}
    m["model_e2e"]["check_failures"] = fails
    md.append("逐项检查未通过的轮数：" + "、".join(f"{c} {v}" for c, v in fails.items()) + "。\n")


def speed(md: list[str], m: dict[str, Any]) -> None:
    rows = load_rows(RESULTS / "model-speed" / "calls.jsonl")
    e2e = load_rows(RESULTS / "model-e2e" / "turns.jsonl")
    summ_p = RESULTS / "model-speed" / "summary.json"
    if not rows:
        return
    dep = json.loads(summ_p.read_text())["deployment"]
    m["deployment"] = dep
    by = groups(rows, lambda r: [r["role"]])
    out = []
    m["speed"] = {}
    for role in ("planner", "replanner", "result_verifier", "goal_verifier", "finalizer"):
        rs = by.get(role)
        if not rs:
            continue
        pre = [r["prompt_tokens"] / (r["prompt_ms"] / 1000) for r in rs if r["prompt_ms"] and r["prompt_tokens"]]
        dec = [r["gen_tokens"] / (r["gen_ms"] / 1000) for r in rs if r["gen_ms"] and r["gen_tokens"]]
        row = {"n": len(rs), "ttft_p50": q([r["ttft_ms"] for r in rs], .5),
               "ttft_p95": q([r["ttft_ms"] for r in rs], .95),
               "total_p50": q([r["total_ms"] for r in rs], .5), "total_p95": q([r["total_ms"] for r in rs], .95),
               "prompt_tokens_p50": q([r["prompt_tokens"] or 0 for r in rs], .5),
               "gen_tokens_p50": q([r["gen_tokens"] or 0 for r in rs], .5),
               "prefill_tps_p50": q(pre, .5), "decode_tps_p50": q(dec, .5)}
        m["speed"][role] = row
        out.append([role, row["n"], f"{row['prompt_tokens_p50']:.0f}", f"{row['gen_tokens_p50']:.0f}",
                    f"{row['ttft_p50']:.0f} / {row['ttft_p95']:.0f}",
                    f"{row['prefill_tps_p50']:.0f}", f"{row['decode_tps_p50']:.1f}",
                    f"{row['total_p50'] / 1000:.2f} / {row['total_p95'] / 1000:.2f}"])
    md.append("### 推理速度\n")
    d = dep.get("details") or {}
    md.append(f"部署：Ollama {dep.get('ollama')}，{dep.get('model')}（{d.get('parameter_size')}，"
              f"{d.get('quantization_level')}），上下文 {dep.get('context_length')} token，"
              f"模型占用 {(dep.get('size_bytes') or 0) / 2**30:.2f} GiB，其中显存 "
              f"{(dep.get('size_vram_bytes') or 0) / 2**30:.2f} GiB。回放 {len(rows)} 条全链路中记下的真实提示词，"
              f"首 token 时延为客户端计时，预填充与解码速率取服务端回报的 token 数与耗时。\n")
    md.append(table(["节点", "次数", "提示 token P50", "生成 token P50", "首 token P50 / P95 (ms)",
                     "预填充 (token/s) P50", "解码 (token/s) P50", "单次总耗时 P50 / P95 (s)"], out))
    md.append("")
    if e2e:
        recs = [x for r in e2e for x in r["llm"]]
        share = groups(recs, lambda x: [x["role"]])
        tot = sum(x["ms"] for x in recs) or 1
        md.append("全链路中各节点的推理耗时占比：" + "、".join(
            f"{k} {sum(x['ms'] for x in v) / tot * 100:.1f}%（{len(v)} 次）" for k, v in
            sorted(share.items(), key=lambda kv: -sum(x["ms"] for x in kv[1]))) + "。\n")


def main() -> None:
    md: list[str] = []
    m: dict[str, Any] = {}
    for section in (framework, memory, concurrency, model_plan, model_e2e, speed):
        section(md, m)
    (RESULTS / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    (RESULTS / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(RESULTS / "tables.md")


if __name__ == "__main__":
    main()
