"""框架侧评测：标注答案驱动的模型跑全量数据集。

    python -m bench.eval.framework                       # 全量，逐条执行
    python -m bench.eval.framework --limit 200           # 前 200 条
    python -m bench.eval.framework --tool-latency-ms 20  # 每次系统调用加 20 ms 等待
    python -m bench.eval.framework --no-verify          # 关掉语义校验（消融）

模型由 OracleLLM 扮演（见 oracle.py），因此这里衡量的是框架本身：调度成功率、
调用执行正确率、框架自身开销与资源占用。每条用例一套独立环境，用例之间串行，
耗时与内存不受其他用例干扰。

产出写到 bench/results/<时间>-framework/：turns.jsonl（每轮一行）、
resources.json（采样序列）与 summary.json（汇总）。
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from miagent.client import client as client_mod
from miagent.graph import build_agent
from miagent.memory import Session
from miagent.mock_server import server as server_mod

from ..suite.check import load
from .env import make_env
from .oracle import OracleLLM
from .resources import Sampler
from .score import score_turn

DATA = Path(__file__).resolve().parents[1] / "data" / "cases.jsonl"
RESULTS = Path(__file__).resolve().parents[1] / "results"

# 图的递归上限：取足以走完「最多 7 个任务 × 重试 × 两次重规划」的量，
# 只防死循环，不参与判定 —— 触顶即记为异常。
RECURSION_LIMIT = 200


def quiet() -> None:
    """服务端与客户端的逐条日志写 stderr，上万条用例下只是噪声与 I/O 开销。"""
    server_mod.log = lambda *a, **k: None
    client_mod.log = lambda *a, **k: None


def run_case(case: dict[str, Any], tool_latency_ms: float = 0,
             verify: bool = True) -> list[dict[str, Any]]:
    faults = [f for t in case["turns"] for f in t["faults"]]
    env = make_env(case["budget"], case["permissions"], faults, tool_latency_ms)
    llm = OracleLLM(case, env.probe)
    agent = build_agent(llm, env.registry, max_concurrent_miclaw=env.budget.max_concurrent_calls,
                        verify=verify)
    session = Session(agent, {"recursion_limit": RECURSION_LIMIT})
    rows = []
    try:
        for i, turn in enumerate(case["turns"]):
            llm.begin_turn(i)
            calls0, ms0, roles0 = llm.call_count, llm.total_elapsed_ms, len(llm.roles)
            repairs0 = llm.repair_count
            t0 = time.perf_counter()
            state, error = None, None
            try:
                state = session.run(turn["request"])
            except Exception as e:                     # 记为该轮失败，不中断全量
                error = f"{type(e).__name__}: {e}"[:300]
                traceback.print_exc()
            wall = (time.perf_counter() - t0) * 1000
            calls = env.probe.of_turn(i)
            score = score_turn(turn, calls, state, env.budget.max_concurrent_calls,
                               env.probe.max_active)
            if error:
                score["success"] = False
            rows.append({
                "case": case["id"], "turn": i, "category": case["category"],
                "level": case["level"], "tags": case["tags"], **score, "error": error,
                "wall_ms": round(wall, 3),
                "llm_calls": llm.call_count - calls0,
                "llm_ms": round(llm.total_elapsed_ms - ms0, 3),
                "llm_roles": llm.roles[roles0:],
                "repairs": llm.repair_count - repairs0,
                "replans": (state or {}).get("replan_count", 0),
                "executions": (state or {}).get("execution_count", 0),
                "max_active": env.probe.max_active,
                "tool_ms": round(sum((c.t1 - c.t0) * 1000 for c in calls), 3),
            })
    finally:
        env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="框架侧评测")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tool-latency-ms", type=float, default=0)
    ap.add_argument("--no-verify", action="store_true", help="关掉语义校验，做消融对照")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    quiet()
    cases = load(args.data)[: args.limit]
    out = args.out or RESULTS / time.strftime("%Y%m%d-%H%M%S-framework")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / "turns.jsonl").open("w", encoding="utf-8") as f, Sampler() as sampler:
        for n, case in enumerate(cases, 1):
            for row in run_case(case, args.tool_latency_ms, not args.no_verify):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sampler.progress = n
            if n % 500 == 0:
                print(f"{n}/{len(cases)}  {time.time() - started:.0f}s", flush=True)
    (out / "resources.json").write_text(json.dumps(sampler.samples), encoding="utf-8")
    meta = {"cases": len(cases), "tool_latency_ms": args.tool_latency_ms,
            "verify": not args.no_verify,
            "data": str(args.data), "resources": sampler.summary()}
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
