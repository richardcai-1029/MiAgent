"""模型侧评测：真实模型（局域网 Ollama）驱动。

    python -m bench.eval.model plan --per-category 100    # 规划准确率，按类别分层抽样
    python -m bench.eval.model e2e  --per-category 30     # 全链路：真实模型跑整张图
    python -m bench.eval.model speed --from <e2e 结果目录>  # 推理速度：回放提示词，取服务端计时

三种模式：

  plan   只调规划节点，把产出的任务图对照标准任务图打分（见 plan_score）。
         多轮用例的后几轮带对话历史：历史里的摘要由参考执行的结果拼成
         （「描述：结果」），与标注答案驱动的模型所写的摘要同一格式。
  e2e    真实模型扮演规划、校验、重规划、收尾全部角色，环境注入用例声明的故障，
         按与框架侧相同的标准打分（见 score）。
  speed  用 e2e 记下的提示词，经 Ollama 原生接口流式回放：首 token 时延取客户端
         计时，预填充与解码的 token 数与耗时取服务端回报。

抽样：每个类别按 seed 取前 N 条（打乱后），类别不足 N 条时取全部。
部署参数（模型、量化、num_ctx、温度）以服务端 /api/show 为准，写进结果。

模型的上下文上限按 8192 token 的部署取 16000 字符（依据见 miagent/llm/ollama.py）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

from miagent.graph import build_agent, initial_state, nodes
from miagent.graph.nodes_meta import Deps
from miagent.graph.state import Turn
from miagent.memory import Session

from ..suite.check import load
from .env import make_env
from .framework import DATA, RECURSION_LIMIT, RESULTS, quiet
from .plan_score import score_plan
from .recording import RecordingOllama
from .resources import Sampler
from .score import score_turn

BASE = os.environ.get("OLLAMA_BASE_URL", "http://192.168.3.121:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-8b:latest")
CONTEXT_LIMIT = 16000
SEED = 20260929


def sample(cases: list[dict[str, Any]], per_category: int, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        by.setdefault(c["category"], []).append(c)
    out = []
    for cat in sorted(by):
        lst = list(by[cat])
        rng.shuffle(lst)
        out += lst[:per_category]
    return sorted(out, key=lambda c: c["id"])


def deployment() -> dict[str, Any]:
    root = BASE.removesuffix("/v1")
    req = urllib.request.Request(f"{root}/api/show", data=json.dumps({"model": MODEL}).encode())
    show = json.load(urllib.request.urlopen(req, timeout=10))
    version = json.load(urllib.request.urlopen(f"{root}/api/version", timeout=10))
    ps = json.load(urllib.request.urlopen(f"{root}/api/ps", timeout=10))
    loaded = next((m for m in ps.get("models", []) if m["name"] == MODEL), {})
    return {"base_url": BASE, "model": MODEL, "ollama": version.get("version"),
            "details": show.get("details"), "parameters": show.get("parameters"),
            "size_bytes": loaded.get("size"), "size_vram_bytes": loaded.get("size_vram"),
            "context_length": loaded.get("context_length")}


def _llm() -> RecordingOllama:
    return RecordingOllama(model=MODEL, base_url=BASE, context_limit=CONTEXT_LIMIT)


def _history(case: dict[str, Any], upto: int) -> list[Turn]:
    """前几轮的对话历史。摘要取参考执行里成功调用的「描述：结果」。"""
    out: list[Turn] = []
    for i, t in enumerate(case["turns"][:upto]):
        desc = {g["id"]: g["description"] for g in t["gold"]["tasks"]}
        summary = "；".join(f"{desc.get(c['id'], c['tool'])}：{c['result']}"
                           for c in t["expected"]["calls"] if c["status"] == "done") or "本轮未调用工具"
        out.append(Turn(number=i + 1, request=t["request"], answer=summary, summary=summary,
                        failure=None, episodes=[]))
    return out


# ---------------- plan ----------------


def run_plan(cases: list[dict[str, Any]], out: Path) -> None:
    llm = _llm()
    with (out / "turns.jsonl").open("w", encoding="utf-8") as f, Sampler(1.0) as sampler:
        for n, case in enumerate(cases, 1):
            env = make_env(case["budget"], case["permissions"])
            deps = Deps(llm=llm, registry=env.registry)
            for i, turn in enumerate(case["turns"]):
                state = initial_state(turn["request"])
                state["history"] = _history(case, i)
                r0, rep0 = len(llm.records), llm.repair_count
                t0 = time.perf_counter()
                error = None
                try:
                    res = nodes.planner(state, deps)
                except Exception as e:
                    res, error = {"tasks": {}, "failure": "exception"}, f"{type(e).__name__}: {e}"[:300]
                wall = (time.perf_counter() - t0) * 1000
                model_tasks = [{"id": t["id"], "required_tool": t["required_tool"],
                                "arguments": t["arguments"], "dependencies": t["dependencies"]}
                               for t in res["tasks"].values()]
                s = score_plan(model_tasks, turn["gold"]["tasks"], turn.get("accept") or {},
                               turn["forbidden"])
                f.write(json.dumps({
                    "case": case["id"], "turn": i, "category": case["category"],
                    "level": case["level"], "tags": case["tags"], **s,
                    "plan_failure": res.get("failure"), "plan_errors": res.get("errors"),
                    "error": error,
                    "wall_ms": round(wall, 1), "repairs": llm.repair_count - rep0,
                    "llm": llm.records[r0:], "model_tasks": model_tasks,
                }, ensure_ascii=False) + "\n")
                f.flush()
            env.close()
            sampler.progress = n
            print(f"plan {n}/{len(cases)}", flush=True)
    _finish(out, sampler, llm)


# ---------------- e2e ----------------


def run_e2e(cases: list[dict[str, Any]], out: Path, keep_prompts: int) -> None:
    llm = _llm()
    llm.keep_prompts = keep_prompts
    with (out / "turns.jsonl").open("w", encoding="utf-8") as f, Sampler(1.0) as sampler:
        for n, case in enumerate(cases, 1):
            faults = [x for t in case["turns"] for x in t["faults"]]
            env = make_env(case["budget"], case["permissions"], faults)
            agent = build_agent(llm, env.registry,
                                max_concurrent_miclaw=env.budget.max_concurrent_calls)
            session = Session(agent, {"recursion_limit": RECURSION_LIMIT})
            for i, turn in enumerate(case["turns"]):
                env.probe.turn = i
                r0, rep0 = len(llm.records), llm.repair_count
                t0 = time.perf_counter()
                state, error = None, None
                try:
                    state = session.run(turn["request"])
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"[:300]
                    traceback.print_exc()
                wall = (time.perf_counter() - t0) * 1000
                calls = env.probe.of_turn(i)
                s = score_turn(turn, calls, state, env.budget.max_concurrent_calls,
                               env.probe.max_active)
                if error:
                    s["success"] = False
                recs = llm.records[r0:]
                f.write(json.dumps({
                    "case": case["id"], "turn": i, "category": case["category"],
                    "level": case["level"], "tags": case["tags"], **s, "error": error,
                    "wall_ms": round(wall, 1), "llm_ms": round(sum(r["ms"] for r in recs), 1),
                    "llm": recs, "repairs": llm.repair_count - rep0,
                    "replans": (state or {}).get("replan_count", 0),
                    "answer": (state or {}).get("final_answer"),
                    "trace": (state or {}).get("trace"),
                }, ensure_ascii=False) + "\n")
                f.flush()
            env.close()
            sampler.progress = n
            print(f"e2e {n}/{len(cases)}", flush=True)
    (out / "prompts.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in llm.prompts), encoding="utf-8")
    _finish(out, sampler, llm)


def _finish(out: Path, sampler: Sampler, llm: RecordingOllama) -> None:
    (out / "resources.json").write_text(json.dumps(sampler.samples), encoding="utf-8")
    meta = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    meta.update(resources=sampler.summary(), llm_stats=llm.stats())
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- speed ----------------


def run_speed(src: Path, out: Path, limit: int) -> None:
    """流式回放记下的提示词：首 token 时延（客户端计时）与服务端的预填充、解码计时。"""
    root = BASE.removesuffix("/v1")
    prompts = [json.loads(line) for line in (src / "prompts.jsonl").open(encoding="utf-8")][:limit]
    with (out / "calls.jsonl").open("w", encoding="utf-8") as f:
        for n, p in enumerate(prompts, 1):
            body = {"model": MODEL, "messages": p["messages"], "stream": True}
            if p.get("format"):
                body["format"] = p["format"]
            req = urllib.request.Request(f"{root}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            ttft = None
            final: dict[str, Any] = {}
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    chunk = json.loads(line)
                    if ttft is None and chunk.get("message", {}).get("content"):
                        ttft = (time.perf_counter() - t0) * 1000
                    if chunk.get("done"):
                        final = chunk
            total = (time.perf_counter() - t0) * 1000
            ns = 1e6
            f.write(json.dumps({
                "role": p["role"], "ttft_ms": round(ttft or total, 1), "total_ms": round(total, 1),
                "load_ms": round(final.get("load_duration", 0) / ns, 1),
                "prompt_tokens": final.get("prompt_eval_count"),
                "prompt_ms": round(final.get("prompt_eval_duration", 0) / ns, 1),
                "gen_tokens": final.get("eval_count"),
                "gen_ms": round(final.get("eval_duration", 0) / ns, 1),
            }) + "\n")
            f.flush()
            print(f"speed {n}/{len(prompts)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="模型侧评测")
    ap.add_argument("mode", choices=("plan", "e2e", "speed"))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--keep-prompts", type=int, default=400, help="e2e 记下多少条提示词供 speed 回放")
    ap.add_argument("--from", dest="src", type=Path, help="speed：e2e 结果目录")
    ap.add_argument("--limit", type=int, default=400, help="speed：回放多少条")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    quiet()
    out = args.out or RESULTS / time.strftime(f"%Y%m%d-%H%M%S-model-{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"mode": args.mode, "deployment": deployment(),
                            "context_limit_chars": CONTEXT_LIMIT, "started": time.strftime("%F %T")}
    if args.mode == "speed":
        meta["source"] = str(args.src)
        (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        run_speed(args.src, out, args.limit)
        print(out)
        return
    cases = sample(load(args.data), args.per_category)
    meta.update(per_category=args.per_category, seed=SEED, cases=len(cases),
                turns=sum(len(c["turns"]) for c in cases))
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "plan":
        run_plan(cases, out)
    else:
        run_e2e(cases, out, args.keep_prompts)
    print(out)


if __name__ == "__main__":
    main()
