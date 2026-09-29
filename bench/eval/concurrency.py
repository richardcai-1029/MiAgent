"""多请求并发评测：经运行时（miagent.runtime）同时提交一批请求。

    python -m bench.eval.concurrency --inflight 4 --cases 1500 --tool-latency-ms 20

与框架侧逐条评测的区别：全部请求共用一套环境 —— 一个 MiClaw 会话、一份
并发配额、一个推理槽，这正是端侧的实际形态。看四件事：

  · 并发下每个请求是否仍然做对（与逐条评测相同的打分）
  · 所有请求合起来，在途的系统调用数是否守住握手下发的并发上限
  · 前台请求是否先于后台请求得到执行（排队时长按优先级分组比较）
  · 吞吐与内存随同时在飞的请求数怎样变化

模型仍由标注答案驱动。多个请求共用一个模型对象，由路由按请求分派：
规划时从提示词里认出是哪条用例，此后同一请求的推理与工具调用都记在它名下
—— 请求的身份取自运行时设定的 request_key，LangGraph 派发节点时会把它
带进工作线程。

只选单轮、无故障、权限齐全的用例：全部请求共用一个会话，按用例下发的
配额与授权在这里无法逐条区分。
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from miagent.llm import FakeLLM, LLMMessage
from miagent.runtime import PRIORITY_RANK, build_runtime
from miagent.runtime.slots import request_key
from miagent.protocol import RequestPriority

from ..suite.catalog import ALL_PERMISSIONS, DEFAULT_BUDGET
from ..suite.check import load
from .env import Call, Probe, make_env
from .framework import DATA, RECURSION_LIMIT, RESULTS, quiet
from .oracle import OracleLLM
from .resources import Sampler
from .score import score_turn

_GOAL = "用户目标："


class RoutedProbe(Probe):
    """全局在途计数，调用记录同时记到发起它的请求名下。"""

    def __init__(self, router: Router) -> None:
        super().__init__()
        self.router = router

    def leave(self, tool: str, args: dict[str, Any], ok: bool, code: str | None, t0: float) -> None:
        super().leave(tool, args, ok, code, t0)
        own = self.router.current()
        if own is not None:
            own.probe.calls.append(Call(len(own.probe.calls), tool, args, ok, code, t0,
                                        time.perf_counter(), 0))


class Router:
    def __init__(self, cases: list[dict[str, Any]]) -> None:
        self.by_request = {c["turns"][0]["request"]: c for c in cases}
        self.oracles: dict[tuple[int, ...], OracleLLM] = {}
        self.started: dict[str, float] = {}
        self._lock = threading.Lock()

    def current(self) -> OracleLLM | None:
        return self.oracles.get(request_key.get())

    def respond(self, messages: list[LLMMessage]) -> str:
        key = request_key.get()
        with self._lock:
            oracle = self.oracles.get(key)
            if oracle is None:
                body = messages[1].content
                request = next(line[len(_GOAL):] for line in body.splitlines()
                               if line.startswith(_GOAL))
                case = self.by_request[request]
                oracle = OracleLLM(case, Probe())
                oracle.begin_turn(0)
                self.oracles[key] = oracle
                self.started[case["id"]] = time.perf_counter()
        return oracle._respond(messages)


def eligible(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if len(c["turns"]) == 1 and not c["turns"][0]["faults"]
            and set(c["permissions"]) == set(ALL_PERMISSIONS)
            and c["budget"]["max_memory_mb"] == DEFAULT_BUDGET["max_memory_mb"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="多请求并发评测")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--inflight", type=int, required=True)
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--tool-latency-ms", type=float, default=20)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    quiet()
    pool = eligible(load(args.data))
    random.Random(args.seed).shuffle(pool)
    cases = pool[: args.cases]
    router = Router(cases)
    probe = RoutedProbe(router)
    env = make_env(DEFAULT_BUDGET, list(ALL_PERMISSIONS), [], args.tool_latency_ms, probe)
    llm = FakeLLM(responder=router.respond, context_limit=64000)
    runtime = build_runtime(llm, env.registry, env.budget, max_inflight=args.inflight,
                            max_idle_sessions=args.inflight,
                            config={"recursion_limit": RECURSION_LIMIT})

    out = args.out or RESULTS / time.strftime(f"%Y%m%d-%H%M%S-concurrency-{args.inflight}")
    out.mkdir(parents=True, exist_ok=True)
    submitted: dict[str, float] = {}
    finished: dict[str, float] = {}
    with Sampler(0.1) as sampler:
        t0 = time.perf_counter()
        futures = []
        for c in cases:
            prio = PRIORITY_RANK[RequestPriority(c["priority"])]
            submitted[c["id"]] = time.perf_counter()
            fut = runtime.submit(c["turns"][0]["request"], session=c["session"], priority=prio)
            fut.add_done_callback(lambda f, cid=c["id"]: finished.__setitem__(cid, time.perf_counter()))
            futures.append((c, fut))
        rows = []
        for n, (c, fut) in enumerate(futures, 1):
            state, error = None, None
            try:
                state = fut.result()
            except Exception as e:
                error = f"{type(e).__name__}: {e}"[:300]
            sampler.progress = n
            oracle = next((o for o in router.oracles.values() if o.case is c), None)
            calls = oracle.probe.calls if oracle else []
            s = score_turn(c["turns"][0], calls, state, env.budget.max_concurrent_calls,
                           probe.max_active)
            if error:
                s["success"] = False
            rows.append({"case": c["id"], "category": c["category"], "priority": c["priority"],
                         **s, "error": error,
                         "wait_ms": round((router.started.get(c["id"], t0) - submitted[c["id"]]) * 1000, 2),
                         "latency_ms": round((finished.get(c["id"], time.perf_counter())
                                              - submitted[c["id"]]) * 1000, 2)})
        makespan = time.perf_counter() - t0
    runtime.close()
    env.close()
    with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"inflight": args.inflight, "cases": len(cases), "tool_latency_ms": args.tool_latency_ms,
            "capacity": env.budget.max_concurrent_calls, "max_active_calls": probe.max_active,
            "makespan_s": round(makespan, 3), "throughput_rps": round(len(cases) / makespan, 2),
            "tool_calls": len(probe.calls), "resources": sampler.summary()}
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("inflight", "max_active_calls", "makespan_s",
                                           "throughput_rps")}))


if __name__ == "__main__":
    main()
