"""框架内存专项：常驻、单请求峰值、长时运行的增长、空闲会话的占用。

    python -m bench.eval.memory --cases 3000

在独立进程里测，用例逐行流式读取，不把整份数据集载入内存 —— 否则数据集
自身的几百 MB 对象会淹没框架的占用。四组量：

  常驻        导入框架之后、跑完预热用例之后的 RSS
  单请求峰值  每条用例执行期间 Python 堆的峰值增量（tracemalloc，从用例开始时清零）
  增长        预热之后每隔一段记一次 Python 堆的当前占用与 RSS，按用例数求斜率；
              堆占用不随处理量上升即没有泄漏。RSS 另记：它含分配器保留与
              系统的内存压缩，在 macOS 上会随系统状态涨落，不适合判泄漏
  空闲会话    跑完多轮用例后把会话留着不放，Python 堆的增量除以会话数

tracemalloc 只计 Python 层的分配，不含解释器本身与 C 扩展的内部缓冲，
所以它衡量的是「框架产生了多少对象」，与 RSS 口径不同，两者分列。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterator

import psutil

from miagent import build_agent
from miagent.memory import Session

from .env import make_env
from .framework import DATA, RECURSION_LIMIT, RESULTS, quiet, run_case
from .oracle import OracleLLM

MB = 2 ** 20


def stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def rss() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def main() -> None:
    ap = argparse.ArgumentParser(description="框架内存专项")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    quiet()
    out = args.out or RESULTS / time.strftime("%Y%m%d-%H%M%S-memory")
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"rss_after_import_mb": round(rss(), 2)}

    tracemalloc.start()
    peaks: list[float] = []
    series: list[dict[str, Any]] = []
    source = stream(args.data)
    for n in range(1, args.cases + 1):
        case = next(source)
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        run_case(case)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append((peak - base) / MB)
        if n == args.warmup:
            gc.collect()
            result["rss_after_warmup_mb"] = round(rss(), 2)
        if n >= args.warmup and n % args.every == 0:
            gc.collect()
            cur, _ = tracemalloc.get_traced_memory()
            series.append({"done": n, "heap_mb": round(cur / MB, 3), "rss_mb": round(rss(), 2),
                           "objects": len(gc.get_objects())})

    peaks.sort()
    q = lambda p: round(peaks[min(len(peaks) - 1, int(p * len(peaks)))], 3)  # noqa: E731
    result["request_peak_heap_mb"] = {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99),
                                      "max": round(peaks[-1], 3)}
    xs = [s["done"] for s in series]
    result["heap_slope_mb_per_1k"] = round(_slope(xs, [s["heap_mb"] for s in series]) * 1000, 4)
    result["objects_slope_per_1k"] = round(_slope(xs, [s["objects"] for s in series]) * 1000, 1)
    result["rss_slope_mb_per_1k"] = round(_slope(xs, [s["rss_mb"] for s in series]) * 1000, 3)
    result["series"] = series

    # 空闲会话：多轮用例跑完后会话不放，看每个会话留下多少
    multi = [c for c in stream(args.data) if len(c["turns"]) >= 2][: args.sessions]
    keep: list[Any] = []
    gc.collect()
    before, _ = tracemalloc.get_traced_memory()
    for case in multi:
        env = make_env(case["budget"], case["permissions"])
        llm = OracleLLM(case, env.probe)
        session = Session(build_agent(llm, env.registry), {"recursion_limit": RECURSION_LIMIT})
        for i, turn in enumerate(case["turns"]):
            llm.begin_turn(i)
            session.run(turn["request"])
        env.close()
        keep.append(session.turns)          # 只留会话的轮次记录，即运行时保留的部分
    gc.collect()
    after, _ = tracemalloc.get_traced_memory()
    result["idle_session"] = {"sessions": len(keep),
                              "turns": sum(len(t) for t in keep),
                              "heap_per_session_kb": round((after - before) / len(keep) / 1024, 2)}
    tracemalloc.stop()
    result["cases"] = args.cases
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "series"}, ensure_ascii=False))


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


if __name__ == "__main__":
    main()
