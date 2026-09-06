"""轻量化基线测量。

产出三项指标，用于任务书第 3 条「优化内存占用与推理耗时」的前后对比：
  · 常驻内存 RSS
  · 冷启动 import 耗时
  · 加载模块数（其中云端/网络相关占比）

每项都在**独立子进程**中测量 —— 同进程内测第二个目标时，第一个的
模块已经在 sys.modules 里了，结果会偏低。

运行：  python bench/baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys

# 端侧不需要的网络/云端包，用于统计"冗余云端模块"占比
CLOUD_PACKAGES = {
    "langsmith", "langgraph_sdk", "requests", "urllib3", "httpx",
    "httpcore", "websockets", "anyio", "certifi", "ssl", "h11",
}

TARGETS = [
    ("裸解释器",          ""),
    ("miagent.protocol",  "import miagent.protocol"),
    ("miagent.client",    "import miagent.client"),
    ("langgraph.graph",   "import langgraph.graph"),
]

PROBE = '''
import json, os, sys, time
try:
    import psutil
    rss0 = psutil.Process(os.getpid()).memory_info().rss
except ImportError:
    psutil = None
base = set(sys.modules)
t = time.perf_counter()
{code}
elapsed = (time.perf_counter() - t) * 1000
loaded = set(sys.modules) - base
tops = {{}}
for m in loaded:
    tops[m.split(".")[0]] = tops.get(m.split(".")[0], 0) + 1
rss = psutil.Process(os.getpid()).memory_info().rss if psutil else 0
print("@@" + json.dumps({{
    "rss_mb": round(rss / 1024 / 1024, 1),
    "elapsed_ms": round(elapsed, 1),
    "modules": len(loaded),
    "tops": tops,
}}))
'''


def measure(code: str, repeat: int = 3) -> dict:
    """跑 repeat 次取内存中位数、耗时最小值（最小值最接近真实开销，噪声只会变大）。"""
    runs = []
    for _ in range(repeat):
        out = subprocess.run(
            [sys.executable, "-c", PROBE.format(code=code or "pass")],
            capture_output=True, text=True, check=True,
        ).stdout
        line = next(l for l in out.splitlines() if l.startswith("@@"))
        runs.append(json.loads(line[2:]))
    runs.sort(key=lambda r: r["rss_mb"])
    best = min(runs, key=lambda r: r["elapsed_ms"])
    return {**runs[len(runs) // 2], "elapsed_ms": best["elapsed_ms"]}


def main() -> None:
    print(f"Python {sys.version.split()[0]}\n")
    print(f"  {'目标':<20}{'RSS':>9}{'冷启动':>11}{'模块数':>9}{'云端模块':>11}")
    print("  " + "-" * 60)

    for label, code in TARGETS:
        r = measure(code)
        cloud = sum(n for pkg, n in r["tops"].items() if pkg in CLOUD_PACKAGES)
        pct = f"({cloud * 100 // r['modules']}%)" if r["modules"] else ""
        print(f"  {label:<20}{r['rss_mb']:>7} MB{r['elapsed_ms']:>9} ms"
              f"{r['modules']:>9}{cloud:>7} {pct}")

    print("\n  云端模块明细（langgraph.graph）:")
    tops = measure("import langgraph.graph")["tops"]
    for pkg in sorted(CLOUD_PACKAGES, key=lambda p: -tops.get(p, 0)):
        if tops.get(pkg):
            print(f"    {pkg:<16}{tops[pkg]:>5} 个子模块")

    print("\n  验收目标：常驻 ≤ 45 MB / 冷启动 ≤ 150 ms / 模块数 < 550")


if __name__ == "__main__":
    main()
