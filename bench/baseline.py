"""端侧资源基线测量。

按【部署形态】而非单个模块测量，因为端侧的实际约束是「这一形态的进程
常驻多少内存」。两种形态：

    瘦客户端  协议 + 传输 + 客户端 + 工具 + 模型层。
              只做系统调用转发与工具调度，不做图调度。
    完整 Agent 追加图引擎，具备任务规划、依赖调度、重规划能力。

每项都在独立子进程中测量：同进程内测第二个目标时，第一个的模块已在
sys.modules 里，结果会偏低。

运行：  python bench/baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys

# 端侧不需要的云端与网络包，用于统计冗余占比
CLOUD_PACKAGES = {
    "langsmith", "langgraph_sdk", "requests", "urllib3", "httpx",
    "httpcore", "websockets", "anyio", "certifi", "ssl", "h11",
}

PROFILES = [
    ("裸解释器", ""),
    ("瘦客户端", "import miagent.client, miagent.tools, miagent.llm"),
    ("完整 Agent", "import miagent.client, miagent.tools, miagent.llm; "
                   "from miagent.graph import build_agent"),
]

# 验收目标
TARGETS = {"瘦客户端": {"rss_mb": 45, "elapsed_ms": 150},
           "完整 Agent": {"rss_mb": 80, "elapsed_ms": 500}}

PROBE = '''
import json, os, sys, time
try:
    import psutil
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
print("@@" + json.dumps({{"rss_mb": round(rss / 1024 / 1024, 1),
                         "elapsed_ms": round(elapsed, 1),
                         "modules": len(loaded), "tops": tops}}))
'''


def measure(code: str, repeat: int = 3) -> dict:
    """内存取中位数，耗时取最小值 —— 噪声只会让耗时变大，不会变小。"""
    runs = []
    for _ in range(repeat):
        out = subprocess.run(
            [sys.executable, "-c", PROBE.format(code=code or "pass")],
            capture_output=True, text=True, check=True,
        ).stdout
        runs.append(json.loads(next(l for l in out.splitlines() if l.startswith("@@"))[2:]))
    runs.sort(key=lambda r: r["rss_mb"])
    return {**runs[len(runs) // 2],
            "elapsed_ms": min(r["elapsed_ms"] for r in runs)}


def main() -> None:
    print(f"Python {sys.version.split()[0]}\n")
    print(f"  {'部署形态':<14}{'常驻内存':>10}{'冷启动':>11}{'模块数':>9}{'云端模块':>12}{'达标':>7}")
    print("  " + "-" * 66)

    results = {}
    for label, code in PROFILES:
        r = measure(code)
        results[label] = r
        cloud = sum(n for pkg, n in r["tops"].items() if pkg in CLOUD_PACKAGES)
        pct = f"{cloud * 100 // r['modules']}%" if r["modules"] else "—"
        target = TARGETS.get(label)
        if target:
            ok = r["rss_mb"] <= target["rss_mb"] and r["elapsed_ms"] <= target["elapsed_ms"]
            mark = "✅" if ok else "✗"
        else:
            mark = "—"
        print(f"  {label:<14}{r['rss_mb']:>7} MB{r['elapsed_ms']:>9} ms"
              f"{r['modules']:>9}{cloud:>7} ({pct})   {mark}")

    print("\n  验收目标：")
    for label, t in TARGETS.items():
        print(f"    {label:<12} 常驻 ≤ {t['rss_mb']} MB，冷启动 ≤ {t['elapsed_ms']} ms")

    full = results["完整 Agent"]
    print(f"\n  完整形态的云端模块构成（端侧不需要，但无法从 LangGraph 中剥离）：")
    for pkg in sorted(CLOUD_PACKAGES, key=lambda p: -full["tops"].get(p, 0)):
        if full["tops"].get(pkg):
            print(f"    {pkg:<16}{full['tops'][pkg]:>5} 个子模块")

    delta = full["rss_mb"] - results["瘦客户端"]["rss_mb"]
    print(f"\n  图引擎的边际成本：常驻 +{delta:.1f} MB，"
          f"冷启动 +{full['elapsed_ms'] - results['瘦客户端']['elapsed_ms']:.0f} ms，"
          f"模块 +{full['modules'] - results['瘦客户端']['modules']}")


if __name__ == "__main__":
    main()
