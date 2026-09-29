"""本进程的资源采样：常驻内存、CPU 占用、线程数，按固定间隔记录。

采样在后台线程里进行，每个样本附上当时已完成的用例数，事后可以看内存
是否随处理量增长（泄漏）。CPU 占用是相对单核的百分比（psutil 口径，
多核满载可超过 100）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import psutil


class Sampler:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval = interval_s
        self.samples: list[dict[str, Any]] = []
        self.progress = 0                       # 由调用方更新：已完成的用例数
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bench-sampler")
        self._t0 = 0.0

    def __enter__(self) -> Sampler:
        self._proc.cpu_percent(None)            # 第一次调用只建立基准
        self._t0 = time.perf_counter()
        self._cpu0 = self._proc.cpu_times()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._take()
        cpu = self._proc.cpu_times()
        self.cpu_seconds = (cpu.user - self._cpu0.user) + (cpu.system - self._cpu0.system)
        self.wall_seconds = time.perf_counter() - self._t0

    def _take(self) -> None:
        mem = self._proc.memory_info()
        self.samples.append({"t": round(time.perf_counter() - self._t0, 3),
                             "done": self.progress,
                             "rss_mb": round(mem.rss / 2**20, 2),
                             "cpu_pct": self._proc.cpu_percent(None),
                             "threads": self._proc.num_threads()})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._take()

    def summary(self) -> dict[str, Any]:
        rss = [s["rss_mb"] for s in self.samples]
        cpu = [s["cpu_pct"] for s in self.samples[1:]] or [0.0]
        # 内存随处理量的增长：按已完成用例数做最小二乘斜率（MB / 千条），
        # 只用后半程的样本 —— 前半程包含首次导入与缓存建立。
        half = [s for s in self.samples if s["done"] >= self.progress / 2]
        slope = _slope([s["done"] for s in half], [s["rss_mb"] for s in half]) * 1000
        return {"samples": len(self.samples), "rss_start_mb": rss[0], "rss_peak_mb": max(rss),
                "rss_end_mb": rss[-1], "rss_slope_mb_per_1k": round(slope, 3),
                "cpu_avg_pct": round(sum(cpu) / len(cpu), 1), "cpu_peak_pct": max(cpu),
                "threads_peak": max(s["threads"] for s in self.samples),
                "cpu_seconds": round(self.cpu_seconds, 2),
                "wall_seconds": round(self.wall_seconds, 2)}


def _slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
