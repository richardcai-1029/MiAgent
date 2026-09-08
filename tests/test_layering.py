"""分层隔离守卫。

端侧存在只需要协议客户端的部署形态：转发系统调用、不做图调度。
这类部署不应为图引擎付出常驻内存与冷启动的代价。

下列模块必须能在完全不加载 langgraph / langchain 的情况下导入。
每个用例都在独立子进程中运行 —— 同进程内 pytest 自身已经加载了
图引擎，测不出真实的导入代价。
"""

import subprocess
import sys

import pytest

# 端侧瘦客户端形态所需的全部模块
LIGHT_MODULES = [
    "miagent.protocol",
    "miagent.transport",
    "miagent.client",
    "miagent.tools",
    "miagent.llm",
    "miagent.graph",        # 包本身惰性导出 build_agent，不触发图引擎
    "miagent.graph.dag",    # 依赖解析是纯函数
    "miagent.graph.state",
    "miagent.graph.schema",
]

FORBIDDEN = ["langgraph", "langchain_core", "langsmith", "requests",
             "httpx", "urllib3", "websockets"]

PROBE = """
import importlib, sys
importlib.import_module({module!r})
loaded = {{m.split('.')[0] for m in sys.modules}}
bad = sorted(loaded & set({forbidden!r}))
print(",".join(bad))
"""


def _heavy_deps_after_importing(module: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out.split(",") if out else []


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_layer_does_not_pull_graph_engine(module):
    assert _heavy_deps_after_importing(module) == []


def test_build_agent_is_lazily_exported():
    """miagent.graph 只在真正访问 build_agent 时才加载图引擎。"""
    assert _heavy_deps_after_importing("miagent.graph") == []
    probe = ("import sys; from miagent.graph import build_agent; "
             "print('langgraph' in {m.split('.')[0] for m in sys.modules})")
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "True"


# 回归哨兵阈值。当前瘦客户端实测约 30 MB；此处取 40 MB 只为捕捉显著退化
# （例如误引入某个重型纯 Python 包），不代表任何外部要求 ——
# MiClaw 侧的真实资源配额尚不可获取。
LIGHT_PROFILE_TRIPWIRE_MB = 40


def test_light_profile_does_not_regress():
    """瘦客户端形态的常驻内存不应显著上升。"""
    probe = ("import os, psutil, miagent.client, miagent.tools, miagent.llm; "
             "print(psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024)")
    rss = int(subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True, check=True).stdout)
    assert rss <= LIGHT_PROFILE_TRIPWIRE_MB, (
        f"瘦客户端常驻 {rss}MB，超过回归哨兵 {LIGHT_PROFILE_TRIPWIRE_MB}MB")
