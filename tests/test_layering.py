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
    "miagent.memory",       # 情景记忆是纯函数
    "miagent.runtime",      # 多请求运行时只用标准库
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


# ============================================================
# 框架接触面守卫
# ============================================================

# 允许 import langgraph 的模块。图的组装与路由必须用框架 API，
# 其余各层都不应该碰到它。
FRAMEWORK_FACING = {"miagent/graph/build.py", "miagent/graph/routers.py"}


def _modules_importing(package: str, root: str = "miagent") -> set[str]:
    """用 AST 找出真正 import 了该包的模块。

    不用文本匹配 —— 文档字符串与注释里出现的包名会被误判。
    """
    import ast
    from pathlib import Path

    hits = set()
    base = Path(__file__).resolve().parent.parent
    for path in sorted((base / root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == package or n.startswith(package + ".") for n in names):
                hits.add(path.relative_to(base).as_posix())
    return hits


def test_framework_surface_is_confined():
    """只有图组装与路由两个模块可以 import langgraph。

    接触面必须可枚举，否则谈不上版本回退：升级失配时需要知道
    到底有哪几处要改。守卫在这里，新增的框架依赖会立刻暴露。
    """
    actual = _modules_importing("langgraph")
    assert actual == FRAMEWORK_FACING, (
        f"框架接触面发生变化。预期 {sorted(FRAMEWORK_FACING)}，实际 {sorted(actual)}。"
        "新增框架依赖需同步评估其版本敏感度，并更新适配改造清单的风险分级。"
    )


# 瘦客户端形态允许加载的第三方顶层包。
# 与 FORBIDDEN 黑名单互补：黑名单只拦已知的重依赖，框架换用新的传递依赖时
# 会静默漏判；白名单则是「多出任何一个都失败」，失效方式是显式的。
ALLOWED_THIRD_PARTY = {"miagent", "pydantic", "pydantic_core",
                       "annotated_types", "typing_extensions", "typing_inspection"}

# 以导入前的模块快照为基线做差：解释器启动时注入的东西（如 sitecustomize）
# 不属于本次导入的代价，也不应因环境不同而影响判定。
WHITELIST_PROBE = """
import sys
baseline = {{m.split('.')[0] for m in sys.modules}}
import importlib
importlib.import_module({module!r})
tops = {{m.split('.')[0] for m in sys.modules if not m.startswith('_')}}
print(",".join(sorted(tops - baseline - sys.stdlib_module_names)))
"""


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_layer_loads_no_unexpected_third_party(module):
    out = subprocess.run(
        [sys.executable, "-c", WHITELIST_PROBE.format(module=module)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    loaded = set(out.split(",")) if out else set()
    assert loaded <= ALLOWED_THIRD_PARTY, (
        f"导入 {module} 后加载了预期之外的第三方包：{sorted(loaded - ALLOWED_THIRD_PARTY)}"
    )
