"""代码规范守卫：模块命名、节点命名、注释形式。

规范写在《代码规范》（docs/07-code-conventions.md）；本文件把其中可机械判定的
几条固化下来，违反时直接指出位置。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "miagent"
SOURCES = sorted(PACKAGE.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE.parent).as_posix()


def test_module_names_are_snake_case():
    bad = [_rel(p) for p in SOURCES
           if not re.fullmatch(r"(__init__|__main__|[a-z][a-z0-9_]*)", p.stem)]
    assert not bad, f"模块名须为小写下划线：{bad}"


def test_every_module_has_docstring():
    bad = [_rel(p) for p in SOURCES if not ast.get_docstring(ast.parse(p.read_text("utf-8")))]
    assert not bad, f"模块缺少文档字符串：{bad}"


def test_public_top_level_definitions_have_docstrings():
    """模块顶层的公开类与函数都有文档字符串。下划线开头的私有定义不要求。"""
    bad = []
    for p in SOURCES:
        for node in ast.parse(p.read_text("utf-8")).body:
            if (isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and not node.name.startswith("_") and not ast.get_docstring(node)):
                bad.append(f"{_rel(p)}:{node.lineno} {node.name}")
    assert not bad, f"公开定义缺少文档字符串：{bad}"


# 分节注释只有三种形式：模块级、类内、字段分组。
_BANNER_LINE = re.compile(r"^\s*# [=-]{3,}")
_ALLOWED_BANNERS = [
    re.compile(r"^# ={60}$"),                       # 模块级分节（上下两行夹标题）
    re.compile(r"^    # -{60}$"),                   # 类内分节（上下两行夹标题）
    re.compile(r"^\s+# -{10} \S.* -{10}$"),         # 字段分组，单行
]


def test_section_banners_follow_convention():
    bad = []
    for p in SOURCES:
        for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
            if _BANNER_LINE.match(line) and not any(r.match(line) for r in _ALLOWED_BANNERS):
                bad.append(f"{_rel(p)}:{i} {line.strip()}")
    assert not bad, f"分节注释形式不合规范：{bad}"


def test_node_ids_match_implementations():
    """图中节点名与实现函数同名；两个执行节点共用 executor，以 source 区分。"""
    from miagent.agent import topology

    for name, node in topology.NODES.items():
        if name.endswith("_executor"):
            assert node.fn.__name__ == "executor"
            assert name == f"{node.bind['source'].value}_executor"
        else:
            assert node.fn.__name__ == name, f"节点 {name} 的实现叫 {node.fn.__name__}"
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", name)


def test_topology_is_closed():
    """边与分叉引用的节点都已声明，且每个节点都可达。"""
    from miagent.agent import topology

    declared = set(topology.NODES) | {topology.START, topology.END}
    referenced = {n for edge in topology.EDGES for n in edge}
    for branch in topology.BRANCHES:
        referenced |= {branch.source, *branch.targets}
    assert referenced <= declared, f"引用了未声明的节点：{referenced - declared}"
    assert set(topology.NODES) <= referenced, f"孤立节点：{set(topology.NODES) - referenced}"


def test_trace_lines_start_with_node_id():
    """执行轨迹的每一行以产生它的节点名开头，轨迹因此可以按节点检索。"""
    from miagent.agent import topology
    from miagent.client import MiClawClient
    from miagent.mock_server import MiClawMockServer
    from miagent.tools import ToolRegistry
    from miagent.transport import LoopbackTransport

    from .test_graph import PERMS, echo, llm_for, plan, run, task

    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("test", PERMS)
    registry = ToolRegistry([echo])
    registry.load_from_miclaw(client)
    out = run(registry, "查电量再回显", llm_for(plan(
        task("t1", "system.get_battery"),
        task("t2", "echo", deps=["t1"], text={"$from": "t1"}))))
    prefixes = {line.split(":", 1)[0] for line in out["trace"]}
    assert prefixes <= set(topology.NODES), f"未知前缀：{prefixes - set(topology.NODES)}"
    assert {"planner", "scheduler", "miclaw_executor", "local_executor", "finalizer"} <= prefixes
    client.close()
