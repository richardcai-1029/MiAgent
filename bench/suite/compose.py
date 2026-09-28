"""任务图的构造与话术渲染。

一张任务图由节点组成，节点的参数要么是用户说出的值（Value），要么是对
同一轮里另一个节点结果的引用（Ref）。另有「顺序依赖」：没有数据流、但用户
明说了先后（「这些都办完之后再……」）。

图先按结构构造，再渲染成一句话：按节点编号的先后逐个取说法模板，引用处
填上游结果的指代，相邻两句按依赖关系选连接词。编号的先后即拓扑序 ——
构造时总是先建上游、后建下游。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .catalog import CATALOG, DINING, Spec, accepts
from .phrases import (JOIN_AFTER_ALL, JOIN_PARALLEL, JOIN_SEQUENTIAL, PHRASES, PREFIXES,
                      REF_ONLY, REFERENCES, SUFFIXES, Value, sample)


class Retry(Exception):
    """这次随机选择凑不成要的结构，换一组选择重来。"""


@dataclass(frozen=True)
class Ref:
    task: str


@dataclass
class Node:
    id: str
    tool: str
    args: dict[str, Value | Ref]
    after: list[str] = field(default_factory=list)   # 顺序依赖，不含数据流
    clause: str = ""                                  # 附在说法后的补充，如备选方案

    @property
    def spec(self) -> Spec:
        return CATALOG[self.tool]

    def refs(self) -> dict[str, str]:
        return {p: v.task for p, v in self.args.items() if isinstance(v, Ref)}

    def dependencies(self) -> list[str]:
        return sorted(set(self.refs().values()) | set(self.after), key=_order)


def _order(tid: str) -> tuple[str, int]:
    m = re.match(r"^(\D*)(\d+)$", tid)
    return (m.group(1), int(m.group(2))) if m else (tid, 0)


class Graph:
    """一轮请求的任务图。节点按加入先后编号 t1、t2……"""

    def __init__(self, prefix: str = "t") -> None:
        self.nodes: dict[str, Node] = {}
        self._prefix = prefix

    def add(self, tool: str, args: dict[str, Value | Ref], after: list[str] | None = None) -> Node:
        node = Node(f"{self._prefix}{len(self.nodes) + 1}", tool, args, list(after or []))
        self.nodes[node.id] = node
        return node

    def children(self, tid: str) -> list[str]:
        return [n.id for n in self.nodes.values() if tid in n.dependencies()]

    def tools(self) -> set[str]:
        return {n.tool for n in self.nodes.values()}

    def harmonize(self, rng: random.Random) -> None:
        """推荐结果交给订餐厅时，推荐的类别改为能订座的类别。"""
        for n in self.nodes.values():
            if n.tool != "system.book_restaurant":
                continue
            for ref in n.refs().values():
                up = self.nodes[ref]
                if up.tool == "system.recommend_place" and up.args["category"].value not in DINING:
                    food = rng.choice(DINING)
                    up.args["category"] = Value(food, food)


# ---------------- 参数取值 ----------------


# 构造普通结构时不使用的工具：录屏的内存开销超出默认配额，必然失败，
# 只出现在资源类故障用例里。
EXCLUDED_FROM_NORMAL = {"system.record_screen"}


def literal_args(tool: str, rng: random.Random, fixed: dict[str, Value | Ref] | None = None
                 ) -> dict[str, Value | Ref]:
    """必填参数全部采样；可选参数一半概率出现。fixed 里给出的参数不采样。"""
    schema = CATALOG[tool].tool.input_schema
    required = set(schema["required"])
    out: dict[str, Value | Ref] = {}
    for p in schema["properties"]:
        if fixed and p in fixed:
            out[p] = fixed[p]
        elif p in required or rng.random() < 0.5:
            if (tool, p) in REF_ONLY:
                raise ValueError(f"{tool}.{p} 只能取自上一步结果")
            out[p] = sample(tool, p, rng)
    return out


def literal_ok(tool: str) -> bool:
    """这个工具能否只凭用户说出的值调用。"""
    return not any(t == tool for t, _ in REF_ONLY) and tool not in EXCLUDED_FROM_NORMAL


# ---------------- 数据流的连接关系 ----------------


def pipes() -> list[tuple[str, str, str]]:
    """全部 (上游工具, 下游工具, 下游参数)：上游的输出能填进下游的这个参数。"""
    return [(p.name, c.name, param)
            for p in CATALOG.values() if p.produces
            for c in CATALOG.values() if c.name != p.name and c.name not in EXCLUDED_FROM_NORMAL
            for param in c.accepts if accepts(c, param, p.produces)]


PIPES = pipes()


def chain_paths(length: int) -> list[list[tuple[str, str | None]]]:
    """长度为 length 的全部数据流链：[(工具, 引用上一步的参数)]，首个参数为 None。
    同一工具不在链上出现两次。"""
    paths: list[list[tuple[str, str | None]]] = [[(t, None)] for t in CATALOG
                                                 if CATALOG[t].produces and literal_ok(t)]
    for _ in range(length - 1):
        paths = [path + [(c, param)] for path in paths for p, c, param in PIPES
                 if p == path[-1][0] and c not in {t for t, _ in path}]
    return paths


# ---------------- 结构 ----------------


def build_single(rng: random.Random, pool: list[str]) -> Graph:
    g = Graph()
    tool = rng.choice(pool)
    g.add(tool, literal_args(tool, rng))
    return g


def build_parallel(rng: random.Random, pool: list[str], k: int) -> Graph:
    g = Graph()
    for tool in rng.sample(pool, k):
        g.add(tool, literal_args(tool, rng))
    return g


def build_chain(rng: random.Random, length: int) -> Graph:
    g = Graph()
    path = rng.choice(CHAINS[length])
    prev: Node | None = None
    for tool, param in path:
        fixed = {param: Ref(prev.id)} if prev is not None and param else None
        prev = g.add(tool, literal_args(tool, rng, fixed))
    return g


# 至少有两个参数能接收引用的工具：汇聚结构的终点
FAN_IN_SINKS = [c for c in CATALOG.values()
                if sum(1 for p in c.accepts) >= 2 and c.name not in EXCLUDED_FROM_NORMAL]


def _producer_subtree(g: Graph, rng: random.Random, value_types: str, depth: int,
                      exclude: str) -> Node:
    """建一个输出属于 value_types 的上游；depth > 1 时上游自己也可以取引用。
    同一张图里不重复用同一个工具，也不用 exclude（汇聚的终点）。"""
    types = value_types.split("|")
    options = [p for p in CATALOG.values() if p.produces in types and literal_ok(p.name)
               and p.name not in g.tools() and p.name != exclude]
    if not options:
        raise Retry
    spec = rng.choice(options)
    fixed: dict[str, Value | Ref] = {}
    if depth > 1:
        refable = [p for p in spec.accepts]
        if refable and rng.random() < 0.6:
            param = rng.choice(refable)
            fixed[param] = Ref(_producer_subtree(g, rng, spec.accepts[param], depth - 1,
                                                 exclude).id)
    return g.add(spec.name, literal_args(spec.name, rng, fixed))


def build_fan_in(rng: random.Random) -> Graph:
    """一个终点同时引用两个以上互不相干的上游。"""
    g = Graph()
    sink = rng.choice(FAN_IN_SINKS)
    params = list(sink.accepts)
    fixed: dict[str, Value | Ref] = {}
    for param in params:
        fixed[param] = Ref(_producer_subtree(g, rng, sink.accepts[param], rng.choice((1, 2)),
                                             sink.name).id)
    g.add(sink.name, literal_args(sink.name, rng, fixed))
    return g


# 一个上游的输出分给两个以上下游。成对出现会显得别扭的下游不同时选
_FAN_OUT_CONFLICTS = {frozenset({"system.navigate", "system.call_taxi"})}


def _consumers_of(value_type: str) -> list[tuple[str, str]]:
    return [(c.name, param) for c in CATALOG.values() if c.name not in EXCLUDED_FROM_NORMAL
            for param in c.accepts if accepts(c, param, value_type)]


def _fan_out_options(producer: Spec, width: int) -> tuple[dict[str, str], list[tuple[str, ...]]]:
    """这个上游的各个下游（工具 → 接收引用的参数），以及可选的下游组合。"""
    by_tool: dict[str, str] = {}
    for c, param in _consumers_of(producer.produces or ""):
        if c != producer.name:
            by_tool.setdefault(c, param)
    combos = [combo for combo in combinations(sorted(by_tool), width)
              if not any(frozenset(pair) in _FAN_OUT_CONFLICTS for pair in combinations(combo, 2))]
    return by_tool, combos


def build_fan_out(rng: random.Random, width: int) -> Graph:
    g = Graph()
    producers = [p for p in CATALOG.values() if p.produces and literal_ok(p.name)
                 and _fan_out_options(p, width)[1]]
    producer = rng.choice(producers)
    root = g.add(producer.name, literal_args(producer.name, rng))
    by_tool, combos = _fan_out_options(producer, width)
    chosen = rng.choice(combos)
    for c in chosen:
        g.add(c, literal_args(c, rng, {by_tool[c]: Ref(root.id)}))
    return g


def build_diamond(rng: random.Random) -> Graph:
    """A → (B ‖ C) → D。D 引用 B、C 之一的结果并在另一个之后执行，
    或不取数据、在两者都完成后执行。"""
    g = build_fan_out(rng, 2)
    b, c = list(g.nodes.values())[1:3]
    options = []
    for x, y in ((b, c), (c, b)):
        if x.spec.produces:
            options += [(x, y, name, param) for name, param in _consumers_of(x.spec.produces)
                        if name not in g.tools()]
    if options and rng.random() < 0.6:
        x, y, name, param = rng.choice(options)
        g.add(name, literal_args(name, rng, {param: Ref(x.id)}), after=[y.id])
    else:
        side = [t for t in CATALOG if CATALOG[t].side_effect and literal_ok(t)
                and t not in g.tools()]
        tool = rng.choice(side)
        g.add(tool, literal_args(tool, rng), after=[b.id, c.id])
    return g


def build_mixed(rng: random.Random, pool: list[str]) -> Graph:
    """一个有依赖的结构，外加一到三个与之无关的独立任务。"""
    base = rng.choice(("chain", "fan_in", "fan_out", "diamond"))
    g = {"chain": lambda: build_chain(rng, rng.choice((3, 4))),
         "fan_in": lambda: build_fan_in(rng),
         "fan_out": lambda: build_fan_out(rng, 2),
         "diamond": lambda: build_diamond(rng)}[base]()
    free = [t for t in pool if t not in g.tools()]
    extra = rng.sample(free, rng.choice((1, 2, 3)))
    at_front = rng.random() < 0.5
    if at_front:
        # 独立任务放在句首：重新编号，保持「编号先后即拓扑序」
        h = Graph()
        for tool in extra:
            h.add(tool, literal_args(tool, rng))
        shift = {n.id: f"t{len(extra) + i}" for i, n in enumerate(g.nodes.values(), 1)}
        for n in g.nodes.values():
            args = {p: Ref(shift[v.task]) if isinstance(v, Ref) else v for p, v in n.args.items()}
            h.add(n.tool, args, after=[shift[a] for a in n.after])
        return h
    for tool in extra:
        g.add(tool, literal_args(tool, rng))
    return g


CHAINS = {n: chain_paths(n) for n in (2, 3, 4)}


# ---------------- 形状 ----------------


def shape(g: Graph) -> dict[str, Any]:
    """结构特征：任务数、层数、各类边数、最大扇入扇出、最宽一层。"""
    nodes = g.nodes
    layers: list[list[str]] = []
    done: set[str] = set()
    remaining = dict(nodes)
    while remaining:
        layer = [t for t, n in remaining.items() if set(n.dependencies()) <= done]
        layers.append(layer)
        done |= set(layer)
        remaining = {t: n for t, n in remaining.items() if t not in done}
    fan_in = max((len(n.dependencies()) for n in nodes.values()), default=0)
    fan_out = max((len(g.children(t)) for t in nodes), default=0)
    return {"tasks": len(nodes), "depth": len(layers),
            "width": max((len(layer) for layer in layers), default=0),
            "data_edges": sum(len(set(n.refs().values())) for n in nodes.values()),
            "order_edges": sum(len(n.after) for n in nodes.values()),
            "max_fan_in": fan_in, "max_fan_out": fan_out}


def classify(g: Graph) -> str:
    """按结构特征判定类别。与构造时的意图对照，二者不一致即是生成器的错误。"""
    s = shape(g)
    if s["tasks"] == 0:
        return "empty"
    if s["tasks"] == 1:
        return "single"
    edges = s["data_edges"] + s["order_edges"]
    if edges == 0:
        return "parallel"
    connected = {t for n in g.nodes.values() for t in [n.id, *n.dependencies()]
                 if n.dependencies() or g.children(n.id)}
    isolated = len(g.nodes) - len(connected)
    if isolated:
        return "mixed"
    if s["max_fan_in"] <= 1 and s["max_fan_out"] <= 1:
        return "chain"
    if s["max_fan_in"] >= 2 and s["max_fan_out"] >= 2 and _has_diamond(g):
        return "diamond"
    if s["max_fan_out"] >= 2 and s["max_fan_in"] <= 1:
        return "fan_out"
    return "fan_in"


def _has_diamond(g: Graph) -> bool:
    """存在某个节点经两条不同路径到达另一个节点。"""
    def reach(t: str) -> set[str]:
        out: set[str] = set()
        stack = g.children(t)
        while stack:
            c = stack.pop()
            if c not in out:
                out.add(c)
                stack += g.children(c)
        return out
    for t in g.nodes:
        kids = g.children(t)
        seen: set[str] = set()
        for k in kids:
            r = reach(k) | {k}
            if seen & r:
                return True
            seen |= r
    return False


# ---------------- 渲染 ----------------

_OPTIONAL = re.compile(r"\{\?(\w+):([^}]*)\}")


def _fill(template: str, surfaces: dict[str, str]) -> str:
    text = _OPTIONAL.sub(lambda m: m.group(2).replace("…", surfaces[m.group(1)])
                         if m.group(1) in surfaces else "", template)
    return text.format(**surfaces)


def reference(node: Node, rng: random.Random) -> str:
    """下游提到这个节点的结果时的说法。"""
    surfaces = {p: v.surface for p, v in node.args.items() if isinstance(v, Value)}
    return _fill(rng.choice(REFERENCES[node.tool]), surfaces)


def phrase(g: Graph, node: Node, rng: random.Random) -> str:
    """一个节点的说法：引用处填上游结果的指代。"""
    templates = PHRASES[node.tool]
    refs = frozenset(p for p, v in node.args.items()
                     if isinstance(v, Ref) or (isinstance(v, Value) and v.referred))
    options = templates.get(refs)
    if options is None:
        raise KeyError(f"{node.tool} 没有引用参数为 {sorted(refs)} 的说法模板")
    surfaces = {p: (reference(g.nodes[v.task], rng) if isinstance(v, Ref) else v.surface)
                for p, v in node.args.items()}
    return _fill(rng.choice(options), surfaces) + node.clause


def render(g: Graph, rng: random.Random, *, prefix: bool = True,
           suffix: bool = True) -> tuple[str, dict[str, str]]:
    """整句话，以及每个节点自己那一段说法（用作任务描述）。
    prefix / suffix 控制句首的称呼与句尾的语气词。"""
    parts: list[str] = []
    pieces: dict[str, str] = {}
    prev: Node | None = None
    for node in g.nodes.values():
        piece = phrase(g, node, rng)
        pieces[node.id] = piece
        if prev is None:
            lead = rng.choice(PREFIXES) if prefix and not piece.startswith("帮我") else ""
            parts.append(lead + piece)
        elif node.after:
            parts.append(rng.choice(JOIN_AFTER_ALL) + "，" + piece)
        elif prev.id in node.dependencies():
            parts.append(rng.choice(JOIN_SEQUENTIAL) + piece)
        elif node.dependencies():
            parts.append("，" + piece)
        else:
            parts.append(rng.choice(JOIN_PARALLEL) + piece)
        prev = node
    return "".join(parts) + (rng.choice(SUFFIXES) if suffix else ""), pieces
