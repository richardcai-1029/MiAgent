"""评测数据集生成器。

    python -m bench.suite.generate                 # 默认 10000 条，写到 bench/data/
    python -m bench.suite.generate -n 500 --seed 7 --out /tmp/x

同一个 seed 与条数永远生成逐字节相同的文件。

一条用例是一次对话（一轮或多轮），每轮带：

    request      用户原话
    gold         首次规划的标准任务图（工具、参数、依赖）
    fallback     有故障且可恢复时，接手失败部分的标准任务图；各任务的 replaces
                 指明它接替的是主计划里的哪个任务
    accept       参数的其他可接受写法（口语换算类参数）
    faults       注入的故障，按 (工具, 实际参数) 匹配
    expected     参考执行的结果：各调用的记录、应发生的副作用、目标能否达成
    forbidden    不应发生的调用（arguments 为 null 表示该工具任何参数都不应调用）
    simulate     仅供标注答案驱动的评测：模拟模型的非理想输出，不改变标准答案

用例级字段：类别（结构）、标签（叠加的故障、模拟与特殊情形）、难度等级、
结构特征、会话标识与优先级、授予的权限、端侧配额。

类别与配比见 MIX；难度等级的规则见 level()。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import execute
from .catalog import ALL_PERMISSIONS, CATALOG, DEFAULT_BUDGET, DINING
from .compose import (EXCLUDED_FROM_NORMAL, Graph, Node, Ref, Retry, build_chain, build_diamond,
                      build_fan_in, build_fan_out, build_mixed, build_parallel, build_single,
                      classify, literal_args, literal_ok, render, shape)
from .phrases import PREFIXES, PRIOR, RESTAURANTS, SUFFIXES, Value, sample

VERSION = "1.0"
DEFAULT_N = 10000
DEFAULT_SEED = 20260928

# 各类别占比。单任务与无依赖并行之外的结构类别合计过半：数据集的重点是
# 有数据流与依赖的多步任务。
MIX: dict[str, float] = {
    "single": 0.14, "parallel": 0.14, "chain": 0.18, "fan_in": 0.09, "fan_out": 0.07,
    "diamond": 0.07, "mixed": 0.11, "multi_turn": 0.10, "no_tool": 0.045,
    "unsupported": 0.025, "permission_denied": 0.03,
}
# 结构类别里叠加故障与模拟偏差的比例
FAULT_RATE = 0.35
SIMULATE_RATE = 0.30

# 普通结构用的工具：单任务与并行可以用全部能只凭用户原话调用的工具
POOL = sorted(t for t in CATALOG if literal_ok(t) or t in EXCLUDED_FROM_NORMAL)

UNSUPPORTED = ("订一张明天去上海的机票", "帮我点一份麻辣烫外卖", "把客厅空调调到 26 度",
               "查一下我的社保余额", "把这张照片修一下", "给张伟转 200 块钱",
               "预约明天下午的理发", "打开扫地机器人", "帮我挂一个协和医院的号",
               "查一下我的信用卡账单", "把这段视频剪辑一下", "买两张今晚的电影票",
               "把卧室的灯关掉", "帮我叫个跑腿送文件", "查一下我的公积金")
CHITCHAT = ("你好", "谢谢", "你叫什么名字", "给我讲个笑话", "今天心情不太好", "晚安",
            "你能做什么", "长城有多长", "推荐几本适合通勤看的书", "光速是多少",
            "怎么煮溏心蛋", "一年有多少周", "帮我想个团建的点子", "什么是量子计算",
            "早上好", "你真棒", "我有点无聊", "水的沸点是多少度", "给宝宝起个名字吧",
            "怎么缓解眼睛疲劳", "说一句鼓励我的话", "中国有多少个省份", "什么是番茄工作法",
            "唐诗三百首里最有名的是哪首", "跑步前要热身吗")
ARITH = ("{a} 加 {b} 等于多少", "{a} 乘以 {b} 是多少", "{a} 减 {b} 得几", "{a} 除以 {b} 等于几")


@dataclass
class Turn:
    graph: Graph
    request: str
    pieces: dict[str, str]
    fallback: list[dict[str, Any]] | None = None
    faults: list[dict[str, Any]] = field(default_factory=list)
    forbidden: list[dict[str, Any]] = field(default_factory=list)
    simulate: dict[str, Any] | None = None
    omitted: set[str] = field(default_factory=set)       # 不可行、不进标准答案的节点
    infeasible: bool = False                              # 句中有做不成的部分


# ---------------- 节点 → 标准答案 ----------------


def _arg(v: Value | Ref) -> Any:
    return {execute.REF: v.task} if isinstance(v, Ref) else v.value


def gold_task(node: Node, description: str) -> dict[str, Any]:
    return {"id": node.id, "description": description, "required_tool": node.tool,
            "dependencies": node.dependencies(),
            "arguments": {p: _arg(v) for p, v in node.args.items()}}


def gold_of(turn: Turn) -> list[dict[str, Any]]:
    return [gold_task(n, turn.pieces[n.id]) for n in turn.graph.nodes.values()
            if n.id not in turn.omitted]


def accept_of(turn: Turn) -> dict[str, dict[str, list[Any]]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for n in turn.graph.nodes.values():
        for p, v in n.args.items():
            if isinstance(v, Value) and v.accept:
                out.setdefault(n.id, {})[p] = [v.value, *v.accept]
    return out


# ---------------- 故障 ----------------


def _downstream(g: Graph, tid: str) -> list[str]:
    out: list[str] = []
    stack = [tid]
    while stack:
        for c in g.children(stack.pop()):
            if c not in out:
                out.append(c)
                stack.append(c)
    return sorted(out, key=lambda t: list(g.nodes).index(t))


def _resolved_args(turn: Turn, node: Node, budget: dict[str, Any], perms: list[str]) -> dict[str, Any] | None:
    """这个节点在参考执行里实际收到的参数（上游全部成功的前提下）。"""
    rec = execute.run(gold_of(turn), faults=[], budget=budget, permissions=perms)[node.id]
    return rec["arguments"]


def _fallback(turn: Turn, failed: Node, replacement: tuple[str, dict[str, Value | Ref]]) -> list[dict[str, Any]]:
    """接替失败节点及其全部下游：替换节点换工具或参数，下游照原样重建并改指新节点。"""
    g = turn.graph
    tool, args = replacement
    ids = {failed.id: "f1"}
    for i, tid in enumerate(_downstream(g, failed.id), 2):
        ids[tid] = f"f{i}"
    out = []
    for old, new in ids.items():
        node = g.nodes[old]
        use_tool, use_args = (tool, args) if old == failed.id else (node.tool, node.args)
        remapped = {p: Ref(ids.get(v.task, v.task)) if isinstance(v, Ref) else v
                    for p, v in use_args.items()}
        clone = Node(new, use_tool, remapped, [ids.get(a, a) for a in node.after])
        task = gold_task(clone, turn.pieces[old] if old != failed.id else _alt_description(clone))
        task["replaces"] = old
        out.append(task)
    return out


def _alt_description(node: Node) -> str:
    if node.tool == "system.navigate":
        return "叫不到车，改为导航前往"
    return f"改订{node.args['name'].surface}" if node.tool == "system.book_restaurant" else node.tool


def add_fault(turn: Turn, rng: random.Random, budget: dict[str, Any], perms: list[str]) -> str | None:
    """给一轮叠加一种故障，返回故障类别；没有适用的节点时返回 None。"""
    g = turn.graph
    nodes = list(g.nodes.values())
    kinds = ["transient_first", "transient_all", "memory_limit"]
    bookable = [n for n in nodes if n.tool == "system.book_restaurant" and isinstance(n.args.get("name"), Value)]
    taxis = [n for n in nodes if n.tool == "system.call_taxi"]
    orders = [n for n in nodes if n.tool in ("system.query_order", "system.track_package")
              and isinstance(n.args.get("order_id"), Value)]
    if bookable or taxis:
        kinds += ["busy_recoverable", "busy_unrecoverable"]
    if orders:
        kinds.append("business_empty")
    kind = rng.choice(kinds)

    if kind in ("transient_first", "transient_all"):
        node = rng.choice(nodes)
        args = _resolved_args(turn, node, budget, perms)
        turn.faults.append({"tool": node.tool, "arguments": args, "kind": kind})
    elif kind == "memory_limit":
        # 端侧内存紧张时握手下发的配额更低：取比目标工具开销少 1 MB 的配额，
        # 开销不低于它的工具都会触发 MC-4001。
        node = rng.choice([n for n in nodes if CATALOG[n.tool].tool.estimated_memory_mb > 4] or nodes)
        budget["max_memory_mb"] = CATALOG[node.tool].tool.estimated_memory_mb - 1
    elif kind.startswith("busy"):
        node = rng.choice(bookable + taxis)
        args = _resolved_args(turn, node, budget, perms)
        turn.faults.append({"tool": node.tool, "arguments": args, "kind": "busy"})
        if kind == "busy_recoverable":
            if node.tool == "system.book_restaurant":
                alt = rng.choice([r for r in RESTAURANTS if r != node.args["name"].value])
                node.clause = rng.choice((f"，订不上就换{alt}", f"，要是满了就订{alt}"))
                repl = (node.tool, {**node.args, "name": Value(alt, alt)})
            else:
                node.clause = rng.choice(("，叫不到车就直接导航过去", "，没车的话就开导航"))
                repl = ("system.navigate", {"destination": node.args["destination"]})
            turn.fallback = _fallback(turn, node, repl)
    else:  # business_empty：用户报了一个不存在的单号
        node = rng.choice(orders)
        bad = "SN" + str(rng.randrange(10**5, 10**6))
        node.args["order_id"] = Value(bad, bad)
    return kind


# ---------------- 模拟模型偏差（只影响标注答案驱动的评测）----------------

NOISE = ("fence", "prose", "trailing_comma", "invalid_once", "stringified_ref_once")


def add_simulation(turn: Turn, rng: random.Random) -> list[str]:
    """planner_noise：规划输出的格式偏差；arg_drift：规划把某个参数写偏，由结果校验
    改正；omit_step：规划漏掉一步，由完成校验指出后补做。后两者只加在无故障的轮次。

    arg_drift 只挑无副作用的调用：写偏的调用会被真实执行，副作用无法由校验撤回。
    """
    g = turn.graph
    sim: dict[str, Any] = {}
    tags: list[str] = []
    has_refs = any(n.refs() for n in g.nodes.values())
    sim["planner_noise"] = rng.choice(NOISE if has_refs else NOISE[:-1])
    tags.append(f"noise:{sim['planner_noise']}")
    if not turn.faults and not turn.fallback:
        roll = rng.random()
        drift = [(n, p) for n in g.nodes.values() if not n.spec.side_effect
                 for p, v in n.args.items() if isinstance(v, Value) and isinstance(v.value, str)]
        leaves = [n for n in g.nodes.values() if not g.children(n.id)]
        if roll < 0.4 and drift:
            node, param = rng.choice(drift)
            for _ in range(20):
                other = sample(node.tool, param, rng).value
                if other != node.args[param].value:
                    sim["arg_drift"] = {"task": node.id, "param": param, "value": other}
                    tags.append("sim:arg_drift")
                    break
        elif roll < 0.7 and len(g.nodes) >= 2 and leaves:
            sim["omit_step"] = {"task": rng.choice(leaves).id}
            tags.append("sim:omit_step")
    turn.simulate = sim
    return tags


# ---------------- 各类别 ----------------


def _plain_turn(g: Graph, rng: random.Random) -> Turn:
    request, pieces = render(g, rng)
    return Turn(g, request, pieces)


def build_structure(category: str, rng: random.Random) -> Graph:
    """按类别构造一张任务图，直到结构判定与类别一致。"""
    while True:
        try:
            g = _build(category, rng)
        except Retry:
            continue
        if classify(g) == category:
            g.harmonize(rng)
            return g


def _build(category: str, rng: random.Random) -> Graph:
    normal = [t for t in POOL if t not in EXCLUDED_FROM_NORMAL]
    return {
        "single": lambda: build_single(rng, POOL),
        "parallel": lambda: build_parallel(rng, POOL, rng.choice((2, 2, 3, 3, 4))),
        "chain": lambda: build_chain(rng, rng.choice((2, 2, 3, 3, 4))),
        "fan_in": lambda: build_fan_in(rng),
        "fan_out": lambda: build_fan_out(rng, rng.choice((2, 2, 3))),
        "diamond": lambda: build_diamond(rng),
        "mixed": lambda: build_mixed(rng, normal),
    }[category]()


def structural_case(category: str, rng: random.Random) -> tuple[list[Turn], list[str], dict[str, Any], list[str]]:
    turn = _plain_turn(build_structure(category, rng), rng)
    budget = dict(DEFAULT_BUDGET, max_concurrent_calls=rng.choice((1, 2, 2, 2, 3, 4)))
    perms = list(ALL_PERMISSIONS)
    tags: list[str] = []
    if rng.random() < FAULT_RATE:
        kind = add_fault(turn, rng, budget, perms)
        if kind:
            tags.append(f"fault:{kind}")
            turn.request, turn.pieces = _rerender(turn, rng)
    if rng.random() < SIMULATE_RATE:
        tags += add_simulation(turn, rng)
    return [turn], tags, budget, perms


def _rerender(turn: Turn, rng: random.Random) -> tuple[str, dict[str, str]]:
    """故障改动了参数或补充了备选方案的说法，重新渲染；备选计划的描述随之更新。"""
    request, pieces = render(turn.graph, rng)
    if turn.fallback:
        for task in turn.fallback:
            if task["replaces"] in pieces and task["id"] != "f1":
                task["description"] = pieces[task["replaces"]]
    return request, pieces


def no_tool_case(rng: random.Random) -> tuple[list[Turn], list[str], dict[str, Any], list[str]]:
    if rng.random() < 0.5:
        text = rng.choice(CHITCHAT)
    else:
        a, b = rng.randrange(2, 1000), rng.randrange(2, 100)
        text = rng.choice(ARITH).format(a=a, b=b)
    request = rng.choice(PREFIXES[:3] + ("",)) + text + rng.choice(SUFFIXES)
    turn = Turn(Graph(), request, {})
    return [turn], [], dict(DEFAULT_BUDGET), list(ALL_PERMISSIONS)


def unsupported_case(rng: random.Random) -> tuple[list[Turn], list[str], dict[str, Any], list[str]]:
    """句中有目录里没有的能力。能做的部分照做，整体判为不可达成。"""
    ask = rng.choice(UNSUPPORTED)
    if rng.random() < 0.4:
        g = Graph()
        request = rng.choice(PREFIXES) + ask + rng.choice(SUFFIXES)
        turn = Turn(g, request, {})
    else:
        normal = [t for t in POOL if t not in EXCLUDED_FROM_NORMAL]
        g = build_single(rng, normal) if rng.random() < 0.6 else build_parallel(rng, normal, 2)
        request, pieces = render(g, rng, suffix=False)
        turn = Turn(g, request + rng.choice(("，另外", "，还有", "，再")) + ask, pieces)
    turn.infeasible = True
    return [turn], ["unsupported"], dict(DEFAULT_BUDGET), list(ALL_PERMISSIONS)


def permission_case(rng: random.Random) -> tuple[list[Turn], list[str], dict[str, Any], list[str]]:
    """用户在授权弹窗里拒绝了某项权限：需要它的工具不可见，这部分做不成。"""
    kind = rng.choice(("single", "parallel", "chain"))
    while True:
        g = build_structure(kind, rng)
        perms_needed = {n.spec.tool.required_permission for n in g.nodes.values()} - {None}
        if perms_needed:
            break
    denied = rng.choice(sorted(perms_needed))
    perms = [p for p in ALL_PERMISSIONS if p != denied]
    turn = _plain_turn(g, rng)
    blocked = [n.id for n in g.nodes.values() if n.spec.tool.required_permission == denied]
    for tid in list(blocked):
        blocked += [d for d in _downstream(g, tid) if d not in blocked]
    turn.omitted = set(blocked)
    turn.infeasible = True
    hidden = sorted(t for t, s in CATALOG.items() if s.tool.required_permission == denied)
    turn.forbidden = [{"tool": t, "arguments": None} for t in hidden]
    return [turn], [f"denied:{denied}", f"shape:{kind}"], dict(DEFAULT_BUDGET), perms


def multi_turn_case(rng: random.Random) -> tuple[list[Turn], list[str], dict[str, Any], list[str]]:
    """后一轮承接上一轮的结论：用上一轮查到的值，且不重查。"""
    budget, perms = dict(DEFAULT_BUDGET), list(ALL_PERMISSIONS)
    carriers = [p for p in CATALOG.values() if p.produces in PRIOR and literal_ok(p.name)
                and _consumers(p.produces)]
    while True:
        first = Graph()
        spec = rng.choice(carriers)
        # 第一轮：一个产出值的查询
        node = first.add(spec.name, literal_args(spec.name, rng))
        t1 = _plain_turn(first, rng)
        rec = execute.run(gold_of(t1), faults=[], budget=budget, permissions=perms)[node.id]
        if rec["status"] == "done":
            break
    value = rec["result"]
    turns = [t1]
    used = {node.tool}
    n_turns = rng.choice((2, 2, 3))
    for _ in range(n_turns - 1):
        options = [(c, p) for c, p in _consumers(spec.produces) if c not in used
                   and not (c == "system.book_restaurant" and node.tool == "system.recommend_place"
                            and node.args["category"].value not in DINING)]
        if not options or rng.random() < 0.2:
            # 话题切换：与上文无关的新请求，不应带入上文的值
            g = build_single(rng, [t for t in POOL if t not in EXCLUDED_FROM_NORMAL and t not in used])
        else:
            tool, param = rng.choice(options)
            g = Graph()
            g.add(tool, literal_args(tool, rng, {param: Value(value, rng.choice(PRIOR[spec.produces]), referred=True)}))
        request, pieces = render(g, rng, prefix=False)
        turn = Turn(g, rng.choice(("那", "好的，", "", "嗯，", "那就")) + request, pieces)
        # 上一轮做过的调用不应重做
        turn.forbidden = [{"tool": n.tool, "arguments": execute.run(gold_of(t), faults=[], budget=budget,
                                                                     permissions=perms)[n.id]["arguments"]}
                          for t in turns for n in t.graph.nodes.values()]
        turns.append(turn)
        used |= g.tools()
    return turns, [f"turns:{len(turns)}"], budget, perms


def _consumers(value_type: str) -> list[tuple[str, str]]:
    return [(c.name, p) for c in CATALOG.values() if c.name not in EXCLUDED_FROM_NORMAL
            for p in c.accepts if value_type in c.accepts[p].split("|")]


BUILDERS = {
    "no_tool": no_tool_case, "unsupported": unsupported_case,
    "permission_denied": permission_case, "multi_turn": multi_turn_case,
}


# ---------------- 组装 ----------------


def level(shapes: list[dict[str, Any]], tags: list[str]) -> int:
    """难度等级 1~5。分数 = (任务数-1) + (层数-1)，按各轮累加，
    多一轮加 2，叠加故障加 2，叠加模拟偏差加 1；等级 = 1 + min(4, 分数 // 2)。"""
    score = sum(max(s["tasks"] - 1, 0) + max(s["depth"] - 1, 0) for s in shapes)
    score += 2 * (len(shapes) - 1)
    score += 2 * any(t.startswith("fault:") for t in tags)
    score += any(t.startswith(("noise:", "sim:")) for t in tags)
    return 1 + min(4, score // 2)


def turn_record(turn: Turn, budget: dict[str, Any], perms: list[str]) -> dict[str, Any]:
    gold = gold_of(turn)
    out = execute.outcome(gold, turn.fallback, faults=turn.faults, budget=budget, permissions=perms)
    achievable = out["achievable"] and not turn.infeasible
    return {
        "request": turn.request,
        "gold": {"tasks": gold},
        "fallback": {"tasks": turn.fallback} if turn.fallback else None,
        "accept": accept_of(turn),
        "faults": turn.faults,
        "expected": {"achievable": achievable, "calls": out["calls"], "effects": out["effects"]},
        "forbidden": turn.forbidden,
        "simulate": turn.simulate,
    }


def make_case(index: int, category: str, rng: random.Random) -> dict[str, Any]:
    if category in BUILDERS:
        turns, tags, budget, perms = BUILDERS[category](rng)
    else:
        turns, tags, budget, perms = structural_case(category, rng)
    records = [turn_record(t, budget, perms) for t in turns]
    # 结构特征按标准答案计：权限类用例里不可行的节点不在其中
    shapes = [shape(_gold_graph(t)) for t in turns]
    return {
        "id": f"mb-{index:05d}",
        "category": category,
        "tags": sorted(set(tags)),
        "level": level(shapes, tags),
        "shape": shapes[0],
        "session": f"conv-{index:05d}",
        "priority": "foreground" if rng.random() < 0.7 else "background",
        "permissions": perms,
        "budget": budget,
        "turns": records,
    }


def _gold_graph(turn: Turn) -> Graph:
    g = Graph()
    for n in turn.graph.nodes.values():
        if n.id not in turn.omitted:
            g.nodes[n.id] = n
    return g


def plan_counts(n: int) -> dict[str, int]:
    """按 MIX 把 n 分到各类别，舍入误差补给最大的类别。"""
    counts = {c: int(n * w) for c, w in MIX.items()}
    counts[max(MIX, key=MIX.get)] += n - sum(counts.values())
    return counts


def generate(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    order = [c for c, k in plan_counts(n).items() for _ in range(k)]
    rng.shuffle(order)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, category in enumerate(order, 1):
        for _ in range(200):
            case = make_case(i, category, rng)
            key = case["turns"][0]["request"]
            if key not in seen:
                seen.add(key)
                cases.append(case)
                break
        else:
            raise RuntimeError(f"类别 {category} 生成不出不重复的请求")
    return cases


def manifest(cases: list[dict[str, Any]], seed: int, digest: str) -> dict[str, Any]:
    def count(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in cases:
            for k in key(c):
                out[str(k)] = out.get(str(k), 0) + 1
        return dict(sorted(out.items()))

    turns = [t for c in cases for t in c["turns"]]
    tasks = [task for t in turns for task in t["gold"]["tasks"]]
    return {
        "version": VERSION, "seed": seed, "cases": len(cases), "turns": len(turns),
        "gold_tasks": len(tasks), "sha256": digest,
        "by_category": count(lambda c: [c["category"]]),
        "by_level": count(lambda c: [c["level"]]),
        "by_tag": count(lambda c: c["tags"]),
        "by_task_count": count(lambda c: [c["shape"]["tasks"]]),
        "by_depth": count(lambda c: [c["shape"]["depth"]]),
        "achievable_turns": sum(t["expected"]["achievable"] for t in turns),
        "turns_with_fallback": sum(t["fallback"] is not None for t in turns),
        "data_edges": sum(1 for task in tasks for v in task["arguments"].values()
                          if isinstance(v, dict) and execute.REF in v),
        "tool_usage": dict(sorted({t: sum(1 for task in tasks if task["required_tool"] == t)
                                   for t in CATALOG}.items())),
        "catalog_size": len(CATALOG),
    }


def write(cases: list[dict[str, Any]], out: Path, seed: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)
    path = out / "cases.jsonl"
    path.write_text(data, encoding="utf-8")
    meta = manifest(cases, seed, hashlib.sha256(data.encode()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    meta = write(generate(args.n, args.seed), args.out, args.seed)
    print(json.dumps({k: meta[k] for k in ("cases", "turns", "gold_tasks", "sha256")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
