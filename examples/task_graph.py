"""任务 DAG 版 Agent 演示：线性依赖、并行机会、失败重规划、非法图。

运行：  python examples/task_graph.py
"""

import json

from miagent.client import MiClawClient
from miagent.graph import build_agent, dag, initial_state
from miagent.llm import FakeLLM
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "location.fine"]


def task(tid, desc, tool, deps=None, **args):
    return {"id": tid, "description": desc, "required_tool": tool,
            "dependencies": deps or [], "arguments": args}


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def make_llm(first, replan=None, answer="完成。"):
    def responder(msgs):
        role = msgs[0].content
        if "重规划器" in role:
            return replan or plan()
        if "规划器" in role:
            return first
        # 收尾是结构化输出：回答给用户，摘要给下一轮
        return json.dumps({"answer": answer, "summary": answer}, ensure_ascii=False)
    return FakeLLM(responder=responder)


def run(title, request, llm, registry):
    print(f"\n{'━' * 78}\n{title}\n{'━' * 78}")
    print(f"  用户：{request}\n")
    out = build_agent(llm, registry).invoke(initial_state(request),
                                            {"recursion_limit": 60})
    for line in out["trace"]:
        print(f"    {line}")
    s = out["execution_summary"]
    print(f"\n  任务状态：完成 {s.get('completed')} / 失败 {s.get('failed')}")
    print(f"  回答：{out['final_answer']}")
    if out["failure"]:
        print(f"  失败码：{out['failure']}")
    return out


client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
client.connect("com.xiaomi.miagent.demo", PERMS)
registry = ToolRegistry(); registry.load_from_miclaw(client)

# ---------- ① 线性依赖 A → B → C ----------
run("① 线性依赖：查日历 → 找餐厅 → 建日程（你给的例子）",
    "查看今晚有没有空，如果有空就找一家附近餐厅并安排到日历。",
    make_llm(plan(
        task("task_1", "查询今晚日历是否空闲", "system.query_calendar", when="今晚"),
        task("task_2", "搜索附近餐厅", "system.search_nearby", ["task_1"], category="餐厅"),
        task("task_3", "创建日程", "system.create_event", ["task_2"],
             title="晚餐", when="今晚 19:00"),
    ), answer="今晚 19:00-22:00 空闲，已为你找到附近餐厅并创建了晚餐日程。"),
    registry)

# ---------- ② 菱形依赖：并行机会 ----------
run("② 菱形依赖：A →(B 天气 ‖ C 餐厅)→ D  —— 第二层可并行",
    "看今晚有没有空，同时查天气和餐厅，然后建日程。",
    make_llm(plan(
        task("task_1", "查日历", "system.query_calendar", when="今晚"),
        task("task_2", "查天气", "system.query_weather", ["task_1"], when="今晚"),
        task("task_3", "找餐厅", "system.search_nearby", ["task_1"], category="餐厅"),
        task("task_4", "建日程", "system.create_event", ["task_2", "task_3"],
             title="晚餐", when="今晚 19:00"),
    ), answer="今晚空闲、天气晴，已找到餐厅并创建日程。"),
    registry)

# ---------- ③ 失败 → 重规划，保留已完成任务 ----------
run("③ 餐厅订满（MC-4002）→ 重规划换一家，已完成的任务不重做",
    "订今晚的餐厅并加到日历。",
    make_llm(
        plan(task("task_1", "查日历", "system.query_calendar", when="今晚"),
             task("task_2", "订小馆 A", "system.book_restaurant", ["task_1"], name="小馆 A"),
             task("task_3", "建日程", "system.create_event", ["task_2"],
                  title="晚餐", when="今晚 19:00")),
        replan=plan(task("task_2b", "改订小馆 B", "system.book_restaurant", name="小馆 B"),
                    task("task_3b", "建日程", "system.create_event", ["task_2b"],
                         title="晚餐", when="今晚 19:00")),
        answer="小馆 A 已订满，已改订小馆 B 并创建了日程。"),
    registry)

# ---------- ④ 非法任务图 ----------
run("④ 模型产出了带环的任务图 → AG-1004，一步都不执行",
    "做一件会绕圈的事",
    make_llm(plan(
        task("task_1", "第一步", "system.query_weather", ["task_2"], when="今晚"),
        task("task_2", "第二步", "system.query_calendar", ["task_1"], when="今晚"),
    ), answer="抱歉，我没能规划出可执行的步骤。"),
    registry)

client.close()
