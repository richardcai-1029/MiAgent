"""自主校验演示：工具说成功了，做的是不是要做的事？

三个场景：
  ① 参数偏差 —— 查错了单号，校验给出正确单号，同一个任务带新参数重试
  ② 结果不符 —— 结果确实不对，但修不出参数，如实转重规划，不宣称完成
  ③ 计划漏做 —— 每个任务都成功，合起来却没达成目标，完成校验判未达成并补做

FakeLLM 在这里同时扮演规划器与校验器：校验器的回复是写死的，
这样演示的是【框架怎么处理校验结论】，而不是模型判得准不准。

运行：  python examples/self_check.py
"""

import json

from miagent.client import MiClawClient
from miagent.graph import build_agent, initial_state
from miagent.llm import FakeLLM
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "location.fine"]


def task(tid, desc, tool, **args):
    return {"id": tid, "description": desc, "required_tool": tool,
            "dependencies": [], "arguments": args}


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def reviews(*items):
    """结果校验的判定。不给判定即全部通过。"""
    return json.dumps({"reviews": list(items)}, ensure_ascii=False)


def reject(task_id, reason, corrected=None):
    return {"task_id": task_id, "passed": False, "reason": reason,
            "corrected_arguments": corrected}


def goal(achieved=True, gap=""):
    return json.dumps({"achieved": achieved, "gap": gap}, ensure_ascii=False)


def make_llm(first, replan=None, checked=None, done=None, answer="完成。"):
    checked = list(checked or [])
    done = list(done or [])

    def responder(msgs):
        role = msgs[0].content
        if "结果校验器" in role:
            return checked.pop(0) if checked else reviews()
        if "完成校验器" in role:
            return done.pop(0) if done else goal()
        if "重规划器" in role:
            return replan or plan()
        if "规划器" in role:
            return first
        return json.dumps({"answer": answer, "summary": answer}, ensure_ascii=False)

    return FakeLLM(responder=responder)


def run(title, request, llm, registry):
    print(f"\n{'━' * 78}\n{title}\n{'━' * 78}")
    print(f"  用户：{request}\n")
    out = build_agent(llm, registry, retry_delay_ms=0).invoke(
        initial_state(request), {"recursion_limit": 60})
    for line in out["trace"]:
        print(f"    {line}")
    s = out["execution_summary"]
    print(f"\n  任务状态：完成 {s.get('completed')} / 失败 {s.get('failed')}")
    print(f"  工具调用次数：{out['execution_count']}    重规划次数：{out['replan_count']}")
    print(f"  回答：{out['final_answer']}")
    if out["failure"]:
        print(f"  失败码：{out['failure']}")
    return out


client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
client.connect("com.xiaomi.miagent.selfcheck", PERMS)
registry = ToolRegistry()
registry.load_from_miclaw(client)

# ---------- ① 参数偏差就地修正 ----------
# 规划把单号写成了 SN002。工具如实报「查无此单」，
# 校验从用户目标里读出正确单号，修正后的参数过了工具 schema 才被派发。
out = run("① 参数偏差：修正参数后重试同一个任务，不重规划",
          "查一下订单 SN001 到哪儿了",
          make_llm(plan(task("t1", "查订单状态", "system.query_order", order_id="SN002")),
                   checked=[reviews(reject("t1", "查的是 SN002，用户要的是 SN001",
                                           {"order_id": "SN001"})),
                            reviews()],
                   answer="订单 SN001 已发货。"),
          registry)
print(f"  任务最终用的参数：{out['tasks']['t1']['arguments']}")

# ---------- ② 结果不符，修不出参数 ----------
# 日历查的是明晚，用户问的是今晚。校验判未通过但给不出修正
# （这里让它不给修正），于是按失败落库，转重规划换个方案。
run("② 结果不符且修不出参数：如实转重规划，不宣称完成",
    "今晚有空吗",
    make_llm(plan(task("t1", "查日历", "system.query_calendar", when="明晚")),
             replan=plan(task("t2", "查今晚日历", "system.query_calendar", when="今晚")),
             checked=[reviews(reject("t1", "查的是明晚，用户问的是今晚")), reviews()],
             answer="今晚 19:00 之后有空。"),
    registry)

# ---------- ③ 计划漏做一步 ----------
# 任务全部成功，任务图这一层没有任何异常迹象；
# 「合起来算不算做完」只有对照目标才知道，这正是完成校验要回答的。
run("③ 计划漏做：任务全成功，完成校验判未达成并补做",
    "查下今晚有没有空，有空就订个餐厅",
    make_llm(plan(task("t1", "查日历", "system.query_calendar", when="今晚")),
             replan=plan(task("t2", "订餐厅", "system.book_restaurant", name="小馆 B")),
             done=[goal(False, "查了日历，还没有订餐厅"), goal(True)],
             answer="今晚有空，已订小馆 B。"),
    registry)

client.close()
