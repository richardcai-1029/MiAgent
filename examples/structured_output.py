"""演示结构化输出：schema 约束 + 自修复重试。

运行：  python examples/structured_output.py
"""

from miagent.client import MiClawClient
from miagent.graph import build_agent, initial_state
from miagent.graph.schema import plan_model_for
from miagent.llm import FakeLLM, plan_json, step
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
client.connect("demo", ["alarm.write"])
reg = ToolRegistry(); reg.load_from_miclaw(client)


def title(t):
    print(f"\n{'━' * 76}\n{t}\n{'━' * 76}")


title("① 模型看到的 schema（由 Pydantic 模型自动生成，工具名是枚举）")
import json
sch = plan_model_for([t.name for t in reg]).model_json_schema()
print(json.dumps(sch["$defs"]["PlanStepConstrained"]["properties"]["tool"],
                 ensure_ascii=False, indent=2))

title("② 模型第一次输出不合规 → 自修复重试 → 救回来")
print("  max_repairs 默认为 1，即「原始一次 + 修复一次」共两次机会。\n")
outputs = iter([
    '{"plan":[{"tool":"system.get_battery"}]}',              # 第 1 次：字段名写成了 plan
    plan_json(step("system.get_battery", reason="查电量")),   # 第 2 次：被纠正后改对
])
def responder(msgs):
    role = msgs[0].content
    return next(outputs) if "规划器" in role else "电量 63%，未在充电。"


def title_note():
    pass

llm = FakeLLM(responder=responder)
out = build_agent(llm, reg).invoke(initial_state("查一下电量"), {"recursion_limit": 40})
for line in out["trace"]:
    print(f"    {line}")
print(f"\n  回答：{out['answer']}")
print(f"  统计：LLM 调用 {llm.call_count} 次，其中自修复重试 {llm.repair_count} 次")

title("③ 幻觉的工具名：在规划阶段就被拒，一步都不执行")
llm2 = FakeLLM(responder=lambda m: (
    plan_json(step("system.open_wechat")) if "规划器" in m[0].content
    else "抱歉，我没有打开微信的能力。"))
# 注：这里规划两次都会输出同一个非法工具名，所以自修复也救不回来，
#     最终以 AG-1001 收场 —— 这正是期望行为：工具不存在就是不存在。
out2 = build_agent(llm2, reg).invoke(initial_state("帮我打开微信"), {"recursion_limit": 40})
for line in out2["trace"]:
    print(f"    {line}")
print(f"\n  失败码：{out2['failure']}    已执行步骤数：{len(out2['results'])}")
print(f"  回答：{out2['answer']}")
print("\n  ★ 改造前：会白跑一步，执行时才报 AG-2001")
print("    改造后：schema 校验阶段就拒，省下一次工具调用和一轮循环")

client.close()
