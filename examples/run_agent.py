"""跑通 Agent 图的三条路径：一次成功 / 失败重试 / 失败重规划。

运行：  python examples/run_agent.py
"""

from datetime import datetime, timedelta

from miagent.client import MiClawClient
from miagent.graph import build_agent, initial_state
from miagent.llm import FakeLLM, plan_json, step
from miagent.mock_server import MiClawMockServer
from miagent.mock_server.tools import reset_flaky
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport


@tool()
def add_days(start: str, days: int) -> str:
    """在给定日期上加若干天。

    Args:
        start: 起始日期，格式 YYYY-MM-DD
        days: 天数
    """
    return (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def make_llm(first_plan: str, replan: str | None, answer: str) -> FakeLLM:
    """按调用方的身份返回不同内容 —— 比固定脚本更稳，
    因为图的 LLM 调用次数会随分支变化。"""
    def responder(messages):
        role = messages[0].content
        if "重规划器" in role:
            return replan or plan_json()
        if "规划器" in role:
            return first_plan
        return answer
    return FakeLLM(responder=responder)


def run(title, task, llm, registry):
    print(f"\n{'━' * 76}\n{title}\n{'━' * 76}")
    app = build_agent(llm, registry)
    out = app.invoke(initial_state(task), {"recursion_limit": 50})
    print(f"  任务：{task}\n")
    for line in out["trace"]:
        print(f"    {line}")
    print(f"\n  最终回答：{out['answer']}")
    print(f"  失败标记：{out['failure'] or '无'}    LLM 调用 {llm.call_count} 次")
    return out


srv = MiClawMockServer()
client = MiClawClient(transport=LoopbackTransport(srv))
client.connect("com.xiaomi.miagent.demo",
               permissions=["alarm.write", "screen.capture", "sms.send"])
registry = ToolRegistry([add_days])
registry.load_from_miclaw(client)
print(f"可用工具：{[t.name for t in registry]}")

# ---------- 场景一：一次成功 ----------
run("场景一：全部成功（本地工具 + MiClaw 工具混合）",
    "查一下电量，然后帮我定一个明早八点的闹钟",
    make_llm(
        plan_json(step("system.get_battery", reason="先看电量"),
                  step("system.set_alarm", reason="设置闹钟", time="08:00", label="晨会")),
        None,
        "电量 63%，已为你创建 08:00 的闹钟。"),
    registry)

# ---------- 场景二：临时故障 → 重试 ----------
reset_flaky()
run("场景二：MC-1003 临时故障 → 段位 1 → BACKOFF → 重试后成功",
    "把我的设置同步一下",
    make_llm(
        plan_json(step("system.sync_settings", reason="同步设置")),
        None,
        "设置已同步完成。"),
    registry)

# ---------- 场景三：资源不足 → 重规划 ----------
run("场景三：MC-4001 资源超限 → 段位 4 → DEGRADE → 重规划换轻量方案",
    "帮我看看屏幕上现在显示什么",
    make_llm(
        plan_json(step("system.capture_screen", reason="截屏看内容")),
        plan_json(step("system.get_battery", reason="截屏超内存配额，改为读取可获取的设备状态")),
        "无法截屏（超出端侧内存配额），已改为读取设备状态：电量 63%，未在充电。"),
    registry)

client.close()
