"""演示工具层：本地工具与 MiClaw 系统工具的统一抽象。

运行：  python examples/unified_tools.py
"""

import json
from datetime import datetime, timedelta

from miagent.client import MiClawClient
from miagent.tools import ToolRegistry, ToolSource, tool


def title(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ============ 一、本地工具：进程内的普通 Python 函数 ============

@tool()
def add_days(start: str, days: int) -> str:
    """在给定日期上加若干天，返回新日期。

    Args:
        start: 起始日期，格式 YYYY-MM-DD
        days: 要增加的天数，可以是负数
    """
    return (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


@tool()
def json_field(text: str, key: str) -> str:
    """从一段 JSON 文本中取出指定字段的值。

    Args:
        text: JSON 格式的字符串
        key: 要提取的字段名
    """
    return str(json.loads(text)[key])


@tool()
def compare(left: float, right: float) -> str:
    """比较两个数的大小，返回 greater / less / equal，用于任务分支判断。

    Args:
        left: 左侧数值
        right: 右侧数值
    """
    return "greater" if left > right else ("less" if left < right else "equal")


# ============ 二、组装：本地 + 远端进同一个注册表 ============

title("① 组装工具注册表")
client = MiClawClient()
client.connect("com.xiaomi.miagent.general",
               permissions=["sms.send", "alarm.write", "screen.capture", "contacts.read"])

registry = ToolRegistry([add_days, json_field, compare])
remote = registry.load_from_miclaw(client)

print(f"  本地工具 {len(registry.by_source(ToolSource.LOCAL))} 个，"
      f"MiClaw 工具 {len(remote)} 个，合计 {len(registry)} 个\n")
print("  " + registry.describe().replace("\n", "\n  "))

title("② 模型视角：看不到来源、权限、开销")
schemas = registry.to_model_schemas()
print(f"  模型拿到 {len(schemas)} 个工具，每个只有三个字段: "
      f"{list(schemas[0].keys())}")
print(f"\n  举例（system.send_sms，模型无从得知它走 IPC 且需要权限）:")
sms = next(s for s in schemas if s["name"] == "system.send_sms")
print("   ", json.dumps(sms, ensure_ascii=False, indent=2).replace("\n", "\n    "))

title("③ 统一调用：本地和远端用同一个入口，调用方无需区分")
for name, args in [
    ("add_days",           {"start": "2026-09-07", "days": 30}),
    ("json_field",         {"text": '{"battery": 63, "charging": false}', "key": "battery"}),
    ("compare",            {"left": 63, "right": 20}),
    ("system.get_battery", {}),
    ("system.set_alarm",   {"time": "08:00", "label": "晨会"}),
]:
    r = registry.invoke(name, args)
    src = registry.get(name).source
    print(f"  [{src:6}] {name:20} → {r.content}")

title("④ 失败处理：错误码决定重试策略，模型收到的是「怎么办」而非「错了」")
failures = [
    ("模型幻觉的工具名",     "system.open_wechat",    {}),
    ("本地工具缺必填参数",   "add_days",              {"start": "2026-09-07"}),
    ("本地工具参数值非法",   "add_days",              {"start": "昨天", "days": 1}),
    ("远端超端侧内存配额",   "system.capture_screen", {}),
    ("权限被拒的工具",       "system.read_contacts",  {"name": "张三"}),
]
for label, name, args in failures:
    r = registry.invoke(name, args)
    print(f"\n  【{label}】")
    print(f"    错误码: {r.error_code}    重试策略: {r.retry_policy}")
    print(f"    喂给模型: {r.for_model()}")

print("""
  ★ 注意最后一条：system.read_contacts 报的是 AG-2001「工具未注册」，
    而不是 MC-5001「权限不足」—— 因为服务端在 tools/list 阶段就把它过滤掉了，
    它从未进入注册表，模型也从未见过它。

    这说明权限过滤是有效的：无权限的工具不会被模型规划，
    MC-5001 只在「权限中途被撤销」这种少见情况下才会出现。
""")
client.close()
