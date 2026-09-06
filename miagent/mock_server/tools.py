"""模拟 MiClaw 提供的系统级工具。

真实 MiClaw 的工具会去调 Android 系统服务；这里全部返回假数据，
但**声明部分是真的** —— 每个工具都要申报：
  · 需要什么权限        -> 服务端据此做授权检查（MC-5001）
  · 预计占用多少内存    -> 服务端据此做配额检查（MC-4001）

把这些约束写成工具的声明，而不是散落在处理函数里，好处是
服务端可以统一执行检查，加新工具时不会漏掉任何一项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SystemTool:
    name: str
    description: str
    input_schema: dict[str, Any]      # JSON Schema，能直接喂给模型做 function calling
    required_permission: str | None   # None 表示不需要授权
    estimated_memory_mb: int
    handler: Callable[[dict[str, Any]], str]


def _schema(props: dict[str, str], required: list[str]) -> dict[str, Any]:
    """构造一个最简 JSON Schema。props 是 {字段名: 说明}，全按字符串处理。"""
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": required,
    }


SYSTEM_TOOLS: list[SystemTool] = [
    SystemTool(
        name="system.get_battery",
        description="查询设备当前电量与充电状态",
        input_schema=_schema({}, []),
        required_permission=None,               # 公开信息，不需要授权
        estimated_memory_mb=4,
        handler=lambda a: "电量 63%，未在充电",
    ),
    SystemTool(
        name="system.set_alarm",
        description="创建一个闹钟",
        input_schema=_schema({"time": "闹钟时间，如 08:00", "label": "备注"}, ["time"]),
        required_permission="alarm.write",
        estimated_memory_mb=8,
        handler=lambda a: f"已创建 {a['time']} 的闹钟（{a.get('label', '无备注')}）",
    ),
    SystemTool(
        name="system.send_sms",
        description="发送短信",
        input_schema=_schema({"to": "收件人号码", "text": "短信正文"}, ["to", "text"]),
        required_permission="sms.send",
        estimated_memory_mb=12,
        handler=lambda a: f"已向 {a['to']} 发送短信：{a['text']}",
    ),
    SystemTool(
        name="system.read_contacts",
        description="按姓名查询通讯录",
        input_schema=_schema({"name": "联系人姓名"}, ["name"]),
        required_permission="contacts.read",    # 这个权限我们会故意拒绝，演示 MC-5001
        estimated_memory_mb=16,
        handler=lambda a: f"{a['name']}：13800138000",
    ),
    SystemTool(
        name="system.capture_screen",
        description="截取当前屏幕并返回图像",
        input_schema=_schema({}, []),
        required_permission="screen.capture",
        estimated_memory_mb=200,                # 故意超出端侧配额，演示 MC-4001
        handler=lambda a: "<screenshot bytes>",
    ),
]

TOOL_REGISTRY: dict[str, SystemTool] = {t.name: t for t in SYSTEM_TOOLS}
