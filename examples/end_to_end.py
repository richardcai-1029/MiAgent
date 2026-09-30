"""端到端演示：Agent 客户端 <-> MiClaw mock 服务端，走真实的 stdio 管道。

运行：  python examples/end_to_end.py
输出里 [server] 开头的是服务端日志（走 stderr），其余是本脚本打印的。
"""

from miagent.client import MiClawClient
from miagent.protocol import AgentError, MiClawError


def title(t):
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


title("① 拉起服务端子进程，握手 + 注册")
client = MiClawClient()

# 故意在握手前调工具，验证本地预检
try:
    client.call_tool("system.get_battery")
except AgentError as e:
    print(f"  握手前调工具 → ✗ {e}")
    print("    ↑ 这个错误是【客户端本地】拦下的，报文根本没发出去")

granted = client.connect(
    agent_id="com.xiaomi.miagent.general",
    permissions=["sms.send", "alarm.write", "contacts.read", "screen.capture"],
    intents=["alarm.create", "message.send"],
)
print(f"\n  会话 ID   : {client.session_id}")
print("  申请权限   : ['sms.send', 'alarm.write', 'contacts.read', 'screen.capture']")
print(f"  批准权限   : {granted}")
print(f"  被拒权限   : {[d['permission'] + ' (' + d['code'] + ')' for d in client.denied_permissions]}")
print(f"  资源配额   : 内存上限 {client.budget.max_memory_mb}MB，"
      f"并发 {client.budget.max_concurrent_calls}，超时 {client.budget.max_call_timeout_ms}ms")

title("② 拉取可用工具列表")
for t in client.list_tools():
    required = t.inputSchema.get("required") or []
    print(f"  · {t.name:26} {t.description}"
          + (f"  [必填: {', '.join(required)}]" if required else ""))
print("\n  注意：system.read_contacts 不在列表里 —— 服务端已按权限过滤，")
print("        模型看不到它，就不会规划出注定失败的步骤。")

title("③ 正常调用")
for name, args in [("system.get_battery", {}),
                   ("system.set_alarm", {"time": "08:00", "label": "晨会"}),
                   ("system.send_sms", {"to": "10086", "text": "查话费"})]:
    print(f"  {name:22} → {client.call_tool(name, args)}")

title("④ 四种失败，各对应一个错误码")
cases = [
    ("工具不存在",        "system.nope",           {}),
    ("缺必填参数",        "system.send_sms",       {"to": "10086"}),
    ("权限被用户拒绝",     "system.read_contacts",  {"name": "张三"}),
    ("超出端侧内存配额",   "system.capture_screen", {}),
]
for label, name, args in cases:
    try:
        client.call_tool(name, args)
    except MiClawError as e:
        print(f"  {label:16} → [{e.code}] {e.message}")
        print(f"  {'':16}    detail: {e.detail}")

title("⑤ 查询资源占用，然后注销退出")
print(f"  {client.query_resource()}")
client.close()
print("\n  已注销并回收子进程。")
