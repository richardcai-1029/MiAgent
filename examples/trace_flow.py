"""把「连接 → 握手 → 注册 → 调工具 → 注销」全流程的真实函数调用打印出来。

用进程内回环传输，让客户端与服务端在同一进程里，这样一处就能看到两侧的
完整调用链。报文仍然真实地走 write_message / read_message 分帧路径。

运行：  python examples/trace_flow.py
标记：  [客户端] / [传输层] / [服务端] / [协议层]
"""

import inspect

import miagent.mock_server.server as server_mod
import miagent.transport as transport_mod
from miagent.client.client import MiClawClient
from miagent.mock_server import MiClawMockServer
from miagent.transport import LoopbackTransport

DEPTH = 0
QUIET = False


def wrap(owner, name, tag, show=None):
    orig = getattr(owner, name)

    def wrapper(*a, **kw):
        global DEPTH
        if QUIET:
            return orig(*a, **kw)
        note = ""
        if show:
            try:
                note = show(*a, **kw)
            except Exception:
                note = ""
        print(f"  {'│  ' * DEPTH}{tag} {name}{note}")
        DEPTH += 1
        try:
            return orig(*a, **kw)
        finally:
            DEPTH -= 1

    # 原本是 staticmethod/classmethod 的，包装后要还原回去，
    # 否则 setattr 会把它变成普通方法，调用时多收一个 self。
    static = inspect.getattr_static(owner, name) if inspect.isclass(owner) else None
    if isinstance(static, staticmethod):
        setattr(owner, name, staticmethod(wrapper))
    elif isinstance(static, classmethod):
        setattr(owner, name, classmethod(wrapper))
    else:
        setattr(owner, name, wrapper)


C, T, S, P = "🟦客户端", "🟨传输层", "🟩服务端", "⬜协议层"

# ---- 客户端 ----
wrap(MiClawClient, "initialize", C)
wrap(MiClawClient, "register", C, lambda s, aid, perms, intents=None: f"(申请权限={perms})")
wrap(MiClawClient, "list_tools", C)
wrap(MiClawClient, "call_tool", C, lambda s, n, a=None: f"({n})")
wrap(MiClawClient, "unregister", C)
wrap(MiClawClient, "_require", C, lambda s, f, w: f"(检查本地状态: {'通过' if f else '不通过'})")
wrap(MiClawClient, "_request", C, lambda s, m, p=None: f"(method={m}, 分配 id={s._next_id + 1})")
wrap(MiClawClient, "_notify", C, lambda s, m, p=None: f"(method={m}, 无 id 不等回复)")
wrap(MiClawClient, "_to_exception", C, lambda e: f"(把 {e.miclaw_code} 还原成异常)")

# ---- 传输层 ----
wrap(LoopbackTransport, "send", T)
wrap(LoopbackTransport, "receive", T)
wrap(transport_mod, "write_message", T, lambda st, m: "(序列化 + 换行分帧 + flush)")
wrap(transport_mod, "read_message", T, lambda st: "(readline + 解析 JSON)")

# ---- 服务端 ----
wrap(MiClawMockServer, "handle", S)
wrap(MiClawMockServer, "_resolve_method", S, lambda s, m: f"(Method('{m}') 反查，不认识就 MC-2004)")
wrap(MiClawMockServer, "_check_state", S, lambda s, m: f"(查状态表: 当前 {s.state})")
wrap(MiClawMockServer, "_dispatch", S)
for h in ["_on_initialize", "_on_register", "_on_tools_list", "_on_tools_call", "_on_unregister"]:
    wrap(MiClawMockServer, h, S)

# ---- 协议层（在 server 模块命名空间里）----
wrap(server_mod, "parse_incoming", P, lambda raw: "(校验 JSON-RPC 结构)")
wrap(server_mod, "success_response", P)
wrap(server_mod, "error_response", P)


def step(n, t):
    print(f"\n{'━' * 78}\n{n}. {t}\n{'━' * 78}")


step(1, "连接：构造服务端对象 + 回环传输，注入客户端")
QUIET = True
srv = MiClawMockServer()
client = MiClawClient(transport=LoopbackTransport(srv))
QUIET = False
print(f"  服务端状态 = {srv.state}    客户端 _handshaked={client._handshaked}")
print("  （真实部署时这一步是 SubprocessTransport 拉起子进程）")

step(2, "握手 initialize")
budget = client.initialize()
print(f"  → 拿到配额: 内存上限 {budget.max_memory_mb}MB    服务端状态 = {srv.state}")

step(3, "注册 miclaw/agent.register")
granted = client.register("com.xiaomi.miagent.demo", ["sms.send", "contacts.read"])
print(f"  → 批准权限 {granted}，拒绝 {[d['permission'] for d in client.denied_permissions]}")
print(f"    服务端状态 = {srv.state}")

step(4, "拉工具列表 tools/list")
tools = client.list_tools()
print(f"  → {len(tools)} 个工具: {[t.name for t in tools]}")

step(5, "调用工具 tools/call（成功）")
print(f"  → {client.call_tool('system.send_sms', {'to': '10086', 'text': '查话费'})}")

step(6, "调用工具 tools/call（失败：超内存配额）")
from miagent.protocol import MiClawError
try:
    client.call_tool("system.capture_screen")
except MiClawError as e:
    print(f"  → 抛出 {e}    detail={e.detail}")

step(7, "注销 miclaw/agent.unregister")
client.unregister()
print(f"  → 服务端状态 = {srv.state}，权限已清空: {srv.granted}")
