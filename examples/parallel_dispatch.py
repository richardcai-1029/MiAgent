"""并行调度演示：依赖结构如何决定实际耗时。

运行：  python examples/parallel_dispatch.py
"""

import json
import time

from miagent.client import MiClawClient
from miagent.graph import build_agent, initial_state
from miagent.llm import FakeLLM
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport

DELAY = 0.3


@tool()
def load_source(name: str) -> str:
    """从本地数据源读取内容（模拟 IO 等待）。

    Args:
        name: 数据源名称
    """
    time.sleep(DELAY)
    return f"{name} 的内容"


def task(tid, tool_name, deps=None, **args):
    return {"id": tid, "description": f"读取 {tid}", "required_tool": tool_name,
            "dependencies": deps or [], "arguments": args}


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def llm_for(p, answer="完成"):
    return FakeLLM(responder=lambda m: p if "规划器" in m[0].content else answer)


def run(title, tasks_json, registry, **kw):
    print(f"\n{'━' * 76}\n{title}\n{'━' * 76}")
    app = build_agent(llm_for(tasks_json), registry, **kw)
    t = time.perf_counter()
    out = app.invoke(initial_state("读取三个数据源"), {"recursion_limit": 60})
    elapsed = (time.perf_counter() - t) * 1000
    for line in out["trace"]:
        print(f"    {line}")
    print(f"\n  分层：{out['execution_summary']['parallel_layers']}")
    print(f"  耗时：{elapsed:.0f} ms   （每个任务 sleep {DELAY * 1000:.0f} ms）")
    return elapsed


client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
client.connect("demo", ["calendar.read"])
registry = ToolRegistry([load_source])
registry.load_from_miclaw(client)

serial = run(
    "① 三个任务串成链：A → B → C（依赖结构强制串行）",
    plan(task("A", "load_source", name="源A"),
         task("B", "load_source", ["A"], name="源B"),
         task("C", "load_source", ["B"], name="源C")),
    registry)

parallel = run(
    "② 同样三个任务，但彼此无依赖 → 同一轮并行派发",
    plan(task("A", "load_source", name="源A"),
         task("B", "load_source", name="源B"),
         task("C", "load_source", name="源C")),
    registry)

print(f"\n  ★ 提速 {serial / parallel:.2f}x —— 同样的工作量，只因依赖结构不同")

run("③ MiClaw 并发配额：4 个就绪，配额只有 2 → 顺延到下一轮",
    plan(*[task(f"t{i}", "system.query_weather", when=f"第{i}天") for i in range(4)]),
    registry, max_concurrent_miclaw=2)

print(f"""
{'━' * 76}
诚实说明：MiClaw 侧目前拿不到实际提速
{'━' * 76}
  图层的并行派发已经生效（上面 ③ 能看到同轮派发多个），但 MiClaw 调用
  会在客户端被锁串行化。原因是客户端目前是同步阻塞的：

    · id 分配不是原子操作，两个线程可能拿到同一个 id
    · 管道写入可能字节交错，破坏按行分帧

  加锁保证了正确性，代价是 MiClaw 调用排队。要拿到真实并发，需要：

    1. 传输层多路复用 —— 连发多个请求不等回复，靠 id 配对响应
       （id 配对逻辑已经有了，缺的是"发了就走"的异步收发）
    2. 服务端并发处理 —— 当前 mock 是单线程循环，一次只处理一条

  两项都尚未实现。本地 IO 型工具不受此限，
  上面 ② 的提速是真实的。
""")
client.close()
