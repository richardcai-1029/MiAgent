"""接真实模型：用通义千问驱动整张 Agent 图，工具仍由 MiClaw mock 服务端提供。

与其他示例唯一的区别是 FakeLLM 换成了 QwenLLM —— 图代码一行不改。

运行：  echo "DASHSCOPE_API_KEY=sk-..." > .env    # 或 export 同名环境变量
        python examples/qwen_agent.py                   # 默认请求
        python examples/qwen_agent.py "明天早上七点叫我起床"
"""

import sys

from miagent.client import MiClawClient
from miagent.graph import build_agent
from miagent.llm.qwen import QwenLLM
from miagent.memory import Session
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "alarm.write", "location.fine"]


def main() -> None:
    requests = sys.argv[1:] or ["今晚有空吗", "那 19 点半安排个晚餐"]

    llm = QwenLLM()      # 模型名、地址均为预设；换模型：QwenLLM(model="qwen-plus")

    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("qwen-demo", PERMS)
    registry = ToolRegistry()
    registry.load_from_miclaw(client)
    print(f"模型：{llm.name}    可用工具：{[t.name for t in registry]}\n")

    session = Session(build_agent(llm, registry))
    for request in requests:
        out = session.run(request)
        print(f"用户：{request}")
        for line in out["trace"]:
            print(f"    {line}")
        print(f"助理：{out['final_answer']}")
        if out["failure"]:
            print(f"失败码：{out['failure']}")
        print()

    print(f"模型统计：{llm.stats()}")
    client.close()


if __name__ == "__main__":
    main()
