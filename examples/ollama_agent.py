"""接本地模型：用局域网里 Ollama 上的 Qwen3 驱动整张 Agent 图，工具仍由 MiClaw mock 服务端提供。

与其他示例唯一的区别是 FakeLLM 换成了 OllamaLLM —— 图代码一行不改。
默认关掉 Qwen3 的思考模式（见 miagent/llm/ollama.py），一次规划约 1~3 s。

运行：  export OLLAMA_BASE_URL=http://192.168.3.121:11434/v1   # 部署机地址，缺省 localhost
        python examples/ollama_agent.py                 # 默认请求
        python examples/ollama_agent.py "明天早上七点叫我起床"
"""

import os
import sys

from miagent.client import MiClawClient
from miagent import build_agent
from miagent.llm.ollama import OllamaLLM
from miagent.memory import Session
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport

PERMS = ["calendar.read", "calendar.write", "alarm.write", "location.fine"]


def main() -> None:
    requests = sys.argv[1:] or ["今晚有空吗", "那 19 点半安排个晚餐"]

    llm = OllamaLLM(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        # 部署侧 Modelfile num_ctx=8192，按实测最保守的 2.0 字符/token 折算
        # （依据见 miagent/llm/ollama.py 模块文档）。改了 num_ctx 要同步改这里。
        context_limit=16000,
    )

    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("ollama-demo", PERMS)
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
