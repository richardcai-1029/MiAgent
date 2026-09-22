"""进程内多轮会话：第二轮承接第一轮的结论。

运行：  python examples/multi_turn.py
"""

import json

from miagent.client import MiClawClient
from miagent.graph import build_agent
from miagent.llm import FakeLLM
from miagent.memory import Session
from miagent.mock_server import MiClawMockServer
from miagent.tools import ToolRegistry
from miagent.transport import LoopbackTransport


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def final(answer, summary):
    return json.dumps({"answer": answer, "summary": summary}, ensure_ascii=False)


def passed():
    """结果校验：不给判定即全部通过。"""
    return json.dumps({"reviews": []}, ensure_ascii=False)


def achieved():
    """完成校验：目标已达成。"""
    return json.dumps({"achieved": True, "gap": ""}, ensure_ascii=False)


def main() -> None:
    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("multi-turn-demo", ["calendar.read", "calendar.write"])
    registry = ToolRegistry()
    registry.load_from_miclaw(client)

    # 脚本按顺序回复：每轮依次是规划、结果校验、完成校验、收尾
    llm = FakeLLM([
        plan({"id": "t1", "description": "查今晚日程", "required_tool": "system.query_calendar",
              "dependencies": [], "arguments": {"when": "今晚"}}),
        passed(), achieved(),
        final("今晚 19:00 之后有空。", "查过日历，今晚 19:00 之后空闲"),
        plan({"id": "t1", "description": "创建晚餐日程", "required_tool": "system.create_event",
              "dependencies": [], "arguments": {"title": "晚餐", "when": "19:30"}}),
        passed(), achieved(),
        final("已建好 19:30 的晚餐日程。", "创建了 19:30 的晚餐日程"),
    ])

    session = Session(build_agent(llm, registry))
    for request in ["今晚有空吗", "那 19 点半安排个晚餐"]:
        out = session.run(request)
        print(f"用户：{request}\n助理：{out['final_answer']}\n")

    print("规划器在第二轮看到的对话历史：")
    second = next(m for m in llm.seen if "规划器" in m[0].content
                  and m[1].content.endswith("用户目标：那 19 点半安排个晚餐"))
    body = second[1].content
    print("  " + body[body.index("上一轮"):body.index("用户目标")].strip().replace("\n", "\n  "))

    print("\n会话记录：")
    for i, t in enumerate(session.turns, 1):
        print(f"  第 {i} 轮  {t['request']} → {t['summary']}（{len(t['episodes'])} 条执行记录）")

    client.close()


if __name__ == "__main__":
    main()
