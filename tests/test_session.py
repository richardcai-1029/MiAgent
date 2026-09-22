"""进程内多轮会话：上一轮的摘要进入下一轮的规划提示词。

本文件按序脚本回复，断言的是提示词里有什么。语义校验会在脚本中间插入
与多轮上下文无关的回复，因此这里一律关掉；校验自身见 test_verify.py。
"""

import json

import pytest

from miagent.client import MiClawClient
from miagent.graph import build_agent
from miagent.graph.nodes import _tools_section
from miagent.llm import FakeLLM
from miagent.llm.context import fit
from miagent.memory import Session, Turn
from miagent.memory.session import render_history
from miagent.mock_server import MiClawMockServer
from miagent.protocol import ErrorCode
from miagent.tools import ToolRegistry, tool
from miagent.transport import LoopbackTransport


@tool()
def echo(text: str) -> str:
    """原样返回输入。

    Args:
        text: 任意文本
    """
    return text


@pytest.fixture
def registry():
    client = MiClawClient(transport=LoopbackTransport(MiClawMockServer()))
    client.connect("test", ["calendar.read", "calendar.write", "location.fine"])
    reg = ToolRegistry([echo])
    reg.load_from_miclaw(client)
    yield reg
    client.close()


def plan(*tasks):
    return json.dumps({"tasks": list(tasks)}, ensure_ascii=False)


def task(tid, tool_name, **args):
    return {"id": tid, "description": f"任务 {tid}", "required_tool": tool_name,
            "dependencies": [], "arguments": args}


def final(answer, summary):
    return json.dumps({"answer": answer, "summary": summary}, ensure_ascii=False)


def scripted(*replies):
    """按顺序回复：每轮一次规划、一次收尾。"""
    llm = FakeLLM(list(replies))
    llm.prompts = []
    original = llm._complete

    def spy(messages):
        llm.prompts.append(messages)
        return original(messages)

    llm._complete = spy
    return llm


def planner_prompt(llm, request):
    """规划器为该请求收到的 user 消息（自修复重试用的是同一份）。"""
    return next(m[1].content for m in llm.prompts
                if "规划器" in m[0].content and m[1].content.endswith(f"用户目标：{request}"))


def turn(request, summary, failure=None):
    return Turn(request=request, answer="", summary=summary, failure=failure, episodes=[])


class TestSessionRecordsTurns:
    def test_each_run_appends_a_turn(self, registry):
        llm = scripted(plan(task("t1", "echo", text="A")), final("好了", "回显了 A"),
                       plan(task("t2", "echo", text="B")), final("也好了", "回显了 B"))
        session = Session(build_agent(llm, registry, verify=False))
        session.run("回显 A")
        session.run("回显 B")
        assert [t["request"] for t in session.turns] == ["回显 A", "回显 B"]
        assert [t["summary"] for t in session.turns] == ["回显了 A", "回显了 B"]
        assert session.turns[0]["episodes"][0]["task_id"] == "t1"

    def test_failed_turn_is_recorded_with_its_code(self, registry):
        llm = scripted("我不知道", "还是不知道", final("抱歉", "没能规划"))
        session = Session(build_agent(llm, registry, verify=False))
        out = session.run("随便说说")
        assert out["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert session.turns[0]["failure"] == ErrorCode.AG_PLAN_PARSE_FAILED.value
        assert session.turns[0]["summary"] == "没能规划"

    def test_clear_forgets_everything(self, registry):
        llm = scripted(plan(), final("你好", "打了招呼"), plan(), final("你好", "又打了招呼"))
        session = Session(build_agent(llm, registry, verify=False))
        session.run("你好")
        session.clear()
        session.run("再说一次")
        assert len(session.turns) == 1
        assert "上一轮" not in planner_prompt(llm, "再说一次")


class TestHistoryReachesThePlanner:
    def test_first_turn_has_no_history(self, registry):
        llm = scripted(plan(), final("你好", "打了招呼"))
        Session(build_agent(llm, registry, verify=False)).run("你好")
        assert "上一轮" not in planner_prompt(llm, "你好")

    def test_second_turn_sees_the_previous_summary(self, registry):
        llm = scripted(plan(task("t1", "system.query_calendar", when="今晚")),
                       final("今晚有空", "查过日历，今晚 19 点后有空"),
                       plan(), final("好", "建了日程"))
        session = Session(build_agent(llm, registry, verify=False))
        session.run("今晚有空吗")
        session.run("那帮我建个日程")
        prompt = planner_prompt(llm, "那帮我建个日程")
        assert "上一轮\n  用户：今晚有空吗\n  结论：查过日历，今晚 19 点后有空" in prompt
        assert prompt.index("上一轮") < prompt.index("用户目标：那帮我建个日程")

    def test_failed_turn_is_marked_in_history(self, registry):
        llm = scripted("我不知道", "还是不知道", final("抱歉", "没能规划"),
                       plan(), final("好", "这次可以"))
        session = Session(build_agent(llm, registry, verify=False))
        session.run("随便说说")
        session.run("再试试")
        assert "上一轮（未完成，AG-1001）" in planner_prompt(llm, "再试试")

    def test_history_does_not_leak_into_the_replanner(self, registry):
        """重规划有目标锚做参照，不再重复带对话历史。"""
        llm = scripted(plan(), final("你好", "打了招呼"),
                       plan(task("t1", "system.book_restaurant", name="小馆 A")),
                       plan(task("t1b", "system.book_restaurant", name="小馆 B")),
                       final("订好了", "订了小馆 B"))
        session = Session(build_agent(llm, registry, retry_delay_ms=0, verify=False))
        session.run("你好")
        session.run("订餐")
        replan = next(m[1].content for m in llm.prompts if "重规划器" in m[0].content)
        assert "上一轮" not in replan


class TestHistoryIsTrimmedOldestFirst:
    def _turns(self):
        return [turn("第一个请求", "第一轮结论"), turn("第二个请求", "第二轮结论"),
                turn("第三个请求", "第三轮结论")]

    def test_priorities_grow_with_age(self):
        sections = render_history(self._turns())
        assert [s.priority for s in sections] == [4, 3, 2]
        assert [s.name for s in sections] == ["对话历史·第 1 轮", "对话历史·第 2 轮", "对话历史·上一轮"]

    def test_only_the_latest_turn_has_a_compact_form(self):
        sections = render_history(self._turns())
        assert [s.compact is not None for s in sections] == [False, False, True]
        assert sections[-1].compact == "上一轮结论：第三轮结论"

    def test_oldest_is_dropped_before_the_latest_is_compacted(self, registry):
        sections = render_history(self._turns())
        full = "\n\n".join(s.text for s in sections)
        body, notes = fit(sections, limit=len(full) - 1, estimate=len)
        assert notes == ["对话历史·第 1 轮→已丢弃"]
        assert "第三轮结论" in body and "第一个请求" not in body

    def test_latest_turn_survives_as_its_conclusion(self, registry):
        sections = render_history(self._turns())
        latest = sections[-1]
        body, notes = fit(sections, limit=len(latest.compact), estimate=len)
        assert notes == ["对话历史·第 1 轮→已丢弃", "对话历史·第 2 轮→已丢弃",
                         "对话历史·上一轮→紧凑形式"]
        assert body == "上一轮结论：第三轮结论"

    def test_history_is_cut_before_tool_descriptions(self, registry):
        tools = _tools_section(registry)
        sections = [tools, *render_history(self._turns())]
        _, notes = fit(sections, limit=len(tools.text) + 10, estimate=len)
        assert notes[:3] == ["对话历史·第 1 轮→已丢弃", "对话历史·第 2 轮→已丢弃",
                             "对话历史·上一轮→紧凑形式"]
        assert "工具描述" not in "".join(notes[:3])

    def test_no_turns_renders_nothing(self):
        assert render_history([]) == []
