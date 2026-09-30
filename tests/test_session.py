"""进程内多轮会话：上一轮的摘要进入下一轮的规划提示词。

本文件按序脚本回复，断言的是提示词里有什么。语义校验会在脚本中间插入
与多轮上下文无关的回复，因此这里一律关掉；校验自身见 test_verify.py。
"""

import json

import pytest

from miagent.client import MiClawClient
from miagent import build_agent
from miagent.agent.prompting import tools_section
from miagent.llm import FakeLLM
from miagent.llm.context import Section, fit
from miagent.memory import Session, Turn
from miagent.memory.session import reachable, render_history
from miagent.mock_server import MiClawMockServer
from miagent.protocol import AgentError, ErrorCode
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


def turn(request, summary, failure=None, number=1):
    return Turn(number=number, request=request, answer="", summary=summary,
                failure=failure, episodes=[])


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
        return [turn("第一个请求", "第一轮结论", number=1),
                turn("第二个请求", "第二轮结论", number=2),
                turn("第三个请求", "第三轮结论", number=3)]

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
        tools = tools_section(registry)
        sections = [tools, *render_history(self._turns())]
        _, notes = fit(sections, limit=len(tools.text) + 10, estimate=len)
        assert notes[:3] == ["对话历史·第 1 轮→已丢弃", "对话历史·第 2 轮→已丢弃",
                             "对话历史·上一轮→紧凑形式"]
        assert "工具描述" not in "".join(notes[:3])

    def test_no_turns_renders_nothing(self):
        assert render_history([]) == []


class EchoAgent:
    """只实现 invoke 的替身：回答与摘要都是请求原文，记录每轮看到的历史。"""

    def __init__(self):
        self.seen: list[list[int]] = []

    def invoke(self, state, config=None):
        self.seen.append([t["number"] for t in state["history"]])
        req = state["user_request"]
        return {**state, "final_answer": req, "turn_summary": f"{req}的结论",
                "failure": None, "episodes": []}


class TestTurnsAreBounded:
    """单个会话保留几轮由上下文预算推出：永远进不了提示词的轮次不留。"""

    def _turns(self, n):
        # 长短不一，使「连同更新的各轮放不下」发生在不同位置
        return [turn(f"请求{i}" * (1 + i % 4), f"结论{i}" * (1 + i % 3), number=i + 1)
                for i in range(n)]

    def test_reachable_edges(self):
        assert reachable([], 100) == 0
        one = self._turns(1)
        assert reachable(one, 0) == 1                 # 最新一轮总有紧凑形式可留
        many = self._turns(6)
        texts = [s.text for s in render_history(many)]
        assert reachable(many, 10**6) == 6
        assert reachable(many, len("\n\n".join(texts[-2:]))) == 2
        assert reachable(many, len("\n\n".join(texts[-2:])) - 1) == 1

    def test_pruning_never_changes_the_prompt(self, registry):
        """削与不削，规划器拿到的提示词逐字相同：削掉的都是必然被丢弃的。"""
        tools = tools_section(registry)
        goal = Section("用户目标", "用户目标：再订一次")
        turns = self._turns(12)
        history_limit = len("\n\n".join(s.text for s in render_history(turns))) // 2
        pruned = turns[len(turns) - reachable(turns, history_limit):]
        assert 1 < len(pruned) < len(turns), "用例前提：确有轮次被削、也有轮次留下"

        def prompt(ts, limit):
            try:
                return fit([tools, *render_history(ts), goal], limit, len)[0]
            except AgentError as e:
                return e.code
        for limit in range(0, history_limit + 1):
            assert prompt(turns, limit) == prompt(pruned, limit), limit

    def test_reachable_is_exactly_what_fit_keeps(self):
        """上界是紧的：只有历史片段时，fit 留下的轮数恰好是 reachable 给出的轮数。
        多削一轮会让本该出现的一轮消失，少削一轮则白占内存。"""
        turns = self._turns(12)
        total = len("\n\n".join(s.text for s in render_history(turns)))
        for limit in range(0, total + 1):
            try:
                _, notes = fit(render_history(turns), limit, len)
            except AgentError:
                continue                  # 连最新一轮的紧凑形式都放不下
            present = len(turns) - sum(n.endswith("已丢弃") for n in notes)
            assert present == reachable(turns, limit), limit

    def test_numbers_survive_pruning(self):
        agent = EchoAgent()
        session = Session(agent, history_limit=60)
        for i in range(1, 9):
            session.run(f"第{i}个请求")
        numbers = [t["number"] for t in session.turns]
        assert numbers[-1] == 8 and numbers == list(range(numbers[0], 9))
        assert 1 < len(numbers) < 8
        labels = [s.name for s in render_history(session.turns)]
        assert labels[0] == f"对话历史·第 {numbers[0]} 轮"   # 编号不因前面被削而改变

    def test_turn_count_stays_bounded_over_a_long_conversation(self):
        """每轮渲染后至少有固定模板那么长，保留的轮数因此不超过 预算 / 模板长度 + 1。"""
        session = Session(EchoAgent(), history_limit=200)
        most = 0
        for i in range(300):
            session.run(f"请求{i}")
            most = max(most, len(session.turns))
        assert most <= 200 // len("上一轮\n  用户：\n  结论：") + 1

    def test_without_a_limit_every_turn_is_kept(self):
        session = Session(EchoAgent())
        for i in range(20):
            session.run(f"请求{i}")
        assert len(session.turns) == 20

    def test_clear_restarts_numbering(self):
        session = Session(EchoAgent())
        session.run("一")
        session.clear()
        session.run("二")
        assert session.turns[0]["number"] == 1
