"""模型层测试。"""

import pytest

from miagent.llm import FakeLLM, plan_json, step, system, user
from miagent.protocol import AgentError, ErrorCode


class TestFakeLLM:
    def test_script_returns_in_order(self):
        llm = FakeLLM(script=["一", "二"])
        assert llm.complete([user("x")]).content == "一"
        assert llm.complete([user("x")]).content == "二"

    def test_exhausted_script_raises(self):
        """脚本用完通常意味着图的循环次数超出预期，必须报错而非静默返回。"""
        llm = FakeLLM(script=[])
        with pytest.raises(AgentError) as ei:
            llm.complete([user("x")])
        assert ei.value.code is ErrorCode.AG_LLM_UNAVAILABLE

    def test_responder_mode(self):
        llm = FakeLLM(responder=lambda msgs: f"收到 {len(msgs)} 条消息")
        assert llm.complete([system("s"), user("u")]).content == "收到 2 条消息"

    def test_records_what_it_saw(self):
        llm = FakeLLM(script=["ok"])
        llm.complete([user("你好")])
        assert llm.seen[-1][0].content == "你好"


class TestContextLimit:
    """端侧模型窗口小，超限必须在发出前拦住。"""

    def test_overflow_raises_ag_3001(self):
        llm = FakeLLM(script=["x"], context_limit=10)
        with pytest.raises(AgentError) as ei:
            llm.complete([user("这是一段很长的输入" * 5)])
        assert ei.value.code is ErrorCode.AG_CONTEXT_OVERFLOW
        assert "limit" in ei.value.detail

    def test_within_limit_passes(self):
        assert FakeLLM(script=["x"], context_limit=100).complete([user("短")]).content == "x"


class TestStats:
    def test_counts_and_timing(self):
        llm = FakeLLM(responder=lambda m: "ok")
        for _ in range(3):
            llm.complete([user("hi")])
        s = llm.stats()
        assert s["calls"] == 3 and s["total_ms"] >= 0

    def test_response_carries_metrics(self):
        r = FakeLLM(script=["ok"]).complete([user("hello")])
        assert r.prompt_chars == 5 and r.model == "fake"


class TestPlanHelpers:
    def test_plan_json_is_text_not_dict(self):
        """真实模型返回文本，helper 也返回文本，
        让 Planner 的解析逻辑在测试中走生产路径。"""
        import json
        raw = plan_json(step("system.get_battery", reason="查电量"))
        parsed = json.loads(raw)
        assert parsed["steps"][0]["tool"] == "system.get_battery"

    def test_step_collects_arguments(self):
        assert step("system.send_sms", to="10086", text="hi")["arguments"] == {
            "to": "10086", "text": "hi"}


class TestStructuredOutput:
    """结构化输出：schema 由 Pydantic 模型导出，与校验同源。"""

    def test_valid_output_parses(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = FakeLLM(script=['{"tasks":[{"id":"a","description":"d","required_tool":"echo"}]}'])
        assert llm.complete_structured([user("t")], Plan).tasks[0].required_tool == "echo"

    def test_schema_is_injected_into_prompt(self):
        """模型看到的格式说明来自 model_json_schema()，不是手写的一段话。"""
        from miagent.graph.schema import TaskPlan as Plan
        llm = FakeLLM(script=['{"tasks":[]}'])
        llm.complete_structured([user("t")], Plan)
        assert "JSON Schema" in llm.seen[-1][-1].content

    def test_self_repair_recovers_from_bad_output(self):
        """第一次不合规 -> 把错误喂回去 -> 第二次改对。"""
        from miagent.graph.schema import TaskPlan as Plan
        outs = iter(["我建议先查电量。", '{"tasks":[{"id":"a","description":"d","required_tool":"echo"}]}'])
        llm = FakeLLM(responder=lambda m: next(outs))
        assert llm.complete_structured([user("t")], Plan).tasks[0].required_tool == "echo"
        assert llm.repair_count == 1

    def test_repair_prompt_carries_the_actual_error(self):
        """自修复的关键是告诉模型「错在哪」，而不是原样重试。"""
        from miagent.graph.schema import TaskPlan as Plan
        outs = iter(['{"plan":[]}', '{"tasks":[]}'])
        llm = FakeLLM(responder=lambda m: next(outs))
        llm.complete_structured([user("t")], Plan)
        assert "tasks" in llm.seen[-1][-1].content     # 修复提示里指出了缺失字段

    def test_gives_up_after_max_repairs(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = FakeLLM(responder=lambda m: "永远不合规")
        with pytest.raises(AgentError) as ei:
            llm.complete_structured([user("t")], Plan, max_repairs=2)
        assert ei.value.code is ErrorCode.AG_LLM_INVALID_RESPONSE
        assert llm.call_count == 3                     # 原始 1 次 + 修复 2 次

    def test_tool_name_enum_rejects_hallucination(self):
        """工具名收进 enum 后，幻觉在校验阶段即被拒。"""
        from miagent.graph.schema import task_plan_model_for as plan_model_for
        M = plan_model_for(["echo", "system.get_battery"])
        llm = FakeLLM(responder=lambda m: '{"tasks":[{"id":"a","description":"d","required_tool":"system.open_wechat"}]}')
        with pytest.raises(AgentError):
            llm.complete_structured([user("t")], M, max_repairs=0)

    def test_native_support_is_declared_per_implementation(self):
        """FakeLLM 靠 prompt+校验；云端实现声明原生支持并覆写策略。"""
        from miagent.llm.openai_compatible import OpenAICompatibleLLM
        assert FakeLLM(script=[]).supports_native_structured_output is False
        assert OpenAICompatibleLLM.supports_native_structured_output is True
