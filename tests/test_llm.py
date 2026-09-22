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


class TestDirtyOutputCleaning:
    """端侧模型不保证每次都输出规整 JSON。解析入口吸收无歧义的脏输出，
    其余交给自修复重试 —— 不能拿到什么就直接当计划执行。"""

    def _extract(self, raw):
        from miagent.llm.base import extract_json
        return extract_json(raw)

    @pytest.mark.parametrize("raw", [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '好的，这是计划：\n{"a": 1}\n希望有帮助！',
        '思考：需要一步\n```json\n{"a": 1}\n```\n完成',
    ])
    def test_wrappers_are_stripped(self, raw):
        import json
        assert json.loads(self._extract(raw)) == {"a": 1}

    def test_only_the_first_object_is_taken(self):
        """输出里有两个 JSON 块时，取「第一个 { 到最后一个 }」会把两块连同
        中间的文字一起截出来，既不是前者也不是后者。"""
        import json
        assert json.loads(self._extract('先看 {"a": 1} 再看 {"a": 2}')) == {"a": 1}

    @pytest.mark.parametrize("raw,expected", [
        ('{"xs": [1, 2,]}', {"xs": [1, 2]}),
        ('{"a": 1,}', {"a": 1}),
        ('{"o": {"k": 1,},}', {"o": {"k": 1}}),
    ])
    def test_structural_trailing_commas_are_removed(self, raw, expected):
        import json
        assert json.loads(self._extract(raw)) == expected

    def test_commas_inside_strings_are_untouched(self):
        """扫描区分字符串内外，去掉的一定是结构上的逗号。"""
        import json
        raw = '{"s": "a, b", "t": "c,"}'
        assert json.loads(self._extract(raw)) == {"s": "a, b", "t": "c,"}

    @pytest.mark.parametrize("raw", [
        '{"a": [{"id": "t1"',        # 截断
        "{'a': 1}",                  # 单引号
        '{“a”: 1}',                  # 全角引号
        '不需要调用工具，电量是 63%。',   # 纯文本回答
    ])
    def test_ambiguous_forms_are_left_to_self_repair(self, raw):
        """这些的修复方式不唯一。补出来的括号位置只是猜测，而猜出来的计划
        会被真实执行 —— 让它解析失败并触发自修复更安全。"""
        import json
        with pytest.raises(ValueError):
            json.loads(self._extract(raw))

    def test_unparseable_text_is_returned_as_is(self):
        """原文要连同错误一起喂回给模型，截过的文本会让它看不出错在哪。"""
        assert self._extract("完全不是 JSON") == "完全不是 JSON"


class TestOpenAICompatible:
    """云端实现：不联网，注入假 client 验证请求形态与流式拼接。"""

    class _Client:
        """最小假 openai client，记录收到的请求参数，按 stream 返回不同形态。"""

        def __init__(self, text="hi", chunks=None):
            self.text, self.chunks, self.calls = text, chunks, []
            self.chat = self
            self.completions = self

        def create(self, **kw):
            from types import SimpleNamespace as NS
            self.calls.append(kw)
            if kw.get("stream"):
                pieces = [NS(choices=[NS(delta=NS(content=c))]) for c in self.chunks]
                pieces.append(NS(choices=[]))          # include_usage 时的收尾 chunk
                return iter(pieces)
            return NS(choices=[NS(message=NS(content=self.text))])

    def _llm(self, **kw):
        from miagent.llm.openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM(model="m", base_url="http://x", api_key="k", **kw)

    def test_non_stream_returns_message_content(self):
        llm = self._llm()
        llm._client = self._Client(text="答")
        assert llm.complete([user("q")]).content == "答"
        assert "stream" not in llm._client.calls[0]

    def test_stream_joins_deltas_and_skips_usage_chunk(self):
        llm = self._llm(stream=True)
        llm._client = self._Client(chunks=["你", "好", None, "！"])
        assert llm.complete([user("q")]).content == "你好！"
        assert llm._client.calls[0]["stream"] is True

    def test_extra_body_is_passed_through(self):
        llm = self._llm(extra_body={"modalities": ["text"]})
        llm._client = self._Client()
        llm.complete([user("q")])
        assert llm._client.calls[0]["extra_body"] == {"modalities": ["text"]}

    def test_native_structured_uses_response_format(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = self._llm()
        llm._client = self._Client(text='{"tasks":[]}')
        llm.complete_structured([user("t")], Plan)
        assert llm._client.calls[0]["response_format"]["type"] == "json_schema"

    def test_non_native_structured_falls_back_to_prompt_schema(self):
        """关掉原生模式后：不带 response_format，schema 由基类注入 prompt。"""
        from miagent.graph.schema import TaskPlan as Plan
        llm = self._llm(native_structured_output=False)
        llm._client = self._Client(text='{"tasks":[]}')
        llm.complete_structured([user("t")], Plan)
        call = llm._client.calls[0]
        assert "response_format" not in call
        assert "JSON Schema" in call["messages"][-1]["content"]

    def test_sdk_errors_become_ag_5001(self):
        class Boom(self._Client):
            def create(self, **kw):
                raise ConnectionError("refused")
        llm = self._llm()
        llm._client = Boom()
        with pytest.raises(AgentError) as ei:
            llm.complete([user("q")])
        assert ei.value.code is ErrorCode.AG_LLM_UNAVAILABLE
        assert ei.value.detail["cause"] == "ConnectionError"


class TestQwen:
    """Qwen 预设：只检查开关是否按 DashScope 的要求预置，不发请求。"""

    def test_omni_presets(self):
        from miagent.llm.qwen import QwenLLM
        llm = QwenLLM(api_key="k")
        assert llm.name == "qwen2.5-omni-7b"
        assert llm._stream is True
        assert llm.supports_native_structured_output is False
        assert llm._extra_body == {"modalities": ["text"]}

    def test_non_omni_model_has_no_modalities(self):
        from miagent.llm.qwen import QwenLLM
        assert QwenLLM(model="qwen-plus", api_key="k")._extra_body == {}

    def test_api_key_from_env(self, monkeypatch):
        from miagent.llm.qwen import QwenLLM
        monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
        assert QwenLLM()._api_key == "from-env"

    def test_api_key_from_dotenv(self, monkeypatch, tmp_path):
        """环境变量没设时，退回读当前目录的 .env；注释、引号、export 前缀都要能处理。"""
        from miagent.llm.qwen import QwenLLM
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "# 注释\nOTHER=1\nexport DASHSCOPE_API_KEY='from-dotenv'\n", encoding="utf-8")
        assert QwenLLM()._api_key == "from-dotenv"

    def test_env_var_beats_dotenv(self, monkeypatch, tmp_path):
        from miagent.llm.qwen import QwenLLM
        monkeypatch.setenv("DASHSCOPE_API_KEY", "from-env")
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=from-dotenv\n", encoding="utf-8")
        assert QwenLLM()._api_key == "from-env"

    def test_missing_api_key_raises_ag_5001(self, monkeypatch, tmp_path):
        from miagent.llm.qwen import QwenLLM
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)                 # 没有 .env 的目录
        with pytest.raises(AgentError) as ei:
            QwenLLM()
        assert ei.value.code is ErrorCode.AG_LLM_UNAVAILABLE


class TestStructuredGoesThroughGuard:
    """结构化补全与普通补全共用同一段检查与计数，原生路径也不例外。"""

    def _native(self, text='{"tasks":[]}', **kw):
        from miagent.llm.openai_compatible import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(model="m", base_url="http://x", api_key="k", **kw)
        llm._client = TestOpenAICompatible._Client(text=text)
        return llm

    def test_native_structured_is_counted(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = self._native()
        llm.complete_structured([user("t")], Plan)
        assert llm.stats()["calls"] == 1

    def test_non_native_structured_is_counted_once(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = self._native(native_structured_output=False)
        llm.complete_structured([user("t")], Plan)
        assert llm.stats()["calls"] == 1

    def test_native_structured_respects_context_limit(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = self._native(context_limit=10)
        with pytest.raises(AgentError) as ei:
            llm.complete_structured([user("x" * 11)], Plan)
        assert ei.value.code is ErrorCode.AG_CONTEXT_OVERFLOW
        assert llm._client.calls == []            # 没发出去

    def test_repair_round_is_a_second_call(self):
        from miagent.graph.schema import TaskPlan as Plan
        llm = FakeLLM(["not json", '{"tasks":[]}'])
        llm.complete_structured([user("t")], Plan)
        assert llm.stats() == {**llm.stats(), "calls": 2, "repairs": 1}
