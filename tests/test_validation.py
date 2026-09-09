"""工具参数的校验与归一化。

重点覆盖「允许什么、禁止什么」这条边界 —— 放宽了会让形式合法、语义错误的
调用被真实执行，收紧了则会把本可自动吸收的格式漂移变成一轮多余的端侧推理。
"""

import pytest

from miagent.protocol import AgentError, ErrorCode
from miagent.tools import ToolRegistry, tool
from miagent.tools.validation import normalize


def schema(props, required=None):
    return {"type": "object", "properties": props, "required": required or []}


def reject(args, sch):
    with pytest.raises(AgentError) as e:
        normalize(args, sch)
    assert e.value.code is ErrorCode.AG_TOOL_SCHEMA_INVALID
    return e.value


class TestLosslessCoercion:
    """无歧义的漂移就地吸收，不必多烧一轮推理让模型重写。"""

    @pytest.mark.parametrize("given,expected", [
        ("3", 3), (" 3 ", 3), ("-7", -7), (3, 3), (3.0, 3),
    ])
    def test_integer_accepts_lossless_forms(self, given, expected):
        out = normalize({"n": given}, schema({"n": {"type": "integer"}}))
        assert out["n"] == expected and isinstance(out["n"], int)

    @pytest.mark.parametrize("given,expected", [("3.5", 3.5), (3, 3), (3.5, 3.5)])
    def test_number_accepts_lossless_forms(self, given, expected):
        assert normalize({"x": given}, schema({"x": {"type": "number"}}))["x"] == expected

    @pytest.mark.parametrize("given,expected", [
        ("true", True), ("false", False), ("True", True), (True, True),
    ])
    def test_boolean_accepts_json_and_python_spellings(self, given, expected):
        assert normalize({"b": given}, schema({"b": {"type": "boolean"}}))["b"] is expected

    def test_scalar_is_wrapped_into_array(self):
        """只有一个元素时模型常省掉数组包装，补回来是无歧义的。"""
        out = normalize({"xs": "a"}, schema({"xs": {"type": "array"}}))
        assert out["xs"] == ["a"]

    def test_array_is_left_alone(self):
        out = normalize({"xs": ["a", "b"]}, schema({"xs": {"type": "array"}}))
        assert out["xs"] == ["a", "b"]

    def test_untyped_property_passes_through(self):
        out = normalize({"anything": {"k": 1}}, schema({"anything": {}}))
        assert out["anything"] == {"k": 1}


class TestAmbiguousInputIsRejected:
    """有歧义的一律打回。猜错会产生一个被真实执行的错误调用，比失败危险。"""

    def test_unconvertible_text_is_rejected(self):
        assert "应为整数" in reject({"n": "三"}, schema({"n": {"type": "integer"}})).message

    def test_fractional_float_is_not_truncated(self):
        reject({"n": 3.5}, schema({"n": {"type": "integer"}}))

    def test_boolean_is_not_an_integer(self):
        """bool 是 int 的子类，不显式排除就会被当作合法整数放行。"""
        reject({"n": True}, schema({"n": {"type": "integer"}}))

    def test_number_is_not_stringified(self):
        """63 与 63.0 会得到不同的文本，该由模型明确写出它要哪一个。"""
        reject({"s": 63}, schema({"s": {"type": "string"}}))

    def test_unknown_field_is_rejected_not_dropped(self):
        """静默丢弃会让调用以模型没有预期的方式执行。"""
        e = reject({"a": 1, "extra": "x"}, schema({"a": {"type": "integer"}}))
        assert "extra" in e.message

    def test_missing_required_is_never_guessed(self):
        assert "缺少必填参数 a" in reject({}, schema({"a": {"type": "string"}}, ["a"])).message

    def test_enum_violation_is_rejected(self):
        sch = schema({"mode": {"type": "string", "enum": ["on", "off"]}})
        assert "必须是" in reject({"mode": "maybe"}, sch).message

    def test_enum_check_runs_after_coercion(self):
        sch = schema({"n": {"type": "integer", "enum": [1, 2]}})
        assert normalize({"n": "2"}, sch)["n"] == 2
        reject({"n": "3"}, sch)


class TestProblemsAreReportedTogether:
    """错误信息会被喂回给模型做自修复，一次说清才可能一次改对。"""

    def test_all_problems_in_one_error(self):
        e = reject({"extra": "x", "n": "三"},
                   schema({"a": {"type": "string"}, "n": {"type": "integer"}}, ["a"]))
        assert len(e.detail["problems"]) == 3
        for fragment in ("缺少必填参数 a", "没有名为 extra 的参数", "参数 n"):
            assert fragment in e.message


class TestThroughToolInvoke:
    """经 Tool.invoke 的完整路径：工具拿到的是归一化之后的值。"""

    @pytest.fixture
    def registry(self):
        @tool()
        def add_days(date: str, days: int) -> str:
            """把日期加上若干天。

            Args:
                date: 起始日期
                days: 天数
            """
            return f"{date}+{days}:{type(days).__name__}"

        return ToolRegistry([add_days])

    def test_tool_receives_the_declared_type(self, registry):
        res = registry.invoke("add_days", {"date": "2026-09-09", "days": "3"})
        assert not res.is_error
        assert res.content.endswith(":int"), "工具拿到的仍是字符串，归一化没生效"

    def test_extra_argument_is_a_schema_error_not_an_internal_error(self, registry):
        """此前多余参数会以 TypeError 的形式落到 AG-2004（工具内部错误），
        分类错了，给模型的行动建议也就不对。"""
        res = registry.invoke("add_days", {"date": "d", "days": 1, "extra": "x"})
        assert res.error_code is ErrorCode.AG_TOOL_SCHEMA_INVALID

    def test_error_detail_names_the_tool(self, registry):
        res = registry.invoke("add_days", {"date": "d"})
        assert res.detail["tool"] == "add_days"
