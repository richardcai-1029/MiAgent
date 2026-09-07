"""工具层测试。"""

import pytest

from miagent.client import MiClawClient
from miagent.protocol import ErrorCode, RetryPolicy, retry_policy
from miagent.tools import ToolRegistry, ToolResult, ToolSource, tool
from miagent.tools.local import schema_from_signature


@tool()
def sample(name: str, count: int, unit: str = "个") -> str:
    """演示用工具。

    Args:
        name: 物品名称
        count: 数量
        unit: 单位
    """
    return f"{name} {count}{unit}"


@tool()
def boom(x: str) -> str:
    """总是抛异常的工具。

    Args:
        x: 随便什么
    """
    raise RuntimeError("炸了")


class TestSchemaGeneration:
    """schema 由函数签名生成，避免签名与 schema 两份拷贝不同步。"""

    def test_types_are_mapped(self):
        s = schema_from_signature(sample._fn)
        assert s["properties"]["name"]["type"] == "string"
        assert s["properties"]["count"]["type"] == "integer"

    def test_params_without_default_are_required(self):
        assert schema_from_signature(sample._fn)["required"] == ["name", "count"]

    def test_arg_docs_come_from_docstring(self):
        s = schema_from_signature(sample._fn)
        assert s["properties"]["name"]["description"] == "物品名称"

    def test_description_excludes_args_section(self):
        """给模型的描述只取首段，不该混进 Args: 列表。"""
        assert sample.description == "演示用工具。"


class TestLocalTool:
    def test_success(self):
        r = sample.invoke({"name": "苹果", "count": 3})
        assert r.is_error is False and r.content == "苹果 3个"

    def test_missing_required_param(self):
        r = sample.invoke({"name": "苹果"})
        assert r.error_code is ErrorCode.AG_TOOL_SCHEMA_INVALID

    def test_exception_becomes_result_not_raise(self):
        """工具炸了不能把图带崩 —— 必须变成结果回给模型。"""
        r = boom.invoke({"x": "1"})
        assert r.is_error and r.error_code is ErrorCode.AG_TOOL_EXECUTION_FAILED

    def test_source_is_local(self):
        assert sample.source is ToolSource.LOCAL


class TestRetryPolicyByBand:
    """段位决定策略，不查逐条表 —— 新增错误码自动继承。"""

    @pytest.mark.parametrize("code, expected", [
        (ErrorCode.MC_REQUEST_TIMEOUT,       RetryPolicy.BACKOFF),
        (ErrorCode.MC_NOT_INITIALIZED,       RetryPolicy.REHANDSHAKE),
        (ErrorCode.MC_TOOL_NOT_FOUND,        RetryPolicy.NONE),
        (ErrorCode.MC_RESOURCE_MEMORY_LIMIT, RetryPolicy.DEGRADE),
        (ErrorCode.MC_PERMISSION_DENIED,     RetryPolicy.ASK_USER),
        (ErrorCode.AG_PLAN_PARSE_FAILED,     RetryPolicy.NONE),
    ])
    def test_policy(self, code, expected):
        assert retry_policy(code) is expected

    def test_every_mc_code_has_a_policy(self):
        """任何 MC- 码都必须能推出策略，不能落到未知分支。"""
        for c in ErrorCode:
            if c.startswith("MC-"):
                assert retry_policy(c) in set(RetryPolicy)


class TestToolResultForModel:
    """给模型的文本必须回答「那我该怎么办」，只说失败会让它原样重试。"""

    def test_success_is_plain_content(self):
        assert ToolResult("好了").for_model() == "好了"

    def test_resource_error_tells_model_to_degrade(self):
        r = ToolResult("内存不够", is_error=True,
                       error_code=ErrorCode.MC_RESOURCE_MEMORY_LIMIT)
        assert "不要重试" in r.for_model() and "MC-4001" in r.for_model()

    def test_transient_error_tells_model_to_retry(self):
        r = ToolResult("超时", is_error=True, error_code=ErrorCode.MC_REQUEST_TIMEOUT)
        assert "重试" in r.for_model() and "不要重试" not in r.for_model()


class TestRegistry:
    def test_duplicate_name_rejected(self):
        """重名必然是 bug，静默覆盖会让其中一个工具永远调不到。"""
        r = ToolRegistry([sample])
        with pytest.raises(ValueError, match="重复"):
            r.register(sample)

    def test_unknown_tool_returns_result_not_raise(self):
        """模型幻觉出不存在的工具名是常事，属于正常剧情。"""
        res = ToolRegistry([sample]).invoke("nope")
        assert res.error_code is ErrorCode.AG_TOOL_NOT_REGISTERED
        assert "sample" in res.content          # 顺带告诉模型有哪些可用

    def test_model_schema_hides_framework_fields(self):
        """★ 对模型统一：只给 name/description/inputSchema 三样。"""
        for s in ToolRegistry([sample]).to_model_schemas():
            assert set(s) == {"name", "description", "inputSchema"}


class TestMiClawToolEndToEnd:
    """远端工具走完整链路：服务端错误码要能一路传到重试策略。"""

    @pytest.fixture
    def registry(self):
        client = MiClawClient()
        client.connect("com.xiaomi.test",
                       permissions=["sms.send", "alarm.write", "screen.capture"])
        reg = ToolRegistry()
        reg.load_from_miclaw(client)
        yield reg
        client.close()

    def test_remote_call_succeeds(self, registry):
        assert "电量" in registry.invoke("system.get_battery").content

    def test_source_is_miclaw(self, registry):
        assert registry.get("system.get_battery").source is ToolSource.MICLAW

    def test_mc_code_survives_all_the_way_to_retry_policy(self, registry):
        """MC-4001 从服务端 -> 网络 -> 客户端 -> 工具层 -> 重试策略，全程不丢。"""
        r = registry.invoke("system.capture_screen")
        assert r.error_code is ErrorCode.MC_RESOURCE_MEMORY_LIMIT
        assert r.retry_policy is RetryPolicy.DEGRADE
        assert "不要重试" in r.for_model()

    def test_local_and_remote_share_one_entrypoint(self, registry):
        """统一抽象的意义：调用方不需要知道工具在哪。"""
        registry.register(sample)
        assert registry.invoke("sample", {"name": "梨", "count": 2}).is_error is False
        assert registry.invoke("system.get_battery").is_error is False

    def test_business_failure_differs_from_call_failure(self, registry):
        """MCP 的 isError（工具跑了但业务失败）不能被当成成功。

        查无此单 -> MC-3003 -> 段位 3 -> 重试无用，让模型换方案。
        这与「调用根本没成立」（JSON-RPC error）是两回事。
        """
        ok = registry.invoke("system.query_order", {"order_id": "SN001"})
        assert ok.is_error is False

        biz = registry.invoke("system.query_order", {"order_id": "SN999"})
        assert biz.is_error is True
        assert biz.error_code is ErrorCode.MC_TOOL_EXECUTION_FAILED
        assert biz.retry_policy is RetryPolicy.NONE

    def test_unpermitted_tool_never_enters_registry(self, registry):
        """服务端已过滤，所以模型根本看不到无权限的工具。"""
        assert "system.read_contacts" not in registry
