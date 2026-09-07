"""MiClaw 系统工具：经协议调用的系统能力。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..protocol import ToolDescriptor
from .base import Tool, ToolSource

if TYPE_CHECKING:                      # 只为类型注解导入，运行时不加载，
    from ..client import MiClawClient  # 保持工具层不硬依赖客户端


class MiClawTool(Tool):
    """把一个远端工具描述包装成本地可调用的 Tool。

    _run 之所以这么短，是因为错误处理已经在两个地方做完了：
      · 客户端把 JSON-RPC error 还原成了 MiClawError
      · Tool.invoke 捕获 MiAgentError，把 MC- 码原样放进 ToolResult
    于是 MC-4001 这类码能一路传到重试策略那里，中途不丢信息。
    """

    source = ToolSource.MICLAW

    def __init__(
        self,
        descriptor: ToolDescriptor,
        client: MiClawClient,
        required_permission: str | None = None,
    ) -> None:
        self.name = descriptor.name
        self.description = descriptor.description
        self.input_schema = descriptor.inputSchema
        self.required_permission = required_permission
        self._client = client

    def _run(self, args: dict[str, Any]) -> str:
        return self._client.call_tool(self.name, args)
