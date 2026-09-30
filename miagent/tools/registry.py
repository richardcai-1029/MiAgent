"""工具注册表：Agent 手里所有工具的唯一入口。

它同时服务两类调用方，给的东西不一样 —— 这正是「对模型统一、对框架分层」
落到代码上的样子：

    模型   ← to_model_schemas()   只有 name/description/inputSchema
    框架   ← get() / by_source()   看得到 source、权限、开销
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from ..protocol import ErrorCode
from .base import Tool, ToolResult, ToolSource
from .remote import MiClawTool

if TYPE_CHECKING:
    from ..client import MiClawClient


class ToolRegistry:
    """工具注册表：按名字登记本地与 MiClaw 工具，对模型给统一的 schema，对框架给来源与开销。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    # ------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            # 重名一定是 bug：要么复制粘贴忘了改名，要么两个来源撞名。
            # 静默覆盖会让其中一个工具永远调不到，且极难排查。
            raise ValueError(
                f"工具名重复: {tool.name}"
                f"（已有 {self._tools[tool.name].source}，欲注册 {tool.source}）"
            )
        self._tools[tool.name] = tool
        return tool

    def register_all(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def load_from_miclaw(self, client: MiClawClient) -> list[Tool]:
        """从 MiClaw 拉取工具并注册。

        服务端已按权限过滤过，这里拿到的都是本 Agent 真能调的。
        """
        loaded = [MiClawTool(d, client) for d in client.list_tools()]
        self.register_all(loaded)
        return loaded

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def by_source(self, source: ToolSource) -> list[Tool]:
        return [t for t in self._tools.values() if t.source is source]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    # ------------------------------------------------------------
    # 两类调用方各取所需
    # ------------------------------------------------------------

    def to_model_schemas(self) -> list[dict[str, Any]]:
        """给模型的工具列表。不含 source / 权限 / 开销。"""
        return [t.to_model_schema() for t in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """按名字调用。工具不存在也返回 ToolResult 而不是抛异常 ——
        模型幻觉出一个不存在的工具名是常事，这属于「正常剧情」。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content=f"没有名为 {name} 的工具。可用工具: {', '.join(sorted(self._tools))}",
                is_error=True,
                error_code=ErrorCode.AG_TOOL_NOT_REGISTERED,
                detail={"requested": name},
            )
        return tool.invoke(arguments)

    def describe(self) -> str:
        """给人看的一览表（调试和文档用），带上模型看不到的那些字段。"""
        lines = [f"{'工具名':<26}{'来源':<9}{'权限':<18}描述"]
        lines.append("-" * 88)
        for t in sorted(self._tools.values(), key=lambda x: (x.source, x.name)):
            # MiClaw 工具能出现在这里，说明服务端已经放行了它的权限，
            # 显示 "-" 会让人误以为它不需要权限。
            perm = t.required_permission or ("服务端已授权" if t.source is ToolSource.MICLAW else "无需权限")
            lines.append(f"{t.name:<26}{t.source:<9}{perm:<18}{t.description}")
        return "\n".join(lines)
