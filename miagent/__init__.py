"""MiAgent：适配 MiClaw 系统级 Agent 生态的轻量化通用 Agent 框架。

常用名字从这里取；每个名字在第一次访问时才导入所在模块，`import miagent`
本身不加载任何子包 —— 瘦客户端形态不为用不到的层付出导入代价。

    from miagent import build_agent, ToolRegistry, tool, FakeLLM, Session

分层（依赖只能自上而下，见 tests/test_layering.py）：

    runtime    多请求运行时：请求级并发、共享资源按优先级仲裁
    adapters   编排框架适配：照 agent.topology 构图（langgraph）
    agent      Agent 节点与图拓扑，框架无关
    memory     上下文记忆：情景记忆、账本、目标锚、多轮会话
    core       任务内核：状态模型、任务调度算法、数据流、校验把关
    llm        模型层：统一契约 + Fake / OpenAI 兼容（Qwen、Ollama）实现
    tools      工具调用模块：本地工具与 MiClaw 系统工具的统一抽象
    client     MiClaw 客户端：握手、调用、收发分流
    transport  传输层：stdio 分帧 + 进程内回环
    protocol   协议层：JSON-RPC 2.0 报文 + MC-/AG- 错误码
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

_EXPORTS = {
    "build_agent": "miagent.adapters.langgraph",
    "Deps": "miagent.agent", "Limits": "miagent.agent",
    "AgentState": "miagent.core", "Task": "miagent.core", "TaskStatus": "miagent.core",
    "initial_state": "miagent.core",
    "Session": "miagent.memory",
    "Runtime": "miagent.runtime", "build_runtime": "miagent.runtime", "serve": "miagent.runtime",
    "LLM": "miagent.llm", "FakeLLM": "miagent.llm",
    "Tool": "miagent.tools", "ToolRegistry": "miagent.tools", "ToolResult": "miagent.tools",
    "ToolSource": "miagent.tools", "LocalTool": "miagent.tools", "MiClawTool": "miagent.tools",
    "tool": "miagent.tools",
    "MiClawClient": "miagent.client",
    "AgentError": "miagent.protocol", "MiClawError": "miagent.protocol",
    "ErrorCode": "miagent.protocol",
}

__all__ = [
    "AgentError", "AgentState", "Deps", "ErrorCode", "FakeLLM", "LLM", "Limits",
    "LocalTool", "MiClawClient", "MiClawError", "MiClawTool", "Runtime", "Session",
    "Task", "TaskStatus", "Tool", "ToolRegistry", "ToolResult", "ToolSource",
    "__version__", "build_agent", "build_runtime", "initial_state", "serve", "tool",
]

if TYPE_CHECKING:
    from .adapters.langgraph import build_agent
    from .agent import Deps, Limits
    from .client import MiClawClient
    from .core import AgentState, Task, TaskStatus, initial_state
    from .llm import LLM, FakeLLM
    from .memory import Session
    from .protocol import AgentError, ErrorCode, MiClawError
    from .runtime import Runtime, build_runtime, serve
    from .tools import LocalTool, MiClawTool, Tool, ToolRegistry, ToolResult, ToolSource, tool


def __getattr__(name: str) -> Any:
    """PEP 562 惰性导出。"""
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
