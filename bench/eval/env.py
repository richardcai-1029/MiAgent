"""评测环境：一条用例一套 MiClaw mock 服务端 + 客户端 + 工具注册表。

服务端提供评测工具目录（bench.suite.catalog），按用例声明的故障改写处理函数；
配额与授权按用例下发，因此内存超配额与权限被拒走的是服务端真实的检查路径。

客户端一侧给每个工具套上探针，记下每一次调用：工具、实际参数（已经过参数
归一化，与服务端收到的相同）、成败与错误码、起止时刻，以及同一时刻在途的
调用数。打分只看探针的记录，不看框架自己的状态 —— 框架说自己做了什么，
与工具实际被调了什么，是两回事。

tool_latency_ms 给每次调用加一段固定等待，放在客户端、服务端锁之外：
回环传输的服务端是串行处理的，不加等待时调用几乎瞬间完成，并行度无从观察。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from miagent.client import MiClawClient
from miagent.mock_server import MiClawMockServer, SystemTool
from miagent.protocol import ErrorCode, MiAgentError, MiClawError, ResourceBudget
from miagent.tools import ToolRegistry
from miagent.tools.remote import MiClawTool
from miagent.transport import LoopbackTransport

from ..suite.catalog import ALL_PERMISSIONS, CATALOG
from ..suite.execute import fault_for


@dataclass
class Call:
    seq: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    code: str | None
    t0: float
    t1: float
    turn: int


class Probe:
    """线程安全的调用记录与在途计数。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[Call] = []
        self.active = 0
        self.max_active = 0
        self.turn = 0

    def enter(self) -> float:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        return time.perf_counter()

    def leave(self, tool: str, args: dict[str, Any], ok: bool, code: str | None, t0: float) -> None:
        t1 = time.perf_counter()
        with self._lock:
            self.active -= 1
            self.calls.append(Call(len(self.calls), tool, args, ok, code, t0, t1, self.turn))

    def of_turn(self, turn: int) -> list[Call]:
        with self._lock:
            return [c for c in self.calls if c.turn == turn]


class ProbedTool(MiClawTool):
    """记下每次调用的 MiClaw 工具。"""

    def __init__(self, inner: MiClawTool, client: MiClawClient, probe: Probe,
                 latency_s: float) -> None:
        self.name, self.description = inner.name, inner.description
        self.input_schema = inner.input_schema
        self.required_permission = inner.required_permission
        self._client, self._probe, self._latency = client, probe, latency_s

    def _run(self, args: dict[str, Any]) -> str:
        t0 = self._probe.enter()
        try:
            if self._latency:
                time.sleep(self._latency)
            out = self._client.call_tool(self.name, args)
        except MiAgentError as e:
            self._probe.leave(self.name, args, False, e.code.value, t0)
            raise
        except Exception:
            self._probe.leave(self.name, args, False, "exception", t0)
            raise
        self._probe.leave(self.name, args, True, None, t0)
        return out


class FaultPlan:
    """按 (工具, 实际参数) 注入故障。第一次失败类故障按调用次数计。"""

    def __init__(self, faults: list[dict[str, Any]]) -> None:
        self.faults = faults
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def wrap(self, tool: SystemTool) -> SystemTool:
        def handler(args: dict[str, Any]) -> str | None:
            fault = fault_for(self.faults, tool.name, args)
            if fault is not None:
                key = tool.name + repr(sorted(args.items()))
                with self._lock:
                    n = self._attempts[key] = self._attempts.get(key, 0) + 1
                kind = fault["kind"]
                if kind == "transient_all" or (kind == "transient_first" and n == 1):
                    raise MiClawError(ErrorCode.MC_REQUEST_TIMEOUT, "与系统服务通信超时")
                if kind == "busy":
                    raise MiClawError(ErrorCode.MC_RESOURCE_BUSY, "资源已被占用")
            return tool.handler(args)

        return SystemTool(tool.name, tool.description, tool.input_schema,
                          tool.required_permission, tool.estimated_memory_mb, handler)


@dataclass
class Env:
    server: MiClawMockServer
    client: MiClawClient
    registry: ToolRegistry
    probe: Probe
    budget: ResourceBudget

    def close(self) -> None:
        self.client.close()


def make_env(budget: dict[str, Any], permissions: list[str],
             faults: list[dict[str, Any]] | None = None, tool_latency_ms: float = 0,
             probe: Probe | None = None) -> Env:
    """按用例的配额、授权与故障装好一套环境。

    Agent 申请全部权限，未授予的由服务端按「用户在弹窗里拒绝」处理：
    被拒的权限出现在注册结果里，需要它们的工具不出现在工具列表中。
    """
    plan = FaultPlan(faults or [])
    tools = {name: plan.wrap(spec.tool) for name, spec in CATALOG.items()}
    rb = ResourceBudget(**budget)
    server = MiClawMockServer(budget=rb, user_denied=set(ALL_PERMISSIONS) - set(permissions),
                              tools=tools)
    client = MiClawClient(transport=LoopbackTransport(server))
    client.connect("com.xiaomi.miagent.bench", list(ALL_PERMISSIONS))
    probe = probe or Probe()
    registry = ToolRegistry([ProbedTool(MiClawTool(d, client), client, probe, tool_latency_ms / 1000)
                             for d in client.list_tools()])
    return Env(server, client, registry, probe, rb)
