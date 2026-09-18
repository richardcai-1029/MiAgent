"""Agent 执行图：任务 DAG + 确定性调度。

图的拓扑：

    START → Planner → Scheduler ─┬→ LocalTool  ─┐
                                 ├→ MCPExecutor ┴→ Evaluator ─┬→ Scheduler
                                 └→ Finalizer → END           ├→ Replanner → Scheduler
                                                              └→ Finalizer

模型负责「拆成什么任务、依赖怎么连、失败后换什么方案」；
依赖是否满足、能否重试、是否完成、是否死锁，全部由 dag.py 的
纯函数确定性地判定。

★ build_agent 采用惰性导入：只有真正要构图时才加载 langgraph。
  依赖解析（dag）、状态定义（state）、计划 schema 都是纯 Python，
  端侧若只需要这几层，不必为图引擎付出常驻内存与冷启动的代价。
"""

from typing import TYPE_CHECKING, Any

from . import dag
from .nodes_meta import Deps  # noqa: F401  轻量转发，见该模块说明
from .schema import FinalOutput, TaskPlan, TaskSpec, task_plan_model_for
from .state import (
    MAX_ATTEMPTS_PER_TASK,
    MAX_REPLANS,
    MAX_TOTAL_EXECUTIONS,
    AgentState,
    DispatchItem,
    Task,
    TaskOutcome,
    TaskStatus,
    initial_state,
    new_task,
)

if TYPE_CHECKING:
    from .build import build_agent

__all__ = [
    "MAX_ATTEMPTS_PER_TASK", "MAX_REPLANS", "MAX_TOTAL_EXECUTIONS",
    "AgentState", "Deps", "DispatchItem", "FinalOutput", "Task", "TaskOutcome",
    "TaskPlan", "TaskSpec", "TaskStatus",
    "build_agent", "dag", "initial_state", "new_task", "task_plan_model_for",
]


def __getattr__(name: str) -> Any:
    """PEP 562 惰性导出：访问 build_agent 时才加载 langgraph。"""
    if name == "build_agent":
        from .build import build_agent

        return build_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
