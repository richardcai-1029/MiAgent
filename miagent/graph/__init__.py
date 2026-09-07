"""Agent 执行图：任务 DAG + 确定性调度。

图的拓扑（与设计稿一致）：

    START → Planner → Scheduler ─┬→ LocalTool  ─┐
                                 ├→ MCPExecutor ┴→ Evaluator ─┬→ Scheduler
                                 └→ Finalizer → END           ├→ Replanner → Scheduler
                                                              └→ Finalizer

模型负责「拆成什么任务、依赖怎么连、失败后换什么方案」；
依赖是否满足、能否重试、是否完成、是否死锁，全部由 dag.py 的
纯函数确定性地判定。
"""

from . import dag
from .build import build_agent
from .nodes import Deps
from .schema import TaskPlan, TaskSpec, task_plan_model_for
from .state import (
    MAX_ATTEMPTS_PER_TASK,
    MAX_REPLANS,
    MAX_TOTAL_EXECUTIONS,
    AgentState,
    Task,
    TaskOutcome,
    TaskStatus,
    initial_state,
    new_task,
)

__all__ = [
    "MAX_ATTEMPTS_PER_TASK", "MAX_REPLANS", "MAX_TOTAL_EXECUTIONS",
    "AgentState", "Deps", "Task", "TaskOutcome", "TaskPlan", "TaskSpec", "TaskStatus",
    "build_agent", "dag", "initial_state", "new_task", "task_plan_model_for",
]
