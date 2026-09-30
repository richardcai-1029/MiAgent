"""任务内核：状态模型、任务调度算法、数据流、校验把关、计划 schema。

Agent 里凡是有确定答案的判断都在这里，全部是纯函数：

    state     AgentState 与任务、执行结果、情景记忆等记录的类型定义
    dag       任务调度算法：依赖解析、就绪判定、级联失败、死锁与完成判定、关键路径排序
    dataflow  任务间数据流：参数里的 $from 引用 → 依赖边，派发前落实为上游结果
    verify    语义校验的效力边界：模型判定能改动什么、修正参数能不能派发
    schema    交给模型的结构化 schema：计划、校验判定、收尾输出

★ 本包不调用模型、不 import 任何编排框架，只依赖 protocol 与 tools。
  memory、agent、runtime 都建立在它之上；移植到别的编排框架时原样复用。
"""

from . import dag, dataflow, verify
from .schema import (FinalOutput, GoalReview, ResultReview, ResultReviewBatch,
                     TaskPlan, TaskSpec, result_review_model_for, task_plan_model_for)
from .state import (
    AgentState,
    Anchor,
    DispatchItem,
    Episode,
    Review,
    Task,
    TaskOutcome,
    TaskStatus,
    Turn,
    initial_state,
    new_task,
)

__all__ = [
    "AgentState", "Anchor", "DispatchItem", "Episode", "FinalOutput", "GoalReview",
    "ResultReview", "ResultReviewBatch", "Review", "Task", "TaskOutcome",
    "TaskPlan", "TaskSpec", "TaskStatus", "Turn",
    "dag", "dataflow", "initial_state", "new_task", "result_review_model_for",
    "task_plan_model_for", "verify",
]
