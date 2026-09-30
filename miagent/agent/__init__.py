"""Agent 节点：规划、调度、执行、校验、评估、重规划、收尾。

每个节点都是「读 State，返回要更新的字段」的普通函数，签名统一为
`(state, deps) -> dict`，不 import 任何编排框架。节点怎样连成图、并行分支
怎样扇出与合并，由 miagent.adapters 下的适配层负责。

    节点（图中名称）                  实现                    调用模型
    Planner         planner           planning.planner        是
    Scheduler       scheduler         scheduler.scheduler     否
    LocalExecutor   local_executor    executor.executor       否
    MiClawExecutor  miclaw_executor   executor.executor       否
    ResultVerifier  result_verifier   verifiers.result_verifier  是
    GoalVerifier    goal_verifier     verifiers.goal_verifier    是
    Evaluator       evaluator         evaluator.evaluator     否
    Replanner       replanner         planning.replanner      是
    Finalizer       finalizer         finalizer.finalizer     是

职责划分的一条硬线，分界是**问题有没有确定答案**：

    交给模型的： 把目标拆成什么任务、任务之间有什么依赖、失败后换什么方案、
                 拿到的结果算不算达成了任务的目的
    绝不交给的： 依赖是否满足、是否超出重试上限、是否全部完成、
                 哪些可以并行、是否死锁、参数合不合 schema

  后者都有确定答案，写在 core.dag 与 tools.validation 里，是纯函数、可穷举测试。
  用模型去猜一个我们确定知道答案的问题，既慢又不可靠。

  前者的最后一项由校验节点承担，但模型给出的只是判定：判定能改动什么、
  修正的参数能不能派发，仍由 core.verify 的纯函数决定（只收紧不放宽）。
"""

from .deps import Deps, Limits
from .evaluator import evaluator
from .executor import executor
from .finalizer import finalizer
from .planning import planner, replanner
from .scheduler import scheduler
from .verifiers import goal_verifier, result_verifier

__all__ = [
    "Deps", "Limits",
    "evaluator", "executor", "finalizer", "goal_verifier", "planner",
    "replanner", "result_verifier", "scheduler",
]
