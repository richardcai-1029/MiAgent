"""账本：终态任务离开任务图时的结算。

Replanner 与 Finalizer 面对的是同一件事：把到了终态的任务从工作记忆（tasks）
结算进情景记忆（episodes），再决定图里还留什么、给模型或用户看什么。
这些步骤有先后，也有一处差一 —— 结算记的是刚结束的那一轮，新任务属于下一轮。
账本把它们收在一处，节点只拿结果写回，不再各自对齐一遍。

三个操作，对应三处调用：

    settle   重规划前：结算终态任务，给出可保留的已完成任务与提示词片段
    accept   重规划后：验收模型给出的新任务图 —— 剪枝、判无进展、定 failure
    close    收尾：结算剩余终态任务，给出执行概况与执行明细
    survey   完成校验：只读地取出执行明细，不结算、不改状态

failure 的规则在这里兑现：成功的重规划清除它（新计划按构造覆盖了全部
失败记录），重规划未产出新任务则取根因码，新计划全是失败过的调用则判无进展。
单个任务成功不清除 failure —— 那是节点侧的约定，见 graph.state。

★ 本模块是纯函数：不调用模型、不 import langgraph、没有副作用。
   episodes 只增不改：返回给节点的是新增的部分，由 reducer 追加；
   渲染所需的全量视图留在结果对象内部。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..graph import dag
from ..graph.state import Episode, Task, TaskStatus
from ..protocol import ErrorCode
from . import episodic

__all__ = ["Settlement", "Acceptance", "Closing", "Survey",
           "settle", "accept", "close", "survey"]


@dataclass(frozen=True)
class Settlement:
    """重规划前的结算结果。"""

    generation: int                 # 新任务所属的轮次：刚结束的那一轮 + 1
    new_episodes: list[Episode]     # 本次新增的记录，节点写回 episodes
    episodes: list[Episode]         # 含新增在内的全量视图，供 accept 使用
    keep: dict[str, Task]           # 可供新任务引用的已完成任务
    done: list[episodic.Line]       # 提示词：已完成，每条记录一行，已去重
    failed: list[episodic.Line]     # 提示词：失败


@dataclass(frozen=True)
class Acceptance:
    """重规划后的验收结果。"""

    tasks: dict[str, Task]          # 剪枝后的任务图；判无进展时为空
    failure: str | None             # None / 根因错误码 / AG-1003
    added: int                      # 新任务数
    kept: int                       # 为供引用而留在图里的已完成任务数
    repeated: list[str]             # 判无进展时被认定为重复的新任务 id；否则为空


@dataclass(frozen=True)
class Survey:
    """只读的执行明细快照。"""

    history: list[episodic.Line]    # 每任务一行
    n_settled: int                  # 已进入情景记忆的记录数


@dataclass(frozen=True)
class Closing:
    """收尾时的结算结果。"""

    tasks: dict[str, Task]          # 级联失败之后的任务图
    new_episodes: list[Episode]     # 本次新增的记录，节点写回 episodes
    summary: dict[str, Any]         # 执行概况
    history: list[episodic.Line]    # 执行明细，每任务一行


def settle(tasks: dict[str, Task], episodes: list[Episode],
           replan_count: int) -> Settlement:
    """结算终态任务。replan_count 是刚结束的那一轮，新任务属于下一轮。

    渲染按「刚结束的那一轮」降级：本轮记录带结果原文，更早的只剩 id 与描述。
    """
    ended = replan_count
    new = episodic.settle(tasks, episodes, ended)
    full = [*episodes, *new]
    done, failed = episodic.render(full, ended)
    return Settlement(
        generation=ended + 1, new_episodes=new, episodes=full,
        keep={tid: t for tid, t in tasks.items() if t["status"] is TaskStatus.DONE},
        done=done, failed=failed,
    )


def accept(settlement: Settlement, merged: dict[str, Task],
           standing: str | None = None, clear: bool = True) -> Acceptance:
    """验收模型给出的任务图。merged 是 keep 与新任务合并后、已通过结构校验的图。

    三步，顺序不可换：

      1. 原地打转：新任务里的每一个调用都失败过，执行只会得到同样的失败，
         判 AG-1003，不派发。
      2. 剪枝：只留新任务引用到的已完成任务（含其上游）——校验要求依赖边
         指向图里存在的任务，派发前要取它们的结果。其余终态任务已在情景记忆里。
      3. 未产出新任务：重规划是恢复机制，跑完却没有新任务说明恢复没发生，
         失败的部分不会再有人接手，据实标记为未达成，错误码取根因；失败记录里
         找不出根因时沿用 standing —— 进入重规划时尚未清除的那个码，
         完成校验判定的未达成就属于这种情况，它没有对应的失败任务。
         产出了新任务则清除 failure —— 新计划看到了全部失败记录并覆盖了剩余工作。

    「覆盖了剩余工作」是模型的一面之词：新计划可能只接手了与失败无关的任务，
    把失败的部分丢下。这时要靠完成校验在收尾前重新判定。clear 为 False
    表示之后没有完成校验兜底（语义校验已关闭），failure 就不因新计划而清除，
    宁可把恢复成功的轮次报为未达成，也不把失败报为完成。
    """
    new_ids = set(merged) - set(settlement.keep)
    new_tasks = {tid: merged[tid] for tid in new_ids}

    if episodic.repeats_calls(new_tasks, settlement.episodes):
        return Acceptance(tasks={}, failure=ErrorCode.AG_PLAN_NO_PROGRESS.value,
                          added=len(new_ids), kept=0, repeated=sorted(new_ids))

    needed = dag.ancestors(merged, new_ids)
    pruned = {tid: t for tid, t in merged.items() if tid in new_ids or tid in needed}
    failure = (_root_failure(settlement.episodes) or standing) if not new_ids or not clear else None
    return Acceptance(tasks=pruned, failure=failure, added=len(new_ids),
                      kept=len(needed), repeated=[])


def close(tasks: dict[str, Task], episodes: list[Episode], replan_count: int) -> Closing:
    """收尾结算。先级联失败 —— 中止路径上被牵连的任务没经过 Evaluator，
    收尾时仍要进入历史 —— 再结算剩余终态任务。"""
    tasks = dag.cascade_failures(tasks)
    new = episodic.settle(tasks, episodes, replan_count)
    full = [*episodes, *new]
    return Closing(tasks=tasks, new_episodes=new,
                   summary=episodic.summarize(tasks, full),
                   history=episodic.history(full, tasks))


def survey(tasks: dict[str, Task], episodes: list[Episode]) -> Survey:
    """执行明细的只读视图。

    完成校验要在收尾之前看到这一轮做了什么，但它不该动状态：结算发生在
    重规划与收尾两处，多一处就多一份「这条记录属于哪一代」的分歧。
    """
    return Survey(history=episodic.history(episodes, tasks), n_settled=len(episodes))


def _root_failure(episodes: list[Episode]) -> str | None:
    """从失败记录里挑出根因的错误码。

    级联失败的错误码统一是 AG-1005（前置任务失败），它只说明「被牵连」，
    对用户没有信息量。优先取自身失败的那一个；同为自身失败时取最近一轮的。
    """
    failed = [e for e in episodes if not e["ok"]]
    if not failed:
        return None
    own = [e for e in failed
           if e["error"] != ErrorCode.AG_DEPENDENCY_UNRESOLVED.value]
    return max(own or failed, key=lambda e: e["generation"])["error"]
