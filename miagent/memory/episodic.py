"""情景记忆：终态任务的压缩记录。

任务一旦 done / failed 就不再参与调度。此后它对流程只剩两个用途：
重规划时告诉模型「哪条路走过了、成没成」，收尾时告诉用户「做了什么」。
这两个用途都不需要完整的 Task —— dependencies、status、retry_count
全是调度期的东西。Episode 只保留剩下的部分。

去重与降级都发生在**渲染**时，不改动状态：episodes 字段只增不减，
同一份记录在不同时刻可以渲染成不同详略，规则确定、可穷举测试。

Episode 的类型定义在 graph.state（它是 State 的元素）；本模块是它的行为。
节点不直接调用这里的函数 —— 结算的步骤由 ledger 编排，这里是账本的内部接缝。

★ 本模块是纯函数：不依赖 LangGraph、不调用模型、没有副作用。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..graph import dag
from ..graph.state import Episode, Task, TaskStatus

__all__ = ["Episode", "args_digest", "settle", "dedupe", "repeats_calls",
           "render", "history", "summarize"]


def args_digest(arguments: dict[str, Any]) -> str:
    """参数的稳定指纹。键序无关，因此参数相同的两次调用指纹一定相同。

    对任务图里的原始参数计算（含 `$from` 引用），不对求值后的参数计算：
    这里要识别的是「模型又拆出了同一个调用」，而引用形式正是模型写下的形式。
    """
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _record(task: Task, generation: int) -> Episode:
    return Episode(
        task_id=task["id"], description=task["description"],
        tool=task["required_tool"], args_digest=args_digest(task["arguments"]),
        ok=task["status"] is TaskStatus.DONE, error=task["error"],
        result=task["result"], generation=generation,
    )


def settle(tasks: dict[str, Task], episodes: list[Episode],
           generation: int) -> list[Episode]:
    """把尚未记录的终态任务转成 Episode，返回**新增**的部分。

    在任务离开任务图之前调用。「已记录」按 task_id 判定，因此保留在图里的
    done 任务（下游还要引用它）不会在下一次结算时被重复记录。
    """
    recorded = {e["task_id"] for e in episodes}
    return [_record(t, generation) for tid, t in tasks.items()
            if tid not in recorded
            and t["status"] in (TaskStatus.DONE, TaskStatus.FAILED)]


def dedupe(episodes: list[Episode]) -> list[Episode]:
    """同 (tool, 参数指纹) 只保留最新一条，保持原有先后顺序。

    重规划常常把同一个调用再拆一次。给模型看两条一样的记录没有信息量，
    只占窗口。
    """
    latest: dict[tuple[str, str], int] = {}
    for i, e in enumerate(episodes):
        latest[(e["tool"], e["args_digest"])] = i
    keep = set(latest.values())
    return [e for i, e in enumerate(episodes) if i in keep]


def repeats_calls(new_tasks: dict[str, Task], episodes: list[Episode]) -> bool:
    """新计划是否只是把这一轮已经调过的调用原样再拆一遍。

    判据：新任务非空，且每一个的 (tool, 参数指纹) 都能在情景记忆里找到。
    只要有一个任务是新的，就不算 —— 模型可能在换方案的同时保留了某一步，
    那一步的结果由执行去回答，这里不猜。

    成败都算重复：重做失败过的调用会得到同样的失败，重做成功过的调用
    只是把已经有结果的事再做一遍，两者都不构成进展，而后者还会让带副作用
    的系统调用发生第二次。

    参数指纹取的是模型写下的原始形式（含 `$from` 引用），因此「同一个引用、
    上游结果却已不同」的情况不会被误判为重复：这种情况下引用指向的 id 本身
    已经变了。
    """
    if not new_tasks:
        return False
    called = {(e["tool"], e["args_digest"]) for e in episodes}
    return all((t["required_tool"], args_digest(t["arguments"])) in called
               for t in new_tasks.values())


def render(episodes: list[Episode], generation: int) -> tuple[str, str]:
    """渲染成重规划提示词里的「已完成」与「失败」两段。

    详略按代降级：本轮（generation 相同）的记录带结果原文；更早的只剩
    id 与描述 —— 早先几轮的结果在当时的重规划里已经被看过，此后模型只需
    知道那个 id 存在、做过什么，以便用 `$from` 引用。
    """
    done_lines: list[str] = []
    failed_lines: list[str] = []
    for e in dedupe(episodes):
        current = e["generation"] == generation
        if e["ok"]:
            tail = f" → {e['result']}" if current else f"（第 {e['generation']} 轮）"
            done_lines.append(f"  {e['task_id']}（已完成）: {e['description']}{tail}")
        else:
            tail = (f" → 失败 {e['error']}：{e['result']}" if current
                    else f" → 失败 {e['error']}（第 {e['generation']} 轮）")
            failed_lines.append(f"  {e['task_id']}: {e['description']}{tail}")
    return ("\n".join(done_lines) or "  （无）", "\n".join(failed_lines) or "  （无）")


def history(episodes: list[Episode], tasks: dict[str, Task]) -> list[str]:
    """完整的执行明细，供收尾使用：情景记忆在前，仍在图里且未记录的任务在后。

    后者出现在中止路径上 —— 预算耗尽、死锁时，图里还留着没跑完的任务，
    以及被级联标记为失败却没经过结算的任务。
    """
    lines = [
        f"  {e['task_id']}: {e['description']} → "
        + (f"成功：{e['result']}" if e["ok"] else f"失败（{e['error']}）：{e['result']}")
        for e in episodes
    ]
    recorded = {e["task_id"] for e in episodes}
    for tid, t in tasks.items():
        if tid in recorded:
            continue
        if t["status"] is TaskStatus.DONE:
            lines.append(f"  {tid}: {t['description']} → 成功：{t['result']}")
        elif t["status"] is TaskStatus.FAILED:
            lines.append(f"  {tid}: {t['description']} → 失败（{t['error']}）：{t['result']}")
        else:
            lines.append(f"  {tid}: {t['description']} → 未执行（{t['status']}）")
    return lines


def summarize(tasks: dict[str, Task], episodes: list[Episode]) -> dict[str, Any]:
    """执行概况。终态部分以情景记忆为准，再补上图里尚未结算的任务。

    同一个任务可能既在情景记忆里、也还留在图里（下游要引用它），按 id 去重。
    """
    completed: list[str] = []
    failed: list[str] = []
    results: dict[str, str] = {}
    seen: set[str] = set()

    for e in episodes:
        if e["task_id"] in seen:
            continue
        seen.add(e["task_id"])
        (completed if e["ok"] else failed).append(e["task_id"])
        if e["ok"] and e["result"] is not None:
            results[e["task_id"]] = e["result"]

    for tid, t in tasks.items():
        if tid in seen:
            continue
        if t["status"] is TaskStatus.DONE:
            completed.append(tid)
            if t["result"] is not None:
                results[tid] = t["result"]
        elif t["status"] is TaskStatus.FAILED:
            failed.append(tid)
        else:
            continue
        seen.add(tid)

    return {
        "total": len(seen | set(tasks)),
        "completed": completed,
        "failed": failed,
        "pending": dag.pending_tasks(tasks),
        "results": results,
        "parallel_layers": dag.parallel_layers(tasks),
    }
