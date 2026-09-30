"""标注答案驱动的模型：按标准答案作答的「理想模型」，外加用例声明的非理想输出。

用它驱动整张图，被测的就只剩框架：调度、数据流、重试、重规划、校验的效力
边界、结算、收尾。模型不会出错，框架出的错才看得见；再按用例的 simulate
字段注入格式偏差、参数写偏、漏掉一步，看框架的防线是否接得住。

各节点的作答规则（理想模型应当怎样回答）：

  规划      给出标准任务图；请求里有做不成的部分（能力不支持、权限被拒）时
            在 unsupported_actions 里列出；按 simulate 漏掉一步或写偏一个参数；
            首次输出按 planner_noise 加格式偏差，自修复时给出干净的输出
  重规划    给出还没做成、且能做成的任务：主计划里参考执行判为能完成而尚未
            成功调用的任务，加上备选计划里尚未成功调用的任务。引用已完成的
            任务时用它在框架里的 id
  结果校验  实际参数与参考执行一致即通过；不一致判未通过，给出标准参数作修正
  完成校验  这一轮能完成的任务都已成功调用、且目标本身可达成时判达成；
            做不成的部分已由规划列出，不计入判断
  收尾      固定格式的回答与摘要

「成功调用」取自环境探针的记录，与框架自己的状态无关。
"""

from __future__ import annotations

import json
import re
from typing import Any

from miagent.llm import FakeLLM, LLMMessage

from ..suite.execute import REF
from .env import Probe

_ITEM = re.compile(r"^任务 (\S+?)：")
_CALL = re.compile(r"^  调用：(\S+)，参数 (\{.*\})$")
_PREFIX = re.compile(r"^r\d+_")


def _canon(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True)


class OracleLLM(FakeLLM):
    name = "oracle"

    def __init__(self, case: dict[str, Any], probe: Probe, context_limit: int = 64000) -> None:
        super().__init__(responder=self._respond, context_limit=context_limit)
        self.case = case
        self.probe = probe
        self.turn_index = 0
        self.roles: list[str] = []          # 每次调用对应的节点，供统计

    # ---------------- 轮次状态 ----------------

    def begin_turn(self, i: int) -> None:
        self.turn_index = i
        self.probe.turn = i
        turn = self.turn
        self._noise_pending = (turn.get("simulate") or {}).get("planner_noise")
        self._generation = 0
        self._ids = {t["id"]: t["id"] for t in turn["gold"]["tasks"]}   # 标注 id → 框架 id
        self._expected = {c["id"]: c for c in turn["expected"]["calls"]}
        fb = (turn.get("fallback") or {}).get("tasks") or []
        self._fallback = fb
        self._tasks = {t["id"]: t for t in turn["gold"]["tasks"] + fb}

    @property
    def turn(self) -> dict[str, Any]:
        return self.case["turns"][self.turn_index]

    def _succeeded(self) -> set[tuple[str, str]]:
        return {(c.tool, _canon(c.arguments)) for c in self.probe.of_turn(self.turn_index) if c.ok}

    def _done(self, tid: str) -> bool:
        rec = self._expected.get(tid)
        return (rec is not None and rec["arguments"] is not None
                and (rec["tool"], _canon(rec["arguments"])) in self._succeeded())

    def _achievable_ids(self) -> list[str]:
        """这一轮应当做成的任务：主计划里参考执行能完成的，加上备选计划。"""
        main = [t["id"] for t in self.turn["gold"]["tasks"]
                if self._expected[t["id"]]["status"] == "done"]
        return main + [t["id"] for t in self._fallback]

    # ---------------- 作答 ----------------

    def _respond(self, messages: list[LLMMessage]) -> str:
        role = messages[0].content
        if "结果校验器" in role:
            self.roles.append("result_verifier")
            return self._review(messages[1].content)
        if "完成校验器" in role:
            self.roles.append("goal_verifier")
            return self._goal()
        if "重规划器" in role:
            self.roles.append("replanner")
            return self._replan()
        if "规划器" in role:
            self.roles.append("planner")
            return self._plan()
        self.roles.append("finalizer")
        return json.dumps({"answer": f"已处理：{self.turn['request']}",
                           "summary": self._summary()}, ensure_ascii=False)

    def _plan(self) -> str:
        sim = self.turn.get("simulate") or {}
        tasks = [_spec(t) for t in self.turn["gold"]["tasks"]]
        if "omit_step" in sim:
            tasks = [t for t in tasks if t["id"] != sim["omit_step"]["task"]]
        if "arg_drift" in sim:
            d = sim["arg_drift"]
            for t in tasks:
                if t["id"] == d["task"]:
                    t["arguments"] = {**t["arguments"], d["param"]: d["value"]}
        clean = json.dumps({"tasks": tasks, "unsupported_actions": self._unsupported_actions()},
                           ensure_ascii=False)
        noise, self._noise_pending = self._noise_pending, None
        return _noisy(clean, tasks, noise) if noise else clean

    def _infeasible(self) -> bool:
        return self.case["category"] in ("unsupported", "permission_denied")

    def _unsupported_actions(self) -> list[str]:
        return ["请求中有没有可用工具能完成的部分"] if self._infeasible() else []

    def _replan(self) -> str:
        self._generation += 1
        pending = [tid for tid in self._achievable_ids() if not self._done(tid)]
        out = []
        for tid in pending:
            spec = _spec(self._tasks[tid])
            spec["arguments"] = {p: ({REF: self._ref(v[REF], pending)} if isinstance(v, dict) and REF in v else v)
                                 for p, v in spec["arguments"].items()}
            spec["dependencies"] = [self._ref(d, pending) for d in spec["dependencies"]]
            out.append(spec)
        for tid in pending:
            self._ids[tid] = f"r{self._generation}_{tid}"
        return json.dumps({"tasks": out}, ensure_ascii=False)

    def _ref(self, tid: str, pending: list[str]) -> str:
        """新计划里的引用：指向同批新任务时用它的标注 id（框架会加前缀），
        指向已完成的任务时用它在框架里的 id。"""
        if tid in pending:
            return tid
        return self._ids.get(tid, tid)

    def _review(self, prompt: str) -> str:
        reviews = []
        current: str | None = None
        for line in prompt.splitlines():
            m = _ITEM.match(line)
            if m:
                current = m.group(1)
                continue
            m = _CALL.match(line)
            if m and current is not None:
                reviews.append(self._judge(current, json.loads(m.group(2))))
                current = None
        return json.dumps({"reviews": reviews}, ensure_ascii=False)

    def _judge(self, fid: str, actual: dict[str, Any]) -> dict[str, Any]:
        tid = _PREFIX.sub("", fid)
        rec = self._expected.get(tid)
        if rec is None or rec["arguments"] is None or actual == rec["arguments"]:
            return {"task_id": fid, "passed": True, "reason": "结果与任务一致",
                    "corrected_arguments": None}
        spec = self._tasks[tid]
        corrected = {p: ({REF: self._ids.get(v[REF], v[REF])} if isinstance(v, dict) and REF in v else v)
                     for p, v in spec["arguments"].items()}
        return {"task_id": fid, "passed": False, "reason": "参数与用户目标不符",
                "corrected_arguments": corrected}

    def _goal(self) -> str:
        missing = [self._tasks[t]["description"] for t in self._achievable_ids() if not self._done(t)]
        if not missing and (self.turn["expected"]["achievable"] or self._infeasible()):
            return json.dumps({"achieved": True, "gap": ""}, ensure_ascii=False)
        gap = "还没有完成：" + "、".join(missing) if missing else "请求中有无法完成的部分"
        return json.dumps({"achieved": False, "gap": gap}, ensure_ascii=False)

    def _summary(self) -> str:
        done = [f"{self._tasks[t]['description']}：{self._expected[t]['result']}"
                for t in self._expected if self._done(t) and self._expected[t]["result"]]
        return "；".join(done) or "本轮未调用工具"


def _spec(task: dict[str, Any]) -> dict[str, Any]:
    return {k: json.loads(json.dumps(task[k], ensure_ascii=False))
            for k in ("id", "description", "dependencies", "required_tool", "arguments")}


def _noisy(clean: str, tasks: list[dict[str, Any]], kind: str) -> str:
    """规划输出的格式偏差。invalid_once 与 stringified_ref_once 的首次输出不合
    schema，靠自修复重试得到干净输出；其余三种应被 JSON 截取直接吸收。"""
    if kind == "fence":
        return f"```json\n{clean}\n```"
    if kind == "prose":
        return f"好的，我来拆解一下。\n{clean}\n以上是计划，请执行。"
    if kind == "trailing_comma":
        return clean[:-2] + ",]}" if clean.endswith("]}") else clean
    if kind == "invalid_once":
        return json.dumps({"tasks": [{"id": t["id"], "tool": t["required_tool"]} for t in tasks]},
                          ensure_ascii=False)
    if kind == "stringified_ref_once":
        broken = json.loads(clean)
        for t in broken["tasks"]:
            for p, v in t["arguments"].items():
                if isinstance(v, dict) and REF in v:
                    t["arguments"][p] = json.dumps(v)
                    return json.dumps(broken, ensure_ascii=False)
        return clean
    raise ValueError(kind)
