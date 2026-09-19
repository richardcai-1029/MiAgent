"""上下文记忆：任务离开任务图之后去哪里。

任务图（AgentState.tasks）是**工作记忆**：只装当前还要调度的东西。
任务到了终态就不再参与调度，继续留在图里只会让每一轮重规划都带着
越来越长的历史 —— 端侧窗口本就小，冗余的历史挤占的是真正需要的信息。

终态任务离开任务图时压缩为 Episode，进入**情景记忆**（AgentState.episodes）。
它只记「调了什么、成没成、结果是什么」，供重规划与收尾引用，
并按规则去重、降级。

**目标锚**（AgentState.anchor）是一次请求里不变的部分：用户目标与首次拆解。
它在每一轮重规划里都不可裁，使新计划始终有一个「原本要做什么」可以对照。

**对话历史**（Session.turns → AgentState.history）跨请求：每轮只留收尾产出的
摘要，下一轮规划据此理解承接关系。进程内保留，不落盘。

★ 本包是纯 Python：不 import langgraph，不调用模型，没有副作用。
"""

from . import anchor, episodic, session
from .anchor import Anchor
from .episodic import Episode
from .session import Session, Turn

__all__ = ["Anchor", "Episode", "Session", "Turn", "anchor", "episodic", "session"]
