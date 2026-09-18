"""上下文记忆：任务离开任务图之后去哪里。

任务图（AgentState.tasks）是**工作记忆**：只装当前还要调度的东西。
任务到了终态就不再参与调度，继续留在图里只会让每一轮重规划都带着
越来越长的历史 —— 端侧窗口本就小，冗余的历史挤占的是真正需要的信息。

终态任务离开任务图时压缩为 Episode，进入**情景记忆**（AgentState.episodes）。
它只记「调了什么、成没成、结果是什么」，供重规划与收尾引用，
并按规则去重、降级。

★ 本包是纯 Python：不 import langgraph，不调用模型，没有副作用。
"""

from .episodic import Episode, dedupe, history, render, settle, summarize

__all__ = ["Episode", "dedupe", "history", "render", "settle", "summarize"]
