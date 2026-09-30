"""编排框架适配层：把 Agent 的节点与拓扑落到具体的编排框架上。

节点（miagent.agent）与拓扑（agent.topology）都不依赖任何框架。一个适配
只需兑现四条约定：

  1. 绑定：每个节点以 `fn(state, deps=deps, **bind)` 调用，返回要更新的字段。
  2. 合并：AgentState 上 Annotated 声明了 reducer 的字段按 reducer 合并，
     其余字段直接覆盖。
  3. 扇出：路由返回 Fanout 列表时，每个分支以 payload 作为全部 state 调用一次，
     各分支并行；全部结束、增量合并之后才进入下游。
  4. 路由：条件边按 Branch.route 的返回值去往下一个节点，START / END 映射为
     框架的起止。

    langgraph/   基于 LangGraph（StateGraph + Send），build_agent 入口

★ 本包自身不 import 任何框架；子包按需导入。
"""
