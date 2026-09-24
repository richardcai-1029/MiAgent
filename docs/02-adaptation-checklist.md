# LangGraph → 小米端侧生态 适配改造清单

| 项目 | MiAgent · 通用 Agent 端侧能力底座 |
|---|---|
| 基础框架 | LangGraph `>=1.2,<2.0`（实测通过 1.2.10、1.2.11） |
| 目标生态 | MiClaw 系统级 Agent 生态 · MiMo 端侧大模型 |
| 文档版本 | v3.4 |

本清单记录已完成并经测试验证的改造项。每项均给出落地位置、验证方式与**框架侵入面**，可逐条核查。

侵入面一列标注该项依赖 LangGraph 到什么程度，据此判断框架升级的影响范围与回退路径，分级标准与高风险项详见第二节。

> **关于 MiClaw 规范来源**：MiClaw 技术文档当前不可获取。清单中所有与 MiClaw
> 交互相关的条目，均依据项目负责人指示——「以标准 MCP (JSON-RPC 2.0) 为基准
> 自行定规约，Agent 做客户端、MiClaw 为服务端，错误码 MC- 代表底层、AG- 代表
> Agent 上层」——推导设计，规约见《MiAgent ↔ MiClaw 通信规约》。待真实规范
> 开放后，差异集中在协议层，上层无需改动。

---

## 一、改造项

### A · 协议与通信适配

LangGraph 的工具调用是进程内 Python 函数调用，不存在跨进程协议、身份注册与时序约束的概念。这一整块从零建立。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| A-1 | 无系统级通信协议 | 建立 JSON-RPC 2.0 报文层，`result`/`error` 互斥等协议不变量由校验器强制 | `protocol/messages.py`；`test_protocol.py` | L0 |
| A-2 | 工具调用为进程内直调，无跨进程能力 | stdio 按行分帧传输，不引入 HTTP 栈 | `transport.py`；`test_transport.py` 含真实子进程用例 | L0 |
| A-3 | 无 Agent 身份注册语义 | 扩展 `miclaw/agent.register`，注册为「申请—批准」协商而非单向声明 | `mock_server/server.py::_on_register` | L0 |
| A-4 | 会话无时序约束，任何时刻均可调用工具 | 握手状态机，以声明式状态表强制 `initialize → register → 调用` | `_REQUIRED_STATE`；违规返回 MC-2002 | L0 |
| A-5 | 厂商扩展方法与标准方法无命名隔离 | 扩展方法统一 `miclaw/` 前缀，避免与未来标准 MCP 方法撞名 | `test_protocol.py::test_extensions_are_namespaced` | L0 |
| A-6 | 无协议版本协商 | `initialize` 比对版本，不匹配返回 MC-2001 | `test_mock_server.py::test_version_mismatch` | L0 |
| A-7 | 协议只有 Agent 调用系统的方向，系统把用户请求派给 Agent 的报文不存在；请求无优先级，多个请求同时到达时 Agent 无从区分谁更急 | 扩展 `miclaw/task.dispatch`（MiClaw → Agent）：对话标识、请求原文、请求优先级。优先级按「有没有人正在等这个回答」分为 `foreground` / `background` 两级而非数值，缺省按前台；响应只给回答与是否完成，AG-* 码不上网络。服务端受理的方法与 Agent 受理的方法分列两个枚举，反向发送返回 MC-2004。客户端接收服务端发起的请求需传输层分流，尚未实现 | `protocol/messages.py::AgentMethod` / `TaskDispatchParams` / `TaskDispatchResult`、`Runtime.dispatch`；`test_protocol.py::TestTaskDispatch`、`test_runtime.py::TestDispatch` | L0 |
| A-8 | 系统无从告知 Agent 一段对话已结束，Agent 侧为对话保留的上下文只能一直留着 | 扩展 `miclaw/conversation.end` 通知（MiClaw → Agent，无回复），参数只有 `conversationId`；结束前已派发的请求照常执行完，此后同一标识视为新对话 | `protocol/messages.py::ConversationEndParams`、`Runtime.end_conversation`；`test_runtime.py::TestSessionReclaim` | L0 |

### B · 错误处理体系

LangChain 工具失败表现为 Python 异常或自由文本，无错误分类，上层无法据此做差异化决策。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| B-1 | 错误无分层归属，无法区分底层与上层故障 | MC-/AG- 双层码；MC- 经网络回传，AG- 不出进程 | `protocol/errors.py`；测试强制 AG-* 无整数码映射 | L0 |
| B-2 | 错误细节混在文本中，上层无法程序化处理 | 结构化 `error.data.detail`，携带机器可读字段 | `test_client.py::test_detail_survives_the_wire` | L0 |
| B-3 | 错误码在网络两侧无法还原为同一异常 | 客户端按 `data.code` 还原为 `MiClawError`；未知码兜底不崩溃 | `client.py::_to_exception` | L0 |
| B-4 | 无按错误类别的差异化重试策略 | 按千位段位决策：1xxx 退避重试、4xxx 延迟或降级、3xxx/5xxx 不重试 | `protocol/errors.py::retry_policy`；`test_tools.py::TestRetryPolicyByBand` | L0 |
| B-6 | 业务失败（`isError`）与调用失败（JSON-RPC error）未在上层区分 | 上层据此选择「让模型换方案」还是「退避重试」 | `client.py::call_tool` 将 isError 映射为 MC-3003；见 `test_business_failure_differs_from_call_failure` | L0 |
| B-8 | 工具内部异常的原始文本直接作为给模型的失败信息 —— 异常消息常带路径、连接串、内部标识，既占端侧本就紧张的窗口，又可能把模型带偏 | 给模型的 content 改为确定性措辞加分级处置建议；原始消息与异常类型移入 `detail`，只进 Agent 侧日志 | `tools/base.py::Tool.invoke`；`test_validation.py::TestErrorTextGivenToTheModel` | L0 |
| B-7 | `RetryPolicy.BACKOFF` 声明「退避后重试」，实现却是下一轮立即重发 —— 传输抖动与资源占用这两类失败在立即重发时状况还没来得及改变 | 重试前等待；等待实现可注入，使退避行为可被测试观察而不必真的等 | `nodes.execute`；`test_graph.py::TestRetryBackoff` | L0 |

### C · 端侧资源约束

LangGraph 面向云端设计，不存在内存配额、并发上限等概念。这是端侧相对云端多出的一整个约束维度。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| C-2 | 无资源配额下发机制 | 握手时下发 `ResourceBudget`，Agent 全程受其约束 | `protocol/messages.py::ResourceBudget` | L0 |
| C-3 | 工具无资源画像，无法预判开销 | 工具申报 `estimated_memory_mb`，服务端执行前校验 | 超配额返回 MC-4001，见 `test_memory_limit` | L0 |
| C-7 | `max_call_timeout_ms` 无执行机制：所有请求一律用连接级超时，一个工具卡住要拖到整条链路超时才被发现 | 工具调用改用配额下发的单次调用超时，其余请求仍用连接级超时；握手未完成时退回连接级 | `client.call_timeout`；`test_client.py::TestPerCallTimeout` | L0 |
| C-6 | `max_concurrent_calls` 无执行机制 | 当前同步实现天然串行；如引入并发需加信号量，否则应移除该字段 | `scheduler` 按 `max_concurrent_calls` 限流单个请求一轮的 MiClaw 派发，超出者顺延；多个请求同时在飞时共用一个同容量、按优先级排队的调用槽（`runtime.SlotPool`），合计不超出配额；本地工具受 GIL 限制不设限 | L1 |
| C-8 | 多请求运行时为每个对话保留一个会话，对话标识不断出现新值时常驻内存只增不减 | 收到对话结束通知即回收；空闲会话（无请求在飞、无排队）数超过 `max_idle_sessions`（默认同 `max_inflight`，暂定）时回收最久没用的；在飞或排队中的会话不回收。请求提交时即绑定会话对象，结束通知不影响已提交的请求；结果在会话状态登记之后才交付 | `runtime.Runtime._release` / `end`；`test_runtime.py::TestSessionReclaim` 8 条 | L0 |

### D · MiMo 模型接入

LangChain 的 LLM 抽象假设云端 API，与端侧模型的运行方式与可靠性特征差异较大。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| D-1 | LLM 抽象面向云端 API | 定义最小 LLM 接口，支持 Fake / MiMo 端侧 / 云端三种实现热切换 | `miagent/llm/`；模板方法统一承担上下文检查与耗时统计，Fake/云端两种实现已可切换 | L0 |
| D-2 | 上下文超限直接拒绝（AG-3001）。端侧窗口小，工具一多、执行历史一长，正常任务也会触顶，拒绝即等于任务失败 | 提示词按优先级片段拼装，超预算时依次削减：工具描述削为只剩工具名、执行历史削为只剩计数；不做按字符截断（切开的 JSON 与历史是语法损坏的文本）；削减内容进 trace；计量口径由 `LLM.estimate` 提供，默认按字符近似，接入真实 tokenizer 后覆写 | `llm/context.py`；`test_context.py` 13 条用例 | L0 |
| D-3 | 解析入口以「第一个 `{` 到最后一个 `}`」截取 JSON：输出含两个 JSON 块时会把两块连同中间的文字一并截出，得到的既不是前者也不是后者 | 改为逐字符扫描取第一个括号配对完整的对象，同时去掉结构上的尾随逗号；扫描区分字符串内外，不会动到字符串值里的逗号。引号错用与截断不修补 —— 修复方式不唯一，猜出来的计划会被真实执行，交由自修复重试 | `llm/base.py::first_json_object`；`test_llm.py::TestDirtyOutputCleaning` | L0 |
| D-4 | 工具描述格式与模型 function calling 格式未打通 | MCP `inputSchema` 本就是 JSON Schema，可直接喂模型，无需转换 | `test_client.py::test_tool_schema_survives_the_wire` | L0 |
| D-5 | 提示词无法保证模型输出符合预期结构，一次格式失误即导致任务失败 | 计划改用 Pydantic schema 驱动：schema 由模型定义导出、与校验同源；解析失败带着具体错误自修复重试；工具名收进 enum 使幻觉在解析阶段即被拒 | `graph/schema.py`、`llm/base.py::complete_structured` | L0 |
| D-6 | 无对话上下文：图的一次 invoke 处理一个请求，第二个请求承接第一个时规划器一无所知 | `memory.Session` 持有历次轮次（请求、回答、本轮摘要、错误码、执行记录），run 时填入 `history`；Planner 渲染为每轮一段，越旧越先削减——上一轮先降为只剩结论，更早的整段丢弃，且在工具描述之前被削；进程内保留，不落盘 | `miagent/memory/session.py`、`nodes.planner`；`test_session.py` 13 条 | L0 |
| D-9 | 收尾只产出自由文本回答：格式无约束，且没有可供下一轮引用的本轮记录，多轮之间要么重传全部历史、要么什么都不传 | 收尾改为 schema 驱动的 `FinalOutput(answer, summary)`，一次调用同时给出回答与本轮摘要；预算为 schema 预留位置；自修复失败或窗口放不下时两者一并退回确定性摘要 | `graph/schema.py::FinalOutput`、`nodes.finalizer`；`test_graph.py::TestFinalizerStructuredOutput` | L0 |
| D-8 | 重规划无目标锚：几轮之后提示词里全是局部的成败记录，新计划可偏离原始目标且不可检测；把失败过的调用原样再拆一遍也照常执行 | 首次规划成功后写入目标锚（用户目标 + 首次拆解各步描述），每轮重规划放最前且不可裁；新任务的 (工具, 参数指纹) 全部在失败记录里时判无进展 AG-1003，不派发 | `miagent/memory/anchor.py`、`ledger.accept`（内部用 `episodic.repeats_calls`）、`nodes.planner` / `replanner`；`test_graph.py::TestGoalAnchor`、`test_ledger.py::TestAcceptDetectsSpinning` | L0 |
| D-10 | 工具没报错就算这一步做完了：框架只认调用有没有抛异常，不认调用做的是不是要做的事。单号写成邻近的一个、查的时段与用户问的不是同一个、结果答非所问 —— 这些在错误码这一层没有任何迹象，却会被当作已完成写进执行历史，最后汇进一个声称成功的回答 | 执行与落库之间插入结果校验：把本轮结果连同任务描述与实际参数交给模型判断「这是不是这一步要的东西」，一次调用判全轮。判定只收紧不放宽（失败不会被改判为成功）；失败原因是参数写偏时，模型给出的修正参数须过工具 schema、与原参数确有不同、且不触碰 `$from` 引用，才换上新参数重试同一个任务，不重规划整张图。校验只在能给出新信息时才做：成功与工具层失败（段位 3）送校验，传输/资源/权限类失败的原因已由段位确定，不送 | `graph/verify.py`（三道闸与判定合并，纯函数）、`nodes.verify_results`、`nodes.evaluator`、`build.py` 接入节点；`test_verify.py` 48 条 | L1 |
| D-7 | 执行上下文只增不减：终态任务全部留在任务图里，每一轮重规划都带着越来越长的历史，挤占端侧本就小的窗口 | 任务图只装还要调度的任务。终态任务在重规划时结算为 Episode 进入情景记忆，只有新任务引用到的已完成任务（连同其上游）留在图里；情景记忆渲染时按 (工具, 参数指纹) 去重、按轮次降级——本轮带结果原文，更早的只剩 id 与描述；执行概况由两者合并得出 | `miagent/memory/ledger.py`（结算的编排：settle / accept / close）、`episodic.py`、`dag.ancestors`、`nodes.replanner` / `finalizer`；`test_ledger.py` 23 条、`test_memory.py`、`test_graph.py::TestWorkingMemoryIsPruned` | L0 |

### E · 依赖裁剪与轻量化

端侧存在只需协议客户端的部署形态，不应为图引擎付出常驻内存与冷启动的代价。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| E-1 | 协议层与客户端不应绑定图引擎 | `langgraph` 拆为可选依赖 `[graph]`，协议/传输/客户端仅依赖 pydantic | `pyproject.toml`；60 个测试在无 langgraph 时仍可运行 | L0 |
| E-7 | 图引擎随包导入被无条件加载，纯逻辑模块也要付出其常驻代价 | `miagent.graph` 以 PEP 562 惰性导出 `build_agent`：依赖解析、状态定义、计划 schema 均为纯 Python，不触发图引擎加载 | 该包导入代价由 884 模块 / 69.8 MB 降至 142 模块 / 30.2 MB | L0 |
| E-8 | 缺少防止分层退化的机制 | 双向守卫：瘦客户端形态涉及的十个模块，在独立子进程中导入后既不得出现 langgraph / langchain_core / langsmith / requests 等已知重依赖（黑名单），加载的第三方包也不得超出 pydantic 及其依赖（白名单） | `tests/test_layering.py`，23 条用例 | L0 · 版本敏感 |

### F · 工具体系适配

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| F-1 | 本地工具（进程内函数）与 MiClaw 系统工具（走协议）无统一抽象 | 统一 Tool 基类 + 注册表，屏蔽调用方式差异，保留失败模式差异 | `miagent/tools/`；87 个测试覆盖两类工具的统一入口 | L0 |
| F-6 | 无工具开发规范文档 | 输出标准化开发规范，含声明字段、错误约定、测试要求 | `docs/03-tool-spec.md` | L0 |
| F-7 | 参数只校验必填项，类型不符会静默通过 —— schema 声明 integer 而模型给出 `"3"` 时，工具拿到的是字符串，调用形式合法而语义错误 | 按 schema 校验类型与枚举，吸收无歧义漂移（`"3"`→`3`、单值→单元素数组），拒绝有歧义的输入与未声明字段；问题一次性全部报出供自修复 | `tools/validation.py`；`test_validation.py` 27 条用例 | L0 |

### G · 任务调度

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| G-2 | 无步数与预算上限，模型可能陷入死循环 | 单任务重试上限、重规划上限、累计执行上限三道闸，映射 AG-1002/AG-1003。累计执行上限由 Scheduler 在派发前检查，因而顺利执行的流程同样受其约束 | `nodes.scheduler`；`test_graph.py::TestExecutionBudgetBoundsSuccessToo` | L1 |
| G-3 | 模型可能把目标拆得过细，每一步都是一次真实调用，白白消耗端侧算力 | 规划产出的待执行任务数超过累计执行预算即拒绝——这样的计划在预算内必然跑不完，与其执行到一半才发现不如当场拒绝。上限直接取执行预算，不另立数字 | `nodes._plan`；`test_graph.py::TestPlanSizeIsChecked` | L0 |
| G-4 | 框架无任务依赖建模，只能线性执行，无法表达分支与汇合 | 引入任务 DAG：显式 dependencies、五态生命周期、依赖解析/就绪判定/完成检测/死锁检测/级联失败全部为确定性纯函数，不交给模型 | `graph/dag.py`；`test_dag.py` 25 条用例覆盖边界 | L0 |
| G-5 | 无依赖关系的任务仍被串行执行，浪费 IPC 等待时间 | 以 LangGraph Send 并行派发同层任务；需为 tasks 定义按 id 合并的 reducer，避免多分支写回互相覆盖 | `routers.py` 返回 Send 列表扇出；outcomes 与 execution_count 用 reducer 汇总；实测无依赖任务提速 3.00x | **L2 · 高** |
| G-6 | 任务图非法（依赖缺失/自依赖/成环）会表现为莫名死锁 | Kahn 拓扑排序在执行前校验，映射 AG-1004；运行期死锁映射 AG-1005 | `dag.validate()`；`test_cyclic_plan_rejected_before_execution` | L0 |
| G-8 | 「所有任务都到了终态」被当作「目标达成」：重规划跑完一个新任务都没产出时，失败任务不会再有人接手，流程却照常收尾，用户拿到声称完成实则漏做的回答 | 重规划产出为空且情景记忆中有失败记录时判定未达成，错误码取根因（跳过级联失败的 AG-1005，同为自身失败取最近一轮）。失败记录本身不作为判据——重规划成功接手时，前一条路失败是正常剧情 | `nodes.replanner`、`_root_failure`；`test_graph.py::TestCompletionIsVerified` | L0 |
| G-9 | Evaluator 每轮无条件写回 `failure: None`。重规划失败（如产出的图依赖缺失）后剩余任务照常执行，任一成功即把那个失败抹掉，收尾时报告成功 | 只在判定中止时写 `failure`；重规划的失败一直保留到收尾 | `nodes.evaluator`；`test_graph.py::test_failed_replan_is_not_masked_by_later_successes` | L0 |
| G-10 | 「没有任务可调度了」被当成「目标达成了」的最后一道判据。计划本身漏掉一步时，每个任务都成功、没有失败记录、重规划也没被触发，任务图这一层完全看不出异常，用户拿到的是一个自信的、漏做的回答 | 判完成与收尾之间插入完成校验：以目标锚（用户目标 + 首次拆解）对照本轮执行记录，判未达成则带着缺口转重规划补做，重规划已达上限则带 AG-1006 收尾。补做同样受无进展判定约束——新任务全是这一轮调过的调用（成败都算）即判 AG-1003 不派发，带副作用的系统调用不会为了补做而发生第二次。一次工具都没调、或已经带着错误码时不做这次推理 | `nodes.verify_goal`、`routers.route_after_goal_verifier`、`build.py` 接入节点与条件边、`ledger.survey` / `accept(standing=)`、`episodic.repeats_calls`；`test_verify.py::TestGoalVerifier`、`TestGoalVerificationEndToEnd` | L1 |
| G-7 | 任务参数由模型一次性写死，下游任务取不到上游结果，多个工具只是多次互不相干的调用 | 参数中以 `{"$from": "任务 id"}` 引用上游结果，派发前确定性求值；引用即依赖，先后关系由引用派生，不依赖模型再声明一遍 | `graph/dataflow.py`；`test_dataflow.py` 23 条用例 | L0 |
| G-11 | 一次 invoke 只处理一个请求，多个用户同时发起请求时只能排队串行；若直接多线程并发 invoke，各请求按配额各自派发，MiClaw 调用合计超出 `max_concurrent_calls`，同一对话的两轮同时执行则后一轮看不到前一轮 | 图之上加多请求运行时：每个请求仍是独立的 invoke；同一会话内串行、不同会话并发；在飞请求数受 `max_inflight` 准入（默认 10，暂定）；按 (优先级, 提交先后) 出队。MiClaw 调用配额与模型推理（默认 1 个推理槽）经按优先级排队的资源槽由全部请求共用，请求的优先级键经 contextvars 传入并行分支。槽只在单次调用期间持有，不会循环等待。防饥饿先不做 | `miagent/runtime/`（`Runtime`、`SlotPool`、`build_runtime`）、`nodes.execute`、`LLM._guarded`；`test_runtime.py`、`test_framework_contract.py` 契约六、七 | **L2** |
| G-12 | 就绪任务按 id 字母序派发：配额放不下全部就绪任务时，关键路径上的任务可能被顺延，整张图多跑一轮；且按字符串排序时 `task_10` 排在 `task_2` 之前，目标锚的步骤排列也因此错位 | 就绪任务按 (-下游最长链长度, 规划序号, id) 排序，下游最长链只沿未终结任务计；任务新增规划序号 `seq`，按模型输出顺序编号、重规划时接续，派发的决胜项与目标锚都改看它 | `dag.downstream_depth` / `ready`、`nodes._build_tasks`、`memory/anchor.py`；`test_dag.py::TestReadyOrder`、`test_graph.py::TestDispatchOrder` | L0 |


### H · 框架版本风险控制

LangGraph 的部分行为约定写在文档而非类型签名里，升级失配时不报错、只表现为行为异常。
本组的目的是把这类风险变成可检测、可定位、影响范围可枚举的东西。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 | 侵入面 |
|---|---|---|---|---|
| H-1 | 框架接触面无约束，任何模块都可 import langgraph，升级的影响范围不可枚举 | AST 扫描守卫，只允许构图与路由两个模块 import langgraph；新增即失败并要求更新风险分级 | `test_layering.py::test_framework_surface_is_confined` | L0 |
| H-2 | 所依赖的框架语义约定无测试覆盖，升级失配时静默失效 | 框架契约测试：以最小图逐条固化 Send payload 范围、reducer 合并时机、条件边返回类型、中断恢复、并行分支继承调用方 contextvars、同一编译图并发 invoke 的状态隔离，不引用业务模块 | `tests/test_framework_contract.py`，7 条用例 | L1（仅测试代码） |
| H-3 | 版本声明无上限，升级可在无人察觉时发生 | 收紧为 `langgraph>=1.2,<2.0`，实测通过版本记录在文档抬头 | `pyproject.toml` | L0 |

---

## 二、框架侵入面与风险分级

改造项对 LangGraph 的依赖程度差异很大：有的完全不碰框架，有的依赖框架未写进类型签名的行为约定。
后者才是版本升级时真正的风险来源，需要单独标记。

### 分级标准

| 级别 | 含义 | 升级失配时的表现 |
|---|---|---|
| **L0** | 不 import langgraph。纯协议、纯函数、打包配置 | 无影响 |
| **L1** | 只用公开 API：`StateGraph` / `add_node` / `add_edge` / `add_conditional_edges` / `compile` | 显式失败：签名或行为变更会直接报错 |
| **L2** | 依赖框架的**语义约定**——写在框架文档里、不在类型签名里 | **静默失败**：不抛异常，只表现为行为异常 |
| **L3** | 侵入内核：改框架源码、monkey patch、依赖私有模块 | 极可能直接崩溃，且无官方兼容承诺 |

分级依据不是改造的工作量，而是**升级失配时能否被发现**。L1 会报错，修就是了；
L2 不报错，问题会以「结果偶尔不对」的形式潜伏，排查成本高得多。

### 分布

| 级别 | 项数 | 编号 |
|---|---|---|
| L0 | 43 | A 组全部、B 组全部、C-2、C-3、C-7、C-8、D 组除 D-10 外全部、E 组全部、F 组全部、G-3、G-4、G-6、G-7、G-8、G-9、G-12、H-1、H-3 |
| L1 | 5 | C-6、D-10、G-2、G-10、H-2 |
| L2 | 2 | **G-5**、**G-11** |
| L3 | 0 | — |

合计 50 项：第 1 周完成 26 项，H 组 3 项为风险分级过程中识别并补齐，
B-7、B-8、C-7、D-2、D-3、F-7、G-3、G-7、G-8 为第 2 周新增，
D-6、D-7、D-8、D-9、D-10、G-9、G-10 为第 3 周新增，
A-7、A-8、C-8、G-11、G-12 为多请求并发与调度优先级改造新增。

**当前没有 L3 项**：未修改框架源码、未做 monkey patch、未引用任何私有模块。
这是选型时「流程可控、不存在隐式框架行为」的直接收益——业务逻辑绝大部分落在框架之外，
升级面因此收敛得很窄。

框架接触面同样是收敛的：整个 `miagent/` 下只有 `graph/build.py`（构图）与 `graph/routers.py`（路由）
两个文件 import langgraph。这一点由 `test_framework_surface_is_confined` 用 AST 扫描强制——
新增第三个框架依赖会立刻测试失败，并要求同步更新本节的风险分级。接触面可枚举，回退才谈得上可行。

### 高风险项 G-5：并行任务派发

G-5 以 `Send` 扇出同层任务，依赖两条**语义约定**：

1. **Send 的 payload 就是被调用节点看到的全部 state。** 执行节点因此读不到 `tasks` 与
   `execution_count`，只能返回增量。
2. **并行分支的返回值经 reducer 合并。** `outcomes` 用 `append_or_reset` 追加，
   `execution_count` 用 `operator.add` 累加。

两条都不在类型签名里。失配后的表现是静默的：

| 约定失效 | 静默表现 |
|---|---|
| payload 改为与主 state 合并 | 节点若改回返回绝对值，多分支写回互相覆盖，计数与结果都偏 |
| 整数 reducer 不再累加 | `execution_count` 计数偏低，`MAX_TOTAL_EXECUTIONS` 这道循环出口失效（G-2 随之失效） |
| 空列表返回被视为「无更新」而跳过合并 | `outcomes` 不再被清空，Evaluator 下一轮重复消费上一轮结果 |

**检测手段**：`tests/test_framework_contract.py` 用最小图逐条固化上述约定，
不引用任何业务模块。升级时先跑这一组，失败能直接指出是哪条约定变了，
不必从业务测试的失败里反推。

**回退路径**：G-5 的框架相关部分只在 `routers.py` 的一个函数里。
调度决策（哪些任务就绪、并发配额如何分配）都在 `scheduler` 节点与 `dag.py` 的纯函数中，
与框架无关。因此降级为串行执行只需改路由返回值，任务图与调度语义不受影响。

### 高风险项 G-11：多请求运行时

G-11 的运行时本身不 import langgraph，但依赖两条**语义约定**：

1. **并行分支继承调用方的 contextvars。** 运行时把请求的优先级键设进 contextvars，
   执行节点在共享资源槽前排队时读它，节点代码因此不需要知道自己属于哪个请求。
2. **同一个编译好的图可被多个线程同时 invoke，状态互不串。** 每个请求一次 invoke，共用一个图。

| 约定失效 | 静默表现 |
|---|---|
| 分支不再继承 contextvars | 分支读到默认键，所有请求同键，优先级不再生效，退化为按排队先后；配额仍守得住 |
| 并发 invoke 共享状态 | 一个请求的结果混进另一个请求 |

**检测手段**：`test_framework_contract.py` 契约六、七。

**回退路径**：优先级键改为显式传递——调度器写进 `DispatchItem`、随 `Send` 的 payload 带到执行节点，
模型调用则由节点从状态里取。改动落在 `scheduler`、`routers.py` 与调模型的节点，资源槽与运行时不变。
第二条若失效，退回每个请求各编译一张图。

### 版本敏感项 E-8：分层守卫的依赖假设

E-8 的黑名单（langgraph / langchain_core / langsmith / requests / httpx / urllib3 / websockets）
编码了一项对 LangGraph 依赖树的假设。框架换用新的传递依赖后，黑名单不会报错，只会漏判——
分层退化了也测不出来，失效方式同样是静默的。

本版本补充了互补的白名单守卫：瘦客户端形态加载的第三方顶层包不得超出 pydantic 及其依赖，
多出任何一个即失败。黑名单负责给出可读的失败原因，白名单负责保证不漏。

### 预留接口

`build_agent` 将 `compile_kwargs` 透传给框架的 `compile()`，用于 `checkpointer` 与
`interrupt_before`——端侧据此在调用系统能力之前暂停、交用户确认后恢复。该路径属 L1，
其可用性由 `test_interrupt_before_pauses_and_resumes` 固化。

### 版本策略

| 措施 | 内容 |
|---|---|
| 版本区间 | `langgraph>=1.2,<2.0`。上限不可省略：L2 项跨大版本失配是静默的 |
| 实测记录 | 通过版本记录在本文档抬头，升级后需更新 |
| 升级流程 | 先跑框架契约测试定位约定变更，再跑接触面守卫确认依赖面未扩大，最后跑全量测试与 `bench/baseline.py` |

> **一次已发生的漂移**：原依赖声明为 `langgraph>=1.0`，无上限。
> 环境中的实际版本已由文档记录的 1.2.10 漂移至 1.2.11，全量测试通过因而未被察觉。
> 本次重测 `bench/baseline.py`，两版本的资源数据一致，故第三节数据仍然有效。
> 这次漂移没有造成问题，但它说明无上限的版本声明会让升级在无人知晓的情况下发生——
> 上限与契约测试的意义正在于此。

---

## 三、实测数据

按部署形态测量。端侧的实际约束是「这一形态的进程常驻多少内存」，按单个模块统计没有意义。
环境：macOS 24.6 / Python 3.13.9 / langgraph 1.2.11。测量脚本 `bench/baseline.py`，可复现；
内存取三次运行的中位数、耗时取最小值，故下表为取整后的近似值。

| 部署形态 | 常驻内存 | 冷启动 | 加载模块数 | 其中云端/网络相关 |
|---|---|---|---|---|
| 裸解释器 | 约 17 MB | — | — | — |
| 瘦客户端 | 约 30 MB | 约 53 ms | 138 | 0（0%） |
| 完整 Agent | 约 70 MB | 约 276 ms | 887 | 334（37%） |

瘦客户端形态涵盖协议、传输、客户端、工具、模型五层，可完成系统调用转发与工具调度，
不含任务规划与依赖调度；完整形态在其上追加图引擎。图引擎的边际成本为常驻约 +40 MB、冷启动约 +223 ms、模块 +749。

> **关于验收阈值**：任务书要求「使其满足 MiMo 端侧大模型的运行资源要求」，
> 但未给出具体数值，MiClaw 侧的真实资源配额亦不可获取。因此本文档只记录实测值，
> 不设定达标线。阈值需由 MiClaw 规范或项目决策给定后再行补充。

依赖体积（site-packages 实测）：

| 包 | 体积 | 端侧是否需要 |
|---|---|---|
| langsmith | 5.5 MB | 云端追踪服务，端侧不需要 |
| pydantic | 3.0 MB | 协议校验必需 |
| langchain_core | 2.8 MB | 图引擎的必需依赖 |
| langgraph | 1.6 MB | 图调度核心 |
| langgraph_sdk | 0.8 MB | 面向 LangGraph Cloud，端侧不需要 |
| websockets | 0.8 MB | 端侧不需要 |
| httpx / requests / urllib3 | 约 0.4 MB（147 模块） | 端侧走 stdio，不需要 |

完整 Agent 形态加载的 887 个模块中，334 个（37%）属于上表中端侧不需要的云端与网络栈。
这些模块由 `langchain_core` 与 `langgraph_sdk` 在模块层面具名引入，是 LangGraph 的
传递依赖。轻量化因此采取限制影响范围的路径：图引擎降为可选依赖（E-1）、包级惰性
导出使纯逻辑模块不触发其加载（E-7）、并以自动化守卫防止分层退化（E-8）。

---

## 四、验证方式

规约与改造的全部约定均以测试代码固化，当前累计 447 条用例。

| 验证项 | 方式 |
|---|---|
| 协议合规 | 报文可被标准 MCP 客户端解析；错误码整数部分符合 JSON-RPC 2.0 |
| 错误码完备 | 遍历全部错误码，MC-* 必有整数映射、AG-* 必无；任一 MC-* 均可推导出重试策略 |
| 时序正确 | 违反 initialize → register → 调用 时序的调用被拒绝并返回 MC-2002 |
| 错误闭环 | 服务端抛出的每个 MC- 码，均可在客户端还原为同一 ErrorCode 并保留结构化细节 |
| 资源约束 | 超出下发配额的工具调用被拦截，detail 中给出实际值与上限 |
| 依赖解析 | 就绪判定、级联失败、完成与死锁检测、环检测均以纯函数实现，边界情况穷举覆盖 |
| 上下文清理 | 终态任务的结算、去重、降级与合并视图均以纯函数实现；重规划后任务图只含新任务与被引用的已完成任务；对话历史从最旧的一轮开始削减 |
| 分层隔离 | 瘦客户端形态涉及的十个模块在独立子进程中导入后，不出现图引擎及其云端传递依赖 |
| 框架契约 | 所依赖的 LangGraph 语义以最小图逐条固化，升级失配可定位到具体约定 |
| 语义校验 | 什么值得校验、判定能改动什么、修正参数的三道闸均以纯函数实现并穷举覆盖；校验不可用时退回原判定，端到端验证参数偏差就地修正与漏做被补上 |
| 接触面收敛 | AST 扫描确认只有构图与路由两个模块 import langgraph；模型调用范围以 AST 扫描限定在规划、校验、收尾三类环节 |
| 可测试性 | 每一层可独立测试，不依赖上层 |
