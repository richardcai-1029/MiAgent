# LangGraph → 小米端侧生态 适配改造清单

| 项目 | MiAgent · 通用 Agent 端侧能力底座 |
|---|---|
| 基础框架 | LangGraph 1.2.10（langchain-core 1.5.3） |
| 目标生态 | MiClaw 系统级 Agent 生态 · MiMo 端侧大模型 |
| 文档版本 | v2.0 |

本清单记录已完成并经测试验证的改造项。每项均给出落地位置与验证方式，可逐条核查。

> **关于 MiClaw 规范来源**：MiClaw 技术文档当前不可获取。清单中所有与 MiClaw
> 交互相关的条目，均依据项目负责人指示——「以标准 MCP (JSON-RPC 2.0) 为基准
> 自行定规约，Agent 做客户端、MiClaw 为服务端，错误码 MC- 代表底层、AG- 代表
> Agent 上层」——推导设计，规约见《MiAgent ↔ MiClaw 通信规约》。待真实规范
> 开放后，差异集中在协议层，上层无需改动。

---

## 一、改造项

### A · 协议与通信适配

LangGraph 的工具调用是进程内 Python 函数调用，不存在跨进程协议、身份注册与时序约束的概念。这一整块从零建立。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| A-1 | 无系统级通信协议 | 建立 JSON-RPC 2.0 报文层，`result`/`error` 互斥等协议不变量由校验器强制 | `protocol/messages.py`；`test_protocol.py` |
| A-2 | 工具调用为进程内直调，无跨进程能力 | stdio 按行分帧传输，不引入 HTTP 栈 | `transport.py`；`test_transport.py` 含真实子进程用例 |
| A-3 | 无 Agent 身份注册语义 | 扩展 `miclaw/agent.register`，注册为「申请—批准」协商而非单向声明 | `mock_server/server.py::_on_register` |
| A-4 | 会话无时序约束，任何时刻均可调用工具 | 握手状态机，以声明式状态表强制 `initialize → register → 调用` | `_REQUIRED_STATE`；违规返回 MC-2002 |
| A-5 | 厂商扩展方法与标准方法无命名隔离 | 扩展方法统一 `miclaw/` 前缀，避免与未来标准 MCP 方法撞名 | `test_protocol.py::test_extensions_are_namespaced` |
| A-6 | 无协议版本协商 | `initialize` 比对版本，不匹配返回 MC-2001 | `test_mock_server.py::test_version_mismatch` |

### B · 错误处理体系

LangChain 工具失败表现为 Python 异常或自由文本，无错误分类，上层无法据此做差异化决策。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| B-1 | 错误无分层归属，无法区分底层与上层故障 | MC-/AG- 双层码；MC- 经网络回传，AG- 不出进程 | `protocol/errors.py`；测试强制 AG-* 无整数码映射 |
| B-2 | 错误细节混在文本中，上层无法程序化处理 | 结构化 `error.data.detail`，携带机器可读字段 | `test_client.py::test_detail_survives_the_wire` |
| B-3 | 错误码在网络两侧无法还原为同一异常 | 客户端按 `data.code` 还原为 `MiClawError`；未知码兜底不崩溃 | `client.py::_to_exception` |
| B-4 | 无按错误类别的差异化重试策略 | 按千位段位决策：1xxx 退避重试、4xxx 延迟或降级、3xxx/5xxx 不重试 | `protocol/errors.py::retry_policy`；`test_tools.py::TestRetryPolicyByBand` |
| B-6 | 业务失败（`isError`）与调用失败（JSON-RPC error）未在上层区分 | 上层据此选择「让模型换方案」还是「退避重试」 | `client.py::call_tool` 将 isError 映射为 MC-3003；见 `test_business_failure_differs_from_call_failure` |

### C · 端侧资源约束

LangGraph 面向云端设计，不存在内存配额、并发上限等概念。这是端侧相对云端多出的一整个约束维度。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| C-2 | 无资源配额下发机制 | 握手时下发 `ResourceBudget`，Agent 全程受其约束 | `protocol/messages.py::ResourceBudget` |
| C-3 | 工具无资源画像，无法预判开销 | 工具申报 `estimated_memory_mb`，服务端执行前校验 | 超配额返回 MC-4001，见 `test_memory_limit` |
| C-6 | `max_concurrent_calls` 无执行机制 | 当前同步实现天然串行；如引入并发需加信号量，否则应移除该字段 | `scheduler` 按 `max_concurrent_calls` 限流 MiClaw 派发，超出者顺延；本地工具受 GIL 限制不设限 |

### D · MiMo 模型接入

LangChain 的 LLM 抽象假设云端 API，与端侧模型的运行方式与可靠性特征差异较大。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| D-1 | LLM 抽象面向云端 API | 定义最小 LLM 接口，支持 Fake / MiMo 端侧 / 云端三种实现热切换 | `miagent/llm/`；模板方法统一承担上下文检查与耗时统计，Fake/云端两种实现已可切换 |
| D-4 | 工具描述格式与模型 function calling 格式未打通 | MCP `inputSchema` 本就是 JSON Schema，可直接喂模型，无需转换 | `test_client.py::test_tool_schema_survives_the_wire` |
| D-5 | 提示词无法保证模型输出符合预期结构，一次格式失误即导致任务失败 | 计划改用 Pydantic schema 驱动：schema 由模型定义导出、与校验同源；解析失败带着具体错误自修复重试；工具名收进 enum 使幻觉在解析阶段即被拒 | `graph/schema.py`、`llm/base.py::complete_structured` |

### E · 依赖裁剪与轻量化

端侧存在只需协议客户端的部署形态，不应为图引擎付出常驻内存与冷启动的代价。

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| E-1 | 协议层与客户端不应绑定图引擎 | `langgraph` 拆为可选依赖 `[graph]`，协议/传输/客户端仅依赖 pydantic | `pyproject.toml`；60 个测试在无 langgraph 时仍可运行 |
| E-7 | 图引擎随包导入被无条件加载，纯逻辑模块也要付出其常驻代价 | `miagent.graph` 以 PEP 562 惰性导出 `build_agent`：依赖解析、状态定义、计划 schema 均为纯 Python，不触发图引擎加载 | 该包导入代价由 873 模块 / 69.2 MB 降至 141 模块 / 29.2 MB |
| E-8 | 缺少防止分层退化的机制 | 自动化守卫：瘦客户端形态涉及的九个模块，在独立子进程中导入后不得出现 langgraph / langchain_core / langsmith / requests 等任一重依赖 | `tests/test_layering.py`，11 条用例 |

### F · 工具体系适配

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| F-1 | 本地工具（进程内函数）与 MiClaw 系统工具（走协议）无统一抽象 | 统一 Tool 基类 + 注册表，屏蔽调用方式差异，保留失败模式差异 | `miagent/tools/`；87 个测试覆盖两类工具的统一入口 |
| F-6 | 无工具开发规范文档 | 输出标准化开发规范，含声明字段、错误约定、测试要求 | `docs/03-tool-spec.md` |

### G · 任务调度

| 编号 | 差异点 | 改造内容 | 落地位置 / 验证 |
|---|---|---|---|
| G-2 | 无步数与预算上限，模型可能陷入死循环 | 落地 AG-1002（超步数）、AG-1003（无进展）、AG-4002（超预算） | 单任务重试上限、重规划上限、累计执行上限三道闸，映射 AG-1002/AG-1003 |
| G-4 | 框架无任务依赖建模，只能线性执行，无法表达分支与汇合 | 引入任务 DAG：显式 dependencies、五态生命周期、依赖解析/就绪判定/完成检测/死锁检测/级联失败全部为确定性纯函数，不交给模型 | `graph/dag.py`；`test_dag.py` 25 条用例覆盖边界 |
| G-5 | 无依赖关系的任务仍被串行执行，浪费 IPC 等待时间 | 以 LangGraph Send 并行派发同层任务；需为 tasks 定义按 id 合并的 reducer，避免多分支写回互相覆盖 | `routers.py` 返回 Send 列表扇出；outcomes 与 execution_count 用 reducer 汇总；实测无依赖任务提速 3.00x |
| G-6 | 任务图非法（依赖缺失/自依赖/成环）会表现为莫名死锁 | Kahn 拓扑排序在执行前校验，映射 AG-1004；运行期死锁映射 AG-1005 | `dag.validate()`；`test_cyclic_plan_rejected_before_execution` |

---

## 二、实测数据

按部署形态测量。端侧的实际约束是「这一形态的进程常驻多少内存」，按单个模块统计没有意义。
环境：macOS 24.6 / Python 3.13.9 / langgraph 1.2.10。测量脚本 `bench/baseline.py`，可复现；
内存取三次运行的中位数、耗时取最小值，故下表为取整后的近似值。

| 部署形态 | 常驻内存 | 冷启动 | 加载模块数 | 其中云端/网络相关 |
|---|---|---|---|---|
| 裸解释器 | 约 17 MB | — | — | — |
| 瘦客户端 | 约 30 MB | 约 55 ms | 137 | 0（0%） |
| 完整 Agent | 约 70 MB | 约 270 ms | 875 | 333（38%） |

瘦客户端形态涵盖协议、传输、客户端、工具、模型五层，可完成系统调用转发与工具调度，
不含任务规划与依赖调度；完整形态在其上追加图引擎。图引擎的边际成本为常驻约 +40 MB、冷启动约 +215 ms、模块 +738。

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

完整 Agent 形态加载的 875 个模块中，333 个（38%）属于上表中端侧不需要的云端与网络栈。
这些模块由 `langchain_core` 与 `langgraph_sdk` 在模块层面具名引入，是 LangGraph 的
传递依赖。轻量化因此采取限制影响范围的路径：图引擎降为可选依赖（E-1）、包级惰性
导出使纯逻辑模块不触发其加载（E-7）、并以自动化守卫防止分层退化（E-8）。

---

## 三、验证方式

规约与改造的全部约定均以测试代码固化，当前累计 167 条用例。

| 验证项 | 方式 |
|---|---|
| 协议合规 | 报文可被标准 MCP 客户端解析；错误码整数部分符合 JSON-RPC 2.0 |
| 错误码完备 | 遍历全部错误码，MC-* 必有整数映射、AG-* 必无；任一 MC-* 均可推导出重试策略 |
| 时序正确 | 违反 initialize → register → 调用 时序的调用被拒绝并返回 MC-2002 |
| 错误闭环 | 服务端抛出的每个 MC- 码，均可在客户端还原为同一 ErrorCode 并保留结构化细节 |
| 资源约束 | 超出下发配额的工具调用被拦截，detail 中给出实际值与上限 |
| 依赖解析 | 就绪判定、级联失败、完成与死锁检测、环检测均以纯函数实现，边界情况穷举覆盖 |
| 分层隔离 | 瘦客户端形态涉及的九个模块在独立子进程中导入后，不出现图引擎及其云端传递依赖 |
| 可测试性 | 每一层可独立测试，不依赖上层 |
