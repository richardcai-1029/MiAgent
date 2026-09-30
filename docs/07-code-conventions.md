# 代码规范

| 项目 | MiAgent · 通用 Agent 端侧能力底座 |
|---|---|
| 范围 | `miagent/` 下全部代码的分层、命名与注释 |
| 文档版本 | v1.0 |
| 对应检查 | `tests/test_layering.py`、`tests/test_conventions.py` |

本规范中可以机械判定的条目都由测试检查，违反时测试直接指出位置；不能机械判定的条目（如注释写什么）靠评审。工具本身的声明、命名与错误处理见《工具开发规范》。

---

## 一、分层与依赖方向

| 包 | 职责 | 允许依赖 |
|---|---|---|
| `protocol` | JSON-RPC 2.0 报文、MC- / AG- 错误码 | — |
| `transport` | stdio 分帧收发、进程内回环 | protocol |
| `client` | MiClaw 客户端：握手、调用、收发分流 | protocol、transport |
| `mock_server` | 按通信规约实现的 MiClaw 服务端，联调与测试用 | protocol、transport |
| `tools` | 工具调用模块：本地工具与系统工具的统一抽象、参数校验 | protocol |
| `llm` | 模型契约、结构化输出、上下文预算与裁剪 | protocol |
| `core` | 任务内核：状态模型、任务调度算法、数据流、校验把关、计划 schema | protocol、tools |
| `memory` | 情景记忆、账本、目标锚、多轮会话 | protocol、llm、core |
| `agent` | 节点与图拓扑 | protocol、tools、llm、core、memory |
| `adapters` | 编排框架适配，照拓扑构图 | tools、llm、core、agent |
| `runtime` | 多请求运行时 | protocol、llm、core、memory、adapters |

- 依赖只能指向表中允许的包，整体无环。只统计运行时生效的导入；`if TYPE_CHECKING:` 块里只用于类型标注的导入不计。
- 只有 `adapters/langgraph/build.py` 可以 import langgraph。新增框架依赖须同步更新《适配改造清单》的风险分级。
- `core`、`memory`、`agent`、`llm`、`tools` 及以下各层在不装 langgraph 时都能导入。
- 顶层 `miagent/__init__.py` 只做惰性导出，不参与分层；新增对外名字时同时登记 `_EXPORTS` 与 `__all__`。

检查：`test_package_dependencies_follow_layering`、`test_framework_surface_is_confined`、`test_light_layer_does_not_pull_graph_engine`、`test_top_level_exports_resolve`。

---

## 二、命名

### 包与模块

- 小写下划线，用名词说职责：`dag`、`dataflow`、`ledger`、`topology`。不用 `utils`、`common`、`misc`、`helpers` 这类不说明职责的名字。
- 一个模块放一类职责。节点按职责分文件：规划与重规划共用规划通道，同在 `planning.py`；两个校验节点共用调用实现，同在 `verifiers.py`。
- 私有模块与私有定义以下划线开头；跨模块使用的不加下划线。

检查：`test_module_names_are_snake_case`。

### 节点

| 场合 | 形式 | 例 |
|---|---|---|
| 文档与注释里的称呼 | 大驼峰 | `ResultVerifier`、`MiClawExecutor` |
| 图中的节点名 | 小写下划线 | `result_verifier`、`miclaw_executor` |
| 实现函数 | 与节点名相同 | `def result_verifier(state, deps)` |
| trace 行 | 以节点名加冒号开头 | `result_verifier: 校验 2 项 → 全部通过` |

两个执行节点共用实现 `executor`，节点名为 `<source>_executor`。

检查：`test_node_ids_match_implementations`、`test_topology_is_closed`、`test_trace_lines_start_with_node_id`。

### 其他

| 对象 | 形式 | 例 |
|---|---|---|
| 类、TypedDict、枚举 | 大驼峰 | `TaskOutcome`、`RetryPolicy` |
| 函数、方法、变量 | 小写下划线 | `cascade_failures` |
| 模块级常量 | 全大写下划线 | `PROTOCOL_VERSION` |
| 错误码成员 | `<段>_<含义>`，值为 `MC-nxxx` / `AG-nxxx` | `AG_GOAL_NOT_ACHIEVED = "AG-1006"` |
| 协议扩展方法 | `miclaw/` 前缀 | `miclaw/task.dispatch` |
| 系统工具 / 本地工具 | 见《工具开发规范》第八节 | `system.get_battery`、`add_days` |

与 MiClaw 相关的标识一律用 `miclaw`（`ToolSource.MICLAW`、`max_concurrent_miclaw`、`miclaw_executor`）。MCP 只在指称协议基准时使用。

---

## 三、注释与文档字符串

### 写什么

- 注释只写**当前设计**：为什么这样做、不这样做会怎样。不写已废弃的做法、未采纳的方案和修改经过——这些属于提交记录与周报。
- 需求没有给出的数值不写成阈值。占位默认值须标明无外部依据（见下文 ⚠️）。
- 引用改造项时写清单编号（如「清单 C-6」），引用错误码时写码值（如 `AG-1003`）。
- 注释与文档字符串用中文；标识符、代码片段、协议字段保持原文。

### 文档字符串

- 每个模块都有文档字符串：第一行说模块是什么，之后说设计要点。
- 模块顶层的公开类与函数都有文档字符串；一句话能说清的就写一句。私有定义按需。
- 纯函数模块在文档字符串里声明「不调用模型、不 import 编排框架、没有副作用」。

检查：`test_every_module_has_docstring`、`test_public_top_level_definitions_have_docstrings`。

### 标记

| 标记 | 用途 |
|---|---|
| `★` | 设计要点或不显然的不变量：改动它之前必须读懂的地方 |
| `⚠️` | 占位值或危险点：默认值无外部依据、stdout 是数据通道之类 |

不使用 `TODO`、`FIXME`、`XXX`。未做的事记在文档的待办里，不留在代码中。

### 分节注释

只有三种形式：

```python
# ============================================================
# 模块级分节标题
# ============================================================

class Foo:
    # ------------------------------------------------------------
    # 类内分节标题
    # ------------------------------------------------------------

    # ---------- 字段分组 ----------
    bar: int
```

检查：`test_section_banners_follow_convention`。

---

## 四、新增代码的检查顺序

1. 放在哪个包：按第一节的职责表，依赖只能指向允许的包。
2. 名字：按第二节；新节点同时登记到 `agent/topology.py` 的 `NODES` 与边。
3. 文档字符串与注释：按第三节。
4. 运行 `tests/test_layering.py` 与 `tests/test_conventions.py`，再跑全量测试。
