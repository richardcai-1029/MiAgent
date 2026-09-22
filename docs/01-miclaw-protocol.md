# MiAgent ↔ MiClaw 通信规约

| 项目 | MiAgent · 通用 Agent 端侧能力底座 |
|---|---|
| 协议基线 | MCP over JSON-RPC 2.0 |
| 角色 | Agent 为客户端，MiClaw 为服务端 |
| 协议版本 | `2026-01-01` |
| 文档版本 | v1.1 |
| 对应实现 | `miagent/protocol/`、`miagent/transport.py`、`miagent/mock_server/` |

> **规约来源说明**
>
> MiClaw 技术文档当前不可获取。本规约依据项目负责人指示自行制定：以标准
> MCP（JSON-RPC 2.0）为基准，Agent 作客户端、MiClaw 作服务端，错误码以
> `MC-` 表示底层、`AG-` 表示 Agent 上层。规约配套实现了 mock 服务端用于
> 联调，全部约定均以测试代码固化。真实规范开放后，差异集中于本层，
> 上层模块无需改动。

---

## 一、设计依据

**以标准 MCP 为基准而非自创报文格式**，核心考量是生态兼容性：本框架产生的报文可被任何标准 MCP 客户端解析，未来若有其他 Agent 接入 MiClaw、或本 Agent 需接入其他 MCP 服务端，均无需协议转换层。仅在标准确实缺失的地方（Agent 注册、端侧资源约束）作扩展，且扩展方法统一加 `miclaw/` 前缀作命名空间隔离——即便未来标准 MCP 出现同名方法也能共存。

**传输选用 stdio 管道而非 HTTP**，理由有四：零依赖，无需 HTTP 服务器与客户端库；不占端口，因而没有端口冲突与本机网络暴露面；父进程退出时子进程被自动回收，生命周期天然可控；端侧不需要跨主机通信，HTTP 的能力是纯粹的冗余。其中生命周期一条对 MiClaw 需随时回收 Agent 内存的调度需求尤为关键。

---

## 二、传输层

### 分帧

一条报文一行，以 `\n` 结尾。接收方按行读取，每行独立解析为一条 JSON-RPC 报文。

管道传输的是连续字节流，不存在天然的消息边界，因此必须约定分隔符。选用换行符的前提是 JSON 序列化会将正文中的换行转义为 `\\n` 两个字符，不会破坏分帧。

发送方每写完一行必须 flush。管道带缓冲，不 flush 会使报文滞留缓冲区，对端持续等待，双方互锁。

### 通道用途

服务端进程的 stdout 是数据通道，任何非协议输出都会插入协议流导致对端解析失败。服务端的日志一律走 stderr。

---

## 三、报文格式

JSON-RPC 2.0 定义三种报文，本规约不作扩展。

**Request** — 携带 `id`，对方必须回复一条 Response。

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "system.send_sms", "arguments": {"to": "10086", "text": "查话费"}}}
```

**Notification** — 无 `id`，单向通知，对方不得回复。

```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

**Response** — 携带请求方的 `id`，`result` 与 `error` **必须恰好存在一个**。

```json
{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "已发送"}], "isError": false}}
```

`result` 与 `error` 互斥是 JSON-RPC 2.0 的硬性规定，实现中由模型校验器强制。序列化时须排除值为 `null` 的字段，否则会出现 `"result": null` 与 `error` 并存的违规报文。协议交互遵循「发送时严格、接收时宽容」：自身发出的报文必须干净，对端发来的尽量兼容。

报文层采用严格模式，出现协议未定义的字段直接判为非法（`MC-2003`）。业务代码通常宜宽容，协议层则相反——静默忽略未知字段会使版本不一致的问题潜伏到线上才暴露。

---

## 四、方法集

共 8 个方法，其中 5 个为标准 MCP，3 个为 MiClaw 扩展。

| method | 类型 | 允许调用的会话状态 |
|---|---|---|
| `initialize` | 标准 MCP | `connected` |
| `notifications/initialized` | 标准 MCP | `handshaked` |
| `tools/list` | 标准 MCP | `registered` |
| `tools/call` | 标准 MCP | `registered` |
| `ping` | 标准 MCP | `connected`、`handshaked`、`registered` |
| `miclaw/agent.register` | MiClaw 扩展 | `handshaked` |
| `miclaw/agent.unregister` | MiClaw 扩展 | `registered` |
| `miclaw/resource.query` | MiClaw 扩展 | `handshaked`、`registered` |

### `initialize`

握手。交换协议版本与能力，并由服务端下发端侧资源配额。

请求 `params`：`protocolVersion`（字符串）、`clientInfo`（`name`、`version`）。
响应 `result`：`protocolVersion`、`serverInfo`、`resourceBudget`。

版本不匹配返回 `MC-2001` 并中止会话。在版本未确认的情况下继续通信，会产生「报文形式合法但含义不符」的错误，这类问题极难诊断。

### `notifications/initialized`

客户端在成功解析握手响应后发出的确认通知，无 `id`，服务端不回复。它的作用是让服务端确知客户端已认可本次握手——`initialize` 的响应发出后，服务端并不知道客户端是否成功解析。

### `miclaw/agent.register`

Agent 向系统登记身份、能力、权限需求与资源画像。标准 MCP 中无对应概念，是系统级场景的必需扩展：端侧可能同时运行多个 Agent，系统需事先掌握各 Agent 能处理什么意图，才能在用户发起请求时决定任务派给谁。

请求 `params`：

```json
{"agent": {"agentId": "com.xiaomi.miagent.general", "name": "通用助理", "version": "0.1.0"},
 "capabilities": {"intents": ["alarm.create", "message.send"]},
 "permissions": ["alarm.write", "sms.send", "contacts.read"],
 "resourceProfile": {"residency": "on_demand", "estimatedMemoryMb": 96},
 "lifecycle": {"heartbeatIntervalMs": 30000, "autoUnregisterAfterMs": 90000}}
```

响应 `result`：

```json
{"sessionId": "sess-7f3a91",
 "grantedPermissions": ["alarm.write", "sms.send"],
 "deniedPermissions": [{"permission": "contacts.read", "code": "MC-5002", "reason": "用户未授权"}],
 "resourceBudget": {"max_memory_mb": 64, "max_concurrent_calls": 2,
                    "max_call_timeout_ms": 5000, "power_saving": false}}
```

**注册是一次协商而非一次声明。** Agent 提交的是申请，系统返回的是批准结果，两者可能不一致。Agent 必须依据实际批准的权限调整自身可用能力集，而非假定申请必然获准。

### `miclaw/agent.unregister`

主动注销，系统侧释放权限与资源配额。注销后会话进入终止状态，不可复用，需重新建立连接。

### `tools/list`

拉取本 Agent 可用的工具列表。响应 `result.tools` 为工具描述数组，每项含 `name`、`description`、`inputSchema`。

`inputSchema` 使用 JSON Schema 格式，可直接用于大模型的 function calling，无需二次转换。

**服务端按已授权权限过滤列表**，未获授权的工具不出现在结果中。这一设计的价值在于：Agent 会把该列表原样提供给模型，若其中包含无权调用的工具，模型将规划出注定失败的步骤，白白消耗一轮推理——端侧的算力与电量成本都是实在的。

### `tools/call`

调用一个工具。请求 `params`：`name`、`arguments`。响应 `result`：`content`（内容块数组）、`isError`（布尔）。

**`isError` 与 JSON-RPC 层的 `error` 语义不同，不得混用。** `isError` 为真表示工具正常执行但业务上无结果（如查无此单）；JSON-RPC `error` 表示调用本身未能成立（权限不足、资源超限、传输故障）。二者的重试策略截然相反——对前者重试是纯粹浪费，对后者放弃则损失了本可成功的机会。

### `miclaw/resource.query`

查询当前资源配额与占用。允许在握手后、注册前调用，使 Agent 可以先了解自身能获得多少资源，再决定申请哪些权限、声明哪些能力，避免「先承诺、后发现资源不足」而多付一轮往返。

### `ping`

保活探测。在除已注销外的所有会话状态下均允许调用。诊断工具本身若受状态约束，当会话卡在意外状态时将无法确认对端是否存活，这是最不利的情形。

---

## 五、错误码

### 分层原则

分层依据不是严重程度，而是**产生位置**：

- `MC-*` 由 MiClaw 服务端产生，经网络回传给 Agent。排查时定位系统侧日志。
- `AG-*` 由 Agent 自身产生，**永不出现在网络报文中**。排查时定位 Agent 侧日志。

责任边界由此一刀切开。

### 与 JSON-RPC 2.0 的关系

标准规定 `error.code` 必须为整数，与前缀式字符串码存在形式冲突。解决方案是双码并存——整数码保证协议合规，字符串码承载分层语义，置于标准允许的 `error.data` 字段内：

```json
{"jsonrpc": "2.0", "id": 7,
 "error": {"code": -32040, "message": "内存占用超出端侧配额",
           "data": {"code": "MC-4001", "detail": {"limit_mb": 64, "requested_mb": 200}}}}
```

若仅用字符串码则脱离标准生态，仅用整数码则丢失分层语义。`AG-*` 无整数映射，因其从不上网络。

### 编码规则

采用千位分段，段位本身即编码了错误类别。这使上层可仅依据段位推导处置策略，无需逐条查表；新增错误码时策略自动继承，不会因遗漏登记而静默走错分支。错误码一旦出现在线上日志与告警规则中即不可重编号，分段方式同时为每一类预留了扩展空间。

**结构化细节置于 `data.detail`，不得拼接进 `message` 文本**，否则上层丧失程序化处理能力。例如资源超限时同时给出配额上限与实际请求值，供上层判断是否存在可用的降级方案。

### MC-* 底层错误 底层错误（由服务端产生，经网络回传）

| 错误码 | JSON-RPC | 含义 | 重试策略 |
|---|---|---|---|
| **MC-1xxx** | | **传输层** | |
| `MC-1001` | `-32010` | 与 MiClaw 服务端建立连接失败 | `backoff` |
| `MC-1002` | `-32011` | 传输通道已关闭 | `backoff` |
| `MC-1003` | `-32012` | 请求超时未收到响应 | `backoff` |
| **MC-2xxx** | | **协议层** | |
| `MC-2001` | `-32020` | 协议版本不匹配 | `rehandshake` |
| `MC-2002` | `-32021` | 尚未完成 initialize 握手 | `rehandshake` |
| `MC-2003` | `-32600` | 报文不符合 JSON-RPC 2.0 规范 | `rehandshake` |
| `MC-2004` | `-32601` | 服务端不支持该 method | `rehandshake` |
| **MC-3xxx** | | **工具执行** | |
| `MC-3001` | `-32030` | 服务端未注册该工具 | `none` |
| `MC-3002` | `-32602` | 工具入参不合法 | `none` |
| `MC-3003` | `-32031` | 工具执行过程中失败 | `none` |
| **MC-4xxx** | | **端侧资源** | |
| `MC-4001` | `-32040` | 内存占用超出端侧配额 | `degrade` |
| `MC-4002` | `-32041` | 系统资源被占用，请稍后重试 | `degrade` |
| `MC-4003` | `-32042` | 设备处于低电量/省电模式，拒绝执行 | `degrade` |
| **MC-5xxx** | | **权限授权** | |
| `MC-5001` | `-32050` | 权限不足 | `ask_user` |
| `MC-5002` | `-32051` | 用户拒绝了本次授权 | `ask_user` |

### AG-* 上层错误（Agent 自身产生，永不出现在网络报文中）

| 错误码 | 含义 |
|---|---|
| **AG-1xxx** | **规划层** |
| `AG-1001` | 模型输出无法解析为可执行计划 |
| `AG-1002` | 超出单任务最大步数 |
| `AG-1003` | 连续重复同一调用，判定为无进展循环 |
| `AG-1004` | 任务图非法：依赖缺失、自依赖或存在环 |
| `AG-1005` | 存在无法满足的依赖，调度死锁 |
| `AG-1006` | 任务已执行完，但校验判定用户目标未达成 |
| **AG-2xxx** | **工具调度** |
| `AG-2001` | Agent 本地未注册该工具 |
| `AG-2002` | 工具参数未通过 schema 校验 |
| `AG-2003` | 工具返回内容无法解析 |
| `AG-2004` | 工具执行时抛出未预期异常 |
| `AG-2005` | 工具结果未达成任务目标，校验未通过 |
| **AG-3xxx** | **上下文** |
| `AG-3001` | 上下文长度超出模型窗口 |
| `AG-3002` | Agent 状态非法 |
| **AG-4xxx** | **生命周期** |
| `AG-4001` | 任务被取消 |
| `AG-4002` | 超出任务耗时或 token 预算 |
| `AG-4003` | 在错误的会话阶段发起调用 |
| **AG-5xxx** | **模型层** |
| `AG-5001` | 模型服务不可用 |
| `AG-5002` | 模型返回内容异常 |

---

## 六、会话时序

会话被建模为四状态状态机。方法与其允许状态的对应关系见第四节表格。

```
  [connected] ──initialize──→ [handshaked] ──agent.register──→ [registered]
       ↑        版本不匹配 MC-2001      │                             │
   进程启动                             │                      unregister
                                        │                             ↓
                                        └──心跳超时（待实现）──→ [closed]
```

各状态的含义是**系统对该 Agent 掌握了多少信息**，而每个方法的允许范围取决于**该方法需要多少信息才能正确执行**：

- `connected`——管道已通，但协议版本未知。仅允许 `initialize` 与 `ping`。
- `handshaked`——版本已对齐，但系统尚不知道这是哪个 Agent。允许注册与资源查询；不允许调用工具，因为 `tools/list` 需按权限过滤、`tools/call` 需检查授权与配额，二者都依赖注册产生的信息。
- `registered`——身份、权限、配额均已确定，全部业务方法可用。
- `closed`——已注销，任何方法均不可用，包括 `ping`。

违反时序的调用返回 `MC-2002`，错误详情中给出当前状态与所需状态。

客户端侧另实现一份轻量预检，非法调用在本地即被拒绝（`AG-4003`），不跨出进程边界，节省一次进程间往返。其定位类似前端表单校验，服务端始终是权威。

---

## 七、端侧资源约束

云端 Agent 无需关心内存与电量，端侧必须关心。这是端侧相对云端多出的一整个约束维度。

规则由三部分构成：

1. **配额下发** —— 会话建立时由系统下发 `resourceBudget`，含内存上限、并发调用上限、单次调用超时、省电模式标志。
2. **工具申报** —— 每个工具在声明中申报预估内存开销。
3. **执行前校验** —— 服务端比对二者，超限则拦截并返回 `MC-4001`，`detail` 中同时给出配额上限与实际请求值。

权限与资源是两道彼此独立的关卡，通过其一不代表通过其二。

配额中的各字段分别在何处生效：

| 字段 | 执行位置 |
|---|---|
| `max_memory_mb` | 服务端在工具执行前比对工具申报的预估开销，超限返回 `MC-4001` |
| `max_concurrent_calls` | Agent 侧调度器限制同一轮并行派发的 MiClaw 调用数，超出者顺延 |
| `max_call_timeout_ms` | Agent 侧客户端用作单次工具调用的等待上限，超时返回 `MC-1003` |
| `power_saving` | 服务端据此拒绝高开销调用，返回 `MC-4003` |

配额字段若只在报文里声明而无执行机制，等同于没有这条约束。

---

## 八、版本与扩展

协议版本采用日期格式（当前 `2026-01-01`），握手时双方比对，不匹配即中止。

扩展方法一律使用 `miclaw/` 前缀。任何标准 MCP 客户端看到不认识的 `miclaw/xxx` 会明确知道这是厂商扩展，而非协议损坏；未来标准 MCP 若出现同名方法，两者可以共存而无需重命名。错误码与方法名一旦上线即为永久 API，此项隔离是必要的前置设计。
