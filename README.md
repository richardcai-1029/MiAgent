# MiAgent

A lightweight, general-purpose agent foundation for the MiClaw system-level agent ecosystem.

Built on LangGraph (connected through an adapter layer, so nodes and topology stay framework-agnostic).
The protocol is defined with standard MCP (JSON-RPC 2.0) as its baseline, with the agent as the client
and MiClaw as the server. The model layer targets on-device MiMo; under a single contract it already
supports FakeLLM, Qwen (DashScope) and locally deployed Ollama.

## Documentation

The documents are written in Chinese.

| Document | Contents |
|---|---|
| [01 Communication protocol](docs/01-miclaw-protocol.md) | Transport, messages, method set, error codes, session sequence, on-device resource constraints |
| [02 Adaptation checklist](docs/02-adaptation-checklist.md) | Differences, adaptation items, framework intrusion levels and version rollback |
| [03 Tool specification](docs/03-tool-spec.md) | Tool categories, declaration, error handling, naming and testing requirements |
| [04 Agent architecture](docs/04-agent-architecture.md) | AgentState / Task / Tool Schema, node design, loop exits |
| [05 MiMo on-device inference constraints](docs/05-mimo-inference-constraints.md) | Output stability, context budget and trimming, four lines of defence against format drift |
| [06 Benchmark dataset](docs/06-benchmark-dataset.md) | How the 10,000-case suite is built, category mix, source and self-check of reference answers |
| [07 Code conventions](docs/07-code-conventions.md) | Layering and dependency direction, naming, comments and docstrings, and the automated checks for them |

## Modules

Dependencies point downward and are acyclic (see section 1 of the code conventions; checked by `tests/test_layering.py`):

```
miagent/
├── runtime/       Multi-request runtime: concurrency across requests, serial within a session, priority arbitration of shared resources
├── adapters/      Orchestration framework adapters: build the graph from the topology; langgraph/ is the only place that imports langgraph
├── agent/         Nodes and graph topology (framework-agnostic): plan, schedule, execute, verify, evaluate, replan, finalize
├── memory/        Context memory: episodic memory and settlement ledger, goal anchor, in-process multi-turn sessions
├── core/          Task kernel: state model, task scheduling (DAG), data flow, semantic verification gate, plan schema
├── llm/           Model layer: LLM contract + Fake / OpenAI-compatible / Qwen / Ollama implementations
├── tools/         Tool calling: unified abstraction over local and system tools, argument validation
├── client/        MiClaw client: handshake, calls, response demultiplexing
├── mock_server/   MiClaw mock server: handshake state machine, permissions, resource quotas
├── transport.py   Transport layer: stdio framing + in-process loopback
└── protocol/      Protocol layer: JSON-RPC 2.0 messages + MC-/AG- error code scheme

bench/
├── baseline.py    Lightweight baseline measurement
├── suite/         Benchmark dataset: tool catalog, generation, reference execution, self-check
└── eval/          Evaluation: fault-injection environment, reference-answer-driven model, scoring and aggregation
```

Minimal usage:

```python
from miagent import FakeLLM, ToolRegistry, build_agent, initial_state, tool

@tool()
def add_days(date: str, days: int) -> str:
    """Add a number of days to a date."""
    ...

llm = FakeLLM(script=[...])          # scripted model output; see examples/task_graph.py
agent = build_agent(llm, ToolRegistry([add_days]))
out = agent.invoke(initial_state("What's the date three days from now?"))
print(out["final_answer"])
```

## Running

```bash
uv venv && uv pip install -e ".[graph,dev]"

.venv/bin/python -m pytest tests/ -q          # all tests
.venv/bin/python examples/task_graph.py       # task DAG demo
.venv/bin/python examples/self_check.py       # autonomous verification of results and completion state
.venv/bin/python examples/multi_turn.py       # in-process multi-turn session
.venv/bin/python examples/trace_flow.py       # call trace of the full protocol flow
.venv/bin/python examples/unified_tools.py    # unified tool-layer abstraction
.venv/bin/python bench/baseline.py            # lightweight baseline measurement
.venv/bin/python -m bench.suite.generate      # generate the benchmark dataset (10,000 cases, written to bench/data/)
.venv/bin/python -m bench.suite.check         # dataset self-check
```

With a real model (Qwen, OpenAI-compatible mode):

```bash
uv pip install -e ".[graph,dev,cloud]"        # the cloud extra installs openai
echo 'DASHSCOPE_API_KEY=sk-...' > .env          # or export the variable of the same name
.venv/bin/python examples/qwen_agent.py "Am I free tonight?"
```

With a local model (Ollama, OpenAI-compatible endpoint):

```bash
uv pip install -e ".[graph,dev,cloud]"
OLLAMA_BASE_URL=http://192.168.3.121:11434/v1 .venv/bin/python examples/ollama_agent.py "Am I free tonight?"
```

## Status

All 60 items on the adaptation checklist are done and verified by tests (589 tests). Each item is labelled
with its framework intrusion level; none of them modifies the framework core.

Measured with `bench/baseline.py` (reproducible):

| Deployment | Resident memory | Cold start | Modules loaded |
|---|---|---|---|
| Thin client (protocol + client + tools + model) | ~30 MB | ~56 ms | 151 |
| Full agent (with graph engine) | ~71 MB | ~277 ms | 896 |
