# ml-platform

Shared Python library for deploying ML and LLM services with consistent monitoring, experiment tracking, and infrastructure. Supports three service paths:

- **Agent services** (`AgentServiceBase`) -- multi-step LLM workflows with tool use and multiple providers
- **Stateful services** (`StatefulServiceBase`) -- online learning, contextual bandits, and feedback loops
- **Stateless services** (BentoML `PlatformMonitor`) -- inference-only endpoints with consistent monitoring

All paths share built-in `/dashboard`, pluggable metrics backends, experiment tracking, and cloud deployment via `ml-platform deploy`.

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         Your ML Service          │
                         └──────┬──────────┬──────────┬─────┘
                                │          │          │
            ┌───────────────────┘          │          └───────────────────┐
            ▼                              ▼                              ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │  Agent Path          │   │  Stateful Path       │   │  Stateless Path      │
 │                      │   │                      │   │  (BentoML + monitor) │
 │  AgentServiceBase    │   │  StatefulServiceBase │   │                      │
 │  AgentRuntime        │   │  StatefulRuntime     │   │  PlatformMonitor     │
 │  create_agent_app()  │   │  create_stateful_    │   │  with_platform_      │
 │  RunContext          │   │    app()             │   │    monitoring()      │
 │  /run /health        │   │  /predict /feedback  │   │                      │
 │  /metrics /dashboard │   │  /health /metrics    │   │                      │
 └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
            │                          │                          │
            └──────────────────┬───────┴──────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │          Shared Platform Layer        │
            │                                       │
            │  MetricsBackend   StateManager        │
            │  ContextStore     ConversationStore   │
            │  ExperimentTracker  LLMProvider       │
            │  Tool protocol    RunContext          │
            └──────────────────┬────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │       CDK Infrastructure Constructs   │
            └──────────────────────────────────────┘
```

## Which path should I use?

| Scenario | Recommended path |
|---|---|
| Single LLM call per request (chatbot, RAG, summariser) | `LLMServiceBase` |
| Multiple LLM calls + tools per request (agents, plan-execute) | `AgentServiceBase` |
| Online learning with delayed feedback (bandits, dynamic pricing) | `StatefulServiceBase` |
| Existing BentoML model needing monitoring | `PlatformMonitor` |

## Quick Start

### Installation

```bash
pip install ml-platform                    # Core only
pip install "ml-platform[stateful]"        # + FastAPI + boto3
pip install "ml-platform[all]"             # Everything
```

## Examples by use case

### 1. Minimal LLM app (3 lines to a dashboard)

```python
from ml_platform.llm import RunContext

async with RunContext(name="my-app") as ctx:
    result = await ctx.complete(my_provider, messages)
# Cost, tokens, and latency tracked automatically
```

### 2. Multi-step agent (multiple LLM calls + tools)

```python
from ml_platform.config import ServiceConfig, AgentConfig
from ml_platform.serving.agent import AgentServiceBase, create_agent_app
from ml_platform._types import Message, AgentResult

class SearchDocsTool:
    name = "search_docs"
    description = "Search the knowledge base for relevant documents"
    parameters_schema = {"query": {"type": "string"}}

    async def execute(self, query: str = "") -> str:
        return f"Found 3 documents about '{query}'"

class MyResearchAgent(AgentServiceBase):
    async def run(self, messages, *, run_context, **kwargs):
        # Step 1: Plan what to do
        plan = await run_context.complete(
            self.providers["planner"], messages, step_name="plan"
        )
        # Step 2: Search for information
        search_result = await run_context.execute_tool(
            self.tools["search_docs"], query=plan.content
        )
        # Step 3: Synthesize a response
        response = await run_context.complete(
            self.providers["writer"],
            messages + [Message(role="user", content=search_result.content)],
            step_name="synthesize",
        )
        return AgentResult(
            content=response.content,
            steps=run_context.steps,
            messages=messages,
        )

config = ServiceConfig(
    service_name="research-agent",
    agent=AgentConfig(max_steps_per_run=10),
)
app = create_agent_app(
    MyResearchAgent, config,
    providers={"planner": my_openai_provider, "writer": my_anthropic_provider},
    tools=[SearchDocsTool()],
)
# Run: uvicorn my_agent:app --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000/dashboard
```

The dashboard automatically shows per-run metrics, tool usage, and multi-model cost breakdowns -- no configuration needed.

### 3. Stateful bandit service

```python
from ml_platform.config import ServiceConfig
from ml_platform.serving.stateful import (
    StatefulServiceBase, PredictionResult, create_stateful_app,
)

class MyBanditService(StatefulServiceBase):
    def __init__(self) -> None:
        self._total_reward = 0.0

    async def predict(self, payload: dict) -> PredictionResult:
        return PredictionResult(
            request_id="req-001",
            prediction={"arm": "model-a"},
            metadata={"cost_usd": 0.003},
        )

    async def process_feedback(self, request_id: str, feedback: dict) -> None:
        self._total_reward += feedback.get("reward", 0.0)

    def save_state(self, artifact_dir: str) -> None: ...
    def load_state(self, artifact_dir: str) -> None: ...
    def metrics_snapshot(self) -> dict[str, float]:
        return {"total_reward": self._total_reward}

config = ServiceConfig(
    service_name="pareto-bandit",
    s3_checkpoint_bucket="my-checkpoints",
)
app = create_stateful_app(MyBanditService, config)
```

### 4. Using RunContext standalone (no FastAPI)

```python
from ml_platform.llm import RunContext

async def my_workflow():
    async with RunContext(name="research") as ctx:
        plan = await ctx.complete(openai_provider, messages, step_name="plan")
        result = await ctx.execute_tool(search_tool, query="LLMOps")
        summary = await ctx.complete(anthropic_provider, messages, step_name="summarize")

    print(f"Total cost: ${ctx.total_cost_usd:.4f}")
    print(f"Steps: {ctx.llm_call_count} LLM calls, {ctx.tool_call_count} tool calls")
```

## Custom metrics

Agent apps automatically emit per-run aggregates without any configuration:

- `total_tokens`, `total_cost_usd` -- per run
- `steps`, `llm_calls`, `tool_calls` -- per run
- `avg_steps_per_run` -- across all runs
- `total_latency_ms` -- wall-clock per run

For custom business metrics, override `metrics_snapshot()`:

```python
class MyAgent(AgentServiceBase):
    def metrics_snapshot(self) -> dict[str, float]:
        return {"cache_hit_rate": self._cache_hits / max(self._total, 1)}
```

## Monitoring dashboards

The composable dashboard system auto-detects your service type:

```python
from ml_platform.monitoring.dashboards import generate_dashboard

config = ServiceConfig(service_name="my-agent", agent=AgentConfig())
grafana = generate_dashboard(config)  # auto-includes core + llm + agent panels

# Or explicit panel selection:
grafana = generate_dashboard(config, panel_sets=["core", "llm", "agent"])
```

### Agent-specific dashboard panels

| Panel | Description |
|---|---|
| Steps per Run | Histogram of steps per user request |
| Tool Usage Distribution | Which tools are called most, success/error rates |
| Multi-Model Cost Breakdown | Cost by model across runs |
| Run Duration vs LLM Time | Overhead from tool execution, parsing, etc. |

## CLI

```bash
ml-platform init --template agent --name my-agent   # Scaffold a new project
ml-platform bootstrap --service-name svc             # Create AWS resources
ml-platform check --service-name svc                 # Verify AWS configuration
```

### Templates

| Template | Description | Base class |
|---|---|---|
| `agent` | Multi-step agent with tools | `AgentServiceBase` |
| `chatbot` | Single LLM, multi-turn | `LLMServiceBase` |
| `rag` | Retrieval + single LLM | `LLMServiceBase` |
| `bandit` | Online learning with feedback | `StatefulServiceBase` |

## Development

```bash
git clone <repo-url> && cd ml-platform
pip install -e ".[dev]"
pytest                      # 109+ tests
ruff check src/ tests/
ruff format src/ tests/
```

## API reference

- [AgentServiceBase & create_agent_app()](docs/api/serving/agent.md)
- [RunContext](docs/api/llm/run-context.md)
- [Tool protocol, ToolCall, ToolResult](docs/api/llm/tools.md)
