# AgentServiceBase & create_agent_app()

Multi-step agent services that make multiple LLM calls and/or tool executions per user request.

## When to use AgentServiceBase

| Scenario | Recommended path |
|---|---|
| Single LLM call per request | `LLMServiceBase` |
| Multiple LLM calls + tools per request | **`AgentServiceBase`** |
| Online learning with feedback loop | `StatefulServiceBase` |
| Existing BentoML model | `PlatformMonitor` |

## AgentServiceBase ABC

```python
from ml_platform.serving.agent import AgentServiceBase

class MyAgent(AgentServiceBase):
    async def run(
        self,
        messages: list[Message],
        *,
        run_context: RunContext,
        **kwargs,
    ) -> AgentResult:
        """Execute a full agent turn.

        Use run_context.complete() for LLM calls and
        run_context.execute_tool() for tool executions.
        """
        ...

    def metrics_snapshot(self) -> dict[str, float]:
        """Optional: periodic aggregate metrics."""
        return {}
```

### Attributes set by framework

- `providers: dict[str, LLMProvider]` -- named LLM providers
- `tools: dict[str, Tool]` -- named tools keyed by `tool.name`

## create_agent_app()

```python
from ml_platform.serving.agent import create_agent_app

app = create_agent_app(
    service_cls=MyAgent,
    config=ServiceConfig(service_name="my-agent", agent=AgentConfig()),
    providers={"planner": openai_provider, "writer": anthropic_provider},
    tools=[SearchDocsTool(), CalculatorTool()],
    service_kwargs={},  # extra kwargs forwarded to MyAgent()
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `service_cls` | `Type[AgentServiceBase]` | Concrete agent class |
| `config` | `ServiceConfig` | Platform configuration |
| `providers` | `dict[str, LLMProvider]` | Named LLM providers |
| `tools` | `list[Tool]` | Tool instances (keyed by `.name`) |
| `service_kwargs` | `dict` | Extra kwargs for `service_cls()` |

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/run` | Execute agent turn, returns `AgentRunResponse` |
| `GET` | `/health` | Readiness probe (`{"status": "healthy", "service": "..."}`) |
| `GET` | `/metrics` | Agent-specific aggregate metrics |
| `GET` | `/dashboard` | Self-contained HTML dashboard |
| `GET` | `/dashboard/api/metrics` | JSON metrics for the dashboard |

### POST /run

**Request body:**

```json
{
  "messages": [
    {"role": "user", "content": "Research LLMOps best practices"}
  ]
}
```

**Response body:**

```json
{
  "content": "Based on my research...",
  "steps": [
    {"name": "plan", "step_type": "llm_call", "provider": "openai", "model": "gpt-4o", ...},
    {"name": "search", "step_type": "tool_call", ...},
    {"name": "synthesize", "step_type": "llm_call", "provider": "anthropic", ...}
  ],
  "metadata": {},
  "total_tokens": 450,
  "total_cost_usd": 0.008,
  "llm_calls": 2,
  "tool_calls": 1
}
```

**Error responses:**

- `503` -- service not initialised yet
- `429` -- agent exceeded `max_steps_per_run`

## AgentRuntime

Manages per-request `RunContext` creation, tool registration, and aggregate metric tracking.

Created internally by `create_agent_app()`. Access directly only if you need non-HTTP serving:

```python
from ml_platform.serving.runtime import AgentRuntime

runtime = AgentRuntime(
    MyAgent, config,
    providers={"main": provider},
    tools=[tool],
)
await runtime.startup()
result = await runtime.run([Message(role="user", content="hi")])
await runtime.shutdown()
```

## Complete example

```python
from ml_platform.config import ServiceConfig, AgentConfig
from ml_platform.serving.agent import AgentServiceBase, create_agent_app
from ml_platform._types import Message, AgentResult

class SearchDocsTool:
    name = "search_docs"
    description = "Search the knowledge base"
    parameters_schema = {"query": {"type": "string"}}

    async def execute(self, query: str = "") -> str:
        return f"Found 3 documents about '{query}'"

class ResearchAgent(AgentServiceBase):
    async def run(self, messages, *, run_context, **kwargs):
        plan = await run_context.complete(
            self.providers["planner"], messages, step_name="plan"
        )
        search = await run_context.execute_tool(
            self.tools["search_docs"], query=plan.content
        )
        response = await run_context.complete(
            self.providers["writer"],
            messages + [Message(role="tool", content=search.content,
                                tool_call_id=search.tool_call_id)],
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
    ResearchAgent, config,
    providers={"planner": my_openai, "writer": my_anthropic},
    tools=[SearchDocsTool()],
)
# uvicorn my_agent:app --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000/dashboard
```
