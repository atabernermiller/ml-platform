# Tool Protocol

The `Tool` protocol defines how agents interact with external capabilities. The library wraps tool execution with observability but never implements tool logic.

## Protocol definition

```python
from typing import Any, Protocol

class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    async def execute(self, **kwargs: Any) -> str: ...
```

All four members are required. Missing any will raise a `TypeError` at app creation time.

## Implementing a tool

```python
class SearchDocsTool:
    """Search the knowledge base for relevant documents."""

    name = "search_docs"
    description = "Search the knowledge base for relevant documents"
    parameters_schema = {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "default": 5},
    }

    async def execute(self, query: str = "", max_results: int = 5) -> str:
        results = await search_engine.search(query, limit=max_results)
        return "\n".join(f"- {r.title}: {r.snippet}" for r in results)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier for the tool |
| `description` | `str` | What the tool does (shown to LLMs for tool selection) |
| `parameters_schema` | `dict` | JSON-Schema-like dict of accepted kwargs |

### execute()

The `execute` method receives keyword arguments matching `parameters_schema` and returns a string result.

- **Return type**: Always `str`. If your tool produces structured data, serialize it (JSON, Markdown table, etc.).
- **Errors**: Exceptions are caught by `RunContext`, recorded as an error step, and do not crash the agent run. The agent receives a `ToolResult` with `error` set.
- **Async**: Tools must be async (`async def execute`).

## ToolCall and ToolResult types

```python
from ml_platform._types import ToolCall, ToolResult

# ToolCall: represents an LLM's request to invoke a tool
call = ToolCall(
    id="tc_abc123",
    name="search_docs",
    arguments={"query": "LLMOps", "max_results": 3},
)

# ToolResult: the outcome of executing a tool
result = ToolResult(
    tool_call_id="tc_abc123",
    name="search_docs",
    content="- Doc 1: ...\n- Doc 2: ...",
    error=None,       # None on success, error message on failure
    latency_ms=150.0,
)
```

## Registering tools with create_agent_app()

Tools are passed as a list to the factory. They are keyed by `tool.name`:

```python
from ml_platform.serving.agent import create_agent_app

app = create_agent_app(
    MyAgent, config,
    providers={"planner": openai, "writer": anthropic},
    tools=[SearchDocsTool(), CalculatorTool(), DatabaseTool()],
)
```

Inside the agent, access tools by name:

```python
class MyAgent(AgentServiceBase):
    async def run(self, messages, *, run_context, **kwargs):
        result = await run_context.execute_tool(
            self.tools["search_docs"], query="LLMOps"
        )
```

## How tool execution is traced

When `run_context.execute_tool()` is called:

1. A timer starts
2. `tool.execute(**kwargs)` is awaited
3. If the tool raises, the error is caught and recorded
4. An `AgentStep` with `step_type="tool_call"` is appended
5. A `ToolResult` is returned to the caller

The agent dashboard automatically shows tool usage distribution and success/error rates.

## Example tools

### Calculator

```python
class CalculatorTool:
    name = "calculator"
    description = "Evaluate a mathematical expression"
    parameters_schema = {"expression": {"type": "string"}}

    async def execute(self, expression: str = "") -> str:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
```

### Database lookup

```python
class DatabaseLookupTool:
    name = "db_lookup"
    description = "Query the product database"
    parameters_schema = {
        "sql": {"type": "string"},
        "limit": {"type": "integer", "default": 10},
    }

    def __init__(self, connection_pool):
        self._pool = connection_pool

    async def execute(self, sql: str = "", limit: int = 10) -> str:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"{sql} LIMIT {limit}")
            return "\n".join(str(dict(row)) for row in rows)
```
