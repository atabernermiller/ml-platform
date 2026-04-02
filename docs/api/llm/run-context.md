# RunContext

Groups multiple LLM calls and tool executions into one observable run with aggregated metrics.

## Overview

`RunContext` is the observability bridge for multi-step workflows. It wraps each LLM call and tool execution with timing, cost tracking, and step recording. The resulting trace shows the full agent trajectory as a parent span with child spans for each step.

## Class signature

```python
from ml_platform.llm.run_context import RunContext

ctx = RunContext(
    name: str = "run",      # human-readable label for traces
    emitter=None,            # optional MetricsEmitter for per-run aggregates
    max_steps: int = 0,      # 0 = unlimited
)
```

## Usage as async context manager

```python
async with RunContext(name="research") as ctx:
    plan = await ctx.complete(openai_provider, messages, step_name="plan")
    result = await ctx.execute_tool(search_tool, query="LLMOps")
    summary = await ctx.complete(anthropic_provider, messages, step_name="summarize")

print(f"Total cost: ${ctx.total_cost_usd:.4f}")
print(f"Steps: {ctx.llm_call_count} LLM calls, {ctx.tool_call_count} tool calls")
```

`RunContext` **must** be used as an async context manager. Calling `complete()` or `execute_tool()` outside a `with` block raises `RuntimeError`.

## Methods

### complete()

```python
async def complete(
    provider: LLMProvider,
    messages: list[Message],
    *,
    step_name: str = "",
    **kwargs,
) -> Completion
```

Executes an LLM call and records an `AgentStep` with `step_type="llm_call"`.

| Parameter | Description |
|---|---|
| `provider` | Any object satisfying `LLMProvider` protocol |
| `messages` | Conversation history |
| `step_name` | Label for the step (auto-generated if empty) |
| `**kwargs` | Forwarded to `provider.complete()` |

Returns the provider's `Completion` response.

### execute_tool()

```python
async def execute_tool(
    tool: Tool,
    **kwargs,
) -> ToolResult
```

Executes a tool and records an `AgentStep` with `step_type="tool_call"`.

Errors raised by the tool are **caught**, recorded in `ToolResult.error`, and do not crash the run.

| Parameter | Description |
|---|---|
| `tool` | Any object satisfying `Tool` protocol |
| `**kwargs` | Forwarded to `tool.execute()` |

Returns a `ToolResult` with either `content` or `error` populated.

## Aggregate properties

| Property | Type | Description |
|---|---|---|
| `steps` | `list[AgentStep]` | Copy of recorded steps in chronological order |
| `total_tokens` | `int` | Sum across all LLM-call steps |
| `total_cost_usd` | `float` | Sum across all steps |
| `total_latency_ms` | `float` | Sum of wall-clock time |
| `llm_call_count` | `int` | Number of LLM-call steps |
| `tool_call_count` | `int` | Number of tool-call steps |

## MaxStepsExceededError

Raised when the step limit is reached:

```python
from ml_platform.llm.run_context import MaxStepsExceededError

async with RunContext(name="bounded", max_steps=5) as ctx:
    for i in range(10):
        try:
            await ctx.complete(provider, messages)
        except MaxStepsExceededError as e:
            print(f"Stopped at step {e.steps_completed}/{e.max_steps}")
            break
```

## Standalone usage (without AgentServiceBase)

`RunContext` works independently of the serving framework:

```python
from ml_platform.llm import RunContext

async def my_workflow():
    async with RunContext(name="custom-pipeline") as ctx:
        analysis = await ctx.complete(openai, messages, step_name="analyze")
        search = await ctx.execute_tool(search_tool, query=analysis.content)
        report = await ctx.complete(anthropic, messages, step_name="report")

    return {
        "report": report.content,
        "cost": ctx.total_cost_usd,
        "steps": len(ctx.steps),
    }
```

## Trace structure

A single `RunContext` produces this observability structure:

```
agent-turn (parent)
├── plan (llm_call, gpt-4o, 800ms, 1200 tokens)
├── search_docs (tool_call, 300ms)
├── calculate (tool_call, 50ms)
└── synthesize (llm_call, claude-3.5-sonnet, 1100ms, 2000 tokens)
```
