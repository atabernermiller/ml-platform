"""Stress test: tools do not leak state between agent runs.

A stateful counter tool is shared across runs.  Each run should
get the same tool instance (tools are registered once), but
RunContext-level step tracking must be isolated per run.
"""

from __future__ import annotations

from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from ml_platform._types import (
    AgentResult,
    Completion,
    CompletionUsage,
    Message,
)
from ml_platform.config import AgentConfig, ServiceConfig
from ml_platform.llm.run_context import RunContext
from ml_platform.serving.agent import AgentServiceBase, create_agent_app


class MockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="ok",
            model="mock",
            provider="mock",
            usage=CompletionUsage(input_tokens=1, output_tokens=1),
        )


class CounterTool:
    """Tool that increments a counter each time it's called."""

    name = "counter"
    description = "Counts calls"
    parameters_schema: dict[str, Any] = {}

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, **kwargs: Any) -> str:
        self.call_count += 1
        return str(self.call_count)


class CounterAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        result = await run_context.execute_tool(self.tools["counter"])
        return AgentResult(
            content=result.content,
            steps=run_context.steps,
            messages=messages,
            metadata={"counter_value": int(result.content)},
        )


@pytest.mark.asyncio
async def test_run_context_isolation_across_runs() -> None:
    """Each run gets its own step list even though the tool is shared."""
    config = ServiceConfig(
        service_name="isolation-test",
        agent=AgentConfig(),
    )
    counter_tool = CounterTool()
    app = create_agent_app(
        CounterAgent,
        config,
        providers={"main": MockProvider()},
        tools=[counter_tool],
    )

    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            results = []
            for _ in range(10):
                resp = await client.post(
                    "/run",
                    json={"messages": [{"role": "user", "content": "count"}]},
                )
                assert resp.status_code == 200
                results.append(resp.json())

    for body in results:
        assert len(body["steps"]) == 1, "Step list leaked between runs"

    assert counter_tool.call_count == 10
