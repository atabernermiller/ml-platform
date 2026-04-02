"""Stress test: 50 concurrent agent runs with 3 steps each.

That's 150 LLM calls + tool executions running concurrently.
All must succeed with per-run metrics isolated (no cross-contamination).
"""

from __future__ import annotations

import asyncio
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


class SlowMockProvider:
    """Provider with a small async delay to simulate realistic latency."""

    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        await asyncio.sleep(0.001)
        return Completion(
            content="response",
            model="mock",
            provider="mock",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )


class SlowMockTool:
    name = "search"
    description = "Mock search"
    parameters_schema: dict[str, Any] = {"q": {"type": "string"}}

    async def execute(self, **kwargs: Any) -> str:
        await asyncio.sleep(0.001)
        return "found"


class ThreeStepAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        await run_context.complete(self.providers["main"], messages, step_name="plan")
        await run_context.execute_tool(self.tools["search"], q="test")
        c = await run_context.complete(
            self.providers["main"], messages, step_name="synthesize"
        )
        return AgentResult(
            content=c.content,
            steps=run_context.steps,
            messages=messages,
        )


@pytest.mark.asyncio
async def test_50_concurrent_agent_runs() -> None:
    config = ServiceConfig(
        service_name="stress-concurrent",
        agent=AgentConfig(max_steps_per_run=50),
    )
    app = create_agent_app(
        ThreeStepAgent,
        config,
        providers={"main": SlowMockProvider()},
        tools=[SlowMockTool()],
    )

    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:

            async def _run(idx: int) -> dict[str, Any]:
                resp = await client.post(
                    "/run",
                    json={"messages": [{"role": "user", "content": f"task-{idx}"}]},
                )
                assert resp.status_code == 200, f"Run {idx} failed: {resp.text}"
                return resp.json()

            results = await asyncio.gather(*[_run(i) for i in range(50)])

    assert len(results) == 50
    for body in results:
        assert len(body["steps"]) == 3
        assert body["llm_calls"] == 2
        assert body["tool_calls"] == 1
        assert body["steps"][0]["step_type"] == "llm_call"
        assert body["steps"][1]["step_type"] == "tool_call"
        assert body["steps"][2]["step_type"] == "llm_call"
