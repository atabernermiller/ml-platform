"""Integration tests for the full agent request cycle.

These test the end-to-end flow: create_agent_app -> POST /run -> mock providers
and tools -> verify response and per-run metrics.
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


# ---------------------------------------------------------------------------
# Mock providers with distinct identity
# ---------------------------------------------------------------------------


class OpenAIMockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="openai says hi",
            model="gpt-4o",
            provider="openai",
            usage=CompletionUsage(input_tokens=100, output_tokens=50),
            cost_usd=0.003,
        )


class AnthropicMockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="claude says hi",
            model="claude-3.5-sonnet",
            provider="anthropic",
            usage=CompletionUsage(input_tokens=200, output_tokens=100),
            cost_usd=0.005,
        )


class SearchTool:
    name = "search"
    description = "Search"
    parameters_schema: dict[str, Any] = {"query": {"type": "string"}}

    async def execute(self, **kwargs: Any) -> str:
        return f"results for: {kwargs.get('query', '')}"


class FailingSearchTool:
    name = "failing_search"
    description = "Always fails"
    parameters_schema: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> str:
        raise ConnectionError("network down")


# ---------------------------------------------------------------------------
# Multi-provider agent
# ---------------------------------------------------------------------------


class ResearchAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        plan = await run_context.complete(
            self.providers["planner"], messages, step_name="plan"
        )
        search = await run_context.execute_tool(
            self.tools["search"], query=plan.content
        )
        response = await run_context.complete(
            self.providers["writer"],
            messages + [Message(role="tool", content=search.content, tool_call_id=search.tool_call_id)],
            step_name="synthesize",
        )
        return AgentResult(
            content=response.content,
            steps=run_context.steps,
            messages=messages,
        )


class PartialFailureAgent(AgentServiceBase):
    """Uses a failing tool but still produces a result."""

    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        result = await run_context.execute_tool(self.tools["failing_search"])
        c = await run_context.complete(
            self.providers["main"], messages, step_name="fallback"
        )
        return AgentResult(
            content=c.content,
            steps=run_context.steps,
            messages=messages,
            metadata={"used_fallback": True},
        )


async def _managed_client(app: Any) -> AsyncClient:
    """Create an httpx client with lifespan management."""
    manager = LifespanManager(app)
    await manager.__aenter__()
    client = AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    )
    client._lifespan_manager = manager  # type: ignore[attr-defined]
    return client


async def _close_client(client: AsyncClient) -> None:
    await client.aclose()
    manager = getattr(client, "_lifespan_manager", None)
    if manager:
        await manager.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullAgentCycle:
    @pytest.mark.asyncio
    async def test_multi_provider_run(self) -> None:
        config = ServiceConfig(
            service_name="research", agent=AgentConfig()
        )
        app = create_agent_app(
            ResearchAgent,
            config,
            providers={
                "planner": OpenAIMockProvider(),
                "writer": AnthropicMockProvider(),
            },
            tools=[SearchTool()],
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "research LLMOps"}]},
            )

            assert resp.status_code == 200
            body = resp.json()

            assert len(body["steps"]) == 3
            assert body["steps"][0]["provider"] == "openai"
            assert body["steps"][1]["step_type"] == "tool_call"
            assert body["steps"][2]["provider"] == "anthropic"

            assert body["llm_calls"] == 2
            assert body["tool_calls"] == 1
            assert body["total_tokens"] == 450
            assert abs(body["total_cost_usd"] - 0.008) < 1e-6
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_tool_failure_mid_run(self) -> None:
        config = ServiceConfig(
            service_name="partial", agent=AgentConfig()
        )
        app = create_agent_app(
            PartialFailureAgent,
            config,
            providers={"main": OpenAIMockProvider()},
            tools=[FailingSearchTool()],
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "try"}]},
            )

            assert resp.status_code == 200
            body = resp.json()
            assert len(body["steps"]) == 2
            assert body["steps"][0]["step_type"] == "tool_call"
            assert body["steps"][1]["step_type"] == "llm_call"
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_metrics_aggregate_after_runs(self) -> None:
        config = ServiceConfig(
            service_name="metrics-test", agent=AgentConfig()
        )
        app = create_agent_app(
            ResearchAgent,
            config,
            providers={
                "planner": OpenAIMockProvider(),
                "writer": AnthropicMockProvider(),
            },
            tools=[SearchTool()],
        )
        client = await _managed_client(app)
        try:
            for _ in range(3):
                await client.post(
                    "/run",
                    json={"messages": [{"role": "user", "content": "q"}]},
                )
            resp = await client.get("/metrics")

            assert resp.status_code == 200
            body = resp.json()
            assert body["total_runs"] == 3.0
            assert body["total_steps"] == 9.0
            assert body["avg_steps_per_run"] == 3.0
        finally:
            await _close_client(client)
