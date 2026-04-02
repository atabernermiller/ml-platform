"""Tests for ml_platform.serving.runtime -- BaseRuntime and AgentRuntime."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ml_platform._types import AgentResult, Completion, CompletionUsage, Message
from ml_platform.config import AgentConfig, ServiceConfig
from ml_platform.llm.run_context import RunContext
from ml_platform.serving.agent import AgentServiceBase
from ml_platform.serving.runtime import AgentRuntime, BaseRuntime


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="response",
            model="mock",
            provider="mock",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )


class MockTool:
    name = "mock_tool"
    description = "Mock"
    parameters_schema: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class SimpleAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        c = await run_context.complete(self.providers["main"], messages)
        return AgentResult(content=c.content, steps=run_context.steps, messages=messages)


# ---------------------------------------------------------------------------
# BaseRuntime tests
# ---------------------------------------------------------------------------


class TestBaseRuntime:
    @pytest.mark.asyncio
    async def test_not_ready_before_startup(self) -> None:
        config = ServiceConfig(service_name="test")
        rt = BaseRuntime(config)
        assert rt.is_ready is False

    @pytest.mark.asyncio
    async def test_ready_after_startup(self) -> None:
        config = ServiceConfig(service_name="test")
        rt = BaseRuntime(config)
        await rt.startup()
        assert rt.is_ready is True
        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_not_ready_after_shutdown(self) -> None:
        config = ServiceConfig(service_name="test")
        rt = BaseRuntime(config)
        await rt.startup()
        await rt.shutdown()
        assert rt.is_ready is False

    @pytest.mark.asyncio
    async def test_emitter_created_on_startup(self) -> None:
        config = ServiceConfig(service_name="test")
        rt = BaseRuntime(config)
        assert rt.emitter is None
        await rt.startup()
        assert rt.emitter is not None
        await rt.shutdown()


# ---------------------------------------------------------------------------
# AgentRuntime tests
# ---------------------------------------------------------------------------


class TestAgentRuntime:
    @pytest.mark.asyncio
    async def test_creates_service_on_startup(self) -> None:
        config = ServiceConfig(
            service_name="test",
            agent=AgentConfig(),
        )
        rt = AgentRuntime(
            SimpleAgent,
            config,
            providers={"main": MockProvider()},
        )
        await rt.startup()
        assert rt.service is not None
        assert rt.service.providers == {"main": rt._providers["main"]}
        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_run_returns_agent_result(self) -> None:
        config = ServiceConfig(
            service_name="test",
            agent=AgentConfig(),
        )
        rt = AgentRuntime(
            SimpleAgent,
            config,
            providers={"main": MockProvider()},
        )
        await rt.startup()
        messages = [Message(role="user", content="hi")]
        result = await rt.run(messages)
        assert isinstance(result, AgentResult)
        assert result.content == "response"
        assert len(result.steps) == 1
        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_aggregates_metrics(self) -> None:
        config = ServiceConfig(
            service_name="test",
            agent=AgentConfig(),
        )
        rt = AgentRuntime(
            SimpleAgent,
            config,
            providers={"main": MockProvider()},
        )
        await rt.startup()
        await rt.run([Message(role="user", content="a")])
        await rt.run([Message(role="user", content="b")])

        snapshot = rt.metrics_snapshot()
        assert snapshot["total_runs"] == 2.0
        assert snapshot["total_llm_calls"] == 2.0
        assert snapshot["avg_steps_per_run"] == 1.0
        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_registers_tools(self) -> None:
        config = ServiceConfig(
            service_name="test",
            agent=AgentConfig(),
        )
        rt = AgentRuntime(
            SimpleAgent,
            config,
            providers={"main": MockProvider()},
            tools=[MockTool()],
        )
        await rt.startup()
        assert "mock_tool" in rt.service.tools
        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_service_not_started_raises(self) -> None:
        config = ServiceConfig(service_name="test", agent=AgentConfig())
        rt = AgentRuntime(SimpleAgent, config, providers={"main": MockProvider()})
        with pytest.raises(RuntimeError, match="not been started"):
            _ = rt.service
