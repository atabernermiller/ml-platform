"""Tests for ml_platform.llm.run_context -- RunContext Layer 2."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ml_platform._types import Completion, CompletionUsage, Message
from ml_platform.llm.run_context import MaxStepsExceededError, RunContext


# ---------------------------------------------------------------------------
# Mock provider and tool
# ---------------------------------------------------------------------------


class MockProvider:
    """LLM provider that returns canned completions."""

    def __init__(
        self,
        content: str = "mock response",
        model: str = "test-model",
        provider: str = "test-provider",
        input_tokens: int = 10,
        output_tokens: int = 5,
        cost: float = 0.001,
    ) -> None:
        self._content = content
        self._model = model
        self._provider = provider
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost = cost

    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content=self._content,
            model=model or self._model,
            provider=self._provider,
            usage=CompletionUsage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            ),
            cost_usd=self._cost,
        )


class MockTool:
    name = "mock_tool"
    description = "A mock tool"
    parameters_schema: dict[str, Any] = {"input": {"type": "string"}}

    async def execute(self, **kwargs: Any) -> str:
        return f"tool result: {kwargs}"


class FailingTool:
    name = "failing_tool"
    description = "Always fails"
    parameters_schema: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> str:
        raise ValueError("tool broke")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunContextComplete:
    @pytest.mark.asyncio
    async def test_delegates_to_provider_and_records_step(self) -> None:
        provider = MockProvider()
        msgs = [Message(role="user", content="hi")]
        async with RunContext(name="test") as ctx:
            result = await ctx.complete(provider, msgs, step_name="plan")

        assert result.content == "mock response"
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == "plan"
        assert ctx.steps[0].step_type == "llm_call"

    @pytest.mark.asyncio
    async def test_records_provider_and_model(self) -> None:
        provider = MockProvider(provider="openai", model="gpt-4o")
        async with RunContext(name="test") as ctx:
            await ctx.complete(provider, [])

        assert ctx.steps[0].provider == "openai"
        assert ctx.steps[0].model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_records_tokens_and_cost(self) -> None:
        provider = MockProvider(input_tokens=100, output_tokens=50, cost=0.005)
        async with RunContext(name="test") as ctx:
            await ctx.complete(provider, [])

        assert ctx.steps[0].tokens is not None
        assert ctx.steps[0].tokens.total_tokens == 150
        assert ctx.steps[0].cost_usd == 0.005

    @pytest.mark.asyncio
    async def test_auto_labels_steps(self) -> None:
        provider = MockProvider()
        async with RunContext(name="test") as ctx:
            await ctx.complete(provider, [])
            await ctx.complete(provider, [])

        assert ctx.steps[0].name == "llm_call_0"
        assert ctx.steps[1].name == "llm_call_1"

    @pytest.mark.asyncio
    async def test_latency_recorded(self) -> None:
        provider = MockProvider()
        async with RunContext(name="test") as ctx:
            await ctx.complete(provider, [])

        assert ctx.steps[0].latency_ms > 0


class TestRunContextExecuteTool:
    @pytest.mark.asyncio
    async def test_delegates_to_tool_and_records_step(self) -> None:
        tool = MockTool()
        async with RunContext(name="test") as ctx:
            result = await ctx.execute_tool(tool, input="hello")

        assert "hello" in result.content
        assert result.error is None
        assert len(ctx.steps) == 1
        assert ctx.steps[0].step_type == "tool_call"
        assert ctx.steps[0].name == "mock_tool"

    @pytest.mark.asyncio
    async def test_handles_tool_error(self) -> None:
        tool = FailingTool()
        async with RunContext(name="test") as ctx:
            result = await ctx.execute_tool(tool)

        assert result.error is not None
        assert "tool broke" in result.error
        assert len(ctx.steps) == 1

    @pytest.mark.asyncio
    async def test_tool_latency_recorded(self) -> None:
        tool = MockTool()
        async with RunContext(name="test") as ctx:
            await ctx.execute_tool(tool)

        assert ctx.steps[0].latency_ms >= 0


class TestRunContextAggregates:
    @pytest.mark.asyncio
    async def test_total_tokens_sums_llm_steps(self) -> None:
        p = MockProvider(input_tokens=100, output_tokens=50)
        async with RunContext(name="test") as ctx:
            await ctx.complete(p, [])
            await ctx.complete(p, [])

        assert ctx.total_tokens == 300  # 150 * 2

    @pytest.mark.asyncio
    async def test_total_cost_sums_all_steps(self) -> None:
        p = MockProvider(cost=0.01)
        async with RunContext(name="test") as ctx:
            await ctx.complete(p, [])
            await ctx.execute_tool(MockTool())
            await ctx.complete(p, [])

        assert abs(ctx.total_cost_usd - 0.02) < 1e-9

    @pytest.mark.asyncio
    async def test_call_counts(self) -> None:
        p = MockProvider()
        t = MockTool()
        async with RunContext(name="test") as ctx:
            await ctx.complete(p, [])
            await ctx.execute_tool(t)
            await ctx.execute_tool(t)
            await ctx.complete(p, [])

        assert ctx.llm_call_count == 2
        assert ctx.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_zero_steps(self) -> None:
        async with RunContext(name="test") as ctx:
            pass

        assert ctx.total_tokens == 0
        assert ctx.total_cost_usd == 0.0
        assert ctx.llm_call_count == 0
        assert ctx.tool_call_count == 0


class TestRunContextMultiProvider:
    @pytest.mark.asyncio
    async def test_multiple_providers_recorded(self) -> None:
        p1 = MockProvider(provider="openai", model="gpt-4o")
        p2 = MockProvider(provider="anthropic", model="claude-3")
        async with RunContext(name="test") as ctx:
            await ctx.complete(p1, [], step_name="plan")
            await ctx.complete(p2, [], step_name="synthesize")

        assert ctx.steps[0].provider == "openai"
        assert ctx.steps[1].provider == "anthropic"


class TestRunContextStepOrdering:
    @pytest.mark.asyncio
    async def test_steps_in_chronological_order(self) -> None:
        p = MockProvider()
        t = MockTool()
        async with RunContext(name="test") as ctx:
            await ctx.complete(p, [], step_name="step1")
            await ctx.execute_tool(t)
            await ctx.complete(p, [], step_name="step3")

        names = [s.name for s in ctx.steps]
        assert names == ["step1", "mock_tool", "step3"]


class TestRunContextMaxSteps:
    @pytest.mark.asyncio
    async def test_raises_on_max_steps(self) -> None:
        p = MockProvider()
        async with RunContext(name="test", max_steps=2) as ctx:
            await ctx.complete(p, [])
            await ctx.complete(p, [])
            with pytest.raises(MaxStepsExceededError) as exc_info:
                await ctx.complete(p, [])
            assert exc_info.value.max_steps == 2
            assert exc_info.value.steps_completed == 2

    @pytest.mark.asyncio
    async def test_zero_max_means_unlimited(self) -> None:
        p = MockProvider()
        async with RunContext(name="test", max_steps=0) as ctx:
            for _ in range(50):
                await ctx.complete(p, [])
        assert len(ctx.steps) == 50


class TestRunContextStandalone:
    @pytest.mark.asyncio
    async def test_works_without_agent_service(self) -> None:
        p = MockProvider(cost=0.01)
        t = MockTool()
        async with RunContext(name="standalone") as ctx:
            await ctx.complete(p, [Message(role="user", content="hi")])
            await ctx.execute_tool(t, input="test")

        assert ctx.llm_call_count == 1
        assert ctx.tool_call_count == 1
        assert ctx.total_cost_usd == 0.01


class TestRunContextMustBeEntered:
    @pytest.mark.asyncio
    async def test_raises_outside_context_manager(self) -> None:
        ctx = RunContext(name="test")
        p = MockProvider()
        with pytest.raises(RuntimeError, match="async context manager"):
            await ctx.complete(p, [])
