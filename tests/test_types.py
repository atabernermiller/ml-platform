"""Tests for ml_platform._types -- Layer 1 core data types."""

from __future__ import annotations

import pytest

from ml_platform._types import (
    AgentResult,
    AgentStep,
    Completion,
    CompletionUsage,
    Message,
    ToolCall,
    ToolResult,
)


class TestMessage:
    def test_user_message(self) -> None:
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.name is None
        assert m.tool_call_id is None

    def test_tool_message(self) -> None:
        m = Message(role="tool", content="result", tool_call_id="tc_1")
        assert m.tool_call_id == "tc_1"

    def test_serialization_roundtrip(self) -> None:
        m = Message(role="assistant", content="hi", name="bot")
        d = m.model_dump()
        m2 = Message.model_validate(d)
        assert m == m2


class TestCompletionUsage:
    def test_total_tokens(self) -> None:
        u = CompletionUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_zero_tokens(self) -> None:
        u = CompletionUsage(input_tokens=0, output_tokens=0)
        assert u.total_tokens == 0


class TestCompletion:
    def test_minimal(self) -> None:
        c = Completion(content="hello", model="gpt-4o")
        assert c.provider == ""
        assert c.usage is None
        assert c.cost_usd == 0.0

    def test_full(self) -> None:
        c = Completion(
            content="hi",
            model="claude-3",
            provider="anthropic",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
            latency_ms=200.0,
        )
        assert c.usage is not None
        assert c.usage.total_tokens == 15


class TestToolCall:
    def test_basic(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments={"query": "LLM"})
        assert tc.name == "search"
        assert tc.arguments["query"] == "LLM"

    def test_empty_arguments(self) -> None:
        tc = ToolCall(id="tc_2", name="noop")
        assert tc.arguments == {}


class TestToolResult:
    def test_success(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", name="search", content="found 3")
        assert tr.error is None

    def test_error(self) -> None:
        tr = ToolResult(
            tool_call_id="tc_1",
            name="search",
            content="",
            error="TimeoutError: 30s",
        )
        assert tr.error is not None


class TestAgentStep:
    def test_llm_call(self) -> None:
        s = AgentStep(
            name="plan",
            step_type="llm_call",
            provider="openai",
            model="gpt-4o",
            latency_ms=500.0,
            tokens=CompletionUsage(input_tokens=100, output_tokens=50),
            cost_usd=0.002,
        )
        assert s.step_type == "llm_call"
        assert s.tokens is not None
        assert s.tokens.total_tokens == 150

    def test_tool_call(self) -> None:
        s = AgentStep(name="search", step_type="tool_call", latency_ms=200.0)
        assert s.provider is None
        assert s.tokens is None

    def test_invalid_step_type(self) -> None:
        with pytest.raises(Exception):
            AgentStep(name="bad", step_type="invalid")  # type: ignore[arg-type]


class TestAgentResult:
    def _make_result(self) -> AgentResult:
        return AgentResult(
            content="final answer",
            steps=[
                AgentStep(
                    name="plan",
                    step_type="llm_call",
                    tokens=CompletionUsage(input_tokens=100, output_tokens=50),
                    cost_usd=0.002,
                    latency_ms=500.0,
                ),
                AgentStep(
                    name="search",
                    step_type="tool_call",
                    cost_usd=0.0,
                    latency_ms=200.0,
                ),
                AgentStep(
                    name="synthesize",
                    step_type="llm_call",
                    tokens=CompletionUsage(input_tokens=200, output_tokens=100),
                    cost_usd=0.005,
                    latency_ms=800.0,
                ),
            ],
            messages=[Message(role="user", content="hello")],
        )

    def test_total_tokens(self) -> None:
        r = self._make_result()
        assert r.total_tokens == 450  # (100+50) + (200+100)

    def test_total_cost_usd(self) -> None:
        r = self._make_result()
        assert abs(r.total_cost_usd - 0.007) < 1e-9

    def test_total_latency_ms(self) -> None:
        r = self._make_result()
        assert r.total_latency_ms == 1500.0

    def test_llm_call_count(self) -> None:
        r = self._make_result()
        assert r.llm_call_count == 2

    def test_tool_call_count(self) -> None:
        r = self._make_result()
        assert r.tool_call_count == 1

    def test_empty_steps(self) -> None:
        r = AgentResult(content="empty")
        assert r.total_tokens == 0
        assert r.total_cost_usd == 0.0
        assert r.llm_call_count == 0
        assert r.tool_call_count == 0

    def test_serialization(self) -> None:
        r = self._make_result()
        d = r.model_dump()
        r2 = AgentResult.model_validate(d)
        assert r2.total_tokens == r.total_tokens
        assert len(r2.steps) == 3
