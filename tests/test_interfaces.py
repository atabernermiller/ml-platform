"""Tests for ml_platform._interfaces -- Layer 1 protocols and ABCs."""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform._interfaces import LLMProvider, Tool
from ml_platform._types import Completion, CompletionUsage, Message


class _MockProvider:
    """Minimal class satisfying LLMProvider protocol."""

    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(content="mock", model=model or "test-model")


class _IncompleteProvider:
    """Missing the complete method."""

    pass


class _MockTool:
    """Minimal class satisfying Tool protocol."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"query": {"type": "string"}}

    async def execute(self, **kwargs: Any) -> str:
        return "result"


class _IncompleteTool:
    """Missing parameters_schema and execute."""

    name = "broken"
    description = "broken"


class TestLLMProviderProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        provider = _MockProvider()
        assert isinstance(provider, LLMProvider)

    def test_incomplete_does_not_satisfy(self) -> None:
        provider = _IncompleteProvider()
        assert not isinstance(provider, LLMProvider)


class TestToolProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        tool = _MockTool()
        assert isinstance(tool, Tool)

    def test_incomplete_does_not_satisfy(self) -> None:
        tool = _IncompleteTool()
        assert not isinstance(tool, Tool)

    def test_tool_requires_all_members(self) -> None:
        required = {"name", "description", "parameters_schema", "execute"}
        tool = _MockTool()
        for attr in required:
            assert hasattr(tool, attr)
