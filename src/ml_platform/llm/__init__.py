"""LLM instrumentation, provider protocol, and agentic run context.

This package provides the core LLM infrastructure:

- :class:`~ml_platform.llm.run_context.RunContext` -- groups multiple LLM
  calls and tool executions into a single observable run.

Re-exports from :mod:`ml_platform._types` and :mod:`ml_platform._interfaces`
are provided for convenience.
"""

from ml_platform._interfaces import LLMProvider, Tool
from ml_platform._types import (
    AgentResult,
    AgentStep,
    Completion,
    CompletionUsage,
    Message,
    ToolCall,
    ToolResult,
)
from ml_platform.llm.run_context import RunContext

__all__ = [
    "AgentResult",
    "AgentStep",
    "Completion",
    "CompletionUsage",
    "LLMProvider",
    "Message",
    "RunContext",
    "Tool",
    "ToolCall",
    "ToolResult",
]
