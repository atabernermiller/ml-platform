"""Core data types for the ml-platform library.

All types live in Layer 1 (zero optional dependencies beyond ``pydantic``).
They are imported by every layer above: feature modules, serving, and CLI.

LLM-related types (``Message``, ``Completion``) follow the conventions
established by the OpenAI API and the emerging OpenTelemetry GenAI
semantic conventions, but are provider-agnostic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM message types
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]
"""Allowed chat-message roles, following the OpenAI convention."""


class Message(BaseModel):
    """A single message in a conversation.

    Attributes:
        role: Who produced this message.
        content: Text body of the message.
        name: Optional speaker name (useful for multi-agent conversations).
        tool_call_id: If ``role="tool"``, the ID of the tool call this
            message responds to.
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class CompletionUsage(BaseModel):
    """Token-level usage statistics for a single LLM completion.

    Attributes:
        input_tokens: Number of tokens in the prompt / input messages.
        output_tokens: Number of tokens generated in the completion.
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class Completion(BaseModel):
    """Normalised result of an LLM completion call.

    Provider adapters translate provider-specific responses into this
    canonical form so that the instrumentation and serving layers
    never depend on a particular SDK.

    Attributes:
        content: The generated text.
        model: Model identifier as reported by the provider.
        provider: Name of the upstream provider (e.g. ``"openai"``).
        usage: Token counts, if available from the provider.
        stop_reason: Why generation stopped (e.g. ``"stop"``, ``"length"``).
        metadata: Arbitrary per-request data.  Numeric values in this dict
            are automatically emitted as per-request metrics by the serving
            layer.
        latency_ms: Wall-clock time of the provider call in milliseconds.
        cost_usd: Estimated cost in USD, populated by the provider adapter
            or by the instrumentation wrapper from token counts.
    """

    content: str
    model: str
    provider: str = ""
    usage: CompletionUsage | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Tool types (for agentic workflows)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A tool invocation requested by an LLM.

    Attributes:
        id: Unique identifier for correlating with the tool result.
        name: Name of the tool to invoke.
        arguments: Keyword arguments to pass to the tool's ``execute`` method.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of executing a tool.

    Attributes:
        tool_call_id: ID of the originating :class:`ToolCall`.
        name: Name of the tool that was executed.
        content: String output from the tool.
        error: Error message if the tool failed, ``None`` on success.
        latency_ms: Execution time in milliseconds.
    """

    tool_call_id: str
    name: str
    content: str
    error: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent run types (for multi-step agentic workflows)
# ---------------------------------------------------------------------------

StepType = Literal["llm_call", "tool_call"]
"""Discriminator for the kind of work an :class:`AgentStep` represents."""


class AgentStep(BaseModel):
    """One step in an agent run -- either an LLM call or a tool execution.

    Attributes:
        name: Human-readable label (e.g. ``"plan"``, ``"search_docs"``).
        step_type: Whether this step is an LLM call or a tool call.
        provider: Upstream provider name (populated for ``llm_call`` steps).
        model: Model identifier (populated for ``llm_call`` steps).
        latency_ms: Wall-clock time for this step in milliseconds.
        tokens: Token usage (populated for ``llm_call`` steps).
        cost_usd: Estimated cost in USD for this step.
    """

    name: str
    step_type: StepType
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    tokens: CompletionUsage | None = None
    cost_usd: float = 0.0


class AgentResult(BaseModel):
    """Result of a full agent turn, with per-step observability.

    An agent turn may involve multiple LLM calls and tool executions.
    ``AgentResult`` captures the final output alongside a detailed
    breakdown of every step the agent took.

    Attributes:
        content: The final user-facing response text.
        steps: Ordered list of steps the agent executed.
        messages: Full message trajectory, including tool call/result
            messages for debugging and replay.
        metadata: Arbitrary per-run data.  Numeric values are automatically
            emitted as metrics.
    """

    content: str
    steps: list[AgentStep] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Sum of tokens consumed across all LLM-call steps."""
        return sum(
            s.tokens.total_tokens for s in self.steps if s.tokens is not None
        )

    @property
    def total_cost_usd(self) -> float:
        """Sum of estimated costs across all steps."""
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        """Sum of wall-clock time across all steps."""
        return sum(s.latency_ms for s in self.steps)

    @property
    def llm_call_count(self) -> int:
        """Number of LLM-call steps in this run."""
        return sum(1 for s in self.steps if s.step_type == "llm_call")

    @property
    def tool_call_count(self) -> int:
        """Number of tool-call steps in this run."""
        return sum(1 for s in self.steps if s.step_type == "tool_call")
