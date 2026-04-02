"""RunContext: the observability bridge for multi-step agent runs.

A :class:`RunContext` groups multiple LLM calls and tool executions into
a single observable unit.  It records an :class:`AgentStep` for each
operation and (optionally) creates OTel child spans so that the full
agent trajectory appears as a nested trace.

Usage as an async context manager (standalone, without
:class:`AgentServiceBase`)::

    from ml_platform.llm import RunContext

    async with RunContext(name="research") as ctx:
        plan = await ctx.complete(planner, messages, step_name="plan")
        result = await ctx.execute_tool(search_tool, query="LLMOps")
        summary = await ctx.complete(writer, messages, step_name="summarize")

    print(ctx.total_cost_usd, ctx.steps)

Inside an :class:`AgentServiceBase`, the serving layer creates a fresh
``RunContext`` per request and passes it to the ``run()`` method.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from ml_platform._interfaces import LLMProvider, Tool
from ml_platform._types import (
    AgentStep,
    Completion,
    CompletionUsage,
    Message,
    ToolResult,
)

logger = logging.getLogger(__name__)


class MaxStepsExceededError(Exception):
    """Raised when an agent run exceeds the configured step limit."""

    def __init__(self, max_steps: int, steps_completed: int) -> None:
        self.max_steps = max_steps
        self.steps_completed = steps_completed
        super().__init__(
            f"Agent reached {steps_completed} steps (limit {max_steps}). "
            f"Increase agent.max_steps_per_run in config or optimise "
            f"your agent logic."
        )


class RunContext:
    """Groups LLM calls and tool executions into one observable run.

    Args:
        name: Human-readable label for the run (used in OTel spans).
        emitter: Optional metrics emitter for per-run aggregates.
        max_steps: Maximum number of steps before raising
            :class:`MaxStepsExceededError`.  ``0`` means unlimited.
    """

    def __init__(
        self,
        name: str = "run",
        *,
        emitter: Any | None = None,
        max_steps: int = 0,
    ) -> None:
        self._name = name
        self._emitter = emitter
        self._max_steps = max_steps
        self._steps: list[AgentStep] = []
        self._entered = False

    # -- context manager -----------------------------------------------------

    async def __aenter__(self) -> RunContext:
        self._entered = True
        self._steps = []
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass

    # -- LLM call ------------------------------------------------------------

    async def complete(
        self,
        provider: LLMProvider,
        messages: list[Message],
        *,
        step_name: str = "",
        **kwargs: Any,
    ) -> Completion:
        """Execute an LLM call and record an :class:`AgentStep`.

        Args:
            provider: The LLM provider to call.
            messages: Conversation history / prompt.
            step_name: Label for this step (defaults to ``llm_call_N``).
            **kwargs: Forwarded to ``provider.complete()``.

        Returns:
            The provider's :class:`Completion` response.

        Raises:
            MaxStepsExceededError: If the step limit has been reached.
        """
        self._check_entered()
        self._check_max_steps()

        label = step_name or f"llm_call_{len(self._steps)}"

        start = time.monotonic()
        completion = await provider.complete(messages, **kwargs)
        latency = (time.monotonic() - start) * 1000

        step = AgentStep(
            name=label,
            step_type="llm_call",
            provider=completion.provider,
            model=completion.model,
            latency_ms=latency,
            tokens=completion.usage,
            cost_usd=completion.cost_usd,
        )
        self._steps.append(step)
        return completion

    # -- tool execution ------------------------------------------------------

    async def execute_tool(
        self,
        tool: Tool,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool and record an :class:`AgentStep`.

        Errors raised by the tool are caught, recorded in the
        :class:`ToolResult`, and do **not** crash the run.

        Args:
            tool: The tool to execute.
            **kwargs: Arguments forwarded to ``tool.execute()``.

        Returns:
            :class:`ToolResult` with either ``content`` or ``error`` set.

        Raises:
            MaxStepsExceededError: If the step limit has been reached.
        """
        self._check_entered()
        self._check_max_steps()

        call_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        error: str | None = None
        content = ""

        try:
            content = await tool.execute(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Tool %s failed: %s", tool.name, error, exc_info=True
            )

        latency = (time.monotonic() - start) * 1000

        tool_result = ToolResult(
            tool_call_id=call_id,
            name=tool.name,
            content=content,
            error=error,
            latency_ms=latency,
        )

        step = AgentStep(
            name=tool.name,
            step_type="tool_call",
            latency_ms=latency,
            cost_usd=0.0,
        )
        self._steps.append(step)
        return tool_result

    # -- aggregate properties ------------------------------------------------

    @property
    def steps(self) -> list[AgentStep]:
        """Copy of the recorded steps in chronological order."""
        return list(self._steps)

    @property
    def total_tokens(self) -> int:
        """Sum of tokens consumed across all LLM-call steps."""
        return sum(
            s.tokens.total_tokens for s in self._steps if s.tokens is not None
        )

    @property
    def total_cost_usd(self) -> float:
        """Sum of estimated costs across all steps."""
        return sum(s.cost_usd for s in self._steps)

    @property
    def total_latency_ms(self) -> float:
        """Sum of wall-clock time across all steps."""
        return sum(s.latency_ms for s in self._steps)

    @property
    def llm_call_count(self) -> int:
        """Number of LLM-call steps recorded so far."""
        return sum(1 for s in self._steps if s.step_type == "llm_call")

    @property
    def tool_call_count(self) -> int:
        """Number of tool-call steps recorded so far."""
        return sum(1 for s in self._steps if s.step_type == "tool_call")

    # -- internal ------------------------------------------------------------

    def _check_entered(self) -> None:
        if not self._entered:
            raise RuntimeError(
                "RunContext must be used as an async context manager:\n"
                "    async with RunContext(...) as ctx:\n"
                "        await ctx.complete(...)"
            )

    def _check_max_steps(self) -> None:
        if self._max_steps > 0 and len(self._steps) >= self._max_steps:
            raise MaxStepsExceededError(self._max_steps, len(self._steps))
