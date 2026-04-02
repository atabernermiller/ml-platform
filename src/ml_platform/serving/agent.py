"""Agent service base class and FastAPI application factory.

Use this module for services that make **multiple LLM calls and/or tool
executions per user request** -- agentic workflows, multi-step reasoning,
plan-execute-synthesize patterns, etc.

For services that make a single LLM call per request, use
:class:`~ml_platform.serving.llm.LLMServiceBase` instead.
For services with online learning and feedback loops, use
:class:`~ml_platform.serving.stateful.StatefulServiceBase`.

Example::

    from ml_platform.serving.agent import AgentServiceBase, create_agent_app
    from ml_platform._types import AgentResult, Message
    from ml_platform.config import ServiceConfig, AgentConfig

    class MyResearchAgent(AgentServiceBase):
        async def run(self, messages, *, run_context, **kwargs):
            plan = await run_context.complete(
                self.providers["planner"], messages, step_name="plan",
            )
            result = await run_context.execute_tool(
                self.tools["search"], query=plan.content,
            )
            response = await run_context.complete(
                self.providers["writer"],
                messages + [Message(role="tool", content=result.content,
                                    tool_call_id=result.tool_call_id)],
                step_name="synthesize",
            )
            return AgentResult(
                content=response.content,
                steps=run_context.steps,
                messages=messages,
            )

    config = ServiceConfig(
        service_name="research-agent",
        agent=AgentConfig(max_steps_per_run=10),
    )
    app = create_agent_app(
        MyResearchAgent, config,
        providers={"planner": openai, "writer": anthropic},
        tools=[SearchDocsTool()],
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Type

from fastapi import HTTPException

from ml_platform._interfaces import LLMProvider, Tool
from ml_platform._types import AgentResult, Message
from ml_platform.config import ServiceConfig
from ml_platform.llm.run_context import MaxStepsExceededError, RunContext


class AgentServiceBase(ABC):
    """Base class for agent services with multi-step LLM reasoning and tool use.

    Subclass this when your app makes multiple LLM calls per user request
    (e.g., planning + tool use + synthesis).  For single-call apps, use
    ``LLMServiceBase`` instead.

    The framework sets :attr:`providers` and :attr:`tools` before the
    first call to :meth:`run`.  Use ``run_context.complete()`` and
    ``run_context.execute_tool()`` inside :meth:`run` for automatic
    observability.

    Example::

        class MyAgent(AgentServiceBase):
            async def run(self, messages, *, run_context, **kwargs):
                plan = await run_context.complete(
                    self.providers["planner"], messages,
                )
                result = await run_context.execute_tool(
                    self.tools["search"], query="...",
                )
                return AgentResult(
                    content=result.content,
                    steps=run_context.steps,
                    messages=messages,
                )
    """

    providers: dict[str, LLMProvider]
    """Named LLM providers set by the framework before serving begins."""

    tools: dict[str, Tool]
    """Named tools set by the framework before serving begins."""

    @abstractmethod
    async def run(
        self,
        messages: list[Message],
        *,
        run_context: RunContext,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a full agent turn.

        Use ``run_context.complete()`` for LLM calls and
        ``run_context.execute_tool()`` for tool executions so that
        every step is automatically recorded and traced.

        Args:
            messages: Input conversation messages.
            run_context: Instrumented context for this run.  Each call
                to ``complete()`` or ``execute_tool()`` records a step.
            **kwargs: Reserved for future use.

        Returns:
            :class:`AgentResult` with the final response and per-step
            breakdown.
        """
        ...

    def metrics_snapshot(self) -> dict[str, float]:
        """Optional: return periodic aggregate metrics.

        Override to expose custom business metrics (cache hit rates,
        active sessions, etc.) that don't map to a single request.
        """
        return {}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Inbound request for ``POST /run``."""

    messages: list[dict[str, Any]] = Field(
        ..., description="List of message dicts with 'role' and 'content'."
    )


class AgentRunResponse(BaseModel):
    """Outbound response from ``POST /run``."""

    content: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_agent_app(
    service_cls: Type[AgentServiceBase],
    config: ServiceConfig,
    *,
    providers: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
    service_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a production-ready FastAPI app wrapping an agent service.

    The returned app includes:

    - ``POST /run`` -- execute an agent turn (multiple LLM calls + tools).
    - ``GET /health`` -- readiness probe.
    - ``GET /metrics`` -- agent-specific aggregate metrics.
    - ``GET /dashboard`` -- self-contained HTML dashboard with agent panels.

    Args:
        service_cls: Concrete :class:`AgentServiceBase` subclass.
        config: Platform-wide configuration.
        providers: Named LLM providers (e.g. ``{"planner": openai_provider}``).
        tools: List of :class:`Tool` implementations.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.

    Returns:
        FastAPI application suitable for ``uvicorn.run(app)``.

    Raises:
        TypeError: If *service_cls* is not an :class:`AgentServiceBase`
            subclass (with a helpful hint about the correct factory).
    """
    from ml_platform.serving._app_builder import build_base_app
    from ml_platform.serving.runtime import AgentRuntime
    from ml_platform.serving.stateful import StatefulServiceBase

    if isinstance(service_cls, type) and issubclass(service_cls, StatefulServiceBase):
        raise TypeError(
            f"{service_cls.__name__} is a StatefulServiceBase subclass. "
            f"Use create_stateful_app() instead of create_agent_app().\n"
            f"Hint: create_stateful_app({service_cls.__name__}, config)"
        )

    if not (isinstance(service_cls, type) and issubclass(service_cls, AgentServiceBase)):
        raise TypeError(
            f"{getattr(service_cls, '__name__', service_cls)} is not an "
            f"AgentServiceBase subclass. Use create_agent_app() with a "
            f"class that inherits from AgentServiceBase."
        )

    runtime = AgentRuntime(
        service_cls,
        config,
        providers=providers,
        tools=tools,
        service_kwargs=service_kwargs,
    )

    app = build_base_app(
        config,
        readiness_check=lambda: runtime.is_ready,
        metrics_source=lambda: runtime.metrics_snapshot(),
        dashboard_type="agent",
    )

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await runtime.startup()
        yield
        await runtime.shutdown()

    app.router.lifespan_context = lifespan

    @app.post("/run", response_model=AgentRunResponse)
    async def run(request: AgentRunRequest) -> AgentRunResponse:
        if not runtime.is_ready:
            raise HTTPException(status_code=503, detail="Service not initialized")

        messages = [Message(**m) for m in request.messages]

        try:
            result: AgentResult = await runtime.run(messages)
        except MaxStepsExceededError as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "max_steps_exceeded",
                    "detail": str(exc),
                    "steps_completed": exc.steps_completed,
                },
            )

        return AgentRunResponse(
            content=result.content,
            steps=[s.model_dump() for s in result.steps],
            metadata=result.metadata,
            total_tokens=result.total_tokens,
            total_cost_usd=result.total_cost_usd,
            llm_calls=result.llm_call_count,
            tool_calls=result.tool_call_count,
        )

    return app
