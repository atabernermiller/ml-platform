"""Agent runtime for multi-step agentic services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Type

from ml_platform._interfaces import ConversationStore, LLMProvider, Tool
from ml_platform.config import ServiceConfig
from ml_platform.health import HealthCheck
from ml_platform.serving.runtime._base import BaseRuntime

logger = logging.getLogger(__name__)


class AgentRuntime(BaseRuntime):
    """Runtime for :class:`~ml_platform.serving.agent.AgentServiceBase`.

    Creates a fresh :class:`~ml_platform.llm.run_context.RunContext` per
    request and aggregates agent-level metrics (steps per run, tool usage,
    cost by model).

    Args:
        service_cls: Concrete :class:`AgentServiceBase` subclass.
        config: Platform-wide configuration.
        providers: Named LLM providers available to the agent.
        tools: Tools available to the agent.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.
    """

    def __init__(
        self,
        service_cls: Type[Any],
        config: ServiceConfig,
        *,
        providers: dict[str, LLMProvider] | None = None,
        tools: list[Tool] | None = None,
        service_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._service_cls = service_cls
        self._service_kwargs = service_kwargs or {}
        self._providers: dict[str, LLMProvider] = providers or {}
        self._tools_list: list[Tool] = tools or []
        self._tools: dict[str, Tool] = {}
        self._service: Any | None = None

        for t in self._tools_list:
            _validate_tool(t)

        self._total_runs: int = 0
        self._total_steps: int = 0
        self._total_llm_calls: int = 0
        self._total_tool_calls: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._counter_lock = asyncio.Lock()

    @property
    def service(self) -> Any:
        """The underlying agent service instance."""
        if self._service is None:
            raise RuntimeError("AgentRuntime has not been started")
        return self._service

    def _get_service_for_scheduling(self) -> Any | None:
        return self._service

    def _register_health_checks(self) -> None:
        if self._health_registry is None:
            return

        self._health_registry.register(HealthCheck(
            name="service_instance",
            check=lambda: self._service is not None,
            critical=True,
            description="Agent service class instantiated",
        ))
        self._health_registry.register(HealthCheck(
            name="providers",
            check=lambda: len(self._providers) > 0,
            critical=True,
            description="At least one LLM provider registered",
        ))

    async def _on_startup(self) -> None:
        self._tools = {}
        for t in self._tools_list:
            _validate_tool(t)
            self._tools[t.name] = t

        self._service = self._service_cls(**self._service_kwargs)
        self._service.providers = self._providers
        self._service.tools = self._tools
        self._service.conversation_store = self._create_conversation_store()

    async def _on_shutdown(self) -> None:
        pass

    def _create_conversation_store(self) -> ConversationStore | None:
        """Create a conversation store based on agent config."""
        agent_cfg = self._config.agent
        if agent_cfg is None or not agent_cfg.conversation_table_name:
            return None

        try:
            from ml_platform.serving.conversation_store import (
                DynamoDBConversationStore,
            )

            store = DynamoDBConversationStore(
                table_name=agent_cfg.conversation_table_name,
                region=self._config.aws_region,
                ttl_s=agent_cfg.conversation_ttl_s,
            )
            logger.info(
                "Conversation store: DynamoDB table %s",
                agent_cfg.conversation_table_name,
            )
            return store
        except Exception:
            from ml_platform.serving.conversation_store import (
                InMemoryConversationStore,
            )

            logger.info(
                "DynamoDB unavailable for conversations; using in-memory store"
            )
            return InMemoryConversationStore()

    def _metrics_snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        if self._service is not None:
            snapshot.update(self._service.metrics_snapshot())
        snapshot["total_runs"] = float(self._total_runs)
        snapshot["total_steps"] = float(self._total_steps)
        snapshot["total_llm_calls"] = float(self._total_llm_calls)
        snapshot["total_tool_calls"] = float(self._total_tool_calls)
        snapshot["total_tokens"] = float(self._total_tokens)
        snapshot["total_cost_usd"] = self._total_cost_usd
        if self._total_runs > 0:
            snapshot["avg_steps_per_run"] = self._total_steps / self._total_runs
        return snapshot

    async def run(self, messages: list[Any], **kwargs: Any) -> Any:
        """Execute a full agent turn with a fresh RunContext."""
        from ml_platform._types import AgentResult
        from ml_platform.llm.run_context import RunContext

        agent_config = self._config.agent
        max_steps = agent_config.max_steps_per_run if agent_config else 20

        async with RunContext(
            name=f"{self._config.service_name}-run",
            emitter=self._emitter,
            max_steps=max_steps,
        ) as ctx:
            result: AgentResult = await self.service.run(
                messages, run_context=ctx, **kwargs
            )

        async with self._counter_lock:
            self._total_runs += 1
            self._total_steps += len(result.steps)
            self._total_llm_calls += result.llm_call_count
            self._total_tool_calls += result.tool_call_count
            self._total_tokens += result.total_tokens
            self._total_cost_usd += result.total_cost_usd

        if self._emitter:
            self._emitter.emit_event(
                "agent_run",
                dimensions={"service": self._config.service_name},
                values={
                    "steps": float(len(result.steps)),
                    "llm_calls": float(result.llm_call_count),
                    "tool_calls": float(result.tool_call_count),
                    "total_tokens": float(result.total_tokens),
                    "total_cost_usd": result.total_cost_usd,
                    "total_latency_ms": result.total_latency_ms,
                },
            )

        return result

    def metrics_snapshot(self) -> dict[str, float]:
        """Return agent-level aggregate metrics."""
        return self._metrics_snapshot()


def _validate_tool(tool: Any) -> None:
    """Raise ``TypeError`` with a helpful message if *tool* is malformed."""
    required = ("name", "description", "parameters_schema", "execute")
    missing = [attr for attr in required if not hasattr(tool, attr)]
    if missing:
        cls_name = type(tool).__name__
        raise TypeError(
            f"{cls_name} does not satisfy the Tool protocol. "
            f"Missing: {', '.join(repr(m) for m in missing)}. "
            f"Required members: {', '.join(required)}"
        )
