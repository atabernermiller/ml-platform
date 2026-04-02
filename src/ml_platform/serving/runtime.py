"""Runtimes for ml-platform services.

:class:`BaseRuntime` provides the shared lifecycle that all service types
need: metric backend initialisation, background metric-emission loops,
experiment-tracker wiring, and startup/shutdown hooks.

:class:`StatefulRuntime` extends the base with S3 checkpoint loops and
feedback handling for online-learning services.

:class:`AgentRuntime` extends the base with per-request
:class:`~ml_platform.llm.run_context.RunContext` creation and agent-level
metric aggregation.

The companion ``create_*_app()`` factories wire a runtime into FastAPI
routes and ASGI lifespan.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from typing import Any, Type

from ml_platform.config import ServiceConfig
from ml_platform.monitoring.metrics import MetricsEmitter
from ml_platform.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseRuntime -- shared lifecycle for all service types
# ---------------------------------------------------------------------------


class BaseRuntime:
    """Manages shared lifecycle: metrics, tracking, and background tasks.

    Subclasses override :meth:`_on_startup` and :meth:`_on_shutdown` to
    add service-type-specific behaviour (checkpoint loops, RunContext
    setup, etc.).

    Args:
        config: Platform-wide configuration.
    """

    def __init__(self, config: ServiceConfig) -> None:
        self._config = config
        self._emitter: MetricsEmitter | None = None
        self._tracker: ExperimentTracker | None = None
        self._bg_tasks: list[asyncio.Task[None]] = []
        self._is_ready: bool = False

    @property
    def config(self) -> ServiceConfig:
        """The platform configuration."""
        return self._config

    @property
    def emitter(self) -> MetricsEmitter | None:
        """The active :class:`MetricsEmitter`, or ``None`` before startup."""
        return self._emitter

    @property
    def tracker(self) -> ExperimentTracker | None:
        """The active :class:`ExperimentTracker`, or ``None`` if disabled."""
        return self._tracker

    @property
    def is_ready(self) -> bool:
        """``True`` once startup has completed and the service is live."""
        return self._is_ready

    async def startup(self) -> None:
        """Initialise backends and start background loops."""
        self._emitter = MetricsEmitter(
            service_name=self._config.service_name,
            region=self._config.aws_region,
        )

        if self._config.mlflow_tracking_uri:
            from ml_platform.tracking.mlflow import MLflowTracker

            self._tracker = MLflowTracker(
                tracking_uri=self._config.mlflow_tracking_uri,
                experiment_name=self._config.mlflow_experiment_name,
            )

        await self._on_startup()

        self._bg_tasks.append(asyncio.create_task(self._metrics_loop()))
        self._is_ready = True
        logger.info("Service ready: %s", self._config.service_name)

    async def shutdown(self) -> None:
        """Cancel background tasks and clean up."""
        for task in self._bg_tasks:
            task.cancel()
        self._bg_tasks.clear()
        await self._on_shutdown()
        self._is_ready = False
        logger.info("Service stopped: %s", self._config.service_name)

    # -- hooks for subclasses ------------------------------------------------

    async def _on_startup(self) -> None:
        """Override in subclasses for service-type-specific start logic."""

    async def _on_shutdown(self) -> None:
        """Override in subclasses for service-type-specific stop logic."""

    def _metrics_snapshot(self) -> dict[str, float]:
        """Override in subclasses to provide service-specific metrics."""
        return {}

    # -- background loops ----------------------------------------------------

    async def _metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.metrics_interval_s)
            if self._is_ready and self._emitter:
                try:
                    snapshot = self._metrics_snapshot()
                    if snapshot:
                        self._emitter.emit(snapshot)
                    if self._tracker and snapshot:
                        self._tracker.log_metrics(snapshot)
                except Exception:
                    logger.exception("Metric emission failed")


# ---------------------------------------------------------------------------
# StatefulRuntime -- checkpoint loops and feedback handling
# ---------------------------------------------------------------------------


class StatefulRuntime(BaseRuntime):
    """Runtime for :class:`~ml_platform.serving.stateful.StatefulServiceBase`.

    Extends :class:`BaseRuntime` with S3 state restoration, periodic
    checkpoint loops, and feedback delegation.

    Args:
        service_cls: Concrete :class:`StatefulServiceBase` subclass.
        config: Platform-wide configuration.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.
    """

    def __init__(
        self,
        service_cls: Type[Any],
        config: ServiceConfig,
        *,
        service_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._service_cls = service_cls
        self._service_kwargs = service_kwargs or {}
        self._service: Any | None = None
        self._state_mgr: Any | None = None

    @property
    def service(self) -> Any:
        """The underlying service instance.

        Raises:
            RuntimeError: If accessed before :meth:`startup` completes.
        """
        if self._service is None:
            raise RuntimeError("StatefulRuntime has not been started")
        return self._service

    async def _on_startup(self) -> None:
        service = self._service_cls(**self._service_kwargs)
        self._service = service

        bucket = ""
        prefix = "checkpoints/"
        if self._config.stateful:
            bucket = self._config.stateful.s3_checkpoint_bucket
            prefix = self._config.stateful.s3_checkpoint_prefix
        elif self._config.s3_checkpoint_bucket:
            bucket = self._config.s3_checkpoint_bucket
            prefix = self._config.s3_checkpoint_prefix

        if bucket:
            from ml_platform.serving.state_manager import S3StateManager

            self._state_mgr = S3StateManager(
                bucket=bucket,
                prefix=prefix,
                region=self._config.aws_region,
            )
            restored_dir = self._state_mgr.download_latest()
            if restored_dir:
                service.load_state(restored_dir)
                logger.info("State restored from s3://%s", bucket)
            else:
                logger.info("No S3 checkpoint found; starting fresh")

        service.on_startup()

        if self._state_mgr:
            self._bg_tasks.append(asyncio.create_task(self._checkpoint_loop()))

    async def _on_shutdown(self) -> None:
        if self._service is not None:
            self._service.on_shutdown()
            if self._state_mgr:
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._service.save_state(tmpdir)
                    self._state_mgr.upload(tmpdir)
                logger.info("Final checkpoint saved")

    def _metrics_snapshot(self) -> dict[str, float]:
        if self._service is not None:
            return self._service.metrics_snapshot()
        return {}

    # -- request handling ----------------------------------------------------

    async def predict(self, payload: dict[str, Any]) -> Any:
        """Delegate to the service's ``predict`` and emit per-request metrics."""
        from ml_platform.serving.stateful import PredictionResult

        result: PredictionResult = await self.service.predict(payload)
        if self._emitter:
            self._emitter.emit_event(
                "prediction",
                dimensions={"service": self._config.service_name},
                values={
                    k: v
                    for k, v in result.metadata.items()
                    if isinstance(v, (int, float))
                },
            )
        return result

    async def process_feedback(
        self, request_id: str, feedback: dict[str, Any]
    ) -> None:
        """Delegate to the service's ``process_feedback``."""
        await self.service.process_feedback(request_id, feedback)

    def metrics_snapshot(self) -> dict[str, float]:
        """Return the latest business metrics from the service."""
        return self._metrics_snapshot()

    # -- background loops ----------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        interval = self._config.checkpoint_interval_s
        if self._config.stateful:
            interval = self._config.stateful.checkpoint_interval_s
        while True:
            await asyncio.sleep(interval)
            if self._service and self._state_mgr:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        self._service.save_state(tmpdir)
                        self._state_mgr.upload(tmpdir)
                    logger.info("Periodic checkpoint saved")
                except Exception:
                    logger.exception("Checkpoint failed")


# ---------------------------------------------------------------------------
# AgentRuntime -- per-request RunContext and agent-level metrics
# ---------------------------------------------------------------------------


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
        providers: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        service_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._service_cls = service_cls
        self._service_kwargs = service_kwargs or {}
        self._providers = providers or {}
        self._tools_list = tools or []
        self._tools: dict[str, Any] = {}
        self._service: Any | None = None

        for t in self._tools_list:
            _validate_tool(t)

        self._total_runs: int = 0
        self._total_steps: int = 0
        self._total_llm_calls: int = 0
        self._total_tool_calls: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    @property
    def service(self) -> Any:
        """The underlying agent service instance."""
        if self._service is None:
            raise RuntimeError("AgentRuntime has not been started")
        return self._service

    async def _on_startup(self) -> None:
        from ml_platform._interfaces import Tool

        self._tools = {}
        for t in self._tools_list:
            _validate_tool(t)
            self._tools[t.name] = t

        self._service = self._service_cls(**self._service_kwargs)
        self._service.providers = self._providers
        self._service.tools = self._tools

    async def _on_shutdown(self) -> None:
        pass

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
