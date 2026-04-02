"""Base runtime with shared lifecycle for all service types."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ml_platform.alerting import AlertEvaluator, WebhookNotifier
from ml_platform.config import ServiceConfig
from ml_platform.health import HealthCheck, HealthRegistry
from ml_platform.monitoring.metrics import MetricsEmitter
from ml_platform.scheduling import TaskRunner, discover_tasks
from ml_platform.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


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
        self._alert_evaluator: AlertEvaluator | None = None
        self._health_registry: HealthRegistry | None = None
        self._bg_tasks: list[asyncio.Task[None]] = []
        self._is_ready: bool = False
        self._task_runner: TaskRunner | None = None

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
    def alert_evaluator(self) -> AlertEvaluator | None:
        """The active :class:`AlertEvaluator`, or ``None`` if no rules."""
        return self._alert_evaluator

    @property
    def health_registry(self) -> HealthRegistry | None:
        """The active :class:`HealthRegistry`, or ``None`` before startup."""
        return self._health_registry

    @property
    def is_ready(self) -> bool:
        """``True`` once startup has completed and the service is live."""
        return self._is_ready

    async def startup(self) -> None:
        """Initialise backends and start background loops."""
        if self._config.log_format:
            from ml_platform.log import configure_logging

            configure_logging(
                format=self._config.log_format,  # type: ignore[arg-type]
                level=self._config.log_level,
                service_name=self._config.service_name,
            )

        self._init_health_registry()

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

        self._register_health_checks()

        await self._start_scheduled_tasks()
        self._init_alert_evaluator()

        self._bg_tasks.append(asyncio.create_task(self._metrics_loop()))
        self._is_ready = True
        logger.info("Service ready: %s", self._config.service_name)

    async def shutdown(self) -> None:
        """Cancel background tasks and clean up."""
        if self._task_runner is not None:
            await self._task_runner.stop()
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

    def _get_service_for_scheduling(self) -> Any | None:
        """Return the service instance for task discovery.

        Subclasses override to expose their service object.
        """
        return None

    # -- health checks --------------------------------------------------------

    def _init_health_registry(self) -> None:
        """Create the HealthRegistry for this runtime."""
        self._health_registry = HealthRegistry(
            service_name=self._config.service_name,
        )
        self._health_registry.register(HealthCheck(
            name="runtime",
            check=lambda: self._is_ready,
            critical=True,
            description="Service runtime initialised and ready",
        ))

    def _register_health_checks(self) -> None:
        """Register backend-specific health checks after startup.

        Subclasses override to add checks for their backends
        (DynamoDB, S3, model state, etc.).
        """

    # -- scheduled tasks -----------------------------------------------------

    async def _start_scheduled_tasks(self) -> None:
        """Discover @scheduled methods on the service and start them."""
        service = self._get_service_for_scheduling()
        if service is None:
            return

        tasks = discover_tasks(service)
        if not tasks:
            return

        runner = TaskRunner(
            emitter=self._emitter,
            service_name=self._config.service_name,
        )
        runner.register_all(tasks)
        await runner.start()
        self._task_runner = runner

    # -- alerting --------------------------------------------------------------

    def _init_alert_evaluator(self) -> None:
        """Create the AlertEvaluator if alert rules are configured."""
        if not self._config.alerts:
            return

        notifiers: list[Any] = []
        if self._config.alert_webhook_url:
            notifiers.append(WebhookNotifier(self._config.alert_webhook_url))

        self._alert_evaluator = AlertEvaluator(
            rules=self._config.alerts,
            notifiers=notifiers,
            service_name=self._config.service_name,
        )
        logger.info(
            "Alert evaluator initialised with %d rule(s)", len(self._config.alerts)
        )

    # -- background loops ----------------------------------------------------

    async def _metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.metrics_interval_s)
            if self._is_ready and self._emitter:
                try:
                    snapshot = self._metrics_snapshot()
                    if self._task_runner is not None:
                        snapshot.update(self._task_runner.metrics_snapshot())
                    if snapshot:
                        self._emitter.emit(snapshot)
                    if self._tracker and snapshot:
                        self._tracker.log_metrics(snapshot)
                    if self._alert_evaluator and snapshot:
                        await self._alert_evaluator.evaluate(snapshot)
                except Exception:
                    logger.exception("Metric emission failed")
