"""Framework-agnostic runtime for stateful ML services.

:class:`StatefulRuntime` owns the full lifecycle of a
:class:`~ml_platform.serving.stateful.StatefulServiceBase`: state
restoration, background checkpoint / metric loops, and graceful shutdown.
It is intentionally decoupled from any HTTP framework so that the same
orchestration logic can back a FastAPI app, a gRPC server, an AWS Lambda
handler, or any future transport.

The companion :func:`~ml_platform.serving.stateful.create_stateful_app`
factory wires a ``StatefulRuntime`` into FastAPI routes and lifespan as
the default integration.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from typing import Any, Type

from ml_platform.config import ServiceConfig
from ml_platform.monitoring.metrics import MetricsEmitter
from ml_platform.serving.stateful import PredictionResult, StatefulServiceBase
from ml_platform.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


class StatefulRuntime:
    """Manages the lifecycle of a :class:`StatefulServiceBase` instance.

    Responsibilities:

    - Instantiate the service and optionally restore state from S3.
    - Create a :class:`MetricsEmitter` and an optional
      :class:`ExperimentTracker`.
    - Run background ``asyncio`` tasks for periodic checkpoints and
      metric snapshots.
    - Perform a final checkpoint on shutdown.

    This class is transport-agnostic: it exposes ``predict``,
    ``process_feedback``, and ``metrics_snapshot`` as plain async /
    sync methods that any server integration can call.

    Args:
        service_cls: Concrete :class:`StatefulServiceBase` subclass.
        config: Platform-wide configuration.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.
    """

    def __init__(
        self,
        service_cls: Type[StatefulServiceBase],
        config: ServiceConfig,
        *,
        service_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._service_cls = service_cls
        self._config = config
        self._service_kwargs = service_kwargs or {}

        self._service: StatefulServiceBase | None = None
        self._emitter: MetricsEmitter | None = None
        self._tracker: ExperimentTracker | None = None
        self._state_mgr: Any = None
        self._bg_tasks: list[asyncio.Task[None]] = []

    @property
    def service(self) -> StatefulServiceBase:
        """The underlying service instance.

        Raises:
            RuntimeError: If accessed before :meth:`startup` completes.
        """
        if self._service is None:
            raise RuntimeError("StatefulRuntime has not been started")
        return self._service

    @property
    def emitter(self) -> MetricsEmitter | None:
        """The active :class:`MetricsEmitter`, or ``None`` before startup."""
        return self._emitter

    @property
    def tracker(self) -> ExperimentTracker | None:
        """The active :class:`ExperimentTracker`, or ``None`` if tracking is disabled."""
        return self._tracker

    @property
    def is_ready(self) -> bool:
        """``True`` once startup has completed and the service is accepting work."""
        return self._service is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize the service, restore state, and start background loops."""
        service = self._service_cls(**self._service_kwargs)
        self._service = service

        if self._config.s3_checkpoint_bucket:
            from ml_platform.serving.state_manager import S3StateManager

            self._state_mgr = S3StateManager(
                bucket=self._config.s3_checkpoint_bucket,
                prefix=self._config.s3_checkpoint_prefix,
                region=self._config.aws_region,
            )
            restored_dir = self._state_mgr.download_latest()
            if restored_dir:
                service.load_state(restored_dir)
                logger.info("State restored from s3://%s", self._config.s3_checkpoint_bucket)
            else:
                logger.info("No S3 checkpoint found; starting fresh")

        if self._config.mlflow_tracking_uri:
            from ml_platform.tracking.mlflow import MLflowTracker

            self._tracker = MLflowTracker(
                tracking_uri=self._config.mlflow_tracking_uri,
                experiment_name=self._config.mlflow_experiment_name,
            )

        self._emitter = MetricsEmitter(
            service_name=self._config.service_name,
            region=self._config.aws_region,
        )

        service.on_startup()

        if self._state_mgr:
            self._bg_tasks.append(asyncio.create_task(self._checkpoint_loop()))
        self._bg_tasks.append(asyncio.create_task(self._metrics_loop()))

        logger.info("Service ready: %s", self._config.service_name)

    async def shutdown(self) -> None:
        """Cancel background tasks, perform a final checkpoint, and clean up."""
        for task in self._bg_tasks:
            task.cancel()
        self._bg_tasks.clear()

        if self._service is not None:
            self._service.on_shutdown()

            if self._state_mgr:
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._service.save_state(tmpdir)
                    self._state_mgr.upload(tmpdir)
                logger.info("Final checkpoint saved")

        logger.info("Service stopped: %s", self._config.service_name)

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Delegate to the service's ``predict`` and emit per-request metrics.

        Args:
            payload: Deserialized request body.

        Returns:
            The service's :class:`PredictionResult`.
        """
        result = await self.service.predict(payload)
        if self._emitter:
            self._emitter.emit_event(
                "prediction",
                dimensions={"service": self._config.service_name},
                values={
                    k: v for k, v in result.metadata.items() if isinstance(v, (int, float))
                },
            )
        return result

    async def process_feedback(
        self, request_id: str, feedback: dict[str, Any]
    ) -> None:
        """Delegate to the service's ``process_feedback``.

        Args:
            request_id: Identifier from the original prediction.
            feedback: Project-specific signal (e.g., reward, label).
        """
        await self.service.process_feedback(request_id, feedback)

    def metrics_snapshot(self) -> dict[str, float]:
        """Return the latest business metrics from the service.

        Returns:
            Flat mapping of metric names to numeric values.
        """
        return self.service.metrics_snapshot()

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.checkpoint_interval_s)
            if self._service and self._state_mgr:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        self._service.save_state(tmpdir)
                        self._state_mgr.upload(tmpdir)
                    logger.info("Periodic checkpoint saved")
                except Exception:
                    logger.exception("Checkpoint failed")

    async def _metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.metrics_interval_s)
            if self._service and self._emitter:
                try:
                    snapshot = self._service.metrics_snapshot()
                    self._emitter.emit(snapshot)
                    if self._tracker:
                        self._tracker.log_metrics(snapshot)
                except Exception:
                    logger.exception("Metric emission failed")
