"""Stateful ML service base class and FastAPI application factory.

Use this module for services that maintain mutable state and learn from
delayed feedback -- contextual bandits, online RL, dynamic pricing, etc.

The ``create_stateful_app`` factory builds a FastAPI application that handles:

- HTTP endpoints for predict, feedback, health, and metrics
- Periodic state checkpointing to S3
- Periodic metric emission to CloudWatch and MLflow
- Graceful startup (restore from checkpoint) and shutdown (final save)

Projects implement ``StatefulServiceBase`` with their model-specific logic.

Example::

    from ml_platform.serving.stateful import (
        StatefulServiceBase, PredictionResult, create_stateful_app,
    )
    from ml_platform.config import ServiceConfig

    class MyBanditService(StatefulServiceBase):
        def __init__(self) -> None:
            self._router = BanditRouter.create(...)

        async def predict(self, payload: dict) -> PredictionResult:
            log = self._router.route(payload["prompt"])
            return PredictionResult(
                request_id=log.request_id,
                prediction={"model": log.selected_model},
                metadata={"cost_usd": log.cost_usd},
            )

        async def process_feedback(self, request_id, feedback):
            self._router.process_feedback(request_id, feedback["reward"])

        def save_state(self, d):  ...
        def load_state(self, d):  ...
        def metrics_snapshot(self): return {"cumulative_reward": ...}

    config = ServiceConfig(service_name="pareto-bandit")
    app = create_stateful_app(MyBanditService, config)
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Type

from fastapi import FastAPI, HTTPException, status

from ml_platform.config import ServiceConfig
from ml_platform.serving.schemas import FeedbackRequest, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionResult:
    """Structured output returned by ``StatefulServiceBase.predict``.

    Attributes:
        request_id: Unique identifier for feedback correlation.
        prediction: Model output payload (project-specific schema).
        metadata: Optional numeric metadata emitted as per-request metrics.
    """

    request_id: str
    prediction: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class StatefulServiceBase(ABC):
    """Contract for ML services with mutable state and feedback loops.

    Implementors provide model-specific logic; the framework manages HTTP
    serving, state persistence, metric emission, and lifecycle hooks.
    """

    @abstractmethod
    async def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Handle a prediction / routing request.

        Args:
            payload: Deserialized request body (project-specific schema).

        Returns:
            Structured result with a unique ``request_id`` for later feedback.
        """
        ...

    @abstractmethod
    async def process_feedback(
        self, request_id: str, feedback: dict[str, Any]
    ) -> None:
        """Update model state with delayed feedback.

        Args:
            request_id: Identifier from the original ``PredictionResult``.
            feedback: Project-specific signal (e.g., reward, label).
        """
        ...

    @abstractmethod
    def save_state(self, artifact_dir: str) -> None:
        """Serialize model state to a local directory.

        The framework uploads the contents to S3 after this call.

        Args:
            artifact_dir: Writable directory for checkpoint files.
        """
        ...

    @abstractmethod
    def load_state(self, artifact_dir: str) -> None:
        """Deserialize model state from a local directory.

        The framework downloads the latest S3 checkpoint before this call.

        Args:
            artifact_dir: Directory containing checkpoint files.
        """
        ...

    @abstractmethod
    def metrics_snapshot(self) -> dict[str, float]:
        """Return current business metrics for periodic logging.

        Called on a schedule; values are emitted to CloudWatch and MLflow.

        Returns:
            Flat mapping of metric names to numeric values.
        """
        ...

    def on_startup(self) -> None:
        """Optional hook invoked after state restoration, before serving."""

    def on_shutdown(self) -> None:
        """Optional hook invoked before graceful shutdown."""


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_stateful_app(
    service_cls: Type[StatefulServiceBase],
    config: ServiceConfig,
    *,
    service_kwargs: dict[str, Any] | None = None,
) -> FastAPI:
    """Build a production-ready FastAPI app wrapping a stateful ML service.

    Args:
        service_cls: Concrete ``StatefulServiceBase`` subclass.
        config: Platform-wide configuration.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.

    Returns:
        FastAPI application suitable for ``uvicorn.run(app)``.
    """
    kwargs = service_kwargs or {}
    _bg_tasks: list[asyncio.Task[None]] = []

    # Mutable references populated during lifespan startup.
    _state: dict[str, Any] = {}

    # ---- lifespan ----------------------------------------------------------

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        service = service_cls(**kwargs)
        _state["service"] = service

        # -- optional S3 checkpoint restore --
        state_mgr = None
        if config.s3_checkpoint_bucket:
            from ml_platform.serving.state_manager import S3StateManager

            state_mgr = S3StateManager(
                bucket=config.s3_checkpoint_bucket,
                prefix=config.s3_checkpoint_prefix,
                region=config.aws_region,
            )
            _state["state_mgr"] = state_mgr
            restored_dir = state_mgr.download_latest()
            if restored_dir:
                service.load_state(restored_dir)
                logger.info("State restored from s3://%s", config.s3_checkpoint_bucket)
            else:
                logger.info("No S3 checkpoint found; starting fresh")

        # -- optional MLflow tracker --
        if config.mlflow_tracking_uri:
            from ml_platform.tracking.experiment import ExperimentTracker

            _state["tracker"] = ExperimentTracker(
                tracking_uri=config.mlflow_tracking_uri,
                experiment_name=config.mlflow_experiment_name,
            )

        # -- metrics emitter (always active) --
        from ml_platform.monitoring.metrics import MetricsEmitter

        _state["emitter"] = MetricsEmitter(
            service_name=config.service_name,
            region=config.aws_region,
        )

        service.on_startup()

        # -- background loops --
        if state_mgr:
            _bg_tasks.append(asyncio.create_task(_checkpoint_loop()))
        _bg_tasks.append(asyncio.create_task(_metrics_loop()))

        logger.info("Service ready: %s", config.service_name)
        yield

        # -- shutdown --
        service.on_shutdown()
        for task in _bg_tasks:
            task.cancel()
        if state_mgr:
            with tempfile.TemporaryDirectory() as tmpdir:
                service.save_state(tmpdir)
                state_mgr.upload(tmpdir)
            logger.info("Final checkpoint saved")
        logger.info("Service stopped: %s", config.service_name)

    # ---- background tasks --------------------------------------------------

    async def _checkpoint_loop() -> None:
        while True:
            await asyncio.sleep(config.checkpoint_interval_s)
            service = _state.get("service")
            state_mgr = _state.get("state_mgr")
            if service and state_mgr:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        service.save_state(tmpdir)
                        state_mgr.upload(tmpdir)
                    logger.info("Periodic checkpoint saved")
                except Exception:
                    logger.exception("Checkpoint failed")

    async def _metrics_loop() -> None:
        while True:
            await asyncio.sleep(config.metrics_interval_s)
            service = _state.get("service")
            emitter = _state.get("emitter")
            tracker = _state.get("tracker")
            if service and emitter:
                try:
                    snapshot = service.metrics_snapshot()
                    emitter.emit(snapshot)
                    if tracker:
                        tracker.log_metrics(snapshot)
                except Exception:
                    logger.exception("Metric emission failed")

    # ---- FastAPI app -------------------------------------------------------

    app = FastAPI(title=config.service_name, lifespan=lifespan)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        service: StatefulServiceBase | None = _state.get("service")
        if service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        result = await service.predict(request.payload)
        emitter = _state.get("emitter")
        if emitter:
            emitter.emit_event(
                "prediction",
                dimensions={"service": config.service_name},
                values={k: v for k, v in result.metadata.items() if isinstance(v, (int, float))},
            )
        return PredictResponse(
            request_id=result.request_id,
            prediction=result.prediction,
            metadata=result.metadata,
        )

    @app.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
    async def feedback(request: FeedbackRequest) -> dict[str, str]:
        service: StatefulServiceBase | None = _state.get("service")
        if service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        await service.process_feedback(request.request_id, request.feedback)
        return {"status": "accepted"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "service": config.service_name}

    @app.get("/metrics")
    async def metrics() -> dict[str, float]:
        service: StatefulServiceBase | None = _state.get("service")
        if service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return service.metrics_snapshot()

    return app
