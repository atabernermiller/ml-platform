"""Stateful ML service base class and FastAPI application factory.

Use this module for services that maintain mutable state and learn from
delayed feedback -- contextual bandits, online RL, dynamic pricing, etc.

The ``create_stateful_app`` factory builds a FastAPI application that
delegates all lifecycle and orchestration to
:class:`~ml_platform.serving.runtime.StatefulRuntime`.  For non-HTTP
transports, use :class:`StatefulRuntime` directly.

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

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Type

from fastapi import HTTPException, status

from ml_platform.config import ServiceConfig
from ml_platform.serving.schemas import FeedbackRequest, PredictRequest, PredictResponse


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
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_stateful_app(
    service_cls: Type[StatefulServiceBase],
    config: ServiceConfig,
    *,
    service_kwargs: dict[str, Any] | None = None,
) -> "FastAPI":
    """Build a production-ready FastAPI app wrapping a stateful ML service.

    Lifecycle orchestration (S3 restore, checkpoint loops, metric emission)
    is handled by :class:`~ml_platform.serving.runtime.StatefulRuntime`.
    This factory adds HTTP routes and ASGI lifespan wiring on top.

    Args:
        service_cls: Concrete ``StatefulServiceBase`` subclass.
        config: Platform-wide configuration.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.

    Returns:
        FastAPI application suitable for ``uvicorn.run(app)``.
    """
    from ml_platform.serving._app_builder import build_base_app
    from ml_platform.serving.runtime import StatefulRuntime

    runtime = StatefulRuntime(
        service_cls, config, service_kwargs=service_kwargs
    )

    app = build_base_app(
        config,
        readiness_check=lambda: runtime.is_ready,
        metrics_source=lambda: runtime.metrics_snapshot(),
        dashboard_type="stateful",
    )

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
        await runtime.startup()
        yield
        await runtime.shutdown()

    app.router.lifespan_context = lifespan

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if not runtime.is_ready:
            raise HTTPException(status_code=503, detail="Service not initialized")
        result = await runtime.predict(request.payload)
        return PredictResponse(
            request_id=result.request_id,
            prediction=result.prediction,
            metadata=result.metadata,
        )

    @app.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
    async def feedback(request: FeedbackRequest) -> dict[str, str]:
        if not runtime.is_ready:
            raise HTTPException(status_code=503, detail="Service not initialized")
        await runtime.process_feedback(request.request_id, request.feedback)
        return {"status": "accepted"}

    return app
