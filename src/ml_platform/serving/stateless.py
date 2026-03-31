"""BentoML integration helpers for stateless ML services.

Provides a lightweight ``PlatformMonitor`` that adds consistent CloudWatch
metrics and MLflow tracking to standard BentoML services without wrapping
the framework.  BentoML already supplies OpenTelemetry traces and Prometheus
metrics; this module adds *business-level* signals (reward, cost, model
selection distributions, etc.).

Example::

    import bentoml
    import numpy as np
    from ml_platform.serving.stateless import with_platform_monitoring
    from ml_platform.config import ServiceConfig

    config = ServiceConfig(service_name="iris-classifier")

    @bentoml.service(resources={"cpu": "2"})
    class IrisClassifier:
        model = bentoml.models.get("iris_sklearn:latest")

        def __init__(self) -> None:
            self._monitor = with_platform_monitoring(self, config)

        @bentoml.api
        def predict(self, features: np.ndarray) -> np.ndarray:
            result = self.model.predict(features)
            self._monitor.record_prediction({"n_samples": len(features)})
            return result
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ml_platform.config import ServiceConfig

if TYPE_CHECKING:
    from ml_platform.tracking.base import ExperimentTracker

logger = logging.getLogger(__name__)


class PlatformMonitor:
    """Bridge between a BentoML service and ml-platform observability.

    Emits per-prediction events and periodic aggregate metrics to CloudWatch
    and optionally MLflow.

    Args:
        config: Platform configuration (service name, region, MLflow URI, etc.).
    """

    def __init__(self, config: ServiceConfig) -> None:
        from ml_platform.monitoring.metrics import MetricsEmitter

        self.config = config
        self._emitter = MetricsEmitter(
            service_name=config.service_name,
            region=config.aws_region,
        )
        self._tracker: ExperimentTracker | None = None
        if config.mlflow_tracking_uri:
            from ml_platform.tracking.mlflow import MLflowTracker

            self._tracker = MLflowTracker(
                tracking_uri=config.mlflow_tracking_uri,
                experiment_name=config.mlflow_experiment_name,
            )

    def record_prediction(self, metadata: dict[str, float]) -> None:
        """Emit a single-prediction event to CloudWatch.

        Args:
            metadata: Numeric values to log (e.g., latency_ms, confidence).
        """
        self._emitter.emit_event(
            event_name="prediction",
            dimensions={"service": self.config.service_name},
            values=metadata,
        )

    def record_batch_metrics(self, metrics: dict[str, float]) -> None:
        """Emit aggregate metrics to CloudWatch and MLflow.

        Suitable for periodic reporting (e.g., once per minute) of
        service-wide statistics.

        Args:
            metrics: Flat mapping of metric names to numeric values.
        """
        self._emitter.emit(metrics)
        if self._tracker:
            self._tracker.log_metrics(metrics)


def with_platform_monitoring(
    service_instance: Any,
    config: ServiceConfig,
) -> PlatformMonitor:
    """Attach ml-platform monitoring to a BentoML service instance.

    Call this in your service's ``__init__`` method to obtain a
    ``PlatformMonitor`` for recording business metrics.

    Args:
        service_instance: The BentoML service (reserved for future introspection).
        config: Platform configuration.

    Returns:
        Ready-to-use ``PlatformMonitor``.
    """
    logger.info(
        "Platform monitoring attached to %s (service=%s)",
        type(service_instance).__name__,
        config.service_name,
    )
    return PlatformMonitor(config)
