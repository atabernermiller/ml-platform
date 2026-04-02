"""Abstract experiment tracker interface.

Defines the contract that all tracker backends must implement.  The
library ships with :class:`~ml_platform.tracking.mlflow.MLflowTracker`
for MLflow and :class:`NullTracker` as a silent no-op for environments
where experiment tracking is disabled.

Third-party integrations (Weights & Biases, Neptune, Comet, etc.) can
implement :class:`ExperimentTracker` and be passed directly to the
serving runtime.

The canonical :class:`ExperimentTracker` ABC lives in
``ml_platform._interfaces`` and is re-exported here for backward
compatibility.
"""

from __future__ import annotations

from typing import Any

from ml_platform._interfaces import ExperimentTracker


class NullTracker(ExperimentTracker):
    """Silent no-op tracker for when experiment tracking is disabled.

    All methods succeed immediately without side effects.  Useful in local
    development, testing, and production services that only need CloudWatch
    metrics without an MLflow backend.
    """

    @property
    def run_id(self) -> str:
        return "null"

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_subdir: str = "") -> None:
        pass

    def end_run(self) -> None:
        pass
