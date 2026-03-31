"""MLflow experiment tracker for ML platform services.

Wraps the MLflow client with conventions shared across all services:

- Automatic experiment creation on first use.
- One long-lived run per service instance (created at startup, ended at
  shutdown) so that metrics form a continuous time series.
- Convenience methods for logging hyperparameters, periodic metrics,
  and versioned artifacts.

Example::

    from ml_platform.tracking.experiment import ExperimentTracker

    tracker = ExperimentTracker(
        tracking_uri="http://mlflow.internal:5000",
        experiment_name="pareto-bandit",
    )
    tracker.log_params({"alpha": 0.1, "pca_components": 25})
    tracker.log_metrics({"cumulative_reward": 42.5, "avg_cost": 0.003})
    tracker.log_artifact("/tmp/checkpoint/state.joblib")
    tracker.end_run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Thin wrapper around MLflow providing ml-platform conventions.

    A single ``Run`` is started on construction and should be ended via
    ``end_run()`` during service shutdown.  All ``log_*`` calls append to
    this run.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Experiment name (created if absent).
        run_name: Optional human-readable run name.
        tags: Optional tags applied to the run.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._mlflow = mlflow
        self._run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self._run_id: str = self._run.info.run_id
        logger.info(
            "MLflow run started: experiment=%s run_id=%s",
            experiment_name,
            self._run_id,
        )

    @property
    def run_id(self) -> str:
        """Active MLflow run ID."""
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run.

        Safe to call multiple times; MLflow deduplicates by key.

        Args:
            params: Mapping of parameter names to values (will be stringified).
        """
        str_params = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(str_params)
        logger.debug("Logged %d params", len(str_params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a batch of metrics to the active run.

        Args:
            metrics: Metric name-value pairs.
            step: Optional step counter (e.g., total predictions served).
        """
        for name, value in metrics.items():
            self._mlflow.log_metric(name, value, step=step)
        logger.debug("Logged %d metrics (step=%s)", len(metrics), step)

    def log_artifact(self, local_path: str, artifact_subdir: str = "") -> None:
        """Upload a local file or directory as a run artifact.

        Args:
            local_path: Path to the file or directory.
            artifact_subdir: Optional subdirectory within the artifact store.
        """
        path = Path(local_path)
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_subdir or None)
        else:
            self._mlflow.log_artifact(str(path), artifact_subdir or None)
        logger.info("Logged artifact: %s", local_path)

    def end_run(self) -> None:
        """Finalize the active MLflow run."""
        self._mlflow.end_run()
        logger.info("MLflow run ended: %s", self._run_id)
