"""Bridge module for plugging external evaluation libraries into the platform.

This module provides a thin, framework-agnostic layer that standardizes how
evaluation outputs (drift scores, data-quality checks, model-performance
metrics) flow into the platform's monitoring pipeline (``MetricsEmitter``,
``AlertEvaluator``, ``ExperimentTracker``).

The library intentionally does **not** depend on any evaluation framework.
Users ``pip install`` whatever tool they prefer (Evidently, alibi-detect,
whylogs, NannyML, etc.) and wrap their results in an :class:`EvaluationResult`
before handing them to an :class:`EvaluationReporter`.

Typical usage with the ``@scheduled`` decorator::

    from ml_platform.evaluation import EvaluationReporter, EvaluationResult, EvaluationStatus
    from ml_platform.scheduling import scheduled
    from ml_platform.serving.stateful import StatefulServiceBase

    class MyBandit(StatefulServiceBase):
        def __init__(self) -> None:
            self.reporter = EvaluationReporter(emitter=self.metrics)

        @scheduled(interval_s=3600, name="hourly-drift-check")
        async def check_drift(self) -> None:
            # --- user-land code using any evaluation library ---
            drift_share = run_my_drift_check(...)

            result = EvaluationResult(
                name="feature-drift",
                status=EvaluationStatus.FAILED if drift_share > 0.3
                       else EvaluationStatus.PASSED,
                metrics={"drift_share": drift_share, "drifted_features": 4},
            )
            self.reporter.report(result)

See the :doc:`/guides/evaluation` guide for complete examples with
Evidently, alibi-detect, and whylogs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_platform._interfaces import ExperimentTracker, MetricsBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EvaluationStatus(str, Enum):
    """Outcome of a single evaluation run.

    Consumers can branch on status to decide whether to alert, retrain,
    or simply log.
    """

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class EvaluationResult:
    """Framework-agnostic container for evaluation outputs.

    Every field except ``name`` has a sensible default so callers only
    populate what they need.

    Attributes:
        name: Short identifier for this evaluation (e.g. ``"feature-drift"``).
            Used as a metric dimension and log label.
        status: Overall pass / warning / fail outcome.
        metrics: Numeric results forwarded to :class:`MetricsEmitter` and
            eligible for :class:`AlertRule` evaluation.  Keys become
            CloudWatch metric names; keep them stable across runs.
        details: Arbitrary structured data for debugging or artifact
            logging (e.g. per-feature drift scores, column-level quality
            breakdowns).  Not emitted as metrics.
        tags: String key-value pairs added as CloudWatch dimensions when
            the result is emitted via ``emit_event``.  Useful for slicing
            by dataset, model version, or environment.
        timestamp: Unix timestamp of the evaluation.  Defaults to
            ``time.time()`` at construction.
    """

    name: str
    status: EvaluationStatus = EvaluationStatus.PASSED
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def metric_prefix(self, key: str) -> str:
        """Return *key* prefixed with ``eval.<name>.`` for namespacing.

        This avoids collisions when multiple evaluations emit metrics
        into the same CloudWatch namespace.

        Args:
            key: Raw metric name.

        Returns:
            Prefixed metric name, e.g. ``"eval.feature-drift.drift_share"``.
        """
        return f"eval.{self.name}.{key}"

    def prefixed_metrics(self) -> dict[str, float]:
        """Return :attr:`metrics` with each key run through :meth:`metric_prefix`.

        A convenience for callers who want namespaced metric keys without
        manually iterating.

        Returns:
            New dict with prefixed keys and the same float values.
        """
        return {self.metric_prefix(k): v for k, v in self.metrics.items()}


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class EvaluationReporter:
    """Bridges :class:`EvaluationResult` into the platform monitoring stack.

    The reporter performs three optional actions for each result:

    1. **Emit metrics** -- numeric values are forwarded to a
       :class:`~ml_platform._interfaces.MetricsBackend` (typically
       ``MetricsEmitter``).  Both raw and namespaced (``eval.<name>.*``)
       metrics are emitted.
    2. **Log to experiment tracker** -- if an
       :class:`~ml_platform._interfaces.ExperimentTracker` is provided,
       metrics are logged as a step and ``details`` are stored as a
       JSON artifact.
    3. **Log structured records** -- every result is logged at INFO
       (passed), WARNING, or ERROR level for the platform logger.

    Args:
        emitter: Metrics backend for CloudWatch / Prometheus emission.
            ``None`` disables metric emission.
        tracker: Experiment tracker for MLflow / W&B logging.
            ``None`` disables experiment logging.
        prefix_metrics: If ``True`` (default), emit metrics with
            ``eval.<name>.`` prefix in addition to the raw keys.
            Disable if you manage metric namespacing externally.
    """

    def __init__(
        self,
        emitter: MetricsBackend | None = None,
        tracker: ExperimentTracker | None = None,
        *,
        prefix_metrics: bool = True,
    ) -> None:
        self._emitter = emitter
        self._tracker = tracker
        self._prefix = prefix_metrics
        self._history: list[EvaluationResult] = []

    @property
    def history(self) -> list[EvaluationResult]:
        """Chronologically-ordered list of reported results."""
        return list(self._history)

    def report(self, result: EvaluationResult) -> None:
        """Process a single evaluation result through the monitoring pipeline.

        Args:
            result: The evaluation output to report.
        """
        self._history.append(result)
        self._log(result)
        self._emit(result)
        self._track(result)

    def report_all(self, results: Sequence[EvaluationResult]) -> None:
        """Report multiple results in sequence.

        Args:
            results: Evaluation outputs to report.
        """
        for r in results:
            self.report(r)

    def metrics_snapshot(self) -> dict[str, float]:
        """Return the most recent metric values for each evaluation name.

        Merges the prefixed metrics from the last result of each distinct
        ``name`` into a single flat dict.  Useful for feeding the snapshot
        into :meth:`AlertEvaluator.evaluate`.

        Returns:
            Flat dict of all latest evaluation metrics.
        """
        latest: dict[str, EvaluationResult] = {}
        for r in self._history:
            latest[r.name] = r

        merged: dict[str, float] = {}
        for r in latest.values():
            merged.update(r.prefixed_metrics())
            merged[f"eval.{r.name}.status"] = _status_to_float(r.status)
        return merged

    # -- internals -----------------------------------------------------------

    def _log(self, result: EvaluationResult) -> None:
        level = {
            EvaluationStatus.PASSED: logging.INFO,
            EvaluationStatus.WARNING: logging.WARNING,
            EvaluationStatus.FAILED: logging.ERROR,
        }[result.status]

        logger.log(
            level,
            "Evaluation '%s' %s | metrics=%s tags=%s",
            result.name,
            result.status.value,
            result.metrics,
            result.tags,
        )

    def _emit(self, result: EvaluationResult) -> None:
        if self._emitter is None or not result.metrics:
            return
        try:
            metrics_to_emit = dict(result.metrics)
            if self._prefix:
                metrics_to_emit.update(result.prefixed_metrics())
            metrics_to_emit[f"eval.{result.name}.status"] = _status_to_float(result.status)

            self._emitter.emit_event(
                event_name=f"eval.{result.name}",
                dimensions={"evaluation": result.name, **result.tags},
                values=metrics_to_emit,
            )
        except Exception:
            logger.exception("Failed to emit evaluation metrics for '%s'", result.name)

    def _track(self, result: EvaluationResult) -> None:
        if self._tracker is None:
            return
        try:
            prefixed = result.prefixed_metrics()
            prefixed[f"eval.{result.name}.status"] = _status_to_float(result.status)
            self._tracker.log_metrics(prefixed)
        except Exception:
            logger.exception("Failed to log evaluation to tracker for '%s'", result.name)


def _status_to_float(status: EvaluationStatus) -> float:
    """Encode status as a numeric value for metric emission.

    Mapping: PASSED -> 0.0, WARNING -> 1.0, FAILED -> 2.0.
    """
    return {
        EvaluationStatus.PASSED: 0.0,
        EvaluationStatus.WARNING: 1.0,
        EvaluationStatus.FAILED: 2.0,
    }[status]
