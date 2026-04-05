"""Tests for the evaluation bridge module."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml_platform.evaluation import (
    EvaluationReporter,
    EvaluationResult,
    EvaluationStatus,
    _status_to_float,
)

# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_defaults(self) -> None:
        result = EvaluationResult(name="test")
        assert result.status == EvaluationStatus.PASSED
        assert result.metrics == {}
        assert result.details == {}
        assert result.tags == {}
        assert result.timestamp > 0

    def test_metric_prefix(self) -> None:
        result = EvaluationResult(name="drift")
        assert result.metric_prefix("share") == "eval.drift.share"

    def test_prefixed_metrics(self) -> None:
        result = EvaluationResult(
            name="drift",
            metrics={"share": 0.4, "p_value": 0.01},
        )
        prefixed = result.prefixed_metrics()
        assert prefixed == {
            "eval.drift.share": 0.4,
            "eval.drift.p_value": 0.01,
        }

    def test_prefixed_metrics_empty(self) -> None:
        result = EvaluationResult(name="drift")
        assert result.prefixed_metrics() == {}


# ---------------------------------------------------------------------------
# EvaluationStatus
# ---------------------------------------------------------------------------


class TestEvaluationStatus:
    def test_string_values(self) -> None:
        assert EvaluationStatus.PASSED == "passed"
        assert EvaluationStatus.WARNING == "warning"
        assert EvaluationStatus.FAILED == "failed"

    def test_status_to_float(self) -> None:
        assert _status_to_float(EvaluationStatus.PASSED) == 0.0
        assert _status_to_float(EvaluationStatus.WARNING) == 1.0
        assert _status_to_float(EvaluationStatus.FAILED) == 2.0


# ---------------------------------------------------------------------------
# EvaluationReporter -- metric emission
# ---------------------------------------------------------------------------


class _FakeEmitter:
    """Test double recording emit_event calls."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, str], dict[str, float]]] = []

    def emit(self, metrics: dict[str, float]) -> None:
        pass

    def emit_event(
        self,
        event_name: str,
        dimensions: dict[str, str],
        values: dict[str, float],
    ) -> None:
        self.events.append((event_name, dimensions, values))


class TestReporterEmission:
    def test_emits_metrics_with_prefix(self) -> None:
        emitter = _FakeEmitter()
        reporter = EvaluationReporter(emitter=emitter)

        result = EvaluationResult(
            name="drift",
            status=EvaluationStatus.WARNING,
            metrics={"share": 0.4},
            tags={"dataset": "prod"},
        )
        reporter.report(result)

        assert len(emitter.events) == 1
        event_name, dims, values = emitter.events[0]
        assert event_name == "eval.drift"
        assert dims["evaluation"] == "drift"
        assert dims["dataset"] == "prod"
        assert values["share"] == 0.4
        assert values["eval.drift.share"] == 0.4
        assert values["eval.drift.status"] == 1.0

    def test_emits_without_prefix_when_disabled(self) -> None:
        emitter = _FakeEmitter()
        reporter = EvaluationReporter(emitter=emitter, prefix_metrics=False)

        result = EvaluationResult(
            name="drift",
            metrics={"share": 0.4},
        )
        reporter.report(result)

        _, _, values = emitter.events[0]
        assert "share" in values
        assert "eval.drift.share" not in values
        assert "eval.drift.status" in values

    def test_skips_emission_when_no_emitter(self) -> None:
        reporter = EvaluationReporter()
        result = EvaluationResult(name="drift", metrics={"share": 0.4})
        reporter.report(result)
        assert len(reporter.history) == 1

    def test_skips_emission_when_no_metrics(self) -> None:
        emitter = _FakeEmitter()
        reporter = EvaluationReporter(emitter=emitter)
        result = EvaluationResult(name="drift")
        reporter.report(result)
        assert len(emitter.events) == 0

    def test_emitter_failure_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        emitter = _FakeEmitter()
        emitter.emit_event = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
        reporter = EvaluationReporter(emitter=emitter)

        result = EvaluationResult(name="drift", metrics={"share": 0.4})
        reporter.report(result)

        assert len(reporter.history) == 1
        assert "Failed to emit" in caplog.text


# ---------------------------------------------------------------------------
# EvaluationReporter -- experiment tracking
# ---------------------------------------------------------------------------


class _FakeTracker:
    """Test double recording log_metrics calls."""

    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []

    @property
    def run_id(self) -> str:
        return "fake-run"

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged.append(dict(metrics))

    def log_artifact(self, local_path: str, artifact_subdir: str = "") -> None:
        pass

    def end_run(self) -> None:
        pass


class TestReporterTracking:
    def test_logs_prefixed_metrics(self) -> None:
        tracker = _FakeTracker()
        reporter = EvaluationReporter(tracker=tracker)

        result = EvaluationResult(
            name="quality",
            status=EvaluationStatus.PASSED,
            metrics={"missing_rate": 0.01},
        )
        reporter.report(result)

        assert len(tracker.logged) == 1
        logged = tracker.logged[0]
        assert logged["eval.quality.missing_rate"] == 0.01
        assert logged["eval.quality.status"] == 0.0

    def test_skips_tracking_when_no_tracker(self) -> None:
        reporter = EvaluationReporter()
        result = EvaluationResult(name="q", metrics={"x": 1.0})
        reporter.report(result)
        assert len(reporter.history) == 1

    def test_tracker_failure_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracker = _FakeTracker()
        tracker.log_metrics = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
        reporter = EvaluationReporter(tracker=tracker)

        result = EvaluationResult(name="q", metrics={"x": 1.0})
        reporter.report(result)
        assert "Failed to log evaluation to tracker" in caplog.text


# ---------------------------------------------------------------------------
# EvaluationReporter -- logging
# ---------------------------------------------------------------------------


class TestReporterLogging:
    def test_passed_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = EvaluationReporter()
        result = EvaluationResult(name="check", status=EvaluationStatus.PASSED)
        with caplog.at_level(logging.INFO, logger="ml_platform.evaluation"):
            reporter.report(result)
        assert "passed" in caplog.text
        assert "'check'" in caplog.text

    def test_warning_logged_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = EvaluationReporter()
        result = EvaluationResult(name="check", status=EvaluationStatus.WARNING)
        with caplog.at_level(logging.WARNING, logger="ml_platform.evaluation"):
            reporter.report(result)
        assert "warning" in caplog.text

    def test_failed_logged_at_error(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = EvaluationReporter()
        result = EvaluationResult(name="check", status=EvaluationStatus.FAILED)
        with caplog.at_level(logging.ERROR, logger="ml_platform.evaluation"):
            reporter.report(result)
        assert "failed" in caplog.text


# ---------------------------------------------------------------------------
# EvaluationReporter -- history & metrics_snapshot
# ---------------------------------------------------------------------------


class TestReporterHistory:
    def test_history_order(self) -> None:
        reporter = EvaluationReporter()
        r1 = EvaluationResult(name="a", metrics={"x": 1.0})
        r2 = EvaluationResult(name="b", metrics={"y": 2.0})
        reporter.report(r1)
        reporter.report(r2)

        assert len(reporter.history) == 2
        assert reporter.history[0].name == "a"
        assert reporter.history[1].name == "b"

    def test_history_is_a_copy(self) -> None:
        reporter = EvaluationReporter()
        reporter.report(EvaluationResult(name="a"))
        h = reporter.history
        h.clear()
        assert len(reporter.history) == 1

    def test_report_all(self) -> None:
        reporter = EvaluationReporter()
        results = [
            EvaluationResult(name="a", metrics={"x": 1.0}),
            EvaluationResult(name="b", metrics={"y": 2.0}),
        ]
        reporter.report_all(results)
        assert len(reporter.history) == 2


class TestReporterMetricsSnapshot:
    def test_returns_latest_per_name(self) -> None:
        reporter = EvaluationReporter()
        reporter.report(EvaluationResult(
            name="drift", status=EvaluationStatus.WARNING, metrics={"share": 0.2},
        ))
        reporter.report(EvaluationResult(
            name="drift", status=EvaluationStatus.PASSED, metrics={"share": 0.05},
        ))
        reporter.report(EvaluationResult(
            name="quality", status=EvaluationStatus.PASSED, metrics={"missing": 0.01},
        ))

        snap = reporter.metrics_snapshot()
        assert snap["eval.drift.share"] == 0.05
        assert snap["eval.drift.status"] == 0.0
        assert snap["eval.quality.missing"] == 0.01
        assert snap["eval.quality.status"] == 0.0

    def test_empty_when_no_history(self) -> None:
        reporter = EvaluationReporter()
        assert reporter.metrics_snapshot() == {}


# ---------------------------------------------------------------------------
# Integration with AlertEvaluator
# ---------------------------------------------------------------------------


class TestAlertIntegration:
    @pytest.mark.asyncio
    async def test_evaluation_metrics_trigger_alert(self) -> None:
        from ml_platform.alerting import AlertEvaluator, AlertRule, AlertState

        rule = AlertRule(
            metric="eval.drift.share",
            condition=">",
            threshold=0.3,
            name="high-drift",
        )
        evaluator = AlertEvaluator([rule])
        reporter = EvaluationReporter()

        reporter.report(EvaluationResult(
            name="drift",
            status=EvaluationStatus.FAILED,
            metrics={"share": 0.5},
        ))

        events = await evaluator.evaluate(reporter.metrics_snapshot())
        assert len(events) == 1
        assert events[0].state == AlertState.FIRING
        assert events[0].rule.name == "high-drift"

    @pytest.mark.asyncio
    async def test_passing_evaluation_does_not_trigger_alert(self) -> None:
        from ml_platform.alerting import AlertEvaluator, AlertRule

        rule = AlertRule(
            metric="eval.drift.share",
            condition=">",
            threshold=0.3,
            name="high-drift",
        )
        evaluator = AlertEvaluator([rule])
        reporter = EvaluationReporter()

        reporter.report(EvaluationResult(
            name="drift",
            status=EvaluationStatus.PASSED,
            metrics={"share": 0.05},
        ))

        events = await evaluator.evaluate(reporter.metrics_snapshot())
        assert len(events) == 0
