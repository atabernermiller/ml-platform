"""Tests for CloudWatch EMF metric emission."""

from __future__ import annotations

import json

from ml_platform.monitoring.metrics import MetricsEmitter


def test_emit_prints_emf_json(capsys: object) -> None:
    emitter = MetricsEmitter(service_name="test-svc")
    emitter.emit({"accuracy": 0.95, "loss": 0.05})

    import _pytest.capture

    assert isinstance(capsys, _pytest.capture.CaptureFixture)
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())

    assert "_aws" in payload
    assert "CloudWatchMetrics" in payload["_aws"]
    cw_metrics = payload["_aws"]["CloudWatchMetrics"]
    assert len(cw_metrics) == 1
    metric_names = {m["Name"] for m in cw_metrics[0]["Metrics"]}
    assert metric_names == {"accuracy", "loss"}
    assert payload["accuracy"] == 0.95
    assert payload["loss"] == 0.05


def test_emit_event_includes_dimensions(capsys: object) -> None:
    emitter = MetricsEmitter(service_name="test-svc")
    emitter.emit_event(
        event_name="prediction",
        dimensions={"service": "test-svc"},
        values={"latency_ms": 42.0},
    )

    import _pytest.capture

    assert isinstance(capsys, _pytest.capture.CaptureFixture)
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())

    assert payload["service"] == "test-svc"
    assert payload["event"] == "prediction"
    assert payload["latency_ms"] == 42.0


def test_emit_empty_noop(capsys: object) -> None:
    emitter = MetricsEmitter(service_name="test-svc")
    emitter.emit({})

    import _pytest.capture

    assert isinstance(capsys, _pytest.capture.CaptureFixture)
    captured = capsys.readouterr()
    assert captured.out == ""
