"""Tests for ml_platform.alerting -- alert rules, evaluator, and notifiers."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from ml_platform.alerting import (
    AlertEvaluator,
    AlertEvent,
    AlertNotifier,
    AlertRule,
    AlertState,
    LogNotifier,
    WebhookNotifier,
)


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_auto_name(self) -> None:
        rule = AlertRule(metric="latency", condition=">", threshold=100)
        assert rule.name == "latency_>_100"

    def test_explicit_name(self) -> None:
        rule = AlertRule(metric="latency", condition=">", threshold=100, name="slow")
        assert rule.name == "slow"

    def test_evaluate_greater_than(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10)
        assert rule.evaluate(11) is True
        assert rule.evaluate(10) is False
        assert rule.evaluate(9) is False

    def test_evaluate_greater_equal(self) -> None:
        rule = AlertRule(metric="x", condition=">=", threshold=10)
        assert rule.evaluate(10) is True
        assert rule.evaluate(9) is False

    def test_evaluate_less_than(self) -> None:
        rule = AlertRule(metric="x", condition="<", threshold=5)
        assert rule.evaluate(4) is True
        assert rule.evaluate(5) is False

    def test_evaluate_less_equal(self) -> None:
        rule = AlertRule(metric="x", condition="<=", threshold=5)
        assert rule.evaluate(5) is True
        assert rule.evaluate(6) is False

    def test_evaluate_equal(self) -> None:
        rule = AlertRule(metric="x", condition="==", threshold=0)
        assert rule.evaluate(0) is True
        assert rule.evaluate(1) is False

    def test_evaluate_not_equal(self) -> None:
        rule = AlertRule(metric="x", condition="!=", threshold=0)
        assert rule.evaluate(1) is True
        assert rule.evaluate(0) is False

    def test_frozen(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=1)
        with pytest.raises(AttributeError):
            rule.metric = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AlertEvent
# ---------------------------------------------------------------------------


class TestAlertEvent:
    def test_to_dict(self) -> None:
        rule = AlertRule(
            metric="error_rate",
            condition=">",
            threshold=0.05,
            name="high-errors",
            severity="critical",
            description="Error rate too high",
        )
        event = AlertEvent(
            rule=rule,
            state=AlertState.FIRING,
            current_value=0.12,
            timestamp=1000.0,
            service_name="my-svc",
        )
        d = event.to_dict()
        assert d["alert_name"] == "high-errors"
        assert d["metric"] == "error_rate"
        assert d["condition"] == "> 0.05"
        assert d["current_value"] == 0.12
        assert d["state"] == "firing"
        assert d["severity"] == "critical"
        assert d["service"] == "my-svc"
        assert d["timestamp"] == 1000.0


# ---------------------------------------------------------------------------
# LogNotifier
# ---------------------------------------------------------------------------


class TestLogNotifier:
    @pytest.mark.asyncio
    async def test_firing_logged_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="test-alert")
        event = AlertEvent(rule=rule, state=AlertState.FIRING, current_value=15)
        notifier = LogNotifier()
        with caplog.at_level("WARNING", logger="ml_platform.alerting"):
            await notifier.notify(event)
        assert "ALERT FIRING" in caplog.text
        assert "test-alert" in caplog.text

    @pytest.mark.asyncio
    async def test_resolved_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="test-alert")
        event = AlertEvent(rule=rule, state=AlertState.RESOLVED, current_value=5)
        notifier = LogNotifier()
        with caplog.at_level("INFO", logger="ml_platform.alerting"):
            await notifier.notify(event)
        assert "ALERT RESOLVED" in caplog.text


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------


class TestWebhookNotifier:
    @pytest.mark.asyncio
    async def test_sends_json_post(self) -> None:
        notifier = WebhookNotifier("https://example.com/hook")
        rule = AlertRule(metric="x", condition=">", threshold=10, name="test")
        event = AlertEvent(
            rule=rule, state=AlertState.FIRING, current_value=15, service_name="svc"
        )

        with mock.patch("urllib.request.urlopen") as mock_open:
            await notifier.notify(event)

        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert req.get_method() == "POST"
        body = json.loads(req.data)
        assert body["alert_name"] == "test"
        assert body["state"] == "firing"

    @pytest.mark.asyncio
    async def test_slack_format(self) -> None:
        notifier = WebhookNotifier("https://hooks.slack.com/services/T/B/X")
        rule = AlertRule(metric="x", condition=">", threshold=10, name="test")
        event = AlertEvent(
            rule=rule, state=AlertState.FIRING, current_value=15, service_name="svc"
        )

        with mock.patch("urllib.request.urlopen") as mock_open:
            await notifier.notify(event)

        body = json.loads(mock_open.call_args[0][0].data)
        assert "text" in body
        assert "FIRING" in body["text"]

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        notifier = WebhookNotifier("https://example.com/hook")
        rule = AlertRule(metric="x", condition=">", threshold=10, name="test")
        event = AlertEvent(rule=rule, state=AlertState.FIRING, current_value=15)

        with mock.patch("urllib.request.urlopen", side_effect=Exception("network")):
            with caplog.at_level("ERROR"):
                await notifier.notify(event)

        assert "Failed to send webhook" in caplog.text


# ---------------------------------------------------------------------------
# AlertEvaluator -- core state machine
# ---------------------------------------------------------------------------


class _RecordingNotifier:
    """Test double that records every event."""

    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def notify(self, event: AlertEvent) -> None:
        self.events.append(event)


class TestAlertEvaluator:
    @pytest.mark.asyncio
    async def test_fires_immediately_when_no_window(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder], service_name="svc")

        events = await ev.evaluate({"x": 15}, now=100)
        assert len(events) == 1
        assert events[0].state == AlertState.FIRING
        assert events[0].current_value == 15

    @pytest.mark.asyncio
    async def test_resolves_immediately_when_no_window(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder])

        await ev.evaluate({"x": 15}, now=100)
        events = await ev.evaluate({"x": 5}, now=101)
        assert len(events) == 1
        assert events[0].state == AlertState.RESOLVED

    @pytest.mark.asyncio
    async def test_no_re_notification_while_firing(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder])

        await ev.evaluate({"x": 15}, now=100)
        events = await ev.evaluate({"x": 20}, now=101)
        assert len(events) == 0
        assert len(recorder.events) == 1

    @pytest.mark.asyncio
    async def test_window_delays_firing(self) -> None:
        rule = AlertRule(
            metric="x", condition=">", threshold=10, name="r1", window_s=60
        )
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder])

        events = await ev.evaluate({"x": 15}, now=100)
        assert len(events) == 0

        events = await ev.evaluate({"x": 15}, now=130)
        assert len(events) == 0

        events = await ev.evaluate({"x": 15}, now=160)
        assert len(events) == 1
        assert events[0].state == AlertState.FIRING

    @pytest.mark.asyncio
    async def test_window_resets_if_condition_clears(self) -> None:
        rule = AlertRule(
            metric="x", condition=">", threshold=10, name="r1", window_s=60
        )
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder])

        await ev.evaluate({"x": 15}, now=100)
        await ev.evaluate({"x": 5}, now=130)
        await ev.evaluate({"x": 15}, now=160)
        events = await ev.evaluate({"x": 15}, now=220)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_window_delays_resolution(self) -> None:
        rule = AlertRule(
            metric="x", condition=">", threshold=10, name="r1", window_s=60
        )
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([rule], notifiers=[recorder])

        await ev.evaluate({"x": 15}, now=100)
        await ev.evaluate({"x": 15}, now=160)
        assert len(recorder.events) == 1

        events = await ev.evaluate({"x": 5}, now=200)
        assert len(events) == 0

        events = await ev.evaluate({"x": 5}, now=260)
        assert len(events) == 1
        assert events[0].state == AlertState.RESOLVED

    @pytest.mark.asyncio
    async def test_ignores_missing_metric(self) -> None:
        rule = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        ev = AlertEvaluator([rule])
        events = await ev.evaluate({"y": 100}, now=100)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_multiple_rules(self) -> None:
        r1 = AlertRule(metric="a", condition=">", threshold=10, name="r1")
        r2 = AlertRule(metric="b", condition="<", threshold=5, name="r2")
        recorder = _RecordingNotifier()
        ev = AlertEvaluator([r1, r2], notifiers=[recorder])

        events = await ev.evaluate({"a": 15, "b": 3}, now=100)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        r1 = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        ev = AlertEvaluator([r1])
        await ev.evaluate({"x": 15}, now=100)

        status = ev.get_status()
        assert len(status) == 1
        assert status[0]["name"] == "r1"
        assert status[0]["state"] == "firing"

    @pytest.mark.asyncio
    async def test_notifier_failure_does_not_crash_evaluator(self) -> None:
        class _BrokenNotifier:
            async def notify(self, event: AlertEvent) -> None:
                raise RuntimeError("boom")

        rule = AlertRule(metric="x", condition=">", threshold=10, name="r1")
        ev = AlertEvaluator([rule], notifiers=[_BrokenNotifier()])

        events = await ev.evaluate({"x": 15}, now=100)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# ServiceConfig integration
# ---------------------------------------------------------------------------


class TestServiceConfigAlerts:
    def test_defaults_empty(self) -> None:
        from ml_platform.config import ServiceConfig

        c = ServiceConfig(service_name="svc")
        assert c.alerts == []
        assert c.alert_webhook_url == ""

    def test_with_rules(self) -> None:
        from ml_platform.config import ServiceConfig

        rules = [
            AlertRule(metric="latency", condition=">", threshold=500, name="slow"),
        ]
        c = ServiceConfig(
            service_name="svc",
            alerts=rules,
            alert_webhook_url="https://example.com/hook",
        )
        assert len(c.alerts) == 1
        assert c.alert_webhook_url == "https://example.com/hook"


# ---------------------------------------------------------------------------
# CloudFormation alarm generation
# ---------------------------------------------------------------------------


class TestCloudFormationAlarms:
    def test_generates_alarms_from_rules(self) -> None:
        from ml_platform.cli.cfn.template import generate_stack_template
        from ml_platform.cli.manifest import ProjectManifest

        manifest = ProjectManifest(service_name="test-svc")
        rules = [
            AlertRule(
                metric="p99_latency_ms",
                condition=">",
                threshold=500,
                window_s=300,
                name="high-latency",
            ),
            AlertRule(
                metric="error_rate",
                condition=">=",
                threshold=0.05,
                name="high-errors",
                severity="critical",
            ),
        ]
        template = generate_stack_template(
            manifest, ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/test:latest",
            alert_rules=rules,
        )
        resources = template["Resources"]

        assert "Alarmhighlatency" in resources
        alarm = resources["Alarmhighlatency"]
        assert alarm["Type"] == "AWS::CloudWatch::Alarm"
        props = alarm["Properties"]
        assert props["MetricName"] == "p99_latency_ms"
        assert props["Threshold"] == 500
        assert props["ComparisonOperator"] == "GreaterThanThreshold"
        assert props["Period"] == 300

        assert "Alarmhigherrors" in resources
        props2 = resources["Alarmhigherrors"]["Properties"]
        assert props2["ComparisonOperator"] == "GreaterThanOrEqualToThreshold"

    def test_no_alarms_when_no_rules(self) -> None:
        from ml_platform.cli.cfn.template import generate_stack_template
        from ml_platform.cli.manifest import ProjectManifest

        manifest = ProjectManifest(service_name="test-svc")
        template = generate_stack_template(
            manifest, ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/test:latest",
        )
        alarm_keys = [k for k in template["Resources"] if k.startswith("Alarm")]
        assert alarm_keys == []

    def test_skips_unsupported_conditions(self) -> None:
        from ml_platform.cli.cfn.template import generate_stack_template
        from ml_platform.cli.manifest import ProjectManifest

        manifest = ProjectManifest(service_name="test-svc")
        rules = [
            AlertRule(metric="x", condition="!=", threshold=0, name="not-eq"),
        ]
        template = generate_stack_template(
            manifest, ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/test:latest",
            alert_rules=rules,
        )
        alarm_keys = [k for k in template["Resources"] if k.startswith("Alarm")]
        assert alarm_keys == []


# ---------------------------------------------------------------------------
# /alerts endpoint
# ---------------------------------------------------------------------------


class TestAlertsEndpoint:
    @pytest.fixture()
    def alert_app(self) -> Any:
        from ml_platform.config import ServiceConfig
        from ml_platform.serving._app_builder import build_base_app

        rules = [AlertRule(metric="x", condition=">", threshold=10, name="test-rule")]
        evaluator = AlertEvaluator(rules, service_name="svc")

        config = ServiceConfig(service_name="alert-test")
        app = build_base_app(
            config,
            readiness_check=lambda: True,
            metrics_source=lambda: {"x": 1.0},
            alert_status=evaluator.get_status,
        )
        return app, evaluator

    @pytest.mark.asyncio
    async def test_alerts_endpoint_returns_status(self, alert_app: Any) -> None:
        import httpx

        app, evaluator = alert_app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test-rule"
        assert data[0]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_alerts_empty_when_no_evaluator(self) -> None:
        import httpx

        from ml_platform.config import ServiceConfig
        from ml_platform.serving._app_builder import build_base_app

        config = ServiceConfig(service_name="no-alerts")
        app = build_base_app(
            config,
            readiness_check=lambda: True,
            metrics_source=lambda: {},
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/alerts")
        assert resp.status_code == 200
        assert resp.json() == []
