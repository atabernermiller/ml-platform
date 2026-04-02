"""Declarative alerting for ml-platform services.

Users define :class:`AlertRule` instances (either inline in
``ServiceConfig`` or deserialized from ``ml-platform.yaml``).  At
runtime, an :class:`AlertEvaluator` checks every metric snapshot
against the rules and fires notifications through one or more
:class:`AlertNotifier` backends when thresholds are breached.

Quick start::

    from ml_platform.alerting import AlertRule
    from ml_platform.config import ServiceConfig

    config = ServiceConfig(
        service_name="my-chatbot",
        alerts=[
            AlertRule(
                metric="p99_latency_ms",
                condition=">",
                threshold=500,
                window_s=300,
                name="high-latency",
            ),
            AlertRule(
                metric="error_rate",
                condition=">",
                threshold=0.05,
            ),
        ],
        alert_webhook_url="https://hooks.slack.com/services/...",
    )

The evaluator supports hysteresis via ``window_s``: a rule only fires
after the condition has been continuously violated for the configured
duration.  It resolves once the metric returns to safe levels for the
same window duration.

Notification backends:

- :class:`LogNotifier` -- always active; logs state transitions.
- :class:`WebhookNotifier` -- HTTP POST to Slack, Discord, PagerDuty,
  or any URL.  The payload is a JSON object with the alert details.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

Condition = Literal[">", ">=", "<", "<=", "==", "!="]


@dataclass(frozen=True)
class AlertRule:
    """Declarative threshold-based alert rule.

    Attributes:
        metric: The metric name to watch (must match a key in the
            metrics snapshot dict).
        condition: Comparison operator applied as ``metric_value <op> threshold``.
        threshold: The numeric threshold value.
        window_s: Seconds the condition must be continuously violated
            before the alert fires.  ``0`` means fire immediately.
        name: Human-readable name for the alert.  Auto-generated from
            the metric name if empty.
        severity: ``"warning"`` or ``"critical"``.  Informational only;
            included in notifications.
        description: Optional explanation included in notifications.
    """

    metric: str
    condition: Condition
    threshold: float
    window_s: int = 0
    name: str = ""
    severity: Literal["warning", "critical"] = "warning"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(
                self, "name", f"{self.metric}_{self.condition}_{self.threshold}"
            )

    def evaluate(self, value: float) -> bool:
        """Return ``True`` if the condition is violated.

        Args:
            value: The current metric value.

        Returns:
            Whether the threshold is breached.
        """
        ops = {
            ">": value > self.threshold,
            ">=": value >= self.threshold,
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            "==": value == self.threshold,
            "!=": value != self.threshold,
        }
        return ops[self.condition]


class AlertState(str, Enum):
    """Current state of a monitored alert rule."""

    OK = "ok"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertEvent:
    """Notification payload emitted on state transitions.

    Attributes:
        rule: The alert rule that triggered.
        state: The new state (``FIRING`` or ``RESOLVED``).
        current_value: The metric value at the time of the transition.
        timestamp: Unix timestamp of the transition.
        service_name: The service that generated the alert.
    """

    rule: AlertRule
    state: AlertState
    current_value: float
    timestamp: float = field(default_factory=time.time)
    service_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for webhook payloads."""
        return {
            "alert_name": self.rule.name,
            "metric": self.rule.metric,
            "condition": f"{self.rule.condition} {self.rule.threshold}",
            "current_value": self.current_value,
            "state": self.state.value,
            "severity": self.rule.severity,
            "description": self.rule.description,
            "service": self.service_name,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Notifier protocol and implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class AlertNotifier(Protocol):
    """Protocol for alert notification backends."""

    def notify(self, event: AlertEvent) -> None:
        """Send a notification for an alert state transition.

        Args:
            event: The alert event describing what happened.
        """
        ...


class LogNotifier:
    """Logs alert transitions via the standard ``logging`` module.

    Always active -- provides a baseline audit trail regardless of
    whether external notification backends are configured.
    """

    def notify(self, event: AlertEvent) -> None:
        """Log the alert event at WARNING (firing) or INFO (resolved).

        Args:
            event: The alert event.
        """
        if event.state == AlertState.FIRING:
            logger.warning(
                "ALERT FIRING: %s | %s %s %s (current: %s) | severity=%s",
                event.rule.name,
                event.rule.metric,
                event.rule.condition,
                event.rule.threshold,
                event.current_value,
                event.rule.severity,
            )
        elif event.state == AlertState.RESOLVED:
            logger.info(
                "ALERT RESOLVED: %s | %s = %s (threshold: %s %s)",
                event.rule.name,
                event.rule.metric,
                event.current_value,
                event.rule.condition,
                event.rule.threshold,
            )


class WebhookNotifier:
    """Sends alert events as JSON HTTP POST requests.

    Compatible with Slack incoming webhooks, Discord webhooks,
    PagerDuty Events API v2, and any endpoint that accepts JSON.

    The payload is a JSON object with keys: ``alert_name``, ``metric``,
    ``condition``, ``current_value``, ``state``, ``severity``,
    ``description``, ``service``, ``timestamp``.

    For Slack, the notifier wraps the message in the ``text`` field
    expected by the Slack API.

    Args:
        url: The webhook endpoint URL.
        timeout_s: HTTP request timeout in seconds.
    """

    def __init__(self, url: str, *, timeout_s: float = 10.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    @property
    def url(self) -> str:
        """The configured webhook URL."""
        return self._url

    def notify(self, event: AlertEvent) -> None:
        """POST the alert event to the webhook URL.

        Args:
            event: The alert event.
        """
        import urllib.request

        payload = event.to_dict()

        if "hooks.slack.com" in self._url:
            emoji = ":rotating_light:" if event.state == AlertState.FIRING else ":white_check_mark:"
            text = (
                f"{emoji} *{event.state.value.upper()}*: {event.rule.name}\n"
                f"Metric: `{event.rule.metric}` = {event.current_value} "
                f"(threshold: {event.rule.condition} {event.rule.threshold})\n"
                f"Severity: {event.rule.severity} | Service: {event.service_name}"
            )
            payload = {"text": text}

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s):
                pass
            logger.debug("Webhook sent to %s for %s", self._url, event.rule.name)
        except Exception:
            logger.exception("Failed to send webhook to %s", self._url)


# ---------------------------------------------------------------------------
# Alert evaluator
# ---------------------------------------------------------------------------


@dataclass
class _RuleState:
    """Internal tracking state for a single rule."""

    state: AlertState = AlertState.OK
    violation_since: float | None = None
    ok_since: float | None = None


class AlertEvaluator:
    """Evaluates metric snapshots against alert rules and fires notifications.

    Call :meth:`evaluate` on each metrics cycle.  The evaluator handles
    hysteresis (``window_s``), deduplication (only notifies on state
    transitions), and dispatches to all registered notifiers.

    Args:
        rules: Alert rules to evaluate.
        notifiers: Notification backends.  A :class:`LogNotifier` is
            always prepended automatically.
        service_name: Included in alert event payloads.
    """

    def __init__(
        self,
        rules: list[AlertRule],
        notifiers: list[AlertNotifier] | None = None,
        service_name: str = "",
    ) -> None:
        self._rules = list(rules)
        self._notifiers: list[AlertNotifier] = [LogNotifier()]
        if notifiers:
            self._notifiers.extend(notifiers)
        self._service_name = service_name
        self._states: dict[str, _RuleState] = {
            rule.name: _RuleState() for rule in self._rules
        }

    @property
    def rules(self) -> list[AlertRule]:
        """The configured alert rules."""
        return list(self._rules)

    def evaluate(
        self,
        metrics: dict[str, float],
        *,
        now: float | None = None,
    ) -> list[AlertEvent]:
        """Check all rules against a metric snapshot and notify on transitions.

        Args:
            metrics: Current metric name-value pairs.
            now: Override the current timestamp (for testing).

        Returns:
            List of alert events that triggered notifications this cycle.
        """
        if now is None:
            now = time.time()

        events: list[AlertEvent] = []
        for rule in self._rules:
            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]
            violated = rule.evaluate(value)
            rs = self._states[rule.name]

            event = self._update_state(rule, rs, violated, value, now)
            if event is not None:
                events.append(event)

        return events

    def get_status(self) -> list[dict[str, Any]]:
        """Return the current state of every rule for the ``/alerts`` endpoint.

        Returns:
            List of dicts with ``name``, ``metric``, ``condition``,
            ``threshold``, ``state``, ``severity``.
        """
        result = []
        for rule in self._rules:
            rs = self._states[rule.name]
            result.append({
                "name": rule.name,
                "metric": rule.metric,
                "condition": f"{rule.condition} {rule.threshold}",
                "state": rs.state.value,
                "severity": rule.severity,
            })
        return result

    def _update_state(
        self,
        rule: AlertRule,
        rs: _RuleState,
        violated: bool,
        value: float,
        now: float,
    ) -> AlertEvent | None:
        """State machine for a single rule.  Returns an event on transitions."""
        if violated:
            rs.ok_since = None

            if rs.state == AlertState.OK or rs.state == AlertState.RESOLVED:
                if rule.window_s == 0:
                    rs.state = AlertState.FIRING
                    rs.violation_since = now
                    return self._fire(rule, AlertState.FIRING, value, now)
                else:
                    rs.state = AlertState.PENDING
                    rs.violation_since = now
                    return None

            if rs.state == AlertState.PENDING:
                elapsed = now - (rs.violation_since or now)
                if elapsed >= rule.window_s:
                    rs.state = AlertState.FIRING
                    return self._fire(rule, AlertState.FIRING, value, now)
                return None

            # Already FIRING -- no re-notification
            return None

        else:
            rs.violation_since = None

            if rs.state == AlertState.PENDING:
                rs.state = AlertState.OK
                rs.ok_since = now
                return None

            if rs.state == AlertState.FIRING:
                if rule.window_s == 0:
                    rs.state = AlertState.RESOLVED
                    rs.ok_since = now
                    event = self._fire(rule, AlertState.RESOLVED, value, now)
                    rs.state = AlertState.OK
                    return event
                else:
                    if rs.ok_since is None:
                        rs.ok_since = now
                        return None
                    elapsed = now - rs.ok_since
                    if elapsed >= rule.window_s:
                        rs.state = AlertState.RESOLVED
                        event = self._fire(rule, AlertState.RESOLVED, value, now)
                        rs.state = AlertState.OK
                        rs.ok_since = None
                        return event
                    return None

            return None

    def _fire(
        self,
        rule: AlertRule,
        state: AlertState,
        value: float,
        now: float,
    ) -> AlertEvent:
        """Create an event and dispatch to all notifiers."""
        event = AlertEvent(
            rule=rule,
            state=state,
            current_value=value,
            timestamp=now,
            service_name=self._service_name,
        )
        for notifier in self._notifiers:
            try:
                notifier.notify(event)
            except Exception:
                logger.exception(
                    "Notifier %s failed for alert %s",
                    type(notifier).__name__,
                    rule.name,
                )
        return event
