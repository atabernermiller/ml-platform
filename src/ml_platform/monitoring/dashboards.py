"""Dashboard configuration generators for Grafana and CloudWatch.

These functions produce JSON structures that can be imported into
Amazon Managed Grafana or deployed as CloudWatch Dashboard resources
via the AWS API / CDK.

The :func:`generate_dashboard` entry point accepts composable *panel sets*
so that stateful, LLM, and agent dashboards share core panels (health,
request rate, latency) while adding domain-specific panels on top.

Usage::

    from ml_platform.monitoring.dashboards import generate_dashboard
    from ml_platform.config import ServiceConfig

    config = ServiceConfig(service_name="my-agent")
    grafana = generate_dashboard(config, panel_sets=["core", "agent"])

Legacy functions :func:`generate_grafana_dashboard` and
:func:`generate_cloudwatch_dashboard` remain for backward compatibility.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ml_platform.config import ServiceConfig

_NAMESPACE = "MLPlatform"

PanelSet = Literal["core", "llm", "agent", "stateful"]


# ---------------------------------------------------------------------------
# Composable dashboard generator
# ---------------------------------------------------------------------------


def generate_dashboard(
    config: ServiceConfig,
    *,
    panel_sets: list[PanelSet] | None = None,
    region: str = "",
    namespace: str = _NAMESPACE,
) -> dict[str, Any]:
    """Generate a Grafana dashboard JSON model with composable panel sets.

    If *panel_sets* is ``None``, the sets are auto-detected from *config*:

    - ``config.agent`` present -> ``["core", "llm", "agent"]``
    - ``config.llm`` present -> ``["core", "llm"]``
    - ``config.stateful`` present -> ``["core", "stateful"]``
    - otherwise -> ``["core"]``

    Args:
        config: Platform configuration (service name used in dimensions).
        panel_sets: Explicit list of panel sets to include.
        region: AWS region for CloudWatch targets.
        namespace: CloudWatch metrics namespace.

    Returns:
        Grafana dashboard JSON model.
    """
    region = region or config.aws_region
    service_name = config.service_name

    if panel_sets is None:
        panel_sets = _auto_detect_panel_sets(config)

    panels: list[dict[str, Any]] = []
    y_offset = 0

    core = _core_panels(service_name, region, namespace, y_offset)
    panels.extend(core)
    y_offset += _height_of(core)

    if "llm" in panel_sets or "agent" in panel_sets:
        llm = _llm_panels(service_name, region, namespace, y_offset)
        panels.extend(llm)
        y_offset += _height_of(llm)

    if "agent" in panel_sets:
        agent = _agent_panels(service_name, region, namespace, y_offset)
        panels.extend(agent)
        y_offset += _height_of(agent)

    if "stateful" in panel_sets:
        sf = _stateful_panels(service_name, region, namespace, y_offset)
        panels.extend(sf)
        y_offset += _height_of(sf)

    return {
        "dashboard": {
            "title": f"{service_name} — Dashboard",
            "tags": ["ml-platform", service_name],
            "timezone": "utc",
            "refresh": "1m",
            "panels": panels,
        },
        "overwrite": True,
    }


# ---------------------------------------------------------------------------
# Panel set builders
# ---------------------------------------------------------------------------


def _cw_target(
    namespace: str,
    service_name: str,
    metric_name: str,
    region: str,
    stat: str = "Average",
) -> dict[str, Any]:
    return {
        "type": "cloudwatch",
        "namespace": namespace,
        "metricName": metric_name,
        "dimensions": {"service": [service_name]},
        "statistics": [stat],
        "period": "60",
        "region": region,
    }


def _panel(
    title: str,
    panel_type: str,
    targets: list[dict[str, Any]],
    *,
    x: int = 0,
    y: int = 0,
    w: int = 12,
    h: int = 8,
) -> dict[str, Any]:
    return {
        "title": title,
        "type": panel_type,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": targets,
    }


def _core_panels(
    svc: str, region: str, ns: str, y0: int
) -> list[dict[str, Any]]:
    t = lambda m, s="Average": _cw_target(ns, svc, m, region, s)
    return [
        _panel("Request Rate", "timeseries", [t("requests_total", "Sum")],
               x=0, y=y0, w=12),
        _panel("Error Rate", "timeseries", [t("error_count", "Sum")],
               x=12, y=y0, w=12),
        _panel("Latency (ms)", "timeseries",
               [t("latency_ms", "p50"), t("latency_ms", "p95"), t("latency_ms", "p99")],
               x=0, y=y0 + 8, w=24),
    ]


def _llm_panels(
    svc: str, region: str, ns: str, y0: int
) -> list[dict[str, Any]]:
    t = lambda m, s="Average": _cw_target(ns, svc, m, region, s)
    return [
        _panel("Tokens / min", "timeseries",
               [t("total_tokens", "Sum")], x=0, y=y0, w=12),
        _panel("Cost by Model (USD)", "timeseries",
               [t("total_cost_usd", "Sum")], x=12, y=y0, w=12),
        _panel("LLM Latency by Provider", "timeseries",
               [t("llm_latency_ms", "p95")], x=0, y=y0 + 8, w=12),
        _panel("LLM Error Rate", "timeseries",
               [t("llm_error_count", "Sum")], x=12, y=y0 + 8, w=12),
    ]


def _agent_panels(
    svc: str, region: str, ns: str, y0: int
) -> list[dict[str, Any]]:
    t = lambda m, s="Average": _cw_target(ns, svc, m, region, s)
    return [
        _panel("Steps per Run", "timeseries",
               [t("steps", "Average")], x=0, y=y0, w=12),
        _panel("Tool Usage Distribution", "barchart",
               [t("tool_calls", "Sum")], x=12, y=y0, w=12),
        _panel("Multi-Model Cost Breakdown", "timeseries",
               [t("total_cost_usd", "Sum")], x=0, y=y0 + 8, w=12),
        _panel("Run Duration vs LLM Time", "timeseries",
               [t("total_latency_ms", "Average"),
                t("llm_latency_ms", "Average")], x=12, y=y0 + 8, w=12),
    ]


def _stateful_panels(
    svc: str, region: str, ns: str, y0: int
) -> list[dict[str, Any]]:
    t = lambda m, s="Average": _cw_target(ns, svc, m, region, s)
    return [
        _panel("Cumulative Reward", "timeseries",
               [t("cumulative_reward", "Maximum")], x=0, y=y0, w=12),
        _panel("Cost per Request", "timeseries",
               [t("cost_usd")], x=12, y=y0, w=12),
        _panel("Exploration Rate", "timeseries",
               [t("exploration_rate")], x=0, y=y0 + 8, w=12),
        _panel("Budget Utilisation", "gauge",
               [t("pacer_cost_ema")], x=12, y=y0 + 8, w=6),
        _panel("Prediction Latency (ms)", "timeseries",
               [t("latency_ms", "p50"), t("latency_ms", "p95"),
                t("latency_ms", "p99")], x=18, y=y0 + 8, w=6),
        _panel("Feedback Delay (s)", "timeseries",
               [t("feedback_delay_s")], x=0, y=y0 + 16, w=12),
        _panel("Prediction Error", "timeseries",
               [t("prediction_error")], x=12, y=y0 + 16, w=12),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_panel_sets(config: ServiceConfig) -> list[PanelSet]:
    sets: list[PanelSet] = ["core"]
    if config.agent is not None:
        sets.extend(["llm", "agent"])
    elif config.llm is not None:
        sets.append("llm")
    elif config.stateful is not None or config.s3_checkpoint_bucket:
        sets.append("stateful")
    return sets


def _height_of(panels: list[dict[str, Any]]) -> int:
    if not panels:
        return 0
    return max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels) - min(
        p["gridPos"]["y"] for p in panels
    )


# ---------------------------------------------------------------------------
# Legacy API (backward compatible)
# ---------------------------------------------------------------------------


def generate_grafana_dashboard(
    service_name: str,
    region: str = "us-east-1",
    namespace: str = _NAMESPACE,
) -> dict[str, Any]:
    """Generate a Grafana dashboard for a stateful bandit service.

    .. deprecated::
        Use :func:`generate_dashboard` with ``panel_sets=["core", "stateful"]``.
    """
    config = ServiceConfig(service_name=service_name, aws_region=region)
    return generate_dashboard(
        config, panel_sets=["core", "stateful"], region=region, namespace=namespace
    )


def generate_cloudwatch_dashboard(
    service_name: str,
    region: str = "us-east-1",
    namespace: str = _NAMESPACE,
) -> str:
    """Generate a CloudWatch Dashboard body JSON string.

    .. deprecated::
        Use :func:`generate_dashboard` and convert to CloudWatch format.
    """

    def _metric_widget(
        title: str, metric_name: str, stat: str = "Average", width: int = 12
    ) -> dict[str, Any]:
        return {
            "type": "metric",
            "width": width,
            "height": 6,
            "properties": {
                "title": title,
                "region": region,
                "metrics": [
                    [namespace, metric_name, "service", service_name, {"stat": stat}]
                ],
                "period": 60,
                "view": "timeSeries",
            },
        }

    widgets = [
        _metric_widget("Cumulative Reward", "cumulative_reward", "Maximum"),
        _metric_widget("Cost per Request", "cost_usd"),
        _metric_widget("Exploration Rate", "exploration_rate"),
        _metric_widget("Budget Utilisation", "pacer_cost_ema"),
        _metric_widget("Prediction Latency P95 (ms)", "latency_ms", "p95"),
        _metric_widget("Feedback Delay (s)", "feedback_delay_s"),
    ]

    return json.dumps({"widgets": widgets})
