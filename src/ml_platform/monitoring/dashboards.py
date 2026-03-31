"""Dashboard configuration generators for Grafana and CloudWatch.

These functions produce JSON structures that can be imported into
Amazon Managed Grafana or deployed as CloudWatch Dashboard resources
via the AWS API / CDK.

Usage::

    from ml_platform.monitoring.dashboards import (
        generate_grafana_dashboard,
        generate_cloudwatch_dashboard,
    )

    grafana_json = generate_grafana_dashboard("my-service", "us-east-1")
    cw_body = generate_cloudwatch_dashboard("my-service", "us-east-1")
"""

from __future__ import annotations

import json
from typing import Any

_NAMESPACE = "MLPlatform"


def generate_grafana_dashboard(
    service_name: str,
    region: str = "us-east-1",
    namespace: str = _NAMESPACE,
) -> dict[str, Any]:
    """Generate a Grafana dashboard JSON model for an ML service.

    The dashboard includes panels for:

    - Model selection distribution (bar chart)
    - Cumulative reward trend (time series)
    - Cost per request (histogram)
    - Exploration rate (time series)
    - Budget utilisation gauge
    - Prediction latency percentiles (P50/P95/P99)
    - Feedback delay distribution

    Args:
        service_name: Service name matching the CloudWatch ``service`` dimension.
        region: AWS region for the CloudWatch data source.
        namespace: CloudWatch metrics namespace.

    Returns:
        Grafana dashboard JSON model suitable for import via the Grafana API.
    """

    def _cw_target(metric_name: str, stat: str = "Average") -> dict[str, Any]:
        return {
            "type": "cloudwatch",
            "namespace": namespace,
            "metricName": metric_name,
            "dimensions": {"service": [service_name]},
            "statistics": [stat],
            "period": "60",
            "region": region,
        }

    panels: list[dict[str, Any]] = [
        {
            "title": "Cumulative Reward",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [_cw_target("cumulative_reward", "Maximum")],
        },
        {
            "title": "Cost per Request",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [_cw_target("cost_usd", "Average")],
        },
        {
            "title": "Exploration Rate",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "targets": [_cw_target("exploration_rate", "Average")],
        },
        {
            "title": "Budget Utilisation",
            "type": "gauge",
            "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
            "targets": [_cw_target("pacer_cost_ema", "Average")],
        },
        {
            "title": "Prediction Latency (ms)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
            "targets": [
                _cw_target("latency_ms", "p50"),
                _cw_target("latency_ms", "p95"),
                _cw_target("latency_ms", "p99"),
            ],
        },
        {
            "title": "Feedback Delay (s)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            "targets": [_cw_target("feedback_delay_s", "Average")],
        },
        {
            "title": "Prediction Error (actual - predicted)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            "targets": [_cw_target("prediction_error", "Average")],
        },
    ]

    return {
        "dashboard": {
            "title": f"{service_name} — ML Service Dashboard",
            "tags": ["ml-platform", service_name],
            "timezone": "utc",
            "refresh": "1m",
            "panels": panels,
        },
        "overwrite": True,
    }


def generate_cloudwatch_dashboard(
    service_name: str,
    region: str = "us-east-1",
    namespace: str = _NAMESPACE,
) -> str:
    """Generate a CloudWatch Dashboard body JSON string.

    Suitable for passing to ``cloudwatch.put_dashboard()`` or the CDK
    ``aws_cloudwatch.CfnDashboard`` construct.

    Args:
        service_name: Service name dimension value.
        region: AWS region.
        namespace: CloudWatch namespace.

    Returns:
        JSON string for the ``DashboardBody`` parameter.
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
