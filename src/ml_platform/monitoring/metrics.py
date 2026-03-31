"""CloudWatch metric emission using the Embedded Metric Format (EMF).

EMF lets you emit structured metric data via CloudWatch Logs that CloudWatch
automatically extracts into metrics -- no ``put_metric_data`` API calls, no
custom aggregation, and no per-metric cost at low cardinality.

The ``MetricsEmitter`` also supports direct ``put_metric_data`` for cases
where you need immediate metric availability (EMF has ~1-minute ingestion
delay).

Reference: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format.html
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_EMF_NAMESPACE = "MLPlatform"


class MetricsEmitter:
    """Emit ML service metrics to CloudWatch.

    Supports two emission modes:

    1. **EMF (default)**: Writes structured JSON to stdout / CloudWatch Logs.
       CloudWatch extracts metrics automatically. Zero API calls, low cost.
       Requires only standard CloudWatch Logs permissions (granted by default
       to ECS task roles).
    2. **Direct PutMetricData**: For immediate availability. Higher cost at
       scale; use sparingly.  Requires ``cloudwatch:PutMetricData`` IAM
       permission (scoped to the ``MLPlatform`` namespace).

    AWS credentials are resolved via boto3's default credential chain
    (env vars, ``~/.aws/credentials``, ECS task role, EC2 instance
    profile).  No explicit keys are accepted.  The boto3 client is
    created lazily on first ``emit_direct()`` call.

    Args:
        service_name: Service identifier added as a CloudWatch dimension.
        region: AWS region for the CloudWatch client (direct mode only).
        namespace: CloudWatch namespace for all metrics.
    """

    def __init__(
        self,
        service_name: str,
        region: str = "us-east-1",
        namespace: str = _EMF_NAMESPACE,
    ) -> None:
        self._service_name = service_name
        self._region = region
        self._namespace = namespace
        self._cw_client: Any | None = None

    def _get_cw_client(self) -> Any:
        if self._cw_client is None:
            import boto3

            self._cw_client = boto3.client("cloudwatch", region_name=self._region)
        return self._cw_client

    def emit(self, metrics: dict[str, float]) -> None:
        """Emit a batch of metrics via EMF (structured log line).

        Each key-value pair becomes a separate CloudWatch metric under the
        configured namespace, with ``service`` as a dimension.

        Args:
            metrics: Mapping of metric names to numeric values.
        """
        if not metrics:
            return

        emf_payload = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self._namespace,
                        "Dimensions": [["service"]],
                        "Metrics": [
                            {"Name": name, "Unit": "None"} for name in metrics
                        ],
                    }
                ],
            },
            "service": self._service_name,
            **metrics,
        }
        print(json.dumps(emf_payload), flush=True)
        logger.debug("Emitted %d EMF metrics for %s", len(metrics), self._service_name)

    def emit_event(
        self,
        event_name: str,
        dimensions: dict[str, str],
        values: dict[str, float],
    ) -> None:
        """Emit a single event with custom dimensions via EMF.

        Use this for per-request metrics (e.g., prediction latency, cost) where
        you need dimensions beyond just ``service``.

        Args:
            event_name: Logical event name (used for log categorisation).
            dimensions: CloudWatch dimensions (keys and values).
            values: Metric values to emit.
        """
        if not values:
            return

        all_dims = {"service": self._service_name, "event": event_name, **dimensions}
        dim_keys = list(all_dims.keys())

        emf_payload = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self._namespace,
                        "Dimensions": [dim_keys],
                        "Metrics": [
                            {"Name": name, "Unit": "None"} for name in values
                        ],
                    }
                ],
            },
            **all_dims,
            **values,
        }
        print(json.dumps(emf_payload), flush=True)

    def emit_direct(self, metrics: dict[str, float]) -> None:
        """Emit metrics via the CloudWatch PutMetricData API.

        Use when you need metrics available immediately (e.g., for alarms
        that must fire within seconds). More expensive at scale than EMF.

        Args:
            metrics: Mapping of metric names to numeric values.
        """
        if not metrics:
            return

        client = self._get_cw_client()
        metric_data = [
            {
                "MetricName": name,
                "Value": value,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "service", "Value": self._service_name},
                ],
            }
            for name, value in metrics.items()
        ]

        for i in range(0, len(metric_data), 25):
            batch = metric_data[i : i + 25]
            client.put_metric_data(Namespace=self._namespace, MetricData=batch)

        logger.debug(
            "Emitted %d direct metrics for %s", len(metrics), self._service_name
        )
