"""Concrete profile implementations for local and AWS environments.

A :class:`~ml_platform._interfaces.Profile` bundles all backend
implementations for a deployment target.  Pick a profile and the
library wires everything automatically.

- ``LocalProfile`` routes all calls to in-memory / filesystem backends.
- ``AWSProfile`` routes all calls to real AWS services.

Usage::

    from ml_platform.profiles import LocalProfile, AWSProfile
    from ml_platform.config import ServiceConfig

    config = ServiceConfig(service_name="my-svc", profile=LocalProfile())
    # or
    config = ServiceConfig(service_name="my-svc", profile=AWSProfile())
"""

from __future__ import annotations

import logging
from typing import Any

from ml_platform._interfaces import (
    ContextStore,
    ConversationStore,
    MetricsBackend,
    StateManager,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LocalProfile",
    "AWSProfile",
    "ConsoleMetricsBackend",
]


class ConsoleMetricsBackend(MetricsBackend):
    """Metrics backend that logs to the console.

    Useful for local development when CloudWatch is not available.
    Metrics are logged at DEBUG level.
    """

    def __init__(self, service_name: str) -> None:
        self._service_name = service_name

    def emit(self, metrics: dict[str, float]) -> None:
        if metrics:
            logger.debug("[%s] metrics: %s", self._service_name, metrics)

    def emit_event(
        self,
        event_name: str,
        dimensions: dict[str, str],
        values: dict[str, float],
    ) -> None:
        if values:
            logger.debug(
                "[%s] event=%s dims=%s values=%s",
                self._service_name,
                event_name,
                dimensions,
                values,
            )


class LocalProfile:
    """Profile for local development with in-memory / filesystem backends.

    Uses console metrics, in-memory stores, and no state checkpointing.
    No AWS credentials required.

    Args:
        base_dir: Base directory for file-system state storage. If empty,
            a temporary directory is used.
    """

    def __init__(self, base_dir: str = "") -> None:
        self._base_dir = base_dir

    def create_metrics_backend(
        self, service_name: str, region: str
    ) -> MetricsBackend:
        return ConsoleMetricsBackend(service_name)

    def create_state_manager(
        self, bucket: str, prefix: str, region: str
    ) -> StateManager | None:
        return None

    def create_context_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ContextStore:
        from ml_platform.serving.context_store import InMemoryContextStore
        return InMemoryContextStore()

    def create_conversation_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ConversationStore:
        from ml_platform.serving.conversation_store import InMemoryConversationStore
        return InMemoryConversationStore()


class AWSProfile:
    """Profile for AWS deployment with real service backends.

    Uses CloudWatch EMF metrics, DynamoDB stores, and S3 state management.
    Requires appropriate AWS credentials and IAM permissions.
    """

    def create_metrics_backend(
        self, service_name: str, region: str
    ) -> MetricsBackend:
        from ml_platform.monitoring.metrics import MetricsEmitter

        class _EMFBackend(MetricsBackend):
            def __init__(self, emitter: Any) -> None:
                self._emitter = emitter

            def emit(self, metrics: dict[str, float]) -> None:
                self._emitter.emit(metrics)

            def emit_event(
                self,
                event_name: str,
                dimensions: dict[str, str],
                values: dict[str, float],
            ) -> None:
                self._emitter.emit_event(event_name, dimensions, values)

        return _EMFBackend(MetricsEmitter(service_name, region))

    def create_state_manager(
        self, bucket: str, prefix: str, region: str
    ) -> StateManager | None:
        if not bucket:
            return None
        from ml_platform.serving.state_manager import S3StateManager
        return S3StateManager(bucket, prefix, region)

    def create_context_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ContextStore:
        from ml_platform.serving.context_store import DynamoDBContextStore
        return DynamoDBContextStore(table_name, region, ttl_s)

    def create_conversation_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ConversationStore:
        from ml_platform.serving.conversation_store import DynamoDBConversationStore
        return DynamoDBConversationStore(table_name, region, ttl_s)
