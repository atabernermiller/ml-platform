"""EventBridge integration for pub/sub event-driven architectures.

Complements :mod:`ml_platform.queue` (point-to-point SQS) with fan-out
event publishing, pattern-based routing, and scheduled rules.

Two backends are provided:

- ``EventBridgeBus`` -- publishes to an AWS EventBridge event bus.
- ``InMemoryEventBus`` -- local-development bus that dispatches events
  to registered handler callables in-process.

Usage::

    from ml_platform.events import EventBridgeBus, InMemoryEventBus

    bus = EventBridgeBus(bus_name="my-app-events")
    bus.publish(
        source="orders.service",
        detail_type="OrderCreated",
        detail={"order_id": "o-123", "total": 99.99},
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

from ml_platform._interfaces import EventBus
from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_events.client import EventBridgeClient

logger = logging.getLogger(__name__)

__all__ = [
    "EventBridgeBus",
    "InMemoryEventBus",
]

EventHandler = Callable[[dict[str, Any]], None]


class EventBridgeBus(EventBus):
    """AWS EventBridge-backed event bus.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        events:PutEvents – on the event bus ARN

    Args:
        bus_name: EventBridge event bus name. Use ``"default"`` for the
            account-default bus.
        region: AWS region.
    """

    def __init__(
        self,
        bus_name: str = "default",
        region: str | None = None,
    ) -> None:
        import boto3

        self._bus_name = bus_name
        self._client: EventBridgeClient = boto3.client(
            "events", region_name=resolve_region(region)
        )

    def publish(
        self,
        source: str,
        detail_type: str,
        detail: dict[str, Any],
    ) -> str:
        response = self._client.put_events(
            Entries=[
                {
                    "Source": source,
                    "DetailType": detail_type,
                    "Detail": json.dumps(detail, default=str),
                    "EventBusName": self._bus_name,
                }
            ]
        )
        entry = response["Entries"][0]
        event_id: str = entry.get("EventId", "")
        if entry.get("ErrorCode"):
            logger.error(
                "EventBridge publish failed: %s – %s",
                entry["ErrorCode"],
                entry.get("ErrorMessage", ""),
            )
        else:
            logger.debug("Published event %s to %s", event_id, self._bus_name)
        return event_id

    def publish_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        eb_entries = [
            {
                "Source": e["source"],
                "DetailType": e["detail_type"],
                "Detail": json.dumps(e["detail"], default=str),
                "EventBusName": self._bus_name,
            }
            for e in entries
        ]
        ids: list[str] = []
        for i in range(0, len(eb_entries), 10):
            batch = eb_entries[i : i + 10]
            response = self._client.put_events(Entries=batch)
            for entry in response["Entries"]:
                ids.append(entry.get("EventId", ""))
        logger.debug("Published batch of %d events to %s", len(entries), self._bus_name)
        return ids


class InMemoryEventBus(EventBus):
    """In-memory event bus for local development and testing.

    Supports synchronous handler registration so tests can assert on
    events without AWS credentials.  Handlers are registered per
    ``detail_type`` (or ``"*"`` for all events).

    Usage::

        bus = InMemoryEventBus()
        received = []
        bus.subscribe("OrderCreated", received.append)
        bus.publish("orders", "OrderCreated", {"order_id": "1"})
        assert len(received) == 1
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self.published: list[dict[str, Any]] = []

    def subscribe(self, detail_type: str, handler: EventHandler) -> None:
        """Register a handler for a specific event type.

        Args:
            detail_type: Event type to subscribe to, or ``"*"`` for all.
            handler: Callable receiving the full event dict.
        """
        self._handlers[detail_type].append(handler)

    def publish(
        self,
        source: str,
        detail_type: str,
        detail: dict[str, Any],
    ) -> str:
        event_id = uuid.uuid4().hex
        event = {
            "event_id": event_id,
            "source": source,
            "detail_type": detail_type,
            "detail": detail,
        }
        self.published.append(event)
        for handler in self._handlers.get(detail_type, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)
        return event_id

    def publish_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        ids: list[str] = []
        for entry in entries:
            eid = self.publish(
                source=entry["source"],
                detail_type=entry["detail_type"],
                detail=entry["detail"],
            )
            ids.append(eid)
        return ids
