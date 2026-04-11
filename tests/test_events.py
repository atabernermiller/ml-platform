"""Tests for EventBridge integration."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.events import EventBridgeBus, InMemoryEventBus


class TestInMemoryEventBus:
    def test_publish_records_event(self) -> None:
        bus = InMemoryEventBus()
        eid = bus.publish("orders", "OrderCreated", {"order_id": "1"})
        assert isinstance(eid, str)
        assert len(bus.published) == 1
        assert bus.published[0]["detail"]["order_id"] == "1"
        assert bus.published[0]["detail_type"] == "OrderCreated"
        assert bus.published[0]["source"] == "orders"

    def test_subscribe_receives_events(self) -> None:
        bus = InMemoryEventBus()
        received: list[dict[str, Any]] = []
        bus.subscribe("OrderCreated", received.append)
        bus.publish("orders", "OrderCreated", {"order_id": "1"})
        bus.publish("orders", "OrderShipped", {"order_id": "1"})
        assert len(received) == 1
        assert received[0]["detail_type"] == "OrderCreated"

    def test_wildcard_subscriber(self) -> None:
        bus = InMemoryEventBus()
        received: list[dict[str, Any]] = []
        bus.subscribe("*", received.append)
        bus.publish("a", "TypeA", {"x": 1})
        bus.publish("b", "TypeB", {"y": 2})
        assert len(received) == 2

    def test_publish_batch(self) -> None:
        bus = InMemoryEventBus()
        entries = [
            {"source": "svc", "detail_type": "A", "detail": {"n": 1}},
            {"source": "svc", "detail_type": "B", "detail": {"n": 2}},
            {"source": "svc", "detail_type": "C", "detail": {"n": 3}},
        ]
        ids = bus.publish_batch(entries)
        assert len(ids) == 3
        assert len(bus.published) == 3

    def test_multiple_subscribers(self) -> None:
        bus = InMemoryEventBus()
        r1: list[dict[str, Any]] = []
        r2: list[dict[str, Any]] = []
        bus.subscribe("Evt", r1.append)
        bus.subscribe("Evt", r2.append)
        bus.publish("src", "Evt", {"k": "v"})
        assert len(r1) == 1
        assert len(r2) == 1


@mock_aws
class TestEventBridgeBus:
    def test_publish_event(self) -> None:
        client = boto3.client("events", region_name="us-east-1")
        client.create_event_bus(Name="test-bus")
        bus = EventBridgeBus(bus_name="test-bus")
        eid = bus.publish("orders", "OrderCreated", {"order_id": "o-1"})
        assert isinstance(eid, str)

    def test_publish_batch(self) -> None:
        client = boto3.client("events", region_name="us-east-1")
        client.create_event_bus(Name="batch-bus")
        bus = EventBridgeBus(bus_name="batch-bus")
        entries = [
            {"source": "svc", "detail_type": "A", "detail": {"n": i}}
            for i in range(5)
        ]
        ids = bus.publish_batch(entries)
        assert len(ids) == 5

    def test_publish_to_default_bus(self) -> None:
        bus = EventBridgeBus(bus_name="default")
        eid = bus.publish("test", "TestEvent", {"key": "val"})
        assert isinstance(eid, str)
