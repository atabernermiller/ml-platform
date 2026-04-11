"""Tests for in-memory and DynamoDB context stores."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from ml_platform.serving.context_store import DynamoDBContextStore, InMemoryContextStore


def test_in_memory_put_get() -> None:
    store = InMemoryContextStore()
    ctx = {"features": [1.0, 2.0], "model": "v1"}

    store.put("req-1", ctx)
    result = store.get("req-1")
    assert result == ctx

    assert store.get("req-1") is None


def test_in_memory_lru_eviction() -> None:
    store = InMemoryContextStore(maxlen=2)
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    store.put("c", {"v": 3})

    assert store.get("a") is None
    assert store.get("b") == {"v": 2}
    assert store.get("c") == {"v": 3}


def test_dynamodb_put_get(mock_dynamodb: None) -> None:
    store = DynamoDBContextStore(
        table_name="test-context",
        ttl_s=3600,
    )
    ctx = {"features": [1.0, 2.0], "model": "v1"}

    store.put("req-dynamo-1", ctx)
    result = store.get("req-dynamo-1")
    assert result == ctx

    # Verify the item was actually consumed (deleted) by confirming the
    # raw DynamoDB table no longer contains it.
    table = boto3.resource("dynamodb", region_name="us-east-1").Table("test-context")
    raw = table.get_item(Key={"request_id": "req-dynamo-1"})
    assert "Item" not in raw


@mock_aws
def test_dynamodb_get_missing_key() -> None:
    """Verify that ``get`` on a key that was never stored returns ``None``."""
    dynamodb = boto3.client("dynamodb", region_name="us-east-1")
    dynamodb.create_table(
        TableName="missing-key-test",
        KeySchema=[{"AttributeName": "request_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "request_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    store = DynamoDBContextStore(
        table_name="missing-key-test",
        ttl_s=3600,
    )
    assert store.get("never-stored") is None
