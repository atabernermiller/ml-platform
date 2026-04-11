"""Tests for batch/bulk operations on Table."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.data import DynamoDBTable, InMemoryTable


class TestInMemoryTableBatch:
    def test_batch_put_items(self) -> None:
        table = InMemoryTable(partition_key="id")
        items = [{"id": str(i), "val": i} for i in range(10)]
        count = table.batch_put_items(items)
        assert count == 10
        assert len(table.scan()) == 10

    def test_batch_get_items(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.batch_put_items([{"id": str(i), "val": i} for i in range(5)])
        keys = [{"id": str(i)} for i in range(3)]
        results = table.batch_get_items(keys)
        assert len(results) == 3

    def test_batch_get_items_missing(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.put_item({"id": "1", "val": "a"})
        results = table.batch_get_items([{"id": "1"}, {"id": "missing"}])
        assert len(results) == 1

    def test_batch_delete_items(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.batch_put_items([{"id": str(i)} for i in range(5)])
        deleted = table.batch_delete_items([{"id": "0"}, {"id": "2"}, {"id": "4"}])
        assert deleted == 3
        assert len(table.scan()) == 2

    def test_batch_delete_missing(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.put_item({"id": "1"})
        deleted = table.batch_delete_items([{"id": "1"}, {"id": "nonexistent"}])
        assert deleted == 1


@mock_aws
class TestDynamoDBTableBatch:
    @staticmethod
    def _create_table(table_name: str = "batch-test") -> DynamoDBTable:
        dynamodb = boto3.client("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        return DynamoDBTable(
            table_name=table_name,
            partition_key="pk",
        )

    def test_batch_put_and_scan(self) -> None:
        table = self._create_table()
        items = [{"pk": f"item-{i}", "data": f"v{i}"} for i in range(30)]
        count = table.batch_put_items(items)
        assert count == 30
        all_items = table.scan()
        assert len(all_items) == 30

    def test_batch_get_items(self) -> None:
        table = self._create_table("batch-get")
        items = [{"pk": f"k-{i}", "val": i} for i in range(10)]
        table.batch_put_items(items)
        keys = [{"pk": f"k-{i}"} for i in range(5)]
        results = table.batch_get_items(keys)
        assert len(results) == 5

    def test_batch_delete_items(self) -> None:
        table = self._create_table("batch-del")
        table.batch_put_items([{"pk": f"d-{i}"} for i in range(8)])
        deleted = table.batch_delete_items([{"pk": f"d-{i}"} for i in range(4)])
        assert deleted == 4
        remaining = table.scan()
        assert len(remaining) == 4
