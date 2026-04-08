"""Tests for general-purpose data table abstraction."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from ml_platform.data import DynamoDBTable, InMemoryTable


class TestInMemoryTable:
    def test_put_and_get(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.put_item({"id": "1", "name": "Alice"})
        item = table.get_item({"id": "1"})
        assert item is not None
        assert item["name"] == "Alice"

    def test_get_missing_item(self) -> None:
        table = InMemoryTable(partition_key="id")
        assert table.get_item({"id": "missing"}) is None

    def test_put_overwrites(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.put_item({"id": "1", "name": "Alice"})
        table.put_item({"id": "1", "name": "Bob"})
        item = table.get_item({"id": "1"})
        assert item is not None
        assert item["name"] == "Bob"

    def test_delete_item(self) -> None:
        table = InMemoryTable(partition_key="id")
        table.put_item({"id": "1", "name": "Alice"})
        assert table.delete_item({"id": "1"}) is True
        assert table.get_item({"id": "1"}) is None
        assert table.delete_item({"id": "1"}) is False

    def test_query(self) -> None:
        table = InMemoryTable(partition_key="customer_id", sort_key="order_id")
        table.put_item({"customer_id": "c1", "order_id": "o1", "total": 100})
        table.put_item({"customer_id": "c1", "order_id": "o2", "total": 200})
        table.put_item({"customer_id": "c2", "order_id": "o3", "total": 300})

        results = table.query("customer_id", "c1")
        assert len(results) == 2
        assert all(r["customer_id"] == "c1" for r in results)

    def test_query_with_sort_key_condition(self) -> None:
        table = InMemoryTable(partition_key="pk", sort_key="sk")
        table.put_item({"pk": "a", "sk": "2024-01-01"})
        table.put_item({"pk": "a", "sk": "2024-06-01"})
        table.put_item({"pk": "a", "sk": "2025-01-01"})

        results = table.query(
            "pk", "a",
            sort_key_condition=("sk", "gte", "2024-06-01"),
        )
        assert len(results) == 2

    def test_query_begins_with(self) -> None:
        table = InMemoryTable(partition_key="pk", sort_key="sk")
        table.put_item({"pk": "a", "sk": "order#1"})
        table.put_item({"pk": "a", "sk": "order#2"})
        table.put_item({"pk": "a", "sk": "refund#1"})

        results = table.query(
            "pk", "a",
            sort_key_condition=("sk", "begins_with", "order"),
        )
        assert len(results) == 2

    def test_query_with_limit(self) -> None:
        table = InMemoryTable(partition_key="pk", sort_key="sk")
        for i in range(10):
            table.put_item({"pk": "a", "sk": f"item-{i:02d}"})
        results = table.query("pk", "a", limit=3)
        assert len(results) == 3

    def test_scan(self) -> None:
        table = InMemoryTable(partition_key="id")
        for i in range(5):
            table.put_item({"id": str(i), "val": i})
        all_items = table.scan()
        assert len(all_items) == 5

    def test_scan_with_limit(self) -> None:
        table = InMemoryTable(partition_key="id")
        for i in range(10):
            table.put_item({"id": str(i)})
        assert len(table.scan(limit=3)) == 3

    def test_deep_copy_isolation(self) -> None:
        table = InMemoryTable(partition_key="id")
        original = {"id": "1", "data": [1, 2, 3]}
        table.put_item(original)
        original["data"].append(4)
        item = table.get_item({"id": "1"})
        assert item is not None
        assert item["data"] == [1, 2, 3]


@mock_aws
class TestDynamoDBTable:
    @staticmethod
    def _create_table(table_name: str = "test-data") -> DynamoDBTable:
        dynamodb = boto3.client("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        return DynamoDBTable(
            table_name=table_name,
            partition_key="pk",
            sort_key="sk",
            region="us-east-1",
        )

    def test_put_and_get(self) -> None:
        table = self._create_table()
        table.put_item({"pk": "customer-1", "sk": "profile", "name": "Alice"})
        item = table.get_item({"pk": "customer-1", "sk": "profile"})
        assert item is not None
        assert item["name"] == "Alice"

    def test_get_missing(self) -> None:
        table = self._create_table()
        assert table.get_item({"pk": "missing", "sk": "none"}) is None

    def test_delete_item(self) -> None:
        table = self._create_table()
        table.put_item({"pk": "c1", "sk": "s1", "data": "val"})
        assert table.delete_item({"pk": "c1", "sk": "s1"}) is True
        assert table.get_item({"pk": "c1", "sk": "s1"}) is None

    def test_query(self) -> None:
        table = self._create_table()
        table.put_item({"pk": "c1", "sk": "order#1", "total": 100})
        table.put_item({"pk": "c1", "sk": "order#2", "total": 200})
        table.put_item({"pk": "c2", "sk": "order#3", "total": 300})

        results = table.query("pk", "c1")
        assert len(results) == 2

    def test_query_begins_with(self) -> None:
        table = self._create_table()
        table.put_item({"pk": "c1", "sk": "order#1"})
        table.put_item({"pk": "c1", "sk": "order#2"})
        table.put_item({"pk": "c1", "sk": "refund#1"})

        results = table.query(
            "pk", "c1",
            sort_key_condition=("sk", "begins_with", "order"),
        )
        assert len(results) == 2

    def test_scan(self) -> None:
        table = self._create_table()
        for i in range(5):
            table.put_item({"pk": f"item-{i}", "sk": "data"})
        items = table.scan()
        assert len(items) == 5
