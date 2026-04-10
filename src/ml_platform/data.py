"""General-purpose DynamoDB data access with local in-memory fallback.

Unlike the single-purpose :class:`~ml_platform.serving.context_store.DynamoDBContextStore`
or :class:`~ml_platform.serving.conversation_store.DynamoDBConversationStore`,
this module provides a lightweight :class:`Table` abstraction for
standard CRUD operations with type-safe queries.

Usage::

    from ml_platform.data import DynamoDBTable, InMemoryTable

    table = DynamoDBTable(
        table_name="customers",
        partition_key="customer_id",
        region="us-east-1",
    )
    table.put_item({"customer_id": "c-123", "name": "Alice", "email": "alice@ex.com"})
    item = table.get_item({"customer_id": "c-123"})
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from ml_platform._interfaces import Table

logger = logging.getLogger(__name__)

__all__ = [
    "DynamoDBTable",
    "InMemoryTable",
]


class DynamoDBTable(Table):
    """DynamoDB-backed general-purpose table.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        dynamodb:PutItem    – on the table
        dynamodb:GetItem    – on the table
        dynamodb:DeleteItem – on the table
        dynamodb:Query      – on the table
        dynamodb:Scan       – on the table

    Args:
        table_name: DynamoDB table name.
        partition_key: Name of the partition key attribute.
        sort_key: Name of the sort key attribute (if applicable).
        region: AWS region.
    """

    def __init__(
        self,
        table_name: str,
        partition_key: str,
        sort_key: str = "",
        region: str = "us-east-1",
    ) -> None:
        import boto3

        self._table_name = table_name
        self._partition_key = partition_key
        self._sort_key = sort_key
        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table: Any = dynamodb.Table(table_name)

    def put_item(self, item: dict[str, Any]) -> None:
        self._table.put_item(Item=item)
        logger.debug("Put item with %s=%s", self._partition_key, item.get(self._partition_key))

    def get_item(self, key: dict[str, Any]) -> dict[str, Any] | None:
        response = self._table.get_item(Key=key)
        return response.get("Item")

    def delete_item(self, key: dict[str, Any]) -> bool:
        response = self._table.delete_item(Key=key, ReturnValues="ALL_OLD")
        return bool(response.get("Attributes"))

    def query(
        self,
        partition_key: str,
        partition_value: Any,
        *,
        sort_key_condition: tuple[str, str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Key

        key_expr = Key(partition_key).eq(partition_value)

        if sort_key_condition:
            sk_attr, op, value = sort_key_condition
            sk_key = Key(sk_attr)
            op_map: dict[str, Any] = {
                "eq": sk_key.eq,
                "begins_with": sk_key.begins_with,
                "lt": sk_key.lt,
                "lte": sk_key.lte,
                "gt": sk_key.gt,
                "gte": sk_key.gte,
            }
            if op == "between":
                key_expr = key_expr & sk_key.between(*value)
            elif op in op_map:
                key_expr = key_expr & op_map[op](value)
            else:
                raise ValueError(f"Unsupported sort key operator: {op}")

        kwargs: dict[str, Any] = {"KeyConditionExpression": key_expr}
        if limit is not None:
            kwargs["Limit"] = limit

        items: list[dict[str, Any]] = []
        while True:
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))
            if limit is not None and len(items) >= limit:
                items = items[:limit]
                break
            if "LastEvaluatedKey" not in response:
                break
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def scan(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["Limit"] = limit

        items: list[dict[str, Any]] = []
        while True:
            response = self._table.scan(**kwargs)
            items.extend(response.get("Items", []))
            if limit is not None and len(items) >= limit:
                items = items[:limit]
                break
            if "LastEvaluatedKey" not in response:
                break
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def batch_put_items(self, items: list[dict[str, Any]]) -> int:
        """Write up to 25 items per batch using ``BatchWriteItem``.

        DynamoDB limits each ``BatchWriteItem`` call to 25 items.
        This method handles chunking automatically.

        Args:
            items: List of item dicts to write.

        Returns:
            Number of items successfully submitted.
        """
        written = 0
        for i in range(0, len(items), 25):
            batch = items[i : i + 25]
            with self._table.batch_writer() as writer:
                for item in batch:
                    writer.put_item(Item=item)
                    written += 1
        logger.debug("Batch wrote %d items to %s", written, self._table_name)
        return written

    def batch_get_items(self, keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Retrieve multiple items by key using ``BatchGetItem``.

        DynamoDB limits each ``BatchGetItem`` call to 100 keys.
        This method handles chunking automatically.

        Args:
            keys: List of key dicts (each containing partition key and
                optional sort key).

        Returns:
            List of found items (order not guaranteed).
        """
        import boto3

        dynamodb = boto3.resource(
            "dynamodb",
            region_name=self._table.meta.client.meta.region_name,
        )
        results: list[dict[str, Any]] = []
        for i in range(0, len(keys), 100):
            chunk = keys[i : i + 100]
            response = dynamodb.batch_get_item(
                RequestItems={self._table_name: {"Keys": chunk}}
            )
            results.extend(response.get("Responses", {}).get(self._table_name, []))
            unprocessed = response.get("UnprocessedKeys", {}).get(self._table_name)
            while unprocessed:
                response = dynamodb.batch_get_item(
                    RequestItems={self._table_name: unprocessed}
                )
                results.extend(
                    response.get("Responses", {}).get(self._table_name, [])
                )
                unprocessed = response.get("UnprocessedKeys", {}).get(self._table_name)
        logger.debug("Batch got %d items from %s", len(results), self._table_name)
        return results

    def batch_delete_items(self, keys: list[dict[str, Any]]) -> int:
        """Delete multiple items by key using ``BatchWriteItem``.

        Args:
            keys: List of key dicts.

        Returns:
            Number of delete requests submitted.
        """
        deleted = 0
        for i in range(0, len(keys), 25):
            batch = keys[i : i + 25]
            with self._table.batch_writer() as writer:
                for key in batch:
                    writer.delete_item(Key=key)
                    deleted += 1
        logger.debug("Batch deleted %d items from %s", deleted, self._table_name)
        return deleted


class InMemoryTable(Table):
    """In-memory table for local development and testing.

    Uses a dict index for O(1) get/put/delete by key, with a separate
    ordered list to preserve insertion order for scan/query.

    Args:
        partition_key: Name of the partition key attribute.
        sort_key: Name of the sort key attribute (if applicable).
    """

    def __init__(self, partition_key: str, sort_key: str = "") -> None:
        self._partition_key = partition_key
        self._sort_key = sort_key
        self._index: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _key_tuple(self, key: dict[str, Any]) -> tuple[Any, ...]:
        pk = key[self._partition_key]
        if self._sort_key and self._sort_key in key:
            return (pk, key[self._sort_key])
        return (pk,)

    def _make_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        pk = item[self._partition_key]
        if self._sort_key and self._sort_key in item:
            return (pk, item[self._sort_key])
        return (pk,)

    def put_item(self, item: dict[str, Any]) -> None:
        kt = self._make_key(item)
        self._index[kt] = copy.deepcopy(item)

    def get_item(self, key: dict[str, Any]) -> dict[str, Any] | None:
        kt = self._key_tuple(key)
        item = self._index.get(kt)
        if item is not None:
            return copy.deepcopy(item)
        return None

    def delete_item(self, key: dict[str, Any]) -> bool:
        kt = self._key_tuple(key)
        return self._index.pop(kt, None) is not None

    def query(
        self,
        partition_key: str,
        partition_value: Any,
        *,
        sort_key_condition: tuple[str, str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in self._index.values():
            if item.get(partition_key) != partition_value:
                continue
            if sort_key_condition:
                sk_attr, op, value = sort_key_condition
                sk_val = item.get(sk_attr)
                if not _check_condition(sk_val, op, value):
                    continue
            results.append(copy.deepcopy(item))
            if limit is not None and len(results) >= limit:
                break
        return results

    def scan(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        items = list(self._index.values())
        if limit is not None:
            items = items[:limit]
        return [copy.deepcopy(i) for i in items]

    def batch_put_items(self, items: list[dict[str, Any]]) -> int:
        """Write multiple items. Delegates to :meth:`put_item` in a loop.

        Args:
            items: List of item dicts to write.

        Returns:
            Number of items written.
        """
        for item in items:
            self.put_item(item)
        return len(items)

    def batch_get_items(self, keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Retrieve multiple items by key.

        Args:
            keys: List of key dicts.

        Returns:
            List of found items.
        """
        results: list[dict[str, Any]] = []
        for key in keys:
            item = self.get_item(key)
            if item is not None:
                results.append(item)
        return results

    def batch_delete_items(self, keys: list[dict[str, Any]]) -> int:
        """Delete multiple items by key.

        Args:
            keys: List of key dicts.

        Returns:
            Number of items deleted.
        """
        deleted = 0
        for key in keys:
            if self.delete_item(key):
                deleted += 1
        return deleted


def _check_condition(value: Any, op: str, target: Any) -> bool:
    """Evaluate a sort-key condition in-memory."""
    if op == "eq":
        return value == target
    if op == "lt":
        return value < target
    if op == "lte":
        return value <= target
    if op == "gt":
        return value > target
    if op == "gte":
        return value >= target
    if op == "begins_with":
        return isinstance(value, str) and value.startswith(target)
    if op == "between":
        low, high = target
        return low <= value <= high
    return False
