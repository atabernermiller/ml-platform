"""Context stores for correlating predictions with delayed feedback.

When a service calls ``predict()``, it stores the feature context keyed by
``request_id``.  When ``process_feedback()`` arrives later, the context is
retrieved so the model can learn from the (context, reward) pair.

Two backends are provided:

- ``DynamoDBContextStore`` -- serverless, pay-per-request, suitable for most
  workloads.
- ``InMemoryContextStore`` -- for local development and testing.
"""

from __future__ import annotations

import collections
import json
import logging
import time
from typing import Any

from ml_platform._interfaces import ContextStore

logger = logging.getLogger(__name__)

__all__ = [
    "ContextStore",
    "DynamoDBContextStore",
    "InMemoryContextStore",
]


class DynamoDBContextStore(ContextStore):
    """DynamoDB-backed context store with automatic TTL expiry.

    Requires a DynamoDB table with:

    - Partition key: ``request_id`` (String)
    - TTL attribute: ``ttl`` (Number)

    Enable DynamoDB TTL on the ``ttl`` attribute for automatic cleanup.

    AWS credentials are resolved via boto3's default credential chain
    (env vars, ``~/.aws/credentials``, ECS task role, EC2 instance
    profile).  No explicit keys are accepted.

    Required IAM permissions::

        dynamodb:PutItem    – on the context table
        dynamodb:DeleteItem – on the context table

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_s: Time-to-live for stored contexts in seconds.
    """

    def __init__(
        self, table_name: str, region: str = "us-east-1", ttl_s: int = 86_400
    ) -> None:
        import boto3

        self._table_name = table_name
        self._ttl_s = ttl_s
        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = dynamodb.Table(table_name)

    def put(self, request_id: str, context: dict[str, Any]) -> None:
        """Store context with TTL."""
        item = {
            "request_id": request_id,
            "context": json.dumps(context, default=str),
            "ttl": int(time.time()) + self._ttl_s,
        }
        self._table.put_item(Item=item)
        logger.debug("Stored context for request_id=%s", request_id)

    def get(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve and delete context (consume-once)."""
        response = self._table.delete_item(
            Key={"request_id": request_id},
            ReturnValues="ALL_OLD",
        )
        item = response.get("Attributes")
        if not item:
            logger.debug("No context found for request_id=%s", request_id)
            return None

        logger.debug("Retrieved context for request_id=%s", request_id)
        return json.loads(item["context"])


class InMemoryContextStore(ContextStore):
    """In-memory context store for local development and testing.

    Entries are evicted on an LRU basis once ``maxlen`` is reached.

    Args:
        maxlen: Maximum number of entries to retain.  Oldest entries are
            discarded when the limit is exceeded.
    """

    def __init__(self, maxlen: int = 100_000) -> None:
        self._store: collections.OrderedDict[str, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._maxlen = maxlen

    def put(self, request_id: str, context: dict[str, Any]) -> None:
        if request_id in self._store:
            self._store.move_to_end(request_id)
        else:
            if len(self._store) >= self._maxlen:
                self._store.popitem(last=False)
        self._store[request_id] = context

    def get(self, request_id: str) -> dict[str, Any] | None:
        return self._store.pop(request_id, None)
