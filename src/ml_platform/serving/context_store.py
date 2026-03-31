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

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ContextStore(ABC):
    """Abstract interface for context persistence between predict and feedback."""

    @abstractmethod
    def put(self, request_id: str, context: dict[str, Any]) -> None:
        """Store context for a prediction request.

        Args:
            request_id: Unique prediction identifier.
            context: Serialisable context data to store.
        """
        ...

    @abstractmethod
    def get(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve and delete context for a request.

        This is a consume-once operation; the entry is removed after retrieval
        to avoid unbounded storage growth.

        Args:
            request_id: Unique prediction identifier.

        Returns:
            Stored context, or ``None`` if the key does not exist or has expired.
        """
        ...


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

    Entries are never expired; the store grows without bound.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def put(self, request_id: str, context: dict[str, Any]) -> None:
        self._store[request_id] = context

    def get(self, request_id: str) -> dict[str, Any] | None:
        return self._store.pop(request_id, None)
