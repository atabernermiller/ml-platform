"""Conversation stores for multi-turn session history.

When a user sends a message, the store appends it.  Before calling the
LLM, the app retrieves the history (optionally windowed by message count
or token budget) to include as context.

Two backends are provided:

- ``DynamoDBConversationStore`` -- serverless, pay-per-request, suitable
  for production.
- ``InMemoryConversationStore`` -- for local development and testing.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from decimal import Decimal
from typing import Any

from ml_platform._types import Message

logger = logging.getLogger(__name__)


class ConversationStore(ABC):
    """Abstract interface for append-only message history with windowing."""

    @abstractmethod
    def append(self, session_id: str, message: Message) -> None:
        """Append a message to a session's history.

        Args:
            session_id: Session identifier.
            message: Message to append.
        """
        ...

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        """Retrieve conversation history for a session.

        Args:
            session_id: Session identifier.
            max_messages: Return at most this many recent messages.
            max_tokens: Approximate token budget; truncate oldest messages
                to fit.  Uses a rough ``len(content) / 4`` heuristic when
                no tokenizer is available.

        Returns:
            Ordered list of messages (oldest first).
        """
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all messages for a session.

        Args:
            session_id: Session identifier.
        """
        ...


class DynamoDBConversationStore(ConversationStore):
    """DynamoDB-backed conversation store with automatic TTL expiry.

    Table schema::

        Partition key: ``session_id`` (String)
        Sort key:      ``message_idx`` (Number)
        TTL attribute: ``ttl`` (Number)

    Each message is stored as a separate item, enabling efficient
    range queries and automatic cleanup via DynamoDB TTL.

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_s: Time-to-live for messages in seconds (default: 7 days).
    """

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        ttl_s: int = 604_800,
    ) -> None:
        import boto3

        self._table_name = table_name
        self._ttl_s = ttl_s
        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = dynamodb.Table(table_name)

    def append(self, session_id: str, message: Message) -> None:
        """Append a message, auto-assigning the next index."""
        next_idx = self._next_index(session_id)
        item: dict[str, Any] = {
            "session_id": session_id,
            "message_idx": next_idx,
            "role": message.role,
            "content": message.content,
            "ttl": int(time.time()) + self._ttl_s,
        }
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        self._table.put_item(Item=item)
        logger.debug(
            "Appended message %d to session %s", next_idx, session_id
        )

    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        """Query messages in order, applying window constraints."""
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        items = response.get("Items", [])

        messages = [
            Message(
                role=item["role"],
                content=item.get("content", ""),
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
            )
            for item in items
        ]

        if max_messages is not None:
            messages = messages[-max_messages:]

        if max_tokens is not None:
            messages = _apply_token_window(messages, max_tokens)

        return messages

    def clear(self, session_id: str) -> None:
        """Delete all messages for a session."""
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ProjectionExpression="session_id, message_idx",
        )
        with self._table.batch_writer() as batch:
            for item in response.get("Items", []):
                batch.delete_item(
                    Key={
                        "session_id": item["session_id"],
                        "message_idx": item["message_idx"],
                    }
                )
        logger.debug("Cleared session %s", session_id)

    def _next_index(self, session_id: str) -> int:
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,
            Limit=1,
            ProjectionExpression="message_idx",
        )
        items = response.get("Items", [])
        if not items:
            return 0
        last_idx = items[0]["message_idx"]
        if isinstance(last_idx, Decimal):
            last_idx = int(last_idx)
        return last_idx + 1


class InMemoryConversationStore(ConversationStore):
    """In-memory conversation store for local development and testing.

    Sessions are never expired; the store grows without bound.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)

    def append(self, session_id: str, message: Message) -> None:
        self._sessions[session_id].append(message)

    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        messages = list(self._sessions[session_id])
        if max_messages is not None:
            messages = messages[-max_messages:]
        if max_tokens is not None:
            messages = _apply_token_window(messages, max_tokens)
        return messages

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _apply_token_window(
    messages: list[Message], max_tokens: int
) -> list[Message]:
    """Keep the most recent messages that fit within a token budget.

    Uses a rough ``len(content) / 4`` heuristic for token estimation.
    """
    result: list[Message] = []
    budget = max_tokens
    for msg in reversed(messages):
        estimated = len(msg.content) // 4 + 1
        if budget - estimated < 0 and result:
            break
        result.append(msg)
        budget -= estimated
    result.reverse()
    return result
