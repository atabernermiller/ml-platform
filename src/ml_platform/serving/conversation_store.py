"""Conversation stores for multi-turn session history.

When a user sends a message, the store appends it.  Before calling the
LLM, the app retrieves the history (optionally windowed by message count
or token budget) to include as context.

Two backends are provided:

- ``DynamoDBConversationStore`` -- serverless, pay-per-request, suitable
  for production.
- ``InMemoryConversationStore`` -- for local development and testing
  (bounded by ``max_sessions`` to prevent memory leaks).
"""

from __future__ import annotations

import collections
import json
import logging
import time
from typing import Any, Callable

from ml_platform._interfaces import ConversationStore
from ml_platform._types import Message

logger = logging.getLogger(__name__)

TokenizerFn = Callable[[str], int]
"""Signature for a pluggable tokenizer: ``(text) -> token_count``."""

_DEFAULT_CHARS_PER_TOKEN = 4

__all__ = [
    "ConversationStore",
    "DynamoDBConversationStore",
    "InMemoryConversationStore",
    "TokenizerFn",
]


def _default_token_estimate(text: str) -> int:
    """Rough ``len / 4`` heuristic used when no tokenizer is provided."""
    return len(text) // _DEFAULT_CHARS_PER_TOKEN + 1


class DynamoDBConversationStore(ConversationStore):
    """DynamoDB-backed conversation store with automatic TTL expiry.

    Table schema::

        Partition key: ``session_id`` (String)
        Sort key:      ``ts_ns``      (Number)  — ``time.time_ns()``
        TTL attribute: ``ttl``         (Number)

    Using nanosecond timestamps as the sort key eliminates the need to
    query for the current max index before each append (the N+1 problem).

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        ttl_s: Time-to-live for messages in seconds (default: 7 days).
        tokenizer: Optional tokenizer callable for accurate windowing.
    """

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        ttl_s: int = 604_800,
        tokenizer: TokenizerFn | None = None,
    ) -> None:
        import boto3

        self._table_name = table_name
        self._ttl_s = ttl_s
        self._tokenizer = tokenizer or _default_token_estimate
        dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = dynamodb.Table(table_name)

    def append(self, session_id: str, message: Message) -> None:
        """Append a message using ``time.time_ns()`` as the sort key."""
        ts_ns = time.time_ns()
        item: dict[str, Any] = {
            "session_id": session_id,
            "ts_ns": ts_ns,
            "role": message.role,
            "content": message.content,
            "ttl": int(time.time()) + self._ttl_s,
        }
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        self._table.put_item(Item=item)
        logger.debug("Appended message ts=%d to session %s", ts_ns, session_id)

    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        """Query messages with full pagination, applying window constraints.

        When ``max_messages`` is specified, the query uses a reverse scan
        with ``Limit`` to fetch only what is needed. Otherwise, it pages
        forward through all results.
        """
        from boto3.dynamodb.conditions import Key

        if max_messages is not None and max_tokens is None:
            items = self._query_reverse(session_id, limit=max_messages)
            items.reverse()
        else:
            items = self._query_forward_all(session_id)

        messages = [
            Message(
                role=item["role"],
                content=item.get("content", ""),
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
            )
            for item in items
        ]

        if max_messages is not None and max_tokens is not None:
            messages = messages[-max_messages:]

        if max_tokens is not None:
            messages = _apply_token_window(messages, max_tokens, self._tokenizer)

        return messages

    def clear(self, session_id: str) -> None:
        """Delete all messages for a session with full pagination."""
        from boto3.dynamodb.conditions import Key

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("session_id").eq(session_id),
            "ProjectionExpression": "session_id, ts_ns",
        }
        while True:
            response = self._table.query(**kwargs)
            items = response.get("Items", [])
            if items:
                with self._table.batch_writer() as batch:
                    for item in items:
                        batch.delete_item(
                            Key={
                                "session_id": item["session_id"],
                                "ts_ns": item["ts_ns"],
                            }
                        )
            if "LastEvaluatedKey" not in response:
                break
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        logger.debug("Cleared session %s", session_id)

    def _query_forward_all(self, session_id: str) -> list[dict[str, Any]]:
        """Page through all items for a session in ascending order."""
        from boto3.dynamodb.conditions import Key

        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("session_id").eq(session_id),
            "ScanIndexForward": True,
        }
        while True:
            response = self._table.query(**kwargs)
            items.extend(response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def _query_reverse(
        self, session_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Fetch the *last* ``limit`` items efficiently via reverse scan."""
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        return response.get("Items", [])


class InMemoryConversationStore(ConversationStore):
    """In-memory conversation store for local development and testing.

    Sessions are evicted on an LRU basis once ``max_sessions`` is reached.
    Individual sessions have their messages capped at ``max_messages_per_session``.

    Args:
        max_sessions: Maximum number of sessions to retain.
        max_messages_per_session: Maximum messages per session before oldest
            are dropped.
        tokenizer: Optional tokenizer callable for accurate windowing.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 10_000,
        max_messages_per_session: int = 5_000,
        tokenizer: TokenizerFn | None = None,
    ) -> None:
        self._sessions: collections.OrderedDict[str, collections.deque[Message]] = (
            collections.OrderedDict()
        )
        self._max_sessions = max_sessions
        self._max_messages = max_messages_per_session
        self._tokenizer = tokenizer or _default_token_estimate

    def append(self, session_id: str, message: Message) -> None:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = collections.deque(
                maxlen=self._max_messages
            )
        self._sessions[session_id].append(message)

    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        if session_id not in self._sessions:
            return []
        messages = list(self._sessions[session_id])
        if max_messages is not None:
            messages = messages[-max_messages:]
        if max_tokens is not None:
            messages = _apply_token_window(messages, max_tokens, self._tokenizer)
        return messages

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _apply_token_window(
    messages: list[Message],
    max_tokens: int,
    tokenizer: TokenizerFn = _default_token_estimate,
) -> list[Message]:
    """Keep the most recent messages that fit within a token budget.

    Args:
        messages: Ordered messages (oldest first).
        max_tokens: Token budget.
        tokenizer: Callable returning token count for a string.

    Returns:
        Suffix of ``messages`` fitting within the budget.
    """
    result: list[Message] = []
    budget = max_tokens
    for msg in reversed(messages):
        estimated = tokenizer(msg.content)
        if budget - estimated < 0 and result:
            break
        result.append(msg)
        budget -= estimated
    result.reverse()
    return result
