"""Tests for conversation store backends."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from ml_platform._types import Message
from ml_platform.serving.conversation_store import (
    DynamoDBConversationStore,
    InMemoryConversationStore,
    _apply_token_window,
)


# ---------------------------------------------------------------------------
# InMemoryConversationStore
# ---------------------------------------------------------------------------


class TestInMemoryConversationStore:
    def test_append_and_get_history(self) -> None:
        store = InMemoryConversationStore()
        store.append("s1", Message(role="user", content="hello"))
        store.append("s1", Message(role="assistant", content="hi there"))

        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_get_empty_session(self) -> None:
        store = InMemoryConversationStore()
        assert store.get_history("nonexistent") == []

    def test_max_messages_window(self) -> None:
        store = InMemoryConversationStore()
        for i in range(10):
            store.append("s1", Message(role="user", content=f"msg {i}"))

        history = store.get_history("s1", max_messages=3)
        assert len(history) == 3
        assert history[0].content == "msg 7"
        assert history[2].content == "msg 9"

    def test_max_tokens_window(self) -> None:
        store = InMemoryConversationStore()
        store.append("s1", Message(role="user", content="a" * 400))
        store.append("s1", Message(role="user", content="b" * 40))
        store.append("s1", Message(role="user", content="c" * 40))

        history = store.get_history("s1", max_tokens=30)
        assert len(history) == 2
        assert history[0].content.startswith("b")

    def test_clear(self) -> None:
        store = InMemoryConversationStore()
        store.append("s1", Message(role="user", content="hello"))
        store.clear("s1")
        assert store.get_history("s1") == []

    def test_sessions_are_isolated(self) -> None:
        store = InMemoryConversationStore()
        store.append("s1", Message(role="user", content="hello"))
        store.append("s2", Message(role="user", content="world"))
        assert len(store.get_history("s1")) == 1
        assert len(store.get_history("s2")) == 1
        assert store.get_history("s1")[0].content == "hello"


# ---------------------------------------------------------------------------
# DynamoDBConversationStore
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _create_conversation_table(table_name: str = "test-conversations") -> None:
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    ddb.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {"AttributeName": "session_id", "AttributeType": "S"},
            {"AttributeName": "message_idx", "AttributeType": "N"},
        ],
        KeySchema=[
            {"AttributeName": "session_id", "KeyType": "HASH"},
            {"AttributeName": "message_idx", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class TestDynamoDBConversationStore:
    @mock_aws
    def test_append_and_get_history(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        store.append("s1", Message(role="user", content="hello"))
        store.append("s1", Message(role="assistant", content="hi"))

        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0].content == "hello"
        assert history[1].content == "hi"

    @mock_aws
    def test_max_messages_window(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        for i in range(5):
            store.append("s1", Message(role="user", content=f"msg {i}"))

        history = store.get_history("s1", max_messages=2)
        assert len(history) == 2
        assert history[0].content == "msg 3"

    @mock_aws
    def test_clear(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        store.append("s1", Message(role="user", content="hello"))
        store.clear("s1")
        assert store.get_history("s1") == []

    @mock_aws
    def test_empty_session(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        assert store.get_history("nonexistent") == []

    @mock_aws
    def test_preserves_message_fields(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        store.append(
            "s1",
            Message(role="tool", content="result", tool_call_id="tc_123"),
        )
        history = store.get_history("s1")
        assert history[0].role == "tool"
        assert history[0].tool_call_id == "tc_123"

    @mock_aws
    def test_sessions_are_isolated(self) -> None:
        _create_conversation_table()
        store = DynamoDBConversationStore("test-conversations")
        store.append("s1", Message(role="user", content="a"))
        store.append("s2", Message(role="user", content="b"))
        assert len(store.get_history("s1")) == 1
        assert store.get_history("s1")[0].content == "a"


# ---------------------------------------------------------------------------
# Token window utility
# ---------------------------------------------------------------------------


class TestApplyTokenWindow:
    def test_keeps_recent_messages(self) -> None:
        messages = [
            Message(role="user", content="a" * 100),
            Message(role="user", content="b" * 100),
            Message(role="user", content="c" * 20),
        ]
        result = _apply_token_window(messages, max_tokens=30)
        assert len(result) == 1
        assert result[0].content.startswith("c")

    def test_empty_list(self) -> None:
        assert _apply_token_window([], max_tokens=100) == []

    def test_all_fit(self) -> None:
        messages = [
            Message(role="user", content="hello"),
            Message(role="user", content="world"),
        ]
        result = _apply_token_window(messages, max_tokens=1000)
        assert len(result) == 2
