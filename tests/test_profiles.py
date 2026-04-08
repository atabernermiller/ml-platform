"""Tests for cloud profile implementations."""

from __future__ import annotations

from ml_platform._interfaces import ContextStore, ConversationStore, MetricsBackend
from ml_platform.profiles import AWSProfile, ConsoleMetricsBackend, LocalProfile


class TestConsoleMetricsBackend:
    def test_emit(self) -> None:
        backend = ConsoleMetricsBackend("test-svc")
        backend.emit({"latency": 10.5, "requests": 42})

    def test_emit_event(self) -> None:
        backend = ConsoleMetricsBackend("test-svc")
        backend.emit_event("predict", {"model": "v1"}, {"latency_ms": 15.0})

    def test_emit_empty(self) -> None:
        backend = ConsoleMetricsBackend("test-svc")
        backend.emit({})

    def test_is_metrics_backend(self) -> None:
        backend = ConsoleMetricsBackend("test-svc")
        assert isinstance(backend, MetricsBackend)


class TestLocalProfile:
    def test_creates_console_metrics(self) -> None:
        profile = LocalProfile()
        backend = profile.create_metrics_backend("svc", "us-east-1")
        assert isinstance(backend, MetricsBackend)

    def test_state_manager_is_none(self) -> None:
        profile = LocalProfile()
        assert profile.create_state_manager("bucket", "prefix/", "us-east-1") is None

    def test_creates_in_memory_context_store(self) -> None:
        profile = LocalProfile()
        store = profile.create_context_store("table", "us-east-1", 3600)
        assert isinstance(store, ContextStore)
        store.put("r1", {"x": 1})
        assert store.get("r1") == {"x": 1}

    def test_creates_in_memory_conversation_store(self) -> None:
        profile = LocalProfile()
        store = profile.create_conversation_store("table", "us-east-1", 3600)
        assert isinstance(store, ConversationStore)


class TestAWSProfile:
    def test_state_manager_none_when_no_bucket(self) -> None:
        profile = AWSProfile()
        assert profile.create_state_manager("", "prefix/", "us-east-1") is None
