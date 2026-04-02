"""Tests for ml_platform.log -- structured logging and request context."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest import mock

import pytest

from ml_platform.log import (
    JSONFormatter,
    TextFormatter,
    bind,
    clear_context,
    configure_logging,
    get_context,
    unbind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure each test starts with an empty log context."""
    clear_context()
    yield
    clear_context()


@pytest.fixture()
def _restore_root_logger():
    """Snapshot and restore root logger handlers after the test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


class TestContextBindings:
    def test_bind_and_get(self) -> None:
        bind(request_id="abc-123", session_id="sess-1")
        ctx = get_context()
        assert ctx == {"request_id": "abc-123", "session_id": "sess-1"}

    def test_bind_is_additive(self) -> None:
        bind(a="1")
        bind(b="2")
        assert get_context() == {"a": "1", "b": "2"}

    def test_bind_overwrites_existing_key(self) -> None:
        bind(a="1")
        bind(a="2")
        assert get_context() == {"a": "2"}

    def test_unbind_removes_keys(self) -> None:
        bind(a="1", b="2", c="3")
        unbind("a", "c")
        assert get_context() == {"b": "2"}

    def test_unbind_missing_key_is_noop(self) -> None:
        bind(a="1")
        unbind("nonexistent")
        assert get_context() == {"a": "1"}

    def test_clear_context(self) -> None:
        bind(a="1", b="2")
        clear_context()
        assert get_context() == {}

    def test_get_context_returns_copy(self) -> None:
        bind(a="1")
        ctx = get_context()
        ctx["a"] = "mutated"
        assert get_context()["a"] == "1"


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    def _make_record(
        self,
        msg: str = "hello",
        level: int = logging.INFO,
        name: str = "test.logger",
    ) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_basic_structure(self) -> None:
        fmt = JSONFormatter(service_name="my-svc")
        record = self._make_record()
        line = fmt.format(record)
        data = json.loads(line)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "hello"
        assert data["service"] == "my-svc"
        assert "timestamp" in data

    def test_timestamp_format(self) -> None:
        fmt = JSONFormatter()
        record = self._make_record()
        data = json.loads(fmt.format(record))
        ts = data["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts

    def test_includes_context(self) -> None:
        bind(request_id="req-42", custom="val")
        fmt = JSONFormatter()
        data = json.loads(fmt.format(self._make_record()))
        assert data["request_id"] == "req-42"
        assert data["custom"] == "val"

    def test_no_service_when_empty(self) -> None:
        fmt = JSONFormatter(service_name="")
        data = json.loads(fmt.format(self._make_record()))
        assert "service" not in data

    def test_exception_info_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="failed",
            args=(),
            exc_info=exc_info,
        )
        fmt = JSONFormatter()
        data = json.loads(fmt.format(record))
        assert "exc_info" in data
        assert "ValueError: boom" in data["exc_info"]

    def test_output_is_single_line(self) -> None:
        fmt = JSONFormatter()
        line = fmt.format(self._make_record(msg="a\nmultiline\nmessage"))
        assert "\n" not in line


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TestTextFormatter:
    def _make_record(self, msg: str = "hello") -> logging.LogRecord:
        return logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_basic_format(self) -> None:
        fmt = TextFormatter(use_color=False)
        line = fmt.format(self._make_record())
        assert "[test.logger]" in line
        assert "hello" in line
        assert "INFO" in line

    def test_includes_context(self) -> None:
        bind(request_id="req-7")
        fmt = TextFormatter(use_color=False)
        line = fmt.format(self._make_record())
        assert "request_id=req-7" in line

    def test_no_context_no_trailing_space(self) -> None:
        fmt = TextFormatter(use_color=False)
        line = fmt.format(self._make_record())
        assert not line.endswith("  ")


# ---------------------------------------------------------------------------
# configure_logging()
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    @pytest.mark.usefixtures("_restore_root_logger")
    def test_json_format(self) -> None:
        configure_logging(format="json", level="DEBUG", service_name="test-svc")
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    @pytest.mark.usefixtures("_restore_root_logger")
    def test_text_format(self) -> None:
        configure_logging(format="text", level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    @pytest.mark.usefixtures("_restore_root_logger")
    def test_replaces_existing_handlers(self) -> None:
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        configure_logging(format="text", level="INFO")
        assert len(root.handlers) == 1

    @pytest.mark.usefixtures("_restore_root_logger")
    def test_reads_service_name_from_env(self) -> None:
        with mock.patch.dict("os.environ", {"ML_PLATFORM_SERVICE_NAME": "env-svc"}):
            configure_logging(format="json", level="INFO")

        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)
        assert handler.formatter._service_name == "env-svc"

    @pytest.mark.usefixtures("_restore_root_logger")
    def test_explicit_service_name_overrides_env(self) -> None:
        with mock.patch.dict("os.environ", {"ML_PLATFORM_SERVICE_NAME": "env-svc"}):
            configure_logging(
                format="json", level="INFO", service_name="explicit-svc"
            )

        handler = logging.getLogger().handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)
        assert handler.formatter._service_name == "explicit-svc"


# ---------------------------------------------------------------------------
# ServiceConfig integration
# ---------------------------------------------------------------------------


class TestServiceConfigLogFields:
    def test_defaults(self) -> None:
        from ml_platform.config import ServiceConfig

        c = ServiceConfig(service_name="svc")
        assert c.log_level == "INFO"
        assert c.log_format == ""

    def test_explicit(self) -> None:
        from ml_platform.config import ServiceConfig

        c = ServiceConfig(service_name="svc", log_level="DEBUG", log_format="json")
        assert c.log_level == "DEBUG"
        assert c.log_format == "json"

    def test_from_env(self) -> None:
        from ml_platform.config import ServiceConfig

        env = {
            "ML_PLATFORM_SERVICE_NAME": "env-svc",
            "ML_PLATFORM_LOG_LEVEL": "WARNING",
            "ML_PLATFORM_LOG_FORMAT": "json",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            c = ServiceConfig.from_env()
        assert c.log_level == "WARNING"
        assert c.log_format == "json"
