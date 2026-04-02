"""Structured logging configuration for ml-platform services.

Provides a single entry point :func:`configure_logging` that sets up
production-ready logging with two format options:

- **json** -- Structured JSON lines for CloudWatch, ELK, Datadog, etc.
- **text** -- Human-readable coloured output for local development.

Both formats automatically include any :func:`request context <bind>`
that has been set for the current async task (``request_id``,
``session_id``, ``run_name``, or any custom key-value pairs).

Quick start::

    from ml_platform.log import configure_logging

    configure_logging(format="json", level="INFO")

Request context (automatically injected by the FastAPI middleware)::

    from ml_platform.log import bind, get_context

    bind(request_id="abc-123", session_id="sess-456")
    # All subsequent log lines in this async task include these fields.

The module deliberately wraps Python's standard ``logging`` rather than
replacing it, so it is compatible with any library that uses
``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Request-scoped context via contextvars
# ---------------------------------------------------------------------------

_log_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "ml_platform_log_context", default={}
)


def bind(**kwargs: str) -> None:
    """Attach key-value pairs to the current async-task log context.

    All log lines emitted within the same task will include these fields.
    This is called automatically by the request-context middleware for
    ``request_id`` and ``session_id``, but users can call it manually
    to add custom fields.

    Args:
        **kwargs: String key-value pairs to attach.

    Example::

        bind(customer_id="cust-42", experiment="v2")
    """
    current = _log_context.get()
    _log_context.set({**current, **kwargs})


def unbind(*keys: str) -> None:
    """Remove keys from the current log context.

    Args:
        *keys: Keys to remove.
    """
    current = _log_context.get()
    _log_context.set({k: v for k, v in current.items() if k not in keys})


def clear_context() -> None:
    """Remove all keys from the current log context."""
    _log_context.set({})


def get_context() -> dict[str, str]:
    """Return a copy of the current log context."""
    return dict(_log_context.get())


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each line contains:

    - ``timestamp`` -- ISO-8601 with milliseconds.
    - ``level`` -- Log level name.
    - ``logger`` -- Logger name.
    - ``message`` -- Formatted message.
    - ``service`` -- Service name (if configured).
    - Any request-context fields set via :func:`bind`.
    - ``exc_info`` -- Exception traceback (if present).

    This format is natively parsed by CloudWatch Logs Insights,
    Datadog, ELK, and most log aggregators.
    """

    def __init__(self, service_name: str = "") -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": self._format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self._service_name:
            data["service"] = self._service_name

        ctx = _log_context.get()
        if ctx:
            data.update(ctx)

        if record.exc_info and record.exc_info[1] is not None:
            data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(data, default=str)

    @staticmethod
    def _format_time(record: logging.LogRecord) -> str:
        ct = time.gmtime(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", ct) + f".{int(record.msecs):03d}Z"


# ---------------------------------------------------------------------------
# Text formatter (for local development)
# ---------------------------------------------------------------------------


class TextFormatter(logging.Formatter):
    """Human-readable formatter with optional ANSI colours and context fields.

    Format::

        2026-04-02 10:30:00 INFO  [ml_platform.serving.runtime] Service ready  request_id=abc-123
    """

    _COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        level = record.levelname.ljust(5)

        if self._use_color:
            color = self._COLORS.get(record.levelname, "")
            level = f"{color}{level}{self._RESET}"

        msg = record.getMessage()
        ctx = _log_context.get()
        ctx_str = "  ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else ""
        if ctx_str:
            ctx_str = f"  {ctx_str}"

        line = f"{ts} {level} [{record.name}] {msg}{ctx_str}"

        if record.exc_info and record.exc_info[1] is not None:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ---------------------------------------------------------------------------
# Configuration entry point
# ---------------------------------------------------------------------------


def configure_logging(
    *,
    format: Literal["json", "text"] = "text",
    level: str | int = "INFO",
    service_name: str = "",
) -> None:
    """Configure logging for the entire application.

    Sets up the root logger with the chosen format and level.  Call this
    once at application startup (before ``uvicorn.run`` or the first log
    line).

    Args:
        format: ``"json"`` for structured JSON (production) or
            ``"text"`` for human-readable (development).
        level: Log level as string (``"DEBUG"``, ``"INFO"``, etc.) or
            ``logging`` constant.
        service_name: Included in every JSON log line.  If empty, reads
            from ``ML_PLATFORM_SERVICE_NAME`` environment variable.

    Example::

        from ml_platform.log import configure_logging

        configure_logging(format="json", level="INFO", service_name="my-chatbot")
    """
    import os

    if not service_name:
        service_name = os.environ.get("ML_PLATFORM_SERVICE_NAME", "")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if format == "json":
        handler.setFormatter(JSONFormatter(service_name=service_name))
    else:
        handler.setFormatter(TextFormatter())
    handler.setLevel(level)

    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("ml_platform").setLevel(level)
