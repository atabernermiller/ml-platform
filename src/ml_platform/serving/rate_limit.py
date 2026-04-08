"""Request rate limiting middleware for FastAPI services.

Provides a token-bucket rate limiter that can be attached to any
FastAPI application via :func:`add_rate_limit_middleware`.

Usage::

    from ml_platform.serving.rate_limit import add_rate_limit_middleware

    app = create_stateful_app(MyService, config)
    add_rate_limit_middleware(app, requests_per_second=10.0, burst=20)
"""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

__all__ = [
    "TokenBucketLimiter",
    "add_rate_limit_middleware",
]


class TokenBucketLimiter:
    """Thread-safe token-bucket rate limiter.

    Allows ``burst`` requests instantly, then refills at
    ``rate`` tokens per second.

    Args:
        rate: Sustained requests per second.
        burst: Maximum burst size (bucket capacity).
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check whether a request is allowed.

        Returns:
            ``True`` if the request should proceed, ``False`` if
            rate-limited.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (approximate)."""
        return self._tokens


class _PerKeyLimiter:
    """Per-key rate limiter (e.g. per IP address or per API key)."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._limiters: dict[str, TokenBucketLimiter] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = TokenBucketLimiter(self._rate, self._burst)
        return self._limiters[key].allow()


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces rate limits."""

    def __init__(
        self,
        app: Any,
        limiter: TokenBucketLimiter | _PerKeyLimiter,
        key_func: Callable[[Request], str] | None,
        skip_paths: frozenset[str],
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._key_func = key_func
        self._skip_paths = skip_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self._skip_paths:
            return await call_next(request)

        if self._key_func is not None:
            key = self._key_func(request)
            allowed = self._limiter.allow(key)  # type: ignore[arg-type]
        else:
            allowed = self._limiter.allow()  # type: ignore[call-arg]

        if not allowed:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "1"},
            )

        return await call_next(request)


_DEFAULT_SKIP_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/ping",
    "/_lambda/ping",
})


def add_rate_limit_middleware(
    app: FastAPI,
    *,
    requests_per_second: float = 10.0,
    burst: int = 20,
    per_client: bool = False,
    key_func: Callable[[Request], str] | None = None,
    skip_paths: frozenset[str] | None = None,
) -> FastAPI:
    """Add rate limiting middleware to a FastAPI application.

    Args:
        app: FastAPI application.
        requests_per_second: Sustained rate limit.
        burst: Maximum burst size.
        per_client: If ``True``, rate limit per client IP. Overridden
            by ``key_func`` if provided.
        key_func: Optional callable to extract a rate-limit key from the
            request (e.g. API key, user ID). When ``None`` and
            ``per_client=False``, a global limiter is used.
        skip_paths: Paths exempt from rate limiting.

    Returns:
        The same app with middleware added.
    """
    paths = skip_paths if skip_paths is not None else _DEFAULT_SKIP_PATHS

    effective_key_func = key_func
    if effective_key_func is None and per_client:
        effective_key_func = _default_client_key

    if effective_key_func is not None:
        limiter: TokenBucketLimiter | _PerKeyLimiter = _PerKeyLimiter(requests_per_second, burst)
    else:
        limiter = TokenBucketLimiter(requests_per_second, burst)

    app.add_middleware(
        _RateLimitMiddleware,
        limiter=limiter,
        key_func=effective_key_func,
        skip_paths=paths,
    )
    logger.info(
        "Rate limit middleware enabled: %.1f req/s, burst=%d, per_client=%s",
        requests_per_second,
        burst,
        per_client or key_func is not None,
    )
    return app


def _default_client_key(request: Request) -> str:
    """Extract client IP from the request for per-client rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"
