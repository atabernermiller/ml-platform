"""Tests for rate limiting middleware."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ml_platform.serving.rate_limit import TokenBucketLimiter, add_rate_limit_middleware


class TestTokenBucketLimiter:
    def test_allows_burst(self) -> None:
        limiter = TokenBucketLimiter(rate=1.0, burst=5)
        results = [limiter.allow() for _ in range(5)]
        assert all(results)

    def test_rejects_after_burst(self) -> None:
        limiter = TokenBucketLimiter(rate=1.0, burst=3)
        for _ in range(3):
            limiter.allow()
        assert limiter.allow() is False

    def test_refills_over_time(self) -> None:
        limiter = TokenBucketLimiter(rate=1000.0, burst=1)
        assert limiter.allow() is True
        assert limiter.allow() is False
        time.sleep(0.05)
        assert limiter.allow() is True

    def test_available_tokens(self) -> None:
        limiter = TokenBucketLimiter(rate=1.0, burst=10)
        assert limiter.available_tokens == 10.0
        limiter.allow()
        assert limiter.available_tokens < 10.0


@pytest.fixture()
def rate_limited_app() -> FastAPI:
    """Create a FastAPI app with rate limiting."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/data")
    async def data() -> dict[str, str]:
        return {"data": "value"}

    add_rate_limit_middleware(app, requests_per_second=1.0, burst=2)
    return app


@pytest.mark.asyncio
async def test_rate_limit_allows_initial_burst(rate_limited_app: FastAPI) -> None:
    transport = ASGITransport(app=rate_limited_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/api/data")
        r2 = await client.get("/api/data")
        assert r1.status_code == 200
        assert r2.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_rejects_excess(rate_limited_app: FastAPI) -> None:
    transport = ASGITransport(app=rate_limited_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            await client.get("/api/data")
        resp = await client.get("/api/data")
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_rate_limit_skips_health(rate_limited_app: FastAPI) -> None:
    transport = ASGITransport(app=rate_limited_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Exhaust rate limit
        for _ in range(5):
            await client.get("/api/data")
        # Health should still work
        resp = await client.get("/health")
        assert resp.status_code == 200
