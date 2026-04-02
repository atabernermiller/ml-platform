"""Tests for the request-context logging middleware in _app_builder.py."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ml_platform.config import ServiceConfig
from ml_platform.log import clear_context, get_context
from ml_platform.serving._app_builder import build_base_app


def _make_app() -> Any:
    """Build a minimal app that exposes the log context via an endpoint."""
    config = ServiceConfig(service_name="ctx-test")
    app = build_base_app(
        config,
        readiness_check=lambda: True,
        metrics_source=lambda: {"ok": 1.0},
    )

    @app.get("/echo-context")
    async def echo_context() -> dict[str, str]:
        return get_context()

    return app


@pytest.fixture()
def ctx_app():
    return _make_app()


@pytest.fixture()
async def client(ctx_app):
    transport = httpx.ASGITransport(app=ctx_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clean():
    clear_context()
    yield
    clear_context()


@pytest.mark.asyncio
class TestRequestContextMiddleware:
    async def test_auto_generates_request_id(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/echo-context")
        assert resp.status_code == 200
        ctx = resp.json()
        assert "request_id" in ctx
        assert len(ctx["request_id"]) > 0

    async def test_request_id_in_response_header(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/echo-context")
        assert "x-request-id" in resp.headers

    async def test_honours_incoming_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get(
            "/echo-context", headers={"X-Request-ID": "custom-id-42"}
        )
        ctx = resp.json()
        assert ctx["request_id"] == "custom-id-42"
        assert resp.headers["x-request-id"] == "custom-id-42"

    async def test_propagates_session_id(self, client: httpx.AsyncClient) -> None:
        resp = await client.get(
            "/echo-context", headers={"X-Session-ID": "sess-99"}
        )
        ctx = resp.json()
        assert ctx["session_id"] == "sess-99"

    async def test_includes_method_and_path(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/echo-context")
        ctx = resp.json()
        assert ctx["method"] == "GET"
        assert ctx["path"] == "/echo-context"

    async def test_context_isolated_between_requests(
        self, client: httpx.AsyncClient
    ) -> None:
        resp1 = await client.get(
            "/echo-context", headers={"X-Session-ID": "first"}
        )
        resp2 = await client.get("/echo-context")
        assert resp1.json()["session_id"] == "first"
        assert "session_id" not in resp2.json()

    async def test_request_id_unique_per_request(
        self, client: httpx.AsyncClient
    ) -> None:
        resp1 = await client.get("/echo-context")
        resp2 = await client.get("/echo-context")
        id1 = resp1.json()["request_id"]
        id2 = resp2.json()["request_id"]
        assert id1 != id2
