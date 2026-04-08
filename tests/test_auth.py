"""Tests for authentication and authorization middleware."""

from __future__ import annotations

from typing import Any

import jwt
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from ml_platform.auth import (
    APIKeyAuth,
    AuthResult,
    JWTAuth,
    add_auth_middleware,
    require_role,
)


@pytest.fixture()
def app_with_api_key_auth() -> FastAPI:
    """Create a FastAPI app with API key authentication."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/protected")
    async def protected(request: Request) -> dict[str, str]:
        return {"identity": request.state.auth.identity}

    add_auth_middleware(app, auth=APIKeyAuth(valid_keys={"sk-test-key"}))
    return app


@pytest.fixture()
def app_with_jwt_auth() -> FastAPI:
    """Create a FastAPI app with JWT authentication."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/protected")
    async def protected(request: Request) -> dict[str, str]:
        return {"user": request.state.auth.identity}

    @app.get("/admin")
    @require_role("admin")
    async def admin(request: Request) -> dict[str, str]:
        return {"admin": "true"}

    add_auth_middleware(
        app,
        auth=JWTAuth(secret="test-secret", role_claim="roles"),
    )
    return app


class TestAPIKeyAuth:
    def test_valid_key(self) -> None:
        auth = APIKeyAuth(valid_keys={"key-123"})

        class FakeRequest:
            headers = {"X-API-Key": "key-123"}
            query_params: dict[str, str] = {}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is True

    def test_invalid_key(self) -> None:
        auth = APIKeyAuth(valid_keys={"key-123"})

        class FakeRequest:
            headers = {"X-API-Key": "wrong-key"}
            query_params: dict[str, str] = {}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is False

    def test_key_from_query_param(self) -> None:
        auth = APIKeyAuth(valid_keys={"key-abc"})

        class FakeRequest:
            headers: dict[str, str] = {}
            query_params = {"api_key": "key-abc"}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is True


class TestJWTAuth:
    def test_valid_token(self) -> None:
        secret = "test-secret"
        auth = JWTAuth(secret=secret)
        token = jwt.encode({"sub": "user-1", "roles": ["admin"]}, secret, algorithm="HS256")

        class FakeRequest:
            headers = {"authorization": f"Bearer {token}"}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is True
        assert result.identity == "user-1"
        assert "admin" in result.roles

    def test_invalid_token(self) -> None:
        auth = JWTAuth(secret="real-secret")

        class FakeRequest:
            headers = {"authorization": "Bearer invalid.token.here"}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is False

    def test_missing_bearer(self) -> None:
        auth = JWTAuth(secret="secret")

        class FakeRequest:
            headers: dict[str, str] = {}

        result = auth.authenticate(FakeRequest())  # type: ignore[arg-type]
        assert result.authenticated is False


@pytest.mark.asyncio
async def test_api_key_middleware_allows_health(app_with_api_key_auth: FastAPI) -> None:
    transport = ASGITransport(app=app_with_api_key_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_key_middleware_rejects_without_key(app_with_api_key_auth: FastAPI) -> None:
    transport = ASGITransport(app=app_with_api_key_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_key_middleware_accepts_valid_key(app_with_api_key_auth: FastAPI) -> None:
    transport = ASGITransport(app=app_with_api_key_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected", headers={"X-API-Key": "sk-test-key"})
        assert resp.status_code == 200
        assert "apikey:" in resp.json()["identity"]


@pytest.mark.asyncio
async def test_jwt_role_check(app_with_jwt_auth: FastAPI) -> None:
    secret = "test-secret"
    token = jwt.encode({"sub": "u-1", "roles": ["user"]}, secret, algorithm="HS256")
    transport = ASGITransport(app=app_with_jwt_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403


@pytest.mark.asyncio
async def test_jwt_admin_access(app_with_jwt_auth: FastAPI) -> None:
    secret = "test-secret"
    token = jwt.encode({"sub": "u-1", "roles": ["admin"]}, secret, algorithm="HS256")
    transport = ASGITransport(app=app_with_jwt_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/admin", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["admin"] == "true"
