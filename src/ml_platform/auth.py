"""Authentication and authorization middleware for FastAPI services.

Provides pluggable auth layers:

- **API key verification** -- header or query-param based.
- **JWT validation** -- RS256/HS256 token validation with optional
  Cognito integration.
- **Role-based access control (RBAC)** -- decorator for endpoint-level
  authorization.

Usage::

    from ml_platform.auth import (
        APIKeyAuth,
        JWTAuth,
        require_role,
        add_auth_middleware,
    )

    app = create_stateful_app(MyService, config)
    add_auth_middleware(app, auth=APIKeyAuth(valid_keys={"sk-abc123"}))

    @app.get("/admin")
    @require_role("admin")
    async def admin_endpoint(request: Request):
        ...
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Callable, Protocol, Sequence

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

__all__ = [
    "AuthBackend",
    "APIKeyAuth",
    "JWTAuth",
    "AuthResult",
    "add_auth_middleware",
    "require_role",
]

SKIP_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/ping",
    "/_lambda/ping",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class AuthResult:
    """Result of an authentication attempt.

    Attributes:
        authenticated: Whether the request was authenticated.
        identity: Identifier for the authenticated principal (e.g. user_id,
            API key name).
        roles: Set of roles assigned to the principal.
        claims: Full set of claims from the auth token.
    """

    __slots__ = ("authenticated", "identity", "roles", "claims")

    def __init__(
        self,
        *,
        authenticated: bool = False,
        identity: str = "",
        roles: frozenset[str] | None = None,
        claims: dict[str, Any] | None = None,
    ) -> None:
        self.authenticated = authenticated
        self.identity = identity
        self.roles = roles or frozenset()
        self.claims = claims or {}


class AuthBackend(Protocol):
    """Protocol for authentication backends."""

    def authenticate(self, request: Request) -> AuthResult:
        """Authenticate a request.

        Args:
            request: The incoming HTTP request.

        Returns:
            Authentication result.
        """
        ...


class APIKeyAuth:
    """API key authentication via header or query parameter.

    Args:
        valid_keys: Set of accepted API keys.
        header_name: HTTP header to read the key from.
        query_param: Query parameter name (fallback if header missing).
    """

    def __init__(
        self,
        valid_keys: set[str],
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
    ) -> None:
        self._valid_keys = valid_keys
        self._header_name = header_name
        self._query_param = query_param

    def authenticate(self, request: Request) -> AuthResult:
        key = request.headers.get(self._header_name) or request.query_params.get(
            self._query_param, ""
        )
        if key and key in self._valid_keys:
            return AuthResult(authenticated=True, identity=f"apikey:{_hash_key(key)}")
        return AuthResult(authenticated=False)


class JWTAuth:
    """JWT token authentication.

    Validates JWT tokens from the ``Authorization: Bearer <token>`` header.
    Supports HS256 (symmetric) validation with a shared secret.

    For production Cognito/OIDC use, provide a JWKS endpoint and
    use RS256 validation (requires ``PyJWT[crypto]``).

    Args:
        secret: Shared secret for HS256 validation.
        algorithms: Allowed algorithms (default ``["HS256"]``).
        audience: Expected ``aud`` claim.
        issuer: Expected ``iss`` claim.
        role_claim: JWT claim containing user roles.
    """

    def __init__(
        self,
        secret: str,
        algorithms: list[str] | None = None,
        audience: str = "",
        issuer: str = "",
        role_claim: str = "roles",
    ) -> None:
        self._secret = secret
        self._algorithms = algorithms or ["HS256"]
        self._audience = audience
        self._issuer = issuer
        self._role_claim = role_claim

    def authenticate(self, request: Request) -> AuthResult:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return AuthResult(authenticated=False)
        token = auth_header[7:].strip()
        try:
            import jwt

            decode_opts: dict[str, Any] = {
                "algorithms": self._algorithms,
            }
            if self._audience:
                decode_opts["audience"] = self._audience
            if self._issuer:
                decode_opts["issuer"] = self._issuer

            claims = jwt.decode(token, self._secret, **decode_opts)
            roles_raw = claims.get(self._role_claim, [])
            if isinstance(roles_raw, str):
                roles_raw = [roles_raw]
            return AuthResult(
                authenticated=True,
                identity=claims.get("sub", ""),
                roles=frozenset(roles_raw),
                claims=claims,
            )
        except Exception as exc:
            logger.debug("JWT validation failed: %s", exc)
            return AuthResult(authenticated=False)


class _AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces authentication on all non-skip routes."""

    def __init__(self, app: Any, auth: AuthBackend, skip_paths: frozenset[str]) -> None:
        super().__init__(app)
        self._auth = auth
        self._skip_paths = skip_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self._skip_paths:
            return await call_next(request)

        result = self._auth.authenticate(request)
        if not result.authenticated:
            return Response(
                content=json.dumps({"detail": "Unauthorized"}),
                status_code=401,
                media_type="application/json",
            )
        request.state.auth = result
        return await call_next(request)


def add_auth_middleware(
    app: FastAPI,
    *,
    auth: AuthBackend,
    skip_paths: frozenset[str] | None = None,
) -> FastAPI:
    """Add authentication middleware to a FastAPI application.

    Args:
        app: FastAPI application.
        auth: Authentication backend.
        skip_paths: Paths to skip authentication for. Defaults to
            health, docs, and ping endpoints.

    Returns:
        The same app with middleware added.
    """
    paths = skip_paths if skip_paths is not None else SKIP_PATHS
    app.add_middleware(_AuthMiddleware, auth=auth, skip_paths=paths)
    logger.info("Auth middleware enabled, skipping: %s", paths)
    return app


def require_role(*roles: str) -> Callable[..., Any]:
    """Decorator that enforces role-based access on an endpoint.

    Usage::

        @app.get("/admin")
        @require_role("admin")
        async def admin_only(request: Request):
            ...

    Args:
        *roles: One or more required roles. The user must have at least
            one of the specified roles.

    Returns:
        Decorator function.
    """
    required = frozenset(roles)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Request | None = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            if request is None:
                raise HTTPException(status_code=500, detail="No request object found")

            auth_result: AuthResult | None = getattr(request.state, "auth", None)
            if auth_result is None or not auth_result.authenticated:
                raise HTTPException(status_code=401, detail="Unauthorized")
            if not (auth_result.roles & required):
                raise HTTPException(status_code=403, detail="Forbidden")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def _hash_key(key: str) -> str:
    """Return a short hash of an API key for logging (avoids logging secrets)."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]
