"""SageMaker serving adapter for ml-platform services.

SageMaker real-time endpoints expect a container that serves:

- ``GET /ping`` -- health check (must return 200)
- ``POST /invocations`` -- inference endpoint

This module provides :func:`wrap_for_sagemaker` which adds these routes
to an existing FastAPI application by delegating to the library's native
health and predict/run endpoints.

The adapter also forces the server to listen on port **8080**, which is
the SageMaker default.

Usage::

    from ml_platform.serving.sagemaker import wrap_for_sagemaker
    from ml_platform.serving.stateful import create_stateful_app

    app = create_stateful_app(MyService, config)
    app = wrap_for_sagemaker(app, service_type="stateful")

Or launch directly::

    from ml_platform.serving.sagemaker import create_sagemaker_app

    app = create_sagemaker_app(MyAgentService, config, service_type="agent", ...)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)

ServiceKind = Literal["stateful", "agent", "llm"]

SAGEMAKER_PORT = 8080


def wrap_for_sagemaker(
    app: FastAPI,
    *,
    service_type: ServiceKind = "agent",
) -> FastAPI:
    """Add SageMaker-compatible ``/ping`` and ``/invocations`` routes.

    ``/ping`` returns 200 (SageMaker health check).  ``/invocations``
    finds the matching handler for the native endpoint (``/predict`` or
    ``/run``) and invokes it directly -- no internal HTTP round-trip.

    Args:
        app: The existing FastAPI application.
        service_type: Which endpoint to proxy invocations to.

    Returns:
        The same ``app`` instance with additional routes.
    """
    target_path = "/predict" if service_type == "stateful" else "/run"

    @app.get("/ping")
    async def sagemaker_ping() -> Response:
        return Response(content="", status_code=200)

    _target_endpoint = None
    for route in app.routes:
        if hasattr(route, "path") and route.path == target_path:  # type: ignore[union-attr]
            if hasattr(route, "endpoint"):
                _target_endpoint = route.endpoint  # type: ignore[union-attr]
                break

    if _target_endpoint is not None:
        handler = _target_endpoint

        @app.post("/invocations")
        async def sagemaker_invocations(request: Request) -> Response:
            result = await handler(request)
            if isinstance(result, Response):
                return result
            body = json.dumps(result) if not isinstance(result, (str, bytes)) else result
            return Response(
                content=body,
                media_type="application/json",
            )
    else:
        @app.post("/invocations")
        async def sagemaker_invocations_fallback(request: Request) -> Response:
            return Response(
                content=json.dumps({"error": f"No handler found for {target_path}"}),
                status_code=501,
                media_type="application/json",
            )

    logger.info(
        "SageMaker adapter enabled: /ping, /invocations -> %s",
        target_path,
    )
    return app


def create_sagemaker_dockerfile() -> str:
    """Return a Dockerfile suitable for SageMaker real-time endpoints.

    Key differences from the standard Dockerfile:

    - Installs ``curl`` for the SageMaker health check.
    - Exposes port 8080 (SageMaker default).
    - Starts uvicorn on port 8080.

    Returns:
        Dockerfile content as a string.
    """
    return """\
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt* pyproject.toml* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null \\
    || pip install --no-cache-dir -e . 2>/dev/null \\
    || true

COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8080

ENV SAGEMAKER_BIND=true

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
"""
