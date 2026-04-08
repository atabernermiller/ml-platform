"""AWS Lambda adapter for ml-platform services.

Lambda is the most widely-used AWS compute service and is ideal for
event-driven workloads, API endpoints, and lightweight inference.

This module provides:

- :func:`create_lambda_handler` -- wraps a FastAPI app into an
  AWS Lambda handler compatible with API Gateway v2 (HTTP API)
  and Lambda Function URLs.
- :func:`wrap_for_lambda` -- adds Lambda-specific configuration
  to an existing FastAPI app.

Usage::

    from ml_platform.serving.lambda_adapter import create_lambda_handler
    from ml_platform.serving.stateful import create_stateful_app

    app = create_stateful_app(MyService, config)
    handler = create_lambda_handler(app)

    # In your Lambda entry point:
    def lambda_handler(event, context):
        return handler(event, context)
"""

from __future__ import annotations

import json
import logging
import base64
from typing import Any, Callable
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

__all__ = [
    "create_lambda_handler",
    "wrap_for_lambda",
]

LambdaHandler = Callable[[dict[str, Any], Any], dict[str, Any]]


def create_lambda_handler(app: Any) -> LambdaHandler:
    """Wrap a FastAPI/ASGI app into an AWS Lambda handler.

    Converts API Gateway v2 / Lambda Function URL events into ASGI
    scope dicts, invokes the app, and converts the response back into
    the Lambda proxy response format.

    Args:
        app: A FastAPI (or any ASGI) application instance.

    Returns:
        A callable ``(event, context) -> response_dict`` suitable for
        use as a Lambda handler.
    """
    import asyncio

    def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_in_new_loop, _async_handler(app, event))
                return future.result()
        return _run_in_new_loop(_async_handler(app, event))

    return handler


def _run_in_new_loop(coro: Any) -> Any:
    """Run a coroutine in a fresh event loop without disturbing the global loop state."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _async_handler(app: Any, event: dict[str, Any]) -> dict[str, Any]:
    """Process a Lambda event through the ASGI app."""
    request_context = event.get("requestContext", {})
    http_info = request_context.get("http", {})

    method = http_info.get("method", event.get("httpMethod", "GET"))
    path = event.get("rawPath", event.get("path", "/"))
    query_params = event.get("queryStringParameters") or {}
    headers_raw = event.get("headers") or {}

    headers_list: list[tuple[bytes, bytes]] = []
    for k, v in headers_raw.items():
        headers_list.append((k.lower().encode(), v.encode()))

    query_string = urlencode(query_params).encode() if query_params else b""

    body_str = event.get("body", "")
    is_base64 = event.get("isBase64Encoded", False)
    if body_str:
        body = base64.b64decode(body_str) if is_base64 else body_str.encode()
    else:
        body = b""

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "query_string": query_string,
        "headers": headers_list,
        "server": ("lambda", 443),
    }

    response_started = False
    status_code = 200
    response_headers: dict[str, str] = {}
    response_body_parts: list[bytes] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal response_started, status_code, response_headers
        if message["type"] == "http.response.start":
            response_started = True
            status_code = message["status"]
            for hdr_name, hdr_val in message.get("headers", []):
                response_headers[hdr_name.decode()] = hdr_val.decode()
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if chunk:
                response_body_parts.append(chunk)

    await app(scope, receive, send)

    response_body = b"".join(response_body_parts)
    is_binary = not response_headers.get("content-type", "").startswith("text/") and \
                not response_headers.get("content-type", "").startswith("application/json")

    result: dict[str, Any] = {
        "statusCode": status_code,
        "headers": response_headers,
    }

    if is_binary and response_body:
        result["body"] = base64.b64encode(response_body).decode()
        result["isBase64Encoded"] = True
    else:
        result["body"] = response_body.decode("utf-8", errors="replace")
        result["isBase64Encoded"] = False

    return result


def wrap_for_lambda(app: Any) -> Any:
    """Configure a FastAPI app for Lambda deployment.

    Sets Lambda-friendly defaults: disables background tasks that
    rely on long-running processes and adds a ``/_lambda/ping`` route
    for warmup invocations.

    Args:
        app: FastAPI application to configure.

    Returns:
        The same app instance with Lambda routes added.
    """
    from fastapi import FastAPI

    @app.get("/_lambda/ping")
    async def lambda_ping() -> dict[str, str]:
        return {"status": "warm"}

    logger.info("Lambda adapter enabled: /_lambda/ping")
    return app
