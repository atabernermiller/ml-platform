"""Tests for the Lambda adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI

from ml_platform.serving.lambda_adapter import create_lambda_handler, wrap_for_lambda


@pytest.fixture()
def sample_app() -> FastAPI:
    """Create a minimal FastAPI app for testing."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/predict")
    async def predict() -> dict[str, Any]:
        return {"prediction": 42}

    return app


def test_create_lambda_handler_get(sample_app: FastAPI) -> None:
    handler = create_lambda_handler(sample_app)
    event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/health",
        "headers": {},
    }
    response = handler(event, None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "healthy"


def test_create_lambda_handler_post(sample_app: FastAPI) -> None:
    handler = create_lambda_handler(sample_app)
    event = {
        "requestContext": {"http": {"method": "POST"}},
        "rawPath": "/predict",
        "headers": {"content-type": "application/json"},
        "body": "{}",
    }
    response = handler(event, None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["prediction"] == 42


def test_create_lambda_handler_not_found(sample_app: FastAPI) -> None:
    handler = create_lambda_handler(sample_app)
    event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/nonexistent",
        "headers": {},
    }
    response = handler(event, None)
    assert response["statusCode"] == 404


def test_wrap_for_lambda_adds_ping(sample_app: FastAPI) -> None:
    wrap_for_lambda(sample_app)
    routes = [r.path for r in sample_app.routes if hasattr(r, "path")]
    assert "/_lambda/ping" in routes


def test_query_parameters(sample_app: FastAPI) -> None:
    @sample_app.get("/search")
    async def search(q: str = "") -> dict[str, str]:
        return {"query": q}

    handler = create_lambda_handler(sample_app)
    event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/search",
        "queryStringParameters": {"q": "test"},
        "headers": {},
    }
    response = handler(event, None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["query"] == "test"
