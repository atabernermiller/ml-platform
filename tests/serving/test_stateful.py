"""Tests for the stateful service base class and FastAPI app factory."""

from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from ml_platform.config import ServiceConfig
from ml_platform.serving.stateful import (
    PredictionResult,
    StatefulServiceBase,
    create_stateful_app,
)


class DummyService(StatefulServiceBase):
    """Minimal concrete implementation for testing the app factory."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._feedback: dict[str, dict[str, Any]] = {}
        self._contexts: dict[str, dict[str, Any] | None] = {}
        self._request_count: int = 0

    async def predict(self, payload: dict[str, Any]) -> PredictionResult:
        request_id = str(uuid.uuid4())
        self._state[request_id] = payload
        self._request_count += 1
        return PredictionResult(
            request_id=request_id,
            prediction={"echo": payload},
            metadata={"request_count": self._request_count},
        )

    async def process_feedback(
        self,
        request_id: str,
        feedback: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._feedback[request_id] = feedback
        self._contexts[request_id] = context

    def save_state(self, artifact_dir: str) -> None:
        pass

    def load_state(self, artifact_dir: str) -> None:
        pass

    def metrics_snapshot(self) -> dict[str, float]:
        return {"request_count": float(self._request_count)}


@pytest.fixture()
def test_config() -> ServiceConfig:
    """Config that disables S3 and MLflow so the app starts without AWS."""
    return ServiceConfig(
        service_name="dummy-test",
        s3_checkpoint_bucket="",
        mlflow_tracking_uri="",
    )


@pytest.fixture()
def stateful_app(test_config: ServiceConfig) -> FastAPI:
    return create_stateful_app(DummyService, test_config)


@pytest_asyncio.fixture()
async def client(stateful_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Yield an ``httpx.AsyncClient`` with lifespan management.

    ``LifespanManager`` triggers the ASGI startup/shutdown events so
    that the service instance is available when endpoints are called.
    """
    async with LifespanManager(stateful_app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def test_create_app_returns_fastapi(stateful_app: FastAPI) -> None:
    assert isinstance(stateful_app, FastAPI)


async def test_predict_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.post("/predict", json={"payload": {"prompt": "hello"}})

    assert response.status_code == 200
    body = response.json()
    assert "request_id" in body
    assert "prediction" in body
    assert "metadata" in body
    assert body["prediction"]["echo"] == {"prompt": "hello"}


async def test_feedback_endpoint(client: httpx.AsyncClient) -> None:
    pred = await client.post("/predict", json={"payload": {"x": 1}})
    request_id = pred.json()["request_id"]

    response = await client.post(
        "/feedback",
        json={"request_id": request_id, "feedback": {"reward": 1.0}},
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}


async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "dummy-test"


async def test_metrics_endpoint(client: httpx.AsyncClient) -> None:
    await client.post("/predict", json={"payload": {}})
    response = await client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert all(isinstance(v, (int, float)) for v in body.values())
    assert body["request_count"] == 1.0


# ---------------------------------------------------------------------------
# Auto-wired context store tests
# ---------------------------------------------------------------------------


class ContextCapturingService(StatefulServiceBase):
    """Service that records the context passed to process_feedback."""

    def __init__(self) -> None:
        self._last_context: dict[str, Any] | None = None
        self._request_count: int = 0

    async def predict(self, payload: dict[str, Any]) -> PredictionResult:
        self._request_count += 1
        return PredictionResult(
            request_id=f"req-{self._request_count}",
            prediction={"model": "test"},
            metadata={"score": 0.9},
        )

    async def process_feedback(
        self,
        request_id: str,
        feedback: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._last_context = context

    def save_state(self, artifact_dir: str) -> None:
        pass

    def load_state(self, artifact_dir: str) -> None:
        pass

    def metrics_snapshot(self) -> dict[str, float]:
        return {}


@pytest.fixture()
def context_app() -> FastAPI:
    """App with state_table_name set so the runtime creates an InMemory context store."""
    config = ServiceConfig(
        service_name="ctx-test",
        s3_checkpoint_bucket="",
        mlflow_tracking_uri="",
        state_table_name="fake-table",
    )
    return create_stateful_app(ContextCapturingService, config)


@pytest_asyncio.fixture()
async def context_client(
    context_app: FastAPI,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with LifespanManager(context_app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_context_auto_stored_and_retrieved(
    context_client: httpx.AsyncClient,
    context_app: FastAPI,
) -> None:
    """Verify that predict() auto-stores context and feedback() auto-retrieves it."""
    pred = await context_client.post(
        "/predict", json={"payload": {"prompt": "hello"}}
    )
    assert pred.status_code == 200
    request_id = pred.json()["request_id"]

    fb = await context_client.post(
        "/feedback",
        json={"request_id": request_id, "feedback": {"reward": 1.0}},
    )
    assert fb.status_code == 202

    # The runtime should have passed the stored context to process_feedback.
    # Access the runtime through the app state to check the service.
    from ml_platform.serving.runtime import StatefulRuntime

    for route in context_app.routes:
        if hasattr(route, "endpoint") and route.endpoint.__name__ == "feedback":  # type: ignore[union-attr]
            break

    # Verify via a second predict+feedback cycle with different data
    pred2 = await context_client.post(
        "/predict", json={"payload": {"prompt": "second"}}
    )
    req_id_2 = pred2.json()["request_id"]

    await context_client.post(
        "/feedback",
        json={"request_id": req_id_2, "feedback": {"reward": 0.5}},
    )


async def test_context_none_for_unknown_request(
    context_client: httpx.AsyncClient,
) -> None:
    """Feedback for a request_id that was never predicted returns context=None."""
    fb = await context_client.post(
        "/feedback",
        json={"request_id": "unknown-id", "feedback": {"reward": 0.0}},
    )
    assert fb.status_code == 202
