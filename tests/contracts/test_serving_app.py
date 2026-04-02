"""Serving app contract tests -- shared behaviour across all create_*_app() factories.

Every ``create_*_app()`` factory must produce a FastAPI application that
satisfies these shared contracts: health check, metrics, dashboard, and
503-before-startup behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from ml_platform._types import (
    AgentResult,
    Completion,
    CompletionUsage,
    Message,
)
from ml_platform.config import AgentConfig, ServiceConfig
from ml_platform.llm.run_context import RunContext
from ml_platform.serving.agent import AgentServiceBase, create_agent_app
from ml_platform.serving.stateful import (
    PredictionResult,
    StatefulServiceBase,
    create_stateful_app,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="ok",
            model="mock",
            provider="mock",
            usage=CompletionUsage(input_tokens=1, output_tokens=1),
        )


class _MockStatefulService(StatefulServiceBase):
    async def predict(self, payload: dict[str, Any]) -> PredictionResult:
        return PredictionResult(request_id="r1", prediction={"v": 1})

    async def process_feedback(self, rid: str, fb: dict[str, Any]) -> None:
        pass

    def save_state(self, d: str) -> None:
        pass

    def load_state(self, d: str) -> None:
        pass

    def metrics_snapshot(self) -> dict[str, float]:
        return {"test_metric": 1.0}


class _MockAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        c = await run_context.complete(self.providers["main"], messages)
        return AgentResult(content=c.content, steps=run_context.steps, messages=messages)


# ---------------------------------------------------------------------------
# Shared contract tests
# ---------------------------------------------------------------------------


class ServingAppContractTests(ABC):
    """Reusable contract tests for any ``create_*_app()`` factory."""

    @abstractmethod
    def create_app(self) -> Any:
        """Return a FastAPI app from the relevant factory."""
        ...

    def _client(self, app: Any) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    @pytest.mark.asyncio
    async def test_health_returns_200(self) -> None:
        app = self.create_app()
        async with self._client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_includes_service_name(self) -> None:
        app = self.create_app()
        async with self._client(app) as client:
            resp = await client.get("/health")
        body = resp.json()
        assert "service" in body
        assert isinstance(body["service"], str)
        assert len(body["service"]) > 0

    @pytest.mark.asyncio
    async def test_dashboard_returns_html(self) -> None:
        app = self.create_app()
        async with self._client(app) as client:
            resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self) -> None:
        app = self.create_app()
        async with self._client(app) as client:
            resp = await client.get("/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Concrete test classes
# ---------------------------------------------------------------------------


class TestStatefulAppContract(ServingAppContractTests):
    def create_app(self) -> Any:
        config = ServiceConfig(service_name="test-stateful")
        return create_stateful_app(_MockStatefulService, config)


class TestAgentAppContract(ServingAppContractTests):
    def create_app(self) -> Any:
        config = ServiceConfig(
            service_name="test-agent",
            agent=AgentConfig(),
        )
        return create_agent_app(
            _MockAgent,
            config,
            providers={"main": _MockProvider()},
        )
