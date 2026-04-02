"""Tests for ml_platform.serving.agent -- AgentServiceBase and create_agent_app."""

from __future__ import annotations

from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from ml_platform._types import AgentResult, AgentStep, CompletionUsage, Message, Completion
from ml_platform.config import AgentConfig, ServiceConfig
from ml_platform.llm.run_context import RunContext
from ml_platform.serving.agent import AgentServiceBase, create_agent_app


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="mock reply",
            model=model or "mock-model",
            provider="mock",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )


class MockTool:
    name = "calc"
    description = "Mock calculator"
    parameters_schema: dict[str, Any] = {"expr": {"type": "string"}}

    async def execute(self, **kwargs: Any) -> str:
        return "42"


class FailingTool:
    name = "fail"
    description = "Always fails"
    parameters_schema: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("boom")


class SimpleAgent(AgentServiceBase):
    """Agent that does one LLM call."""

    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        c = await run_context.complete(self.providers["main"], messages, step_name="answer")
        return AgentResult(
            content=c.content,
            steps=run_context.steps,
            messages=messages,
        )


class MultiStepAgent(AgentServiceBase):
    """Agent that does plan -> tool -> synthesize."""

    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        plan = await run_context.complete(
            self.providers["planner"], messages, step_name="plan"
        )
        tool_result = await run_context.execute_tool(self.tools["calc"], expr="1+1")
        response = await run_context.complete(
            self.providers["writer"], messages, step_name="synthesize"
        )
        return AgentResult(
            content=response.content,
            steps=run_context.steps,
            messages=messages,
        )


class ToolErrorAgent(AgentServiceBase):
    """Agent that calls a failing tool then continues."""

    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        result = await run_context.execute_tool(self.tools["fail"])
        c = await run_context.complete(self.providers["main"], messages, step_name="fallback")
        return AgentResult(
            content=c.content,
            steps=run_context.steps,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config() -> ServiceConfig:
    return ServiceConfig(
        service_name="test-agent",
        agent=AgentConfig(max_steps_per_run=20),
    )


async def _managed_client(app: Any) -> AsyncClient:
    manager = LifespanManager(app)
    await manager.__aenter__()
    client = AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )
    client._lifespan_manager = manager  # type: ignore[attr-defined]
    return client


async def _close_client(client: AsyncClient) -> None:
    await client.aclose()
    manager = getattr(client, "_lifespan_manager", None)
    if manager:
        await manager.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateAgentApp:
    def test_returns_fastapi_with_routes(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent,
            simple_config,
            providers={"main": MockProvider()},
        )
        routes = {r.path for r in app.routes}
        assert "/run" in routes
        assert "/health" in routes
        assert "/metrics" in routes
        assert "/dashboard" in routes

    def test_rejects_stateful_service(self, simple_config: ServiceConfig) -> None:
        from ml_platform.serving.stateful import StatefulServiceBase

        class FakeStateful(StatefulServiceBase):
            async def predict(self, payload: dict) -> Any: ...
            async def process_feedback(self, rid: str, fb: dict) -> None: ...
            def save_state(self, d: str) -> None: ...
            def load_state(self, d: str) -> None: ...
            def metrics_snapshot(self) -> dict: ...

        with pytest.raises(TypeError, match="StatefulServiceBase"):
            create_agent_app(FakeStateful, simple_config)  # type: ignore[arg-type]

    def test_rejects_non_agent_class(self, simple_config: ServiceConfig) -> None:
        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="AgentServiceBase"):
            create_agent_app(NotAnAgent, simple_config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    @pytest.mark.asyncio
    async def test_simple_agent(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent,
            simple_config,
            providers={"main": MockProvider()},
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["content"] == "mock reply"
            assert len(body["steps"]) == 1
            assert body["llm_calls"] == 1
            assert body["tool_calls"] == 0
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_multi_step_agent(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            MultiStepAgent,
            simple_config,
            providers={"planner": MockProvider(), "writer": MockProvider()},
            tools=[MockTool()],
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "research"}]},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["steps"]) == 3
            assert body["llm_calls"] == 2
            assert body["tool_calls"] == 1
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_tool_error_continues(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            ToolErrorAgent,
            simple_config,
            providers={"main": MockProvider()},
            tools=[FailingTool()],
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "try"}]},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["steps"]) == 2
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self) -> None:
        config = ServiceConfig(
            service_name="test",
            agent=AgentConfig(max_steps_per_run=1),
        )

        class GreedyAgent(AgentServiceBase):
            async def run(self, messages, *, run_context, **kwargs):
                p = self.providers["main"]
                await run_context.complete(p, messages)
                await run_context.complete(p, messages)
                return AgentResult(content="", steps=run_context.steps, messages=messages)

        app = create_agent_app(
            GreedyAgent, config, providers={"main": MockProvider()}
        )
        client = await _managed_client(app)
        try:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "x"}]},
            )
            assert resp.status_code == 429
        finally:
            await _close_client(client)

    @pytest.mark.asyncio
    async def test_503_before_startup(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent, simple_config, providers={"main": MockProvider()}
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "x"}]},
            )
        assert resp.status_code == 503


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent, simple_config, providers={"main": MockProvider()}
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "test-agent"


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_after_run(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent, simple_config, providers={"main": MockProvider()}
        )
        client = await _managed_client(app)
        try:
            await client.post(
                "/run",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            resp = await client.get("/metrics")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total_runs"] == 1.0
        finally:
            await _close_client(client)


class TestDashboardEndpoint:
    @pytest.mark.asyncio
    async def test_dashboard_returns_html(self, simple_config: ServiceConfig) -> None:
        app = create_agent_app(
            SimpleAgent, simple_config, providers={"main": MockProvider()}
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "test-agent" in resp.text


class TestToolValidation:
    def test_invalid_tool_raises(self, simple_config: ServiceConfig) -> None:
        class BadTool:
            name = "bad"

        with pytest.raises(TypeError, match="Tool protocol"):
            create_agent_app(
                SimpleAgent,
                simple_config,
                providers={"main": MockProvider()},
                tools=[BadTool()],
            )
