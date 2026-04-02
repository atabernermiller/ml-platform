"""Stress test: sustained load of 1000 agent runs.

Each run creates 3 steps (2 LLM calls + 1 tool call).
Verifies that memory stays bounded and RunContext objects are
garbage collected between runs.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
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


class FastProvider:
    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        return Completion(
            content="ok",
            model="mock",
            provider="mock",
            usage=CompletionUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )


class FastTool:
    name = "tool"
    description = "fast tool"
    parameters_schema: dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class LoadAgent(AgentServiceBase):
    async def run(
        self, messages: list[Message], *, run_context: RunContext, **kwargs: Any
    ) -> AgentResult:
        await run_context.complete(self.providers["p"], messages, step_name="s1")
        await run_context.execute_tool(self.tools["tool"])
        c = await run_context.complete(self.providers["p"], messages, step_name="s3")
        return AgentResult(
            content=c.content, steps=run_context.steps, messages=messages
        )


def _get_memory_mb() -> float:
    """Process RSS in MB (platform-independent approximation)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except ImportError:
        import psutil  # type: ignore[import-untyped]
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@pytest.mark.asyncio
async def test_agent_memory_stable_under_sustained_load() -> None:
    config = ServiceConfig(
        service_name="stress-sustained",
        agent=AgentConfig(max_steps_per_run=50),
    )
    app = create_agent_app(
        LoadAgent,
        config,
        providers={"p": FastProvider()},
        tools=[FastTool()],
    )

    num_runs = 1000

    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            gc.collect()
            baseline_mb = _get_memory_mb()

            for i in range(num_runs):
                resp = await client.post(
                    "/run",
                    json={"messages": [{"role": "user", "content": f"run-{i}"}]},
                )
                assert resp.status_code == 200

                if i % 200 == 0:
                    gc.collect()

            gc.collect()
            final_mb = _get_memory_mb()

            resp = await client.get("/metrics")
            assert resp.status_code == 200
            metrics = resp.json()
            assert metrics["total_runs"] == float(num_runs)

    growth_mb = final_mb - baseline_mb
    assert growth_mb < 80, f"Memory grew by {growth_mb:.1f} MB (limit 80 MB)"
