"""Project templates for ``ml-platform init``.

Each template generates a minimal but runnable project directory that
demonstrates the corresponding service path.
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Literal

TemplateName = Literal["agent", "chatbot", "bandit"]

_AGENT_APP = string.Template('''\
"""Multi-step research agent with tool use.

Run:
    pip install ml-platform
    python app.py
    # Open http://localhost:8000/dashboard
"""

from __future__ import annotations

import asyncio
from typing import Any

import uvicorn

from ml_platform._types import AgentResult, Completion, CompletionUsage, Message
from ml_platform.config import AgentConfig, ServiceConfig
from ml_platform.llm.run_context import RunContext
from ml_platform.serving.agent import AgentServiceBase, create_agent_app


# ---------------------------------------------------------------------------
# Mock provider (replace with your real LLM provider)
# ---------------------------------------------------------------------------


class MockProvider:
    """Placeholder LLM provider -- replace with OpenAI, Anthropic, etc."""

    async def complete(
        self, messages: list[Message], *, model: str = "", **kwargs: Any
    ) -> Completion:
        last = messages[-1].content if messages else ""
        return Completion(
            content=f"Mock response to: {last[:50]}",
            model=model or "mock-model",
            provider="mock",
            usage=CompletionUsage(input_tokens=20, output_tokens=10),
            cost_usd=0.001,
        )


# ---------------------------------------------------------------------------
# Mock tool (replace with your real tools)
# ---------------------------------------------------------------------------


class SearchDocsTool:
    """Placeholder search tool -- replace with your retrieval logic."""

    name = "search_docs"
    description = "Search the knowledge base for relevant documents"
    parameters_schema: dict[str, Any] = {"query": {"type": "string"}}

    async def execute(self, query: str = "") -> str:
        return f"Found 3 documents about \\'{query}\\'"


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------


class ResearchAgent(AgentServiceBase):
    """A simple plan-search-synthesize agent."""

    async def run(
        self,
        messages: list[Message],
        *,
        run_context: RunContext,
        **kwargs: Any,
    ) -> AgentResult:
        plan = await run_context.complete(
            self.providers["planner"], messages, step_name="plan"
        )

        search = await run_context.execute_tool(
            self.tools["search_docs"], query=plan.content
        )

        response = await run_context.complete(
            self.providers["writer"],
            messages + [Message(role="tool", content=search.content,
                                tool_call_id=search.tool_call_id)],
            step_name="synthesize",
        )

        return AgentResult(
            content=response.content,
            steps=run_context.steps,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


config = ServiceConfig(
    service_name="$name",
    agent=AgentConfig(max_steps_per_run=10),
)

app = create_agent_app(
    ResearchAgent,
    config,
    providers={"planner": MockProvider(), "writer": MockProvider()},
    tools=[SearchDocsTool()],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')


def generate_template(
    name: str,
    template: TemplateName,
    output_dir: str | None = None,
) -> Path:
    """Generate a project from a template.

    Args:
        name: Project / service name.
        template: Which template to use.
        output_dir: Parent directory for the new project.  Defaults to
            the current working directory.

    Returns:
        Path to the created project directory.
    """
    base = Path(output_dir or os.getcwd()) / name
    base.mkdir(parents=True, exist_ok=True)

    if template == "agent":
        (base / "app.py").write_text(_AGENT_APP.substitute(name=name))
    else:
        (base / "app.py").write_text(
            f"# TODO: {template} template -- replace with your service code\n"
        )

    (base / "requirements.txt").write_text("ml-platform\nuvicorn\n")
    return base
