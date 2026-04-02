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

    _write_manifest(base, name, template)
    _write_dockerfile(base)
    _write_docker_compose(base, name)

    return base


def _write_manifest(base: Path, name: str, template: TemplateName) -> None:
    """Generate an ``ml-platform.yaml`` tailored to the template."""
    feature_defaults: dict[TemplateName, dict[str, bool]] = {
        "agent": {
            "conversation_store": True,
            "context_store": False,
            "checkpointing": False,
            "mlflow": False,
        },
        "chatbot": {
            "conversation_store": True,
            "context_store": False,
            "checkpointing": False,
            "mlflow": False,
        },
        "bandit": {
            "conversation_store": False,
            "context_store": True,
            "checkpointing": True,
            "mlflow": False,
        },
    }

    svc_type: dict[TemplateName, str] = {
        "agent": "agent",
        "chatbot": "llm",
        "bandit": "stateful",
    }

    features = feature_defaults.get(template, feature_defaults["chatbot"])
    manifest = (
        f"service_name: {name}\n"
        f"type: {svc_type.get(template, 'llm')}\n"
        f"region: us-east-1\n"
        f"\n"
        f"features:\n"
    )
    for k, v in features.items():
        manifest += f"  {k}: {'true' if v else 'false'}\n"
    manifest += (
        "\n"
        "compute:\n"
        "  size: medium\n"
        "\n"
        "scaling:\n"
        "  min_tasks: 1\n"
        "  max_tasks: 4\n"
        "  scale_up_cpu: 70\n"
        "  scale_down_cpu: 30\n"
    )
    (base / "ml-platform.yaml").write_text(manifest)


def _write_dockerfile(base: Path) -> None:
    (base / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "\n"
        "WORKDIR /app\n"
        "\n"
        "COPY requirements.txt* pyproject.toml* ./\n"
        "RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null \\\n"
        "    || pip install --no-cache-dir -e . 2>/dev/null \\\n"
        "    || true\n"
        "\n"
        "COPY . .\n"
        "\n"
        "RUN pip install --no-cache-dir .\n"
        "\n"
        "EXPOSE 8000\n"
        "\n"
        'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )


def _write_docker_compose(base: Path, name: str) -> None:
    compose = (
        "version: '3.8'\n"
        "\n"
        "services:\n"
        f"  {name}:\n"
        "    build: .\n"
        "    ports:\n"
        '      - "8000:8000"\n'
        "    environment:\n"
        "      - ML_PLATFORM_SERVICE_NAME=" + name + "\n"
        "\n"
        "  prometheus:\n"
        "    image: prom/prometheus:latest\n"
        "    ports:\n"
        '      - "9090:9090"\n'
        "    volumes:\n"
        "      - ./prometheus.yml:/etc/prometheus/prometheus.yml\n"
        "    profiles: [observability]\n"
        "\n"
        "  grafana:\n"
        "    image: grafana/grafana:latest\n"
        "    ports:\n"
        '      - "3000:3000"\n'
        "    profiles: [observability]\n"
        "\n"
        "  jaeger:\n"
        "    image: jaegertracing/all-in-one:latest\n"
        "    ports:\n"
        '      - "16686:16686"\n'
        "    profiles: [observability]\n"
    )
    (base / "docker-compose.dev.yml").write_text(compose)
