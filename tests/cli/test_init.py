"""Tests for ``ml-platform init`` template generation.

Verifies that each template generates the correct files and that
the generated code is syntactically valid Python.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from ml_platform.cli.templates import generate_template


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    return str(tmp_path)


class TestAgentTemplate:
    def test_generates_expected_files(self, output_dir: str) -> None:
        path = generate_template("my-agent", "agent", output_dir)
        assert (path / "app.py").exists()
        assert (path / "requirements.txt").exists()
        assert (path / "ml-platform.yaml").exists()
        assert (path / "Dockerfile").exists()
        assert (path / "docker-compose.dev.yml").exists()

    def test_app_is_valid_python(self, output_dir: str) -> None:
        path = generate_template("my-agent", "agent", output_dir)
        source = (path / "app.py").read_text()
        ast.parse(source)

    def test_manifest_is_valid_yaml(self, output_dir: str) -> None:
        path = generate_template("my-agent", "agent", output_dir)
        data = yaml.safe_load((path / "ml-platform.yaml").read_text())
        assert data["service_name"] == "my-agent"
        assert data["type"] == "agent"
        assert data["features"]["conversation_store"] is True

    def test_dockerfile_has_expose(self, output_dir: str) -> None:
        path = generate_template("my-agent", "agent", output_dir)
        content = (path / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_service_name_substituted(self, output_dir: str) -> None:
        path = generate_template("custom-name", "agent", output_dir)
        source = (path / "app.py").read_text()
        assert "custom-name" in source


class TestChatbotTemplate:
    def test_generates_files(self, output_dir: str) -> None:
        path = generate_template("my-chatbot", "chatbot", output_dir)
        assert (path / "app.py").exists()
        assert (path / "ml-platform.yaml").exists()
        data = yaml.safe_load((path / "ml-platform.yaml").read_text())
        assert data["type"] == "llm"
        assert data["features"]["conversation_store"] is True


class TestBanditTemplate:
    def test_generates_files(self, output_dir: str) -> None:
        path = generate_template("my-bandit", "bandit", output_dir)
        assert (path / "app.py").exists()
        assert (path / "ml-platform.yaml").exists()
        data = yaml.safe_load((path / "ml-platform.yaml").read_text())
        assert data["type"] == "stateful"
        assert data["features"]["context_store"] is True
        assert data["features"]["checkpointing"] is True


class TestDockerCompose:
    def test_contains_service(self, output_dir: str) -> None:
        path = generate_template("my-svc", "agent", output_dir)
        content = (path / "docker-compose.dev.yml").read_text()
        assert "my-svc:" in content
        assert "prometheus:" in content
        assert "grafana:" in content
        assert "jaeger:" in content
