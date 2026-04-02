"""Tests for destroy plan building and resource inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_platform.cli.destroy import _build_plan, _load_resource_manifest, _print_inventory


@pytest.fixture()
def resource_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / ".ml-platform"
    d.mkdir()
    monkeypatch.setattr("ml_platform.cli.destroy._RESOURCE_DIR", str(d))
    return d


class TestLoadResourceManifest:
    def test_returns_empty_when_missing(self, resource_dir: Path) -> None:
        result = _load_resource_manifest("nonexistent")
        assert result == {}

    def test_loads_json(self, resource_dir: Path) -> None:
        data = {
            "service_name": "my-svc",
            "region": "us-east-1",
            "stack_name": "my-svc-stack",
            "ecr_repository": "my-svc",
            "features": {"conversation_store": True, "context_store": False, "checkpointing": True},
        }
        (resource_dir / "my-svc.resources.json").write_text(json.dumps(data))
        result = _load_resource_manifest("my-svc")
        assert result["service_name"] == "my-svc"
        assert result["features"]["checkpointing"] is True


class TestBuildPlan:
    def test_plan_from_manifest(self, resource_dir: Path) -> None:
        data = {
            "service_name": "my-chatbot",
            "stack_name": "my-chatbot-stack",
            "ecr_repository": "my-chatbot",
            "features": {"conversation_store": True, "context_store": False, "checkpointing": False},
        }
        (resource_dir / "my-chatbot.resources.json").write_text(json.dumps(data))

        plan = _build_plan("my-chatbot", "us-east-1")
        assert plan.service_name == "my-chatbot"
        assert plan.stack_name == "my-chatbot-stack"
        assert plan.ecr_repository == "my-chatbot"
        assert any(r.kind == "DynamoDB table" and "conversations" in r.identifier for r in plan.resources)
        assert not any(r.kind == "S3 bucket" for r in plan.resources)

    def test_plan_falls_back_to_naming_convention(self, resource_dir: Path) -> None:
        plan = _build_plan("unknown-svc", "eu-west-1")
        assert plan.stack_name == "unknown-svc-stack"
        assert plan.ecr_repository == "unknown-svc"
        kinds = [r.kind for r in plan.resources]
        assert "CloudFormation stack" in kinds
        assert "ECR repository" in kinds
        assert "CloudWatch dashboard" in kinds

    def test_plan_with_all_features(self, resource_dir: Path) -> None:
        data = {
            "service_name": "full-svc",
            "stack_name": "full-svc-stack",
            "ecr_repository": "full-svc",
            "features": {"conversation_store": True, "context_store": True, "checkpointing": True},
        }
        (resource_dir / "full-svc.resources.json").write_text(json.dumps(data))
        plan = _build_plan("full-svc", "us-east-1")
        assert plan.s3_bucket == "ml-platform-full-svc-ckpt"
        assert len(plan.dynamo_tables) == 2


class TestPrintInventory:
    def test_prints_without_error(self, resource_dir: Path) -> None:
        plan = _build_plan("test-svc", "us-east-1")
        _print_inventory(plan)
