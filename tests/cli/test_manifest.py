"""Tests for ``ml_platform.cli.manifest`` -- YAML manifest reader/writer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml_platform.cli.manifest import (
    FeaturesConfig,
    ProjectManifest,
    ScalingConfig,
    load_manifest,
    save_manifest,
)


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "ml-platform.yaml"


class TestLoadManifest:
    def test_minimal(self, manifest_path: Path) -> None:
        manifest_path.write_text(yaml.dump({"service_name": "test-svc"}))
        m = load_manifest(manifest_path)
        assert m.service_name == "test-svc"
        assert m.service_type == "llm"
        assert m.compute_size == "medium"
        assert m.scaling.min_tasks == 1

    def test_full(self, manifest_path: Path) -> None:
        data = {
            "service_name": "my-agent",
            "type": "agent",
            "region": "eu-west-1",
            "features": {
                "conversation_store": True,
                "context_store": False,
                "checkpointing": True,
                "mlflow": False,
            },
            "compute": {"size": "large"},
            "scaling": {"min_tasks": 2, "max_tasks": 10, "scale_up_cpu": 60, "scale_down_cpu": 20},
        }
        manifest_path.write_text(yaml.dump(data))
        m = load_manifest(manifest_path)
        assert m.service_name == "my-agent"
        assert m.service_type == "agent"
        assert m.region == "eu-west-1"
        assert m.features.conversation_store is True
        assert m.features.checkpointing is True
        assert m.compute_size == "large"
        assert m.cpu == 1024
        assert m.memory == 2048
        assert m.scaling.min_tasks == 2
        assert m.scaling.max_tasks == 10

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_missing_service_name(self, manifest_path: Path) -> None:
        manifest_path.write_text(yaml.dump({"type": "llm"}))
        with pytest.raises(ValueError, match="service_name"):
            load_manifest(manifest_path)


class TestSaveManifest:
    def test_roundtrip(self, manifest_path: Path) -> None:
        m = ProjectManifest(
            service_name="roundtrip-svc",
            service_type="stateful",
            features=FeaturesConfig(context_store=True, checkpointing=True),
            compute_size="xlarge",
            scaling=ScalingConfig(min_tasks=0, max_tasks=20),
            region="ap-southeast-1",
        )
        save_manifest(m, manifest_path)

        loaded = load_manifest(manifest_path)
        assert loaded.service_name == "roundtrip-svc"
        assert loaded.service_type == "stateful"
        assert loaded.features.context_store is True
        assert loaded.compute_size == "xlarge"
        assert loaded.scaling.max_tasks == 20
        assert loaded.region == "ap-southeast-1"


class TestComputeProperties:
    @pytest.mark.parametrize(
        "size,expected_cpu,expected_mem",
        [("small", 256, 512), ("medium", 512, 1024), ("large", 1024, 2048), ("xlarge", 2048, 4096)],
    )
    def test_sizes(self, size: str, expected_cpu: int, expected_mem: int) -> None:
        m = ProjectManifest(service_name="test", compute_size=size)
        assert m.cpu == expected_cpu
        assert m.memory == expected_mem

    def test_unknown_size_defaults(self) -> None:
        m = ProjectManifest(service_name="test", compute_size="unknown")
        assert m.cpu == 512
        assert m.memory == 1024
