"""Tests for SageMaker deployment CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from ml_platform.cli.deploy_sagemaker import (
    _estimate_monthly_cost,
    _print_plan,
)
from ml_platform.cli.manifest import (
    FeaturesConfig,
    ProjectManifest,
    SageMakerConfig,
)


class TestSageMakerCostEstimate:
    def test_realtime_endpoint_cost(self) -> None:
        manifest = ProjectManifest(
            service_name="test-svc",
            deploy_target="sagemaker",
            sagemaker=SageMakerConfig(
                instance_type="ml.m5.large",
                min_instances=1,
                max_instances=4,
            ),
        )
        costs = _estimate_monthly_cost(manifest)
        assert "SageMaker Endpoint (light)" in costs
        assert "SageMaker Endpoint (peak)" in costs
        assert costs["SageMaker Endpoint (light)"] > 0
        assert costs["SageMaker Endpoint (peak)"] > costs["SageMaker Endpoint (light)"]

    def test_serverless_cost(self) -> None:
        manifest = ProjectManifest(
            service_name="test-svc",
            deploy_target="sagemaker",
            sagemaker=SageMakerConfig(serverless=True),
        )
        costs = _estimate_monthly_cost(manifest)
        assert "SageMaker Serverless (est.)" in costs
        assert "SageMaker Endpoint (light)" not in costs

    def test_dynamodb_cost_included(self) -> None:
        manifest = ProjectManifest(
            service_name="test-svc",
            deploy_target="sagemaker",
            features=FeaturesConfig(conversation_store=True, context_store=True),
        )
        costs = _estimate_monthly_cost(manifest)
        assert costs["DynamoDB"] == 2.0

    def test_no_s3_when_disabled(self) -> None:
        manifest = ProjectManifest(
            service_name="test-svc",
            deploy_target="sagemaker",
        )
        costs = _estimate_monthly_cost(manifest)
        assert costs["S3 (checkpoints)"] == 0.0


class TestSageMakerPrintPlan:
    def test_plan_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        manifest = ProjectManifest(
            service_name="test-bot",
            deploy_target="sagemaker",
            sagemaker=SageMakerConfig(
                instance_type="ml.g5.xlarge",
                min_instances=1,
                max_instances=2,
            ),
        )
        _print_plan(manifest)
        output = capsys.readouterr().out
        assert "SageMaker" in output
        assert "ml.g5.xlarge" in output
        assert "/mo" in output

    def test_serverless_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        manifest = ProjectManifest(
            service_name="serverless-bot",
            deploy_target="sagemaker",
            sagemaker=SageMakerConfig(
                serverless=True,
                serverless_memory_mb=4096,
                serverless_max_concurrency=20,
            ),
        )
        _print_plan(manifest)
        output = capsys.readouterr().out
        assert "Serverless" in output
        assert "4096" in output
        assert "Scale-to-zero" in output or "scale-to-zero" in output.lower()


class TestSageMakerManifest:
    def test_sagemaker_config_in_manifest(self, tmp_path: Path) -> None:
        from ml_platform.cli.manifest import load_manifest, save_manifest

        manifest = ProjectManifest(
            service_name="sm-test",
            deploy_target="sagemaker",
            sagemaker=SageMakerConfig(
                instance_type="ml.g5.xlarge",
                serverless=False,
                min_instances=2,
                max_instances=8,
            ),
        )
        path = tmp_path / "ml-platform.yaml"
        save_manifest(manifest, path)

        loaded = load_manifest(path)
        assert loaded.deploy_target == "sagemaker"
        assert loaded.sagemaker.instance_type == "ml.g5.xlarge"
        assert loaded.sagemaker.min_instances == 2
        assert loaded.sagemaker.max_instances == 8

    def test_default_deploy_target_is_ecs(self, tmp_path: Path) -> None:
        from ml_platform.cli.manifest import load_manifest, save_manifest

        manifest = ProjectManifest(service_name="default-test")
        path = tmp_path / "ml-platform.yaml"
        save_manifest(manifest, path)

        loaded = load_manifest(path)
        assert loaded.deploy_target == "ecs"


class TestSageMakerServingAdapter:
    @pytest.mark.asyncio
    async def test_ping_returns_200(self) -> None:
        import httpx
        from fastapi import FastAPI

        from ml_platform.serving.sagemaker import wrap_for_sagemaker

        app = FastAPI()

        @app.get("/health/live")
        async def health():
            return {"status": "healthy"}

        @app.post("/run")
        async def run():
            return {"result": "ok"}

        wrap_for_sagemaker(app, service_type="agent")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ping")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invocations_proxies_to_run(self) -> None:
        import httpx
        from fastapi import FastAPI, Request

        from ml_platform.serving.sagemaker import wrap_for_sagemaker

        app = FastAPI()

        @app.post("/run")
        async def run(request: Request):
            body = await request.json()
            return {"echo": body}

        wrap_for_sagemaker(app, service_type="agent")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/invocations",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["echo"]["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_invocations_proxies_to_predict_for_stateful(self) -> None:
        import httpx
        from fastapi import FastAPI, Request

        from ml_platform.serving.sagemaker import wrap_for_sagemaker

        app = FastAPI()

        @app.post("/predict")
        async def predict(request: Request):
            body = await request.json()
            return {"prediction": body.get("features", [])}

        wrap_for_sagemaker(app, service_type="stateful")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/invocations",
                json={"features": [1, 2, 3]},
            )
        assert resp.status_code == 200
        assert resp.json()["prediction"] == [1, 2, 3]


class TestSageMakerDockerfile:
    def test_dockerfile_generation(self) -> None:
        from ml_platform.serving.sagemaker import create_sagemaker_dockerfile

        content = create_sagemaker_dockerfile()
        assert "8080" in content
        assert "SAGEMAKER_BIND" in content
        assert "curl" in content


class TestCLITargets:
    def test_deploy_accepts_sagemaker(self) -> None:
        from ml_platform.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["deploy", "sagemaker", "--service-name", "test"])
        assert args.target == "sagemaker"

    def test_destroy_accepts_sagemaker(self) -> None:
        from ml_platform.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["destroy", "sagemaker", "--service-name", "test"])
        assert args.target == "sagemaker"
