"""Tests for CI/CD pipeline generation."""

from __future__ import annotations

from ml_platform.cicd import generate_codepipeline_buildspec, generate_github_actions


class TestGenerateGitHubActions:
    def test_basic_workflow(self) -> None:
        yaml = generate_github_actions(service_name="my-api")
        assert "name: my-api CI/CD" in yaml
        assert "pytest" in yaml
        assert "ruff" in yaml
        assert "mypy" in yaml

    def test_with_ecr(self) -> None:
        yaml = generate_github_actions(
            service_name="my-api",
            ecr_repository="123456.dkr.ecr.us-east-1.amazonaws.com/my-api",
        )
        assert "ECR_REPOSITORY" in yaml
        assert "docker build" in yaml

    def test_with_ecs_deploy(self) -> None:
        yaml = generate_github_actions(
            service_name="my-api",
            ecr_repository="repo",
            deploy_ecs=True,
        )
        assert "deploy-ecs" in yaml
        assert "update-service" in yaml

    def test_with_lambda_deploy(self) -> None:
        yaml = generate_github_actions(
            service_name="my-api",
            deploy_lambda=True,
        )
        assert "deploy-lambda" in yaml
        assert "lambda.zip" in yaml

    def test_without_tests(self) -> None:
        yaml = generate_github_actions(service_name="my-api", run_tests=False)
        assert "pytest" not in yaml


class TestGenerateCodePipelineBuildspec:
    def test_basic_buildspec(self) -> None:
        yaml = generate_codepipeline_buildspec(service_name="my-api")
        assert "version: 0.2" in yaml
        assert "pip install" in yaml
        assert "my-api.zip" in yaml

    def test_with_tests(self) -> None:
        yaml = generate_codepipeline_buildspec(service_name="my-api", run_tests=True)
        assert "pytest" in yaml
        assert "ruff" in yaml

    def test_without_tests(self) -> None:
        yaml = generate_codepipeline_buildspec(service_name="my-api", run_tests=False)
        assert "pytest" not in yaml
