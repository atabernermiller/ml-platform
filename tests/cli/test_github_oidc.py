"""Tests for GitHub Actions OIDC bootstrap (IAM provider + deploy role)."""

from __future__ import annotations

import json
from typing import Generator

import boto3
import pytest
from moto import mock_aws

from ml_platform.cli.github_oidc import (
    GITHUB_OIDC_AUDIENCE,
    GITHUB_OIDC_URL,
    OIDCBootstrapResult,
    _build_deploy_policy,
    _build_trust_policy,
    _role_name_for_service,
    _validate_repo,
    bootstrap_github_oidc,
)


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestValidateRepo:
    def test_valid_format(self) -> None:
        owner, name = _validate_repo("myorg/my-service")
        assert owner == "myorg"
        assert name == "my-service"

    def test_strips_whitespace(self) -> None:
        owner, name = _validate_repo("  myorg/my-service  ")
        assert owner == "myorg"
        assert name == "my-service"

    @pytest.mark.parametrize(
        "bad_repo",
        ["just-a-name", "", "a/b/c", "/repo", "owner/"],
    )
    def test_invalid_format_raises(self, bad_repo: str) -> None:
        with pytest.raises(ValueError, match="Invalid repository format"):
            _validate_repo(bad_repo)


class TestRoleNameForService:
    def test_naming_convention(self) -> None:
        assert _role_name_for_service("my-api") == "ml-platform-my-api-deploy"


class TestBuildTrustPolicy:
    def test_structure(self) -> None:
        provider_arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
        policy = _build_trust_policy(provider_arn, "myorg/my-svc")

        assert policy["Version"] == "2012-10-17"
        stmt = policy["Statement"][0]
        assert stmt["Effect"] == "Allow"
        assert stmt["Principal"]["Federated"] == provider_arn
        assert stmt["Action"] == "sts:AssumeRoleWithWebIdentity"

        conditions = stmt["Condition"]
        assert conditions["StringEquals"][
            "token.actions.githubusercontent.com:aud"
        ] == GITHUB_OIDC_AUDIENCE
        assert conditions["StringLike"][
            "token.actions.githubusercontent.com:sub"
        ] == "repo:myorg/my-svc:*"


class TestBuildDeployPolicy:
    def test_has_required_sids(self) -> None:
        policy = _build_deploy_policy("my-svc", "us-east-1", "123456789012")
        sids = {s["Sid"] for s in policy["Statement"]}
        assert {"ECR", "ECRRepo", "CloudFormation", "S3", "DynamoDB",
                "IAMPassRole", "Logs"}.issubset(sids)

    def test_resources_scoped_to_service(self) -> None:
        policy = _build_deploy_policy("chat-api", "eu-west-1", "111111111111")
        for stmt in policy["Statement"]:
            resources = stmt.get("Resource", [])
            if isinstance(resources, str):
                resources = [resources]
            for r in resources:
                if r == "*":
                    continue
                assert "chat-api" in r or "MLPlatform" in r, (
                    f"Statement {stmt.get('Sid')} has unscoped resource: {r}"
                )


# ---------------------------------------------------------------------------
# Integration tests with moto
# ---------------------------------------------------------------------------


@mock_aws
class TestBootstrapGithubOIDC:
    def test_creates_provider_and_role(self) -> None:
        result = bootstrap_github_oidc(
            repo="myorg/my-service",
            service_name="my-service",
        )
        assert isinstance(result, OIDCBootstrapResult)
        assert result.created_provider is True
        assert result.created_role is True
        assert "oidc-provider" in result.provider_arn
        assert result.role_name == "ml-platform-my-service-deploy"
        assert result.role_arn.endswith(result.role_name)

    def test_idempotent_second_run(self) -> None:
        bootstrap_github_oidc(
            repo="myorg/my-service",
            service_name="my-service",
        )
        result = bootstrap_github_oidc(
            repo="myorg/my-service",
            service_name="my-service",
        )
        assert result.created_provider is False
        assert result.created_role is False

    def test_dry_run_creates_nothing(self) -> None:
        result = bootstrap_github_oidc(
            repo="myorg/my-service",
            service_name="my-service",
            dry_run=True,
        )
        assert result.created_provider is True
        assert result.created_role is True

        iam = boto3.client("iam", region_name="us-east-1")
        with pytest.raises(iam.exceptions.NoSuchEntityException):
            iam.get_role(RoleName="ml-platform-my-service-deploy")

    def test_role_trust_policy_scoped_to_repo(self) -> None:
        bootstrap_github_oidc(
            repo="acme/prod-api",
            service_name="prod-api",
        )
        iam = boto3.client("iam", region_name="us-east-1")
        role = iam.get_role(RoleName="ml-platform-prod-api-deploy")
        trust = role["Role"]["AssumeRolePolicyDocument"]

        stmt = trust["Statement"][0]
        sub_condition = stmt["Condition"]["StringLike"][
            "token.actions.githubusercontent.com:sub"
        ]
        assert sub_condition == "repo:acme/prod-api:*"

    def test_role_has_inline_policy(self) -> None:
        bootstrap_github_oidc(
            repo="myorg/svc",
            service_name="svc",
        )
        iam = boto3.client("iam", region_name="us-east-1")
        policies = iam.list_role_policies(RoleName="ml-platform-svc-deploy")
        names = policies["PolicyNames"]
        assert "ml-platform-svc-deploy-policy" in names

    def test_invalid_repo_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid repository format"):
            bootstrap_github_oidc(
                repo="bad-format",
                service_name="svc",
            )

    def test_explicit_region(self) -> None:
        result = bootstrap_github_oidc(
            repo="myorg/svc",
            service_name="svc",
            region="eu-west-1",
        )
        assert result.role_arn is not None

    def test_role_tags(self) -> None:
        bootstrap_github_oidc(
            repo="myorg/tagged-svc",
            service_name="tagged-svc",
        )
        iam = boto3.client("iam", region_name="us-east-1")
        tags = iam.list_role_tags(RoleName="ml-platform-tagged-svc-deploy")
        tag_map = {t["Key"]: t["Value"] for t in tags["Tags"]}
        assert tag_map["ml-platform:service"] == "tagged-svc"
        assert tag_map["ml-platform:repo"] == "myorg/tagged-svc"
        assert tag_map["managed-by"] == "ml-platform-bootstrap"

    def test_updates_trust_on_rerun_with_different_repo(self) -> None:
        bootstrap_github_oidc(
            repo="myorg/svc-v1",
            service_name="evolving-svc",
        )
        bootstrap_github_oidc(
            repo="myorg/svc-v2",
            service_name="evolving-svc",
        )
        iam = boto3.client("iam", region_name="us-east-1")
        role = iam.get_role(RoleName="ml-platform-evolving-svc-deploy")
        trust = role["Role"]["AssumeRolePolicyDocument"]
        sub = trust["Statement"][0]["Condition"]["StringLike"][
            "token.actions.githubusercontent.com:sub"
        ]
        assert sub == "repo:myorg/svc-v2:*"
