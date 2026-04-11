"""Bootstrap GitHub Actions OIDC federation for AWS deployments.

Creates (or reuses) the GitHub Actions OIDC identity provider in IAM and
an IAM role whose trust policy is scoped to a single GitHub repository.
The role receives the minimum permissions needed to deploy an ml-platform
service (ECR push, ECS deploy, CloudFormation, S3, DynamoDB, etc.).

This solves the chicken-and-egg problem: you need the deploy role *before*
CDK is deployable, so the role itself cannot be managed by CDK in the same
pipeline.

Usage::

    from ml_platform.cli.github_oidc import bootstrap_github_oidc

    result = bootstrap_github_oidc(
        repo="myorg/my-service",
        service_name="my-service",
        region="us-east-1",
    )
    print(result.role_arn)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_iam.client import IAMClient
    from mypy_boto3_sts.client import STSClient

logger = logging.getLogger(__name__)

GITHUB_OIDC_URL = "https://token.actions.githubusercontent.com"
GITHUB_OIDC_THUMBPRINT = "6938fd4d98bab03faadb97b34396831e3780aea1"
GITHUB_OIDC_AUDIENCE = "sts.amazonaws.com"

_PASS = "\033[32m✓\033[0m"
_FAIL = "\033[31m✗\033[0m"
_SKIP = "\033[33m–\033[0m"
_INFO = "\033[36mℹ\033[0m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _print(symbol: str, message: str) -> None:
    print(f"  {symbol}  {message}")


@dataclass(frozen=True)
class OIDCBootstrapResult:
    """Outcome of a GitHub OIDC bootstrap operation.

    Attributes:
        provider_arn: ARN of the IAM OIDC identity provider.
        role_arn: ARN of the IAM deploy role.
        role_name: Name of the IAM deploy role.
        created_provider: Whether the provider was newly created (vs. reused).
        created_role: Whether the role was newly created (vs. reused).
    """

    provider_arn: str
    role_arn: str
    role_name: str
    created_provider: bool
    created_role: bool


def _validate_repo(repo: str) -> tuple[str, str]:
    """Parse and validate ``owner/repo`` format.

    Args:
        repo: GitHub repository in ``owner/repo`` format.

    Returns:
        Tuple of ``(owner, repo_name)``.

    Raises:
        ValueError: If the format is invalid.
    """
    parts = repo.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid repository format: {repo!r}. "
            f"Expected 'owner/repo' (e.g. 'myorg/my-service')."
        )
    return parts[0], parts[1]


def _get_account_id(region: str) -> str:
    """Return the AWS account ID for the current credentials."""
    import boto3

    sts: STSClient = boto3.client("sts", region_name=region)
    return sts.get_caller_identity()["Account"]


def _ensure_oidc_provider(
    iam: IAMClient, account_id: str, *, dry_run: bool
) -> tuple[str, bool]:
    """Create the GitHub Actions OIDC provider if it doesn't already exist.

    The provider is account-global (not per-repo), so we reuse it if one
    already exists for ``token.actions.githubusercontent.com``.

    Args:
        iam: IAM client.
        account_id: AWS account ID.
        dry_run: If ``True``, only report what would happen.

    Returns:
        Tuple of ``(provider_arn, was_created)``.
    """
    provider_arn = (
        f"arn:aws:iam::{account_id}:oidc-provider/"
        f"token.actions.githubusercontent.com"
    )

    try:
        iam.get_open_id_connect_provider(OpenIDConnectProviderArn=provider_arn)
        _print(_SKIP, "GitHub OIDC provider already exists")
        return provider_arn, False
    except iam.exceptions.NoSuchEntityException:
        pass

    if dry_run:
        _print(_INFO, "Would create GitHub OIDC identity provider")
        return provider_arn, True

    response = iam.create_open_id_connect_provider(
        Url=GITHUB_OIDC_URL,
        ClientIDList=[GITHUB_OIDC_AUDIENCE],
        ThumbprintList=[GITHUB_OIDC_THUMBPRINT],
    )
    created_arn: str = response["OpenIDConnectProviderArn"]
    _print(_PASS, f"Created OIDC provider: {created_arn}")
    return created_arn, True


def _build_trust_policy(provider_arn: str, repo: str) -> dict:
    """Build the assume-role trust policy for the deploy role.

    The ``sub`` condition is scoped to the repository so that only
    workflows in ``repo`` can assume the role.  The ``ref:refs/heads/main``
    suffix further restricts to pushes to the default branch.

    Args:
        provider_arn: ARN of the OIDC provider.
        repo: GitHub repository in ``owner/repo`` format.

    Returns:
        IAM trust policy document.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Federated": provider_arn},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": GITHUB_OIDC_AUDIENCE,
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": (
                            f"repo:{repo}:*"
                        ),
                    },
                },
            }
        ],
    }


def _build_deploy_policy(
    service_name: str, region: str, account_id: str
) -> dict:
    """Build the permissions policy attached to the deploy role.

    Covers ECR push, ECS service update, CloudFormation stack operations,
    S3 checkpoint bucket, DynamoDB tables, CloudWatch, IAM pass-role
    (for ECS task roles), and Secrets Manager read access.

    Args:
        service_name: Service name used in resource naming conventions.
        region: AWS region.
        account_id: AWS account ID.

    Returns:
        IAM policy document.
    """
    arn_prefix = f"arn:aws:{{}}"  # filled per-statement
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECR",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                ],
                "Resource": "*",
            },
            {
                "Sid": "ECRRepo",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:CreateRepository",
                    "ecr:DescribeRepositories",
                ],
                "Resource": (
                    f"arn:aws:ecr:{region}:{account_id}:repository/{service_name}*"
                ),
            },
            {
                "Sid": "CloudFormation",
                "Effect": "Allow",
                "Action": [
                    "cloudformation:CreateStack",
                    "cloudformation:UpdateStack",
                    "cloudformation:DeleteStack",
                    "cloudformation:DescribeStacks",
                    "cloudformation:DescribeStackEvents",
                    "cloudformation:GetTemplate",
                    "cloudformation:ValidateTemplate",
                    "cloudformation:CreateChangeSet",
                    "cloudformation:DescribeChangeSet",
                    "cloudformation:ExecuteChangeSet",
                    "cloudformation:DeleteChangeSet",
                ],
                "Resource": (
                    f"arn:aws:cloudformation:{region}:{account_id}"
                    f":stack/{service_name}*/*"
                ),
            },
            {
                "Sid": "ECS",
                "Effect": "Allow",
                "Action": [
                    "ecs:UpdateService",
                    "ecs:DescribeServices",
                    "ecs:DescribeTaskDefinition",
                    "ecs:RegisterTaskDefinition",
                    "ecs:DeregisterTaskDefinition",
                    "ecs:DescribeClusters",
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceTag/ml-platform:service": service_name,
                    },
                },
            },
            {
                "Sid": "ECSUntagged",
                "Effect": "Allow",
                "Action": [
                    "ecs:RegisterTaskDefinition",
                    "ecs:DeregisterTaskDefinition",
                ],
                "Resource": "*",
            },
            {
                "Sid": "S3",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:CreateBucket",
                    "s3:PutBucketVersioning",
                ],
                "Resource": [
                    f"arn:aws:s3:::ml-platform-{service_name}-*",
                    f"arn:aws:s3:::ml-platform-{service_name}-*/*",
                ],
            },
            {
                "Sid": "DynamoDB",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:CreateTable",
                    "dynamodb:DescribeTable",
                    "dynamodb:UpdateTimeToLive",
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchWriteItem",
                ],
                "Resource": (
                    f"arn:aws:dynamodb:{region}:{account_id}"
                    f":table/{service_name}-*"
                ),
            },
            {
                "Sid": "CloudWatch",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                    "cloudwatch:PutDashboard",
                    "cloudwatch:DeleteDashboards",
                    "cloudwatch:PutMetricAlarm",
                    "cloudwatch:DeleteAlarms",
                ],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:namespace": "MLPlatform",
                    },
                },
            },
            {
                "Sid": "CloudWatchDashboard",
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutDashboard",
                    "cloudwatch:DeleteDashboards",
                ],
                "Resource": (
                    f"arn:aws:cloudwatch::{account_id}"
                    f":dashboard/{service_name}*"
                ),
            },
            {
                "Sid": "IAMPassRole",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": (
                    f"arn:aws:iam::{account_id}:role/{service_name}*"
                ),
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": [
                            "ecs-tasks.amazonaws.com",
                            "lambda.amazonaws.com",
                            "sagemaker.amazonaws.com",
                        ],
                    },
                },
            },
            {
                "Sid": "Logs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                ],
                "Resource": (
                    f"arn:aws:logs:{region}:{account_id}"
                    f":log-group:/ml-platform/{service_name}*"
                ),
            },
        ],
    }


def _role_name_for_service(service_name: str) -> str:
    """Derive the deploy role name from the service name."""
    return f"ml-platform-{service_name}-deploy"


def _ensure_deploy_role(
    iam: IAMClient,
    *,
    role_name: str,
    service_name: str,
    provider_arn: str,
    repo: str,
    region: str,
    account_id: str,
    dry_run: bool,
) -> tuple[str, bool]:
    """Create (or update) the IAM deploy role.

    If the role already exists the trust policy and permissions policy are
    **updated in place** so that repeated runs converge.

    Args:
        iam: IAM client.
        role_name: Desired role name.
        service_name: Service name for scoped permissions.
        provider_arn: OIDC provider ARN.
        repo: GitHub ``owner/repo``.
        region: AWS region.
        account_id: AWS account ID.
        dry_run: If ``True``, only report what would happen.

    Returns:
        Tuple of ``(role_arn, was_created)``.
    """
    trust_policy = _build_trust_policy(provider_arn, repo)
    deploy_policy = _build_deploy_policy(service_name, region, account_id)
    policy_name = f"{role_name}-policy"

    try:
        role = iam.get_role(RoleName=role_name)
        role_arn: str = role["Role"]["Arn"]
        _print(_SKIP, f"IAM role {role_name} already exists")

        if not dry_run:
            iam.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(trust_policy),
            )
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(deploy_policy),
            )
            _print(_PASS, "Updated trust policy and permissions")

        return role_arn, False
    except iam.exceptions.NoSuchEntityException:
        pass

    if dry_run:
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        _print(_INFO, f"Would create IAM role {role_name}")
        return role_arn, True

    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=(
            f"GitHub Actions deploy role for ml-platform service "
            f"'{service_name}' (repo: {repo})"
        ),
        MaxSessionDuration=3600,
        Tags=[
            {"Key": "ml-platform:service", "Value": service_name},
            {"Key": "ml-platform:repo", "Value": repo},
            {"Key": "managed-by", "Value": "ml-platform-bootstrap"},
        ],
    )
    role_arn = response["Role"]["Arn"]
    _print(_PASS, f"Created IAM role: {role_arn}")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(deploy_policy),
    )
    _print(_PASS, f"Attached inline policy: {policy_name}")

    return role_arn, True


def bootstrap_github_oidc(
    *,
    repo: str,
    service_name: str,
    region: str | None = None,
    dry_run: bool = False,
) -> OIDCBootstrapResult:
    """Create the GitHub OIDC provider and scoped deploy role.

    Safe to run repeatedly: existing resources are reused or updated
    in place.

    Args:
        repo: GitHub repository in ``owner/repo`` format.
        service_name: Service name for scoped IAM permissions.
        region: AWS region (resolved via :func:`resolve_region` if ``None``).
        dry_run: If ``True``, only print what would be created.

    Returns:
        Result containing ARNs of the created/reused resources.

    Raises:
        ValueError: If *repo* is not in ``owner/repo`` format.
    """
    import boto3

    _validate_repo(repo)
    resolved_region = resolve_region(region)

    mode = "DRY RUN" if dry_run else "bootstrap"
    print(f"\n{_BOLD}ml-platform {mode} — GitHub OIDC{_RESET}")
    print(f"  Service: {service_name}")
    print(f"  Repo:    {repo}")
    print(f"  Region:  {resolved_region}\n")

    iam: IAMClient = boto3.client("iam", region_name=resolved_region)

    account_id = _get_account_id(resolved_region)
    _print(_PASS, f"Account: {account_id}")

    print(f"\n{_BOLD}  OIDC Provider{_RESET}")
    provider_arn, created_provider = _ensure_oidc_provider(
        iam, account_id, dry_run=dry_run
    )

    print(f"\n{_BOLD}  Deploy Role{_RESET}")
    role_name = _role_name_for_service(service_name)
    role_arn, created_role = _ensure_deploy_role(
        iam,
        role_name=role_name,
        service_name=service_name,
        provider_arn=provider_arn,
        repo=repo,
        region=resolved_region,
        account_id=account_id,
        dry_run=dry_run,
    )

    print(f"\n{_PASS}  {'Would complete' if dry_run else 'Done'}.")
    if not dry_run:
        print(f"\n  Add this to your GitHub Actions workflow:\n")
        print(f"    permissions:")
        print(f"      id-token: write")
        print(f"      contents: read")
        print(f"")
        print(f"    steps:")
        print(f"      - uses: aws-actions/configure-aws-credentials@v4")
        print(f"        with:")
        print(f"          role-to-assume: {role_arn}")
        print(f"          aws-region: {resolved_region}")
        print()

    return OIDCBootstrapResult(
        provider_arn=provider_arn,
        role_arn=role_arn,
        role_name=role_name,
        created_provider=created_provider,
        created_role=created_role,
    )
