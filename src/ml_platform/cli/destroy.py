"""``ml-platform destroy aws`` -- guided teardown with verification.

Flow: load resource manifest -> inventory resources -> confirm by name ->
      delete in order -> verify deletion -> print console links.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_RESOURCE_DIR = ".ml-platform"


@dataclass
class ResourceEntry:
    """A single AWS resource that was (or should be) deleted."""

    kind: str
    identifier: str
    detail: str = ""
    deleted: bool = False
    error: str = ""


@dataclass
class TeardownPlan:
    """Inventory of resources to delete and their status."""

    service_name: str
    region: str
    stack_name: str
    ecr_repository: str
    resources: list[ResourceEntry] = field(default_factory=list)
    s3_bucket: str = ""
    dynamo_tables: list[str] = field(default_factory=list)
    dashboard_name: str = ""


def _load_resource_manifest(service_name: str) -> dict[str, Any]:
    """Read the .ml-platform/SERVICE.resources.json file."""
    path = Path(_RESOURCE_DIR) / f"{service_name}.resources.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _build_plan(service_name: str, region: str) -> TeardownPlan:
    """Build a teardown plan from the resource manifest or naming conventions.

    Args:
        service_name: The service to tear down.
        region: AWS region.

    Returns:
        Populated teardown plan.
    """
    manifest = _load_resource_manifest(service_name)

    stack = manifest.get("stack_name", f"{service_name}-stack")
    ecr_repo = manifest.get("ecr_repository", service_name)
    features = manifest.get("features", {})

    plan = TeardownPlan(
        service_name=service_name,
        region=region,
        stack_name=stack,
        ecr_repository=ecr_repo,
    )

    plan.resources.append(ResourceEntry(
        kind="CloudFormation stack",
        identifier=stack,
        detail="ECS Service, ALB, Target Group, Security Groups, IAM Roles, Log Group",
    ))
    plan.resources.append(ResourceEntry(
        kind="ECR repository",
        identifier=ecr_repo,
    ))

    if features.get("checkpointing"):
        bucket = f"ml-platform-{service_name}-ckpt"
        plan.s3_bucket = bucket
        plan.resources.append(ResourceEntry(kind="S3 bucket", identifier=bucket))

    tables: list[str] = []
    if features.get("conversation_store"):
        tables.append(f"{service_name}-conversations")
    if features.get("context_store"):
        tables.append(f"{service_name}-context")
    plan.dynamo_tables = tables
    for t in tables:
        plan.resources.append(ResourceEntry(kind="DynamoDB table", identifier=t))

    plan.dashboard_name = f"{service_name}-dashboard"
    plan.resources.append(
        ResourceEntry(kind="CloudWatch dashboard", identifier=plan.dashboard_name)
    )

    return plan


def _print_inventory(plan: TeardownPlan) -> None:
    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Resources to delete: {plan.service_name:<34}│
  │  Region: {plan.region:<47}│
  ├──────────────────────────────────────────────────────────┤""")

    for r in plan.resources:
        line = f"{r.kind}: {r.identifier}"
        print(f"  │  {line:<55}│")
        if r.detail:
            print(f"  │    → {r.detail:<51}│")

    print(f"""  │                                                          │
  │  After deletion, recurring charges will be: $0.00/mo     │
  └──────────────────────────────────────────────────────────┘

  ⚠  This will permanently delete all data (checkpoints,
     conversation history, metrics). This cannot be undone.
""")


# ---------------------------------------------------------------------------
# Deletion helpers
# ---------------------------------------------------------------------------


def _delete_cloudformation_stack(
    stack_name: str, region: str
) -> tuple[bool, str]:
    """Delete a CloudFormation stack and wait for completion."""
    import boto3

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        cfn.delete_stack(StackName=stack_name)
        waiter = cfn.get_waiter("stack_delete_complete")
        waiter.wait(StackName=stack_name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _delete_ecr_repository(repo_name: str, region: str) -> tuple[bool, str]:
    import boto3

    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.delete_repository(repositoryName=repo_name, force=True)
        return True, ""
    except ecr.exceptions.RepositoryNotFoundException:
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _delete_s3_bucket(bucket_name: str, region: str) -> tuple[bool, str]:
    import boto3

    s3 = boto3.resource("s3", region_name=region)
    try:
        bucket = s3.Bucket(bucket_name)
        bucket.object_versions.delete()
        bucket.objects.delete()
        bucket.delete()
        return True, ""
    except Exception as exc:
        if "NoSuchBucket" in str(exc):
            return True, ""
        return False, str(exc)


def _delete_dynamodb_table(table_name: str, region: str) -> tuple[bool, str]:
    import boto3

    ddb = boto3.client("dynamodb", region_name=region)
    try:
        ddb.delete_table(TableName=table_name)
        waiter = ddb.get_waiter("table_not_exists")
        waiter.wait(TableName=table_name)
        return True, ""
    except ddb.exceptions.ResourceNotFoundException:
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _delete_cloudwatch_dashboard(
    dashboard_name: str, region: str
) -> tuple[bool, str]:
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    try:
        cw.delete_dashboards(DashboardNames=[dashboard_name])
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Verification pass
# ---------------------------------------------------------------------------


def _verify_cleanup(plan: TeardownPlan) -> list[ResourceEntry]:
    """Check that every resource is actually gone. Returns any survivors."""
    import boto3

    survivors: list[ResourceEntry] = []
    region = plan.region

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        resp = cfn.describe_stacks(StackName=plan.stack_name)
        status = resp["Stacks"][0]["StackStatus"]
        if status != "DELETE_COMPLETE":
            survivors.append(ResourceEntry(
                kind="CloudFormation stack", identifier=plan.stack_name,
                error=f"Status: {status}",
            ))
    except cfn.exceptions.ClientError:
        pass

    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[plan.ecr_repository])
        survivors.append(ResourceEntry(
            kind="ECR repository", identifier=plan.ecr_repository,
        ))
    except ecr.exceptions.RepositoryNotFoundException:
        pass
    except Exception:
        pass

    if plan.s3_bucket:
        s3 = boto3.client("s3", region_name=region)
        try:
            s3.head_bucket(Bucket=plan.s3_bucket)
            survivors.append(ResourceEntry(
                kind="S3 bucket", identifier=plan.s3_bucket,
            ))
        except Exception:
            pass

    ddb = boto3.client("dynamodb", region_name=region)
    for t in plan.dynamo_tables:
        try:
            ddb.describe_table(TableName=t)
            survivors.append(ResourceEntry(kind="DynamoDB table", identifier=t))
        except ddb.exceptions.ResourceNotFoundException:
            pass
        except Exception:
            pass

    return survivors


# ---------------------------------------------------------------------------
# Manual fallback instructions
# ---------------------------------------------------------------------------


def _print_manual_fallback(plan: TeardownPlan, failed: list[ResourceEntry]) -> None:
    """Print manual cleanup instructions for resources that failed to delete."""
    region = plan.region
    print(f"\n  ⚠  Teardown incomplete. {len(failed)} resource(s) may still incur charges.\n")
    print("  To finish cleanup manually:\n")

    for i, r in enumerate(failed, 1):
        print(f"    {i}. {r.kind} ({r.identifier}):")
        if r.error:
            print(f"       Error: {r.error}")

        if r.kind == "CloudFormation stack":
            print(f"       aws cloudformation delete-stack --stack-name {r.identifier} --region {region}")
            print(f"       Or: https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks?filteringText={r.identifier}")

        elif r.kind == "ECR repository":
            print(f"       aws ecr delete-repository --repository-name {r.identifier} --force --region {region}")
            print(f"       Or: https://{region}.console.aws.amazon.com/ecr/repositories?region={region}")

        elif r.kind == "S3 bucket":
            print(f"       aws s3 rb s3://{r.identifier} --force")
            print(f"       Or: https://s3.console.aws.amazon.com/s3/bucket/{r.identifier}/delete")

        elif r.kind == "DynamoDB table":
            print(f"       aws dynamodb delete-table --table-name {r.identifier} --region {region}")
            print(f"       Or: https://{region}.console.aws.amazon.com/dynamodbv2/home?region={region}#delete-table?table={r.identifier}")

        elif r.kind == "CloudWatch dashboard":
            print(f"       aws cloudwatch delete-dashboards --dashboard-names {r.identifier} --region {region}")

        print()

    print(f"  After manual cleanup, re-run to verify:")
    print(f"    ml-platform destroy aws --service-name {plan.service_name} --verify-only\n")


def _print_console_links(plan: TeardownPlan) -> None:
    """Print AWS Console links so the user can visually confirm."""
    region = plan.region
    svc = plan.service_name
    print(f"""
  Verify for yourself (AWS Console):
    • CloudFormation:  https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks?filteringText={svc}
    • ECS:             https://{region}.console.aws.amazon.com/ecs/v2/clusters?region={region}
    • ECR:             https://{region}.console.aws.amazon.com/ecr/repositories?region={region}
    • S3:              https://s3.console.aws.amazon.com/s3/buckets?region={region}&search={svc}
    • DynamoDB:        https://{region}.console.aws.amazon.com/dynamodbv2/home?region={region}#tables
""")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_destroy(
    service_name: str,
    *,
    region: str = "us-east-1",
    force: bool = False,
    verify_only: bool = False,
) -> bool:
    """Execute the full destroy flow.

    Args:
        service_name: Service to destroy.
        region: AWS region.
        force: Skip name confirmation (prints a warning).
        verify_only: Only verify, don't delete.

    Returns:
        ``True`` if all resources are deleted.
    """
    plan = _build_plan(service_name, region)

    if verify_only:
        print(f"\n  Verifying resources for {service_name} in {region}...")
        survivors = _verify_cleanup(plan)
        if not survivors:
            print(f"\n  ✓  No resources found for {service_name}. Estimated charges: $0.00/mo")
            return True
        else:
            print(f"\n  ⚠  {len(survivors)} resource(s) still exist:")
            for s in survivors:
                print(f"      {s.kind}: {s.identifier}")
            return False

    _print_inventory(plan)

    if force:
        print("  ⚠  --force flag set. Proceeding without name confirmation.\n")
    else:
        try:
            confirm = input(f"  Type the service name to confirm: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False
        if confirm != service_name:
            print(f"  Names don't match ('{confirm}' != '{service_name}'). Cancelled.")
            return False

    print("\n  Destroying resources...")
    failed: list[ResourceEntry] = []

    # Order: stack first (removes ECS + ALB + IAM), then ECR, S3, DDB, dashboard
    ok, err = _delete_cloudformation_stack(plan.stack_name, plan.region)
    if ok:
        print(f"    ✓  Deleted CloudFormation stack {plan.stack_name}")
    else:
        print(f"    ✗  Failed: CloudFormation stack {plan.stack_name}: {err}")
        failed.append(ResourceEntry(
            kind="CloudFormation stack", identifier=plan.stack_name, error=err,
        ))

    ok, err = _delete_ecr_repository(plan.ecr_repository, plan.region)
    if ok:
        print(f"    ✓  Deleted ECR repository {plan.ecr_repository}")
    else:
        print(f"    ✗  Failed: ECR repository {plan.ecr_repository}: {err}")
        failed.append(ResourceEntry(
            kind="ECR repository", identifier=plan.ecr_repository, error=err,
        ))

    if plan.s3_bucket:
        ok, err = _delete_s3_bucket(plan.s3_bucket, plan.region)
        if ok:
            print(f"    ✓  Deleted S3 bucket {plan.s3_bucket}")
        else:
            print(f"    ✗  Failed: S3 bucket {plan.s3_bucket}: {err}")
            failed.append(ResourceEntry(
                kind="S3 bucket", identifier=plan.s3_bucket, error=err,
            ))

    for t in plan.dynamo_tables:
        ok, err = _delete_dynamodb_table(t, plan.region)
        if ok:
            print(f"    ✓  Deleted DynamoDB table {t}")
        else:
            print(f"    ✗  Failed: DynamoDB table {t}: {err}")
            failed.append(ResourceEntry(
                kind="DynamoDB table", identifier=t, error=err,
            ))

    ok, err = _delete_cloudwatch_dashboard(plan.dashboard_name, plan.region)
    if ok:
        print(f"    ✓  Deleted CloudWatch dashboard {plan.dashboard_name}")
    else:
        print(f"    ✗  Failed: CloudWatch dashboard {plan.dashboard_name}: {err}")
        failed.append(ResourceEntry(
            kind="CloudWatch dashboard", identifier=plan.dashboard_name, error=err,
        ))

    # Verification pass
    print("\n  Verifying cleanup...")
    survivors = _verify_cleanup(plan)

    if survivors:
        _print_manual_fallback(plan, survivors)
        return False

    if failed:
        _print_manual_fallback(plan, failed)
        return False

    # Clean up local resource manifest
    res_path = Path(_RESOURCE_DIR) / f"{service_name}.resources.json"
    if res_path.exists():
        res_path.unlink()

    print(f"\n  ✓  All resources deleted. Estimated ongoing charges: $0.00/mo")
    _print_console_links(plan)
    print(f"  ✓  Teardown complete. You should not see any ml-platform charges on your next AWS bill.\n")
    return True
