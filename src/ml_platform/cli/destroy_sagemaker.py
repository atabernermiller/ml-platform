"""``ml-platform destroy sagemaker`` -- tear down a SageMaker deployment.

Reads the resource manifest written by ``deploy sagemaker``, builds a
deletion plan, asks for confirmation, and removes all resources in the
correct dependency order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mypy_boto3_application_autoscaling.client import ApplicationAutoScalingClient
    from mypy_boto3_ecr.client import ECRClient
    from mypy_boto3_iam.client import IAMClient
    from mypy_boto3_sagemaker.client import SageMakerClient

_RESOURCE_DIR = ".ml-platform"


def _load_resource_manifest(service_name: str) -> dict[str, Any] | None:
    path = Path(_RESOURCE_DIR) / f"{service_name}.sagemaker.resources.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _print_inventory(res: dict[str, Any]) -> None:
    svc = res["service_name"]
    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  SageMaker Teardown Plan: {svc:<30} │
  │  Region: {res['region']:<47} │
  ├──────────────────────────────────────────────────────────┤
  │                                                          │
  │  Resources to delete (in order):                         │
  │                                                          │
  │    1. SageMaker Endpoint: {res['endpoint_name']:<30} │
  │    2. EndpointConfig:     {res['config_name']:<30} │
  │    3. SageMaker Model:    {res['model_name']:<30} │
  │    4. IAM Role:           {res['role_name']:<30} │
  │    5. ECR Repository:     {res['ecr_repository']:<30} │
  │                                                          │
  │  After deletion, monthly cost will be $0.                │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
""")


def _delete_endpoint(res: dict[str, Any]) -> bool:
    import boto3

    sm: SageMakerClient = boto3.client("sagemaker", region_name=res["region"])

    try:
        sm.delete_endpoint(EndpointName=res["endpoint_name"])
        print(f"  ✓  Deleting endpoint: {res['endpoint_name']}")

        print("  Waiting for endpoint deletion...")
        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(
            EndpointName=res["endpoint_name"],
            WaiterConfig={"Delay": 10, "MaxAttempts": 60},
        )
        print(f"  ✓  Endpoint deleted")
    except Exception as exc:
        if "Could not find" in str(exc) or "ValidationException" in str(exc):
            print(f"  ✓  Endpoint already deleted")
        else:
            print(f"  ✗  Failed to delete endpoint: {exc}")
            return False
    return True


def _delete_endpoint_config(res: dict[str, Any]) -> bool:
    import boto3

    sm: SageMakerClient = boto3.client("sagemaker", region_name=res["region"])
    try:
        sm.delete_endpoint_config(EndpointConfigName=res["config_name"])
        print(f"  ✓  EndpointConfig deleted: {res['config_name']}")
    except Exception as exc:
        if "Could not find" in str(exc) or "ValidationException" in str(exc):
            print(f"  ✓  EndpointConfig already deleted")
        else:
            print(f"  ✗  Failed to delete EndpointConfig: {exc}")
            return False
    return True


def _delete_model(res: dict[str, Any]) -> bool:
    import boto3

    sm: SageMakerClient = boto3.client("sagemaker", region_name=res["region"])
    try:
        sm.delete_model(ModelName=res["model_name"])
        print(f"  ✓  Model deleted: {res['model_name']}")
    except Exception as exc:
        if "Could not find" in str(exc) or "ValidationException" in str(exc):
            print(f"  ✓  Model already deleted")
        else:
            print(f"  ✗  Failed to delete model: {exc}")
            return False
    return True


def _delete_iam_role(res: dict[str, Any]) -> bool:
    import boto3

    iam: IAMClient = boto3.client("iam", region_name=res["region"])
    role_name = res["role_name"]

    try:
        attached = iam.list_attached_role_policies(RoleName=role_name)
        for pol in attached.get("AttachedPolicies", []):
            iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])

        inline = iam.list_role_policies(RoleName=role_name)
        for pol_name in inline.get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=pol_name)

        iam.delete_role(RoleName=role_name)
        print(f"  ✓  IAM role deleted: {role_name}")
    except Exception as exc:
        if "NoSuchEntity" in str(exc):
            print(f"  ✓  IAM role already deleted")
        else:
            print(f"  ✗  Failed to delete IAM role: {exc}")
            return False
    return True


def _delete_ecr_repository(res: dict[str, Any]) -> bool:
    import boto3

    ecr: ECRClient = boto3.client("ecr", region_name=res["region"])
    repo = res["ecr_repository"]
    try:
        ecr.delete_repository(repositoryName=repo, force=True)
        print(f"  ✓  ECR repository deleted: {repo}")
    except Exception as exc:
        if "RepositoryNotFoundException" in str(exc):
            print(f"  ✓  ECR repository already deleted")
        else:
            print(f"  ✗  Failed to delete ECR repository: {exc}")
            return False
    return True


def _remove_auto_scaling(res: dict[str, Any]) -> None:
    """Deregister the scalable target if it exists."""
    import boto3

    if res.get("serverless"):
        return

    try:
        aas: ApplicationAutoScalingClient = boto3.client("application-autoscaling", region_name=res["region"])
        resource_id = f"endpoint/{res['endpoint_name']}/variant/primary"
        aas.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        print(f"  ✓  Auto-scaling deregistered")
    except Exception:
        pass


def run_destroy_sagemaker(
    service_name: str,
    region: str | None = None,
    *,
    force: bool = False,
    verify_only: bool = False,
) -> bool:
    """Execute the full SageMaker teardown flow.

    Args:
        service_name: Service to destroy.
        region: AWS region.
        force: Skip name confirmation.
        verify_only: Only verify resources are deleted.

    Returns:
        ``True`` on success.
    """
    res = _load_resource_manifest(service_name)
    if res is None:
        print(f"\n  No SageMaker resource manifest found for '{service_name}'.")
        print(f"  Expected: {_RESOURCE_DIR}/{service_name}.sagemaker.resources.json")
        return False

    _print_inventory(res)

    if verify_only:
        print("  --verify-only: checking that resources are deleted...")
        import boto3

        sm: SageMakerClient = boto3.client("sagemaker", region_name=res["region"])
        try:
            sm.describe_endpoint(EndpointName=res["endpoint_name"])
            print(f"  ✗  Endpoint still exists: {res['endpoint_name']}")
            return False
        except Exception:
            print(f"  ✓  Endpoint does not exist")
        print("  ✓  Verification passed: resources appear deleted.")
        return True

    if not force:
        print(f"  Type the service name to confirm deletion: {service_name}")
        try:
            confirm = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False
        if confirm != service_name:
            print("  Name does not match. Aborting.")
            return False

    print("\n  Beginning teardown...\n")

    _remove_auto_scaling(res)
    ok = _delete_endpoint(res)
    ok = _delete_endpoint_config(res) and ok
    ok = _delete_model(res) and ok
    ok = _delete_iam_role(res) and ok
    ok = _delete_ecr_repository(res) and ok

    manifest_path = Path(_RESOURCE_DIR) / f"{service_name}.sagemaker.resources.json"
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"  ✓  Resource manifest removed")

    if ok:
        print(f"""
  ✓  SageMaker teardown complete for '{service_name}'.
     Monthly cost is now $0.

  To verify manually:
    aws sagemaker describe-endpoint --endpoint-name {res['endpoint_name']} --region {res['region']}
    # Should return "Could not find endpoint"
""")
    else:
        print(f"""
  ⚠  Some resources could not be deleted automatically.
  Check the AWS Console and delete manually:
    • SageMaker > Endpoints > {res['endpoint_name']}
    • SageMaker > Models > {res['model_name']}
    • IAM > Roles > {res['role_name']}
    • ECR > Repositories > {res['ecr_repository']}
""")

    return ok
