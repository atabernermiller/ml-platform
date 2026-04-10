"""``ml-platform deploy aws`` -- guided one-command AWS deployment.

Flow: read manifest -> generate plan -> show cost estimate ->
      user approval -> validate credentials -> build Docker image ->
      push to ECR -> apply CloudFormation -> write resource manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mypy_boto3_cloudformation.client import CloudFormationClient
    from mypy_boto3_ecr.client import ECRClient
    from mypy_boto3_sts.client import STSClient

from ml_platform.cli.manifest import (
    ProjectManifest,
    interactive_create,
    load_manifest,
    save_manifest,
)

_RESOURCE_DIR = ".ml-platform"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

_FARGATE_VCPU_HOUR = 0.04048
_FARGATE_GB_HOUR = 0.004445
_ALB_FIXED_HOUR = 0.0225
_DYNAMODB_WRITE_UNIT = 1.25e-6
_CW_LOGS_GB = 0.50
_HOURS_PER_MONTH = 730


def _estimate_monthly_cost(manifest: ProjectManifest) -> dict[str, float]:
    """Rough monthly cost estimate in USD for the deployment plan."""
    costs: dict[str, float] = {}

    vcpu = manifest.cpu / 1024
    gb = manifest.memory / 1024

    avg_tasks_light = max(manifest.scaling.min_tasks, 1)
    avg_tasks_peak = (manifest.scaling.min_tasks + manifest.scaling.max_tasks) / 2

    costs["ECS Fargate (light)"] = (
        (vcpu * _FARGATE_VCPU_HOUR + gb * _FARGATE_GB_HOUR)
        * _HOURS_PER_MONTH
        * avg_tasks_light
    )
    costs["ECS Fargate (peak)"] = (
        (vcpu * _FARGATE_VCPU_HOUR + gb * _FARGATE_GB_HOUR)
        * _HOURS_PER_MONTH
        * avg_tasks_peak
    )
    costs["ALB"] = _ALB_FIXED_HOUR * _HOURS_PER_MONTH

    ddb_cost = 0.0
    if manifest.features.conversation_store:
        ddb_cost += 1.0
    if manifest.features.context_store:
        ddb_cost += 1.0
    costs["DynamoDB"] = ddb_cost

    costs["CloudWatch Logs"] = 3.0
    costs["S3 (checkpoints)"] = 1.0 if manifest.features.checkpointing else 0.0

    return costs


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------


def _print_plan(manifest: ProjectManifest) -> None:
    svc = manifest.service_name
    costs = _estimate_monthly_cost(manifest)

    light_total = sum(
        v for k, v in costs.items() if "(peak)" not in k
    )
    peak_total = sum(
        v for k, v in costs.items() if "(light)" not in k
    )

    vcpu = manifest.cpu / 1024
    gb = manifest.memory / 1024

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Deployment Plan: {svc:<38} │
  │  Region: {manifest.region:<47} │
  ├──────────────────────────────────────────────────────────┤
  │                                                          │
  │  COMPUTE (auto-scaling enabled)                          │
  │    ECS Fargate with Application Load Balancer            │
  │      Size:      {vcpu:.1f} vCPU / {gb:.0f} GB ({manifest.compute_size}){' ' * (27 - len(manifest.compute_size))}│
  │      Min tasks: {manifest.scaling.min_tasks:<40} │
  │      Max tasks: {manifest.scaling.max_tasks:<40} │
  │      Scale up:  CPU > {manifest.scaling.scale_up_cpu}%{' ' * 33}│
  │      Scale down: CPU < {manifest.scaling.scale_down_cpu}% for 5 min{' ' * 22}│
  │                                                          │""")

    features_lines: list[str] = []
    if manifest.features.conversation_store:
        features_lines.append(f"    DynamoDB: {svc}-conversations (on-demand)")
    if manifest.features.context_store:
        features_lines.append(f"    DynamoDB: {svc}-context (on-demand)")
    if manifest.features.checkpointing:
        features_lines.append(f"    S3: ml-platform-{svc}-ckpt (versioned)")

    if features_lines:
        print("  │  STORAGE                                                │")
        for line in features_lines:
            print(f"  │  {line:<55}│")
        print("  │                                                          │")

    skipped: list[str] = []
    if not manifest.features.conversation_store:
        skipped.append("Conversation store (not needed)")
    if not manifest.features.context_store:
        skipped.append("Context store (not needed)")
    if not manifest.features.checkpointing:
        skipped.append("S3 checkpointing (not needed)")
    if skipped:
        print("  │  Skipped:                                                │")
        for s in skipped:
            print(f"  │    {s:<53}│")
        print("  │                                                          │")

    print(f"""  │  OBSERVABILITY                                           │
  │    CloudWatch Logs (EMF metrics extraction)              │
  │    CloudWatch Dashboard ({svc}-dashboard){' ' * max(0, 20 - len(svc))}│
  │    ALB health checks on /health (10s interval)           │
  │                                                          │
  │  RELIABILITY (included by default)                       │
  │    Auto-restart: unhealthy tasks replaced automatically  │
  │    Graceful shutdown: 30s deregistration delay           │
  │    Multi-AZ: tasks spread across availability zones      │
  │                                                          │
  │  ESTIMATED MONTHLY COST                                  │
  │                                                          │""")

    print(f"  │  Light load (~1 task):                                    │")
    for k, v in costs.items():
        if "(peak)" in k or v == 0:
            continue
        label = k.replace(" (light)", "")
        print(f"  │    {label:<30} ~${v:>6.0f}/mo{' ' * 8}│")
    print(f"  │    {'─' * 40}{' ' * 13}│")
    print(f"  │    {'Total':<30} ~${light_total:>6.0f}/mo{' ' * 8}│")
    print(f"  │                                                          │")
    print(f"  │  Peak load (~{manifest.scaling.max_tasks} tasks during spikes):{' ' * (24 - len(str(manifest.scaling.max_tasks)))}│")
    for k, v in costs.items():
        if "(light)" in k or v == 0:
            continue
        label = k.replace(" (peak)", "")
        print(f"  │    {label:<30} ~${v:>6.0f}/mo{' ' * 8}│")
    print(f"  │    {'─' * 40}{' ' * 13}│")
    print(f"  │    {'Total':<30} ~${peak_total:>6.0f}/mo{' ' * 8}│")
    print(f"  │                                                          │")
    print(f"  │  Note: LLM API costs are separate.                       │")
    print(f"  └──────────────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Approval flow
# ---------------------------------------------------------------------------


def _get_approval() -> str:
    """Ask for user approval. Returns 'y', 'n', or 'm'."""
    print("\n  [y] Accept and deploy")
    print("  [n] Cancel")
    print("  [m] Modify plan")
    try:
        choice = input("\n  Your choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"
    return choice


def _modify_plan(manifest: ProjectManifest) -> ProjectManifest:
    """Interactive plan modification."""
    print("\n  What would you like to change?")
    print(f"    [1] Compute size     (currently: {manifest.compute_size})")
    print(f"    [2] Scaling limits   (currently: min {manifest.scaling.min_tasks}, max {manifest.scaling.max_tasks})")
    print(f"    [3] Region           (currently: {manifest.region})")

    try:
        choice = input("\n  Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return manifest

    if choice == "1":
        print("    [1] small   (0.25 vCPU / 0.5 GB)")
        print("    [2] medium  (0.5 vCPU / 1 GB)")
        print("    [3] large   (1 vCPU / 2 GB)")
        print("    [4] xlarge  (2 vCPU / 4 GB)")
        sc = input("  Choice: ").strip()
        sizes = {"1": "small", "2": "medium", "3": "large", "4": "xlarge"}
        manifest.compute_size = sizes.get(sc, manifest.compute_size)

    elif choice == "2":
        try:
            mn = int(input(f"  Min tasks [{manifest.scaling.min_tasks}]: ").strip() or str(manifest.scaling.min_tasks))
            mx = int(input(f"  Max tasks [{manifest.scaling.max_tasks}]: ").strip() or str(manifest.scaling.max_tasks))
            manifest.scaling.min_tasks = mn
            manifest.scaling.max_tasks = max(mx, mn)
        except ValueError:
            print("  Invalid number, keeping current values.")

    elif choice == "3":
        r = input(f"  Region [{manifest.region}]: ").strip()
        if r:
            manifest.region = r

    return manifest


# ---------------------------------------------------------------------------
# Docker build + ECR push
# ---------------------------------------------------------------------------


def _ensure_dockerfile() -> None:
    """Generate a Dockerfile if one doesn't exist."""
    if Path("Dockerfile").exists():
        return

    print("  Generating Dockerfile...")
    dockerfile = """\
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt* pyproject.toml* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null \\
    || pip install --no-cache-dir -e . 2>/dev/null \\
    || true

COPY . .

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    Path("Dockerfile").write_text(dockerfile)
    print("  ✓  Dockerfile created")


def _docker_build_and_push(
    manifest: ProjectManifest, account_id: str
) -> str:
    """Build Docker image and push to ECR. Returns the image URI."""
    import boto3

    region = manifest.region
    svc = manifest.service_name
    repo_name = svc
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"

    ecr: ECRClient = boto3.client("ecr", region_name=region)

    try:
        ecr.create_repository(repositoryName=repo_name)
        print(f"  ✓  Created ECR repository: {repo_name}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"  ✓  ECR repository exists: {repo_name}")

    print("  Building Docker image...")
    _run_cmd(f"docker build -t {image_uri} .")

    print("  Authenticating with ECR...")
    token_resp = ecr.get_authorization_token()
    auth = token_resp["authorizationData"][0]
    registry = auth["proxyEndpoint"]
    _run_cmd(
        f"docker login --username AWS --password-stdin {registry}",
        stdin_data=_decode_ecr_token(auth["authorizationToken"]),
    )

    print("  Pushing image to ECR...")
    _run_cmd(f"docker push {image_uri}")
    print(f"  ✓  Image pushed: {image_uri}")

    return image_uri


def _decode_ecr_token(token: str) -> str:
    import base64

    decoded = base64.b64decode(token).decode()
    return decoded.split(":")[1]


def _run_cmd(cmd: str, stdin_data: str | None = None) -> None:
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        input=stdin_data,
    )
    if result.returncode != 0:
        print(f"  ✗  Command failed: {cmd}")
        print(f"      {result.stderr.strip()}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CloudFormation deployment
# ---------------------------------------------------------------------------


def _deploy_cloudformation(
    manifest: ProjectManifest, ecr_image_uri: str
) -> dict[str, str]:
    """Create or update a CloudFormation stack. Returns stack outputs."""
    import boto3

    from ml_platform.cli.cfn.template import generate_stack_template, stack_name

    cfn: CloudFormationClient = boto3.client("cloudformation", region_name=manifest.region)
    name = stack_name(manifest)
    template = generate_stack_template(manifest, ecr_image_uri=ecr_image_uri)
    template_body = json.dumps(template)

    try:
        cfn.describe_stacks(StackName=name)
        exists = True
    except cfn.exceptions.ClientError:
        exists = False

    print(f"  {'Updating' if exists else 'Creating'} CloudFormation stack: {name}")

    try:
        if exists:
            cfn.update_stack(
                StackName=name,
                TemplateBody=template_body,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            waiter = cfn.get_waiter("stack_update_complete")
        else:
            cfn.create_stack(
                StackName=name,
                TemplateBody=template_body,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            waiter = cfn.get_waiter("stack_create_complete")

        print("  Waiting for stack... (typically 3-5 minutes)")
        waiter.wait(StackName=name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
        print(f"  ✓  Stack {'updated' if exists else 'created'}: {name}")
    except Exception as exc:
        if "No updates are to be performed" in str(exc):
            print("  ✓  Stack is already up to date")
        else:
            raise

    resp = cfn.describe_stacks(StackName=name)
    outputs: dict[str, str] = {}
    for o in resp["Stacks"][0].get("Outputs", []):
        outputs[o["OutputKey"]] = o["OutputValue"]
    return outputs


# ---------------------------------------------------------------------------
# Resource manifest (for destroy)
# ---------------------------------------------------------------------------


def _write_resource_manifest(
    manifest: ProjectManifest, outputs: dict[str, str], ecr_image_uri: str
) -> Path:
    """Write ``.ml-platform/SERVICE.resources.json`` for destroy."""
    resource_dir = Path(_RESOURCE_DIR)
    resource_dir.mkdir(exist_ok=True)

    data = {
        "service_name": manifest.service_name,
        "region": manifest.region,
        "stack_name": f"{manifest.service_name}-stack",
        "ecr_repository": manifest.service_name,
        "ecr_image_uri": ecr_image_uri,
        "features": {
            "conversation_store": manifest.features.conversation_store,
            "context_store": manifest.features.context_store,
            "checkpointing": manifest.features.checkpointing,
        },
        "outputs": outputs,
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    path = resource_dir / f"{manifest.service_name}.resources.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_deploy(
    service_name: str = "",
    *,
    auto_approve: bool = False,
    manifest_path: str = "ml-platform.yaml",
) -> bool:
    """Execute the full deploy flow.

    Args:
        service_name: Override the service name (useful with ``--service-name``).
        auto_approve: Skip the approval prompt (for CI/CD).
        manifest_path: Path to the YAML manifest.

    Returns:
        ``True`` on success.
    """
    # Step 0: Load or create manifest
    try:
        manifest = load_manifest(manifest_path)
        if service_name:
            manifest.service_name = service_name
        print(f"\n  Loaded {manifest_path}")
    except FileNotFoundError:
        manifest = interactive_create(service_name)
        save_manifest(manifest, manifest_path)
        print(f"\n  ✓  Saved {manifest_path}")

    # Step 1: Show plan
    _print_plan(manifest)

    # Step 2: Approval loop
    if not auto_approve:
        while True:
            choice = _get_approval()
            if choice == "y":
                break
            elif choice == "n":
                print("\n  Cancelled.")
                return False
            elif choice == "m":
                manifest = _modify_plan(manifest)
                save_manifest(manifest, manifest_path)
                _print_plan(manifest)
            else:
                print("  Invalid choice.")
    else:
        print("\n  --yes flag set, skipping approval.")

    # Step 3: Validate credentials
    print("\n  Checking AWS credentials...")
    import boto3

    try:
        sts: STSClient = boto3.client("sts", region_name=manifest.region)
        identity = sts.get_caller_identity()
        account_id = identity["Account"]
        arn = identity["Arn"]
        print(f"  ✓  Authenticated as {arn}")
    except Exception as exc:
        print(f"  ✗  AWS credentials failed: {exc}")
        print("     Configure credentials via: aws configure")
        return False

    # Step 4: Docker build + ECR push
    _ensure_dockerfile()
    ecr_image_uri = _docker_build_and_push(manifest, account_id)

    # Step 5: Deploy CloudFormation
    outputs = _deploy_cloudformation(manifest, ecr_image_uri)

    # Step 6: Write resource manifest
    res_path = _write_resource_manifest(manifest, outputs, ecr_image_uri)

    # Step 7: Print results
    service_url = outputs.get("ServiceUrl", "")
    dashboard_url = outputs.get("DashboardUrl", "")

    print(f"""
  ✓  Deployment complete.

  Service URL:  http://{service_url}
  Health check: http://{service_url}/health
  Dashboard:    http://{service_url}/dashboard
  CW Dashboard: {dashboard_url}

  Resource manifest: {res_path}

  Next steps:
    • Verify:    curl http://{service_url}/health
    • Tear down: ml-platform destroy aws --service-name {manifest.service_name}
""")
    return True
