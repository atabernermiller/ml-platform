"""``ml-platform deploy sagemaker`` -- guided SageMaker endpoint deployment.

Flow: read manifest -> generate plan -> show cost estimate ->
      user approval -> validate credentials -> build Docker image ->
      push to ECR -> create IAM role -> create SageMaker Model ->
      create EndpointConfig -> create Endpoint -> optional auto-scaling
      -> write resource manifest.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ml_platform.cli.manifest import (
    ProjectManifest,
    SageMakerConfig,
    interactive_create,
    load_manifest,
    save_manifest,
)

_RESOURCE_DIR = ".ml-platform"

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

_SM_INSTANCE_COST: dict[str, float] = {
    "ml.t2.medium": 0.065,
    "ml.m5.large": 0.134,
    "ml.m5.xlarge": 0.269,
    "ml.m5.2xlarge": 0.538,
    "ml.c5.large": 0.119,
    "ml.c5.xlarge": 0.238,
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 1.691,
    "ml.g5.4xlarge": 2.387,
    "ml.p3.2xlarge": 4.284,
}

_HOURS_PER_MONTH = 730


def _estimate_monthly_cost(manifest: ProjectManifest) -> dict[str, float]:
    """Rough monthly cost estimate in USD for a SageMaker endpoint."""
    sm = manifest.sagemaker
    costs: dict[str, float] = {}

    if sm.serverless:
        costs["SageMaker Serverless (est.)"] = 15.0
    else:
        hourly = _SM_INSTANCE_COST.get(sm.instance_type, 0.134)
        costs["SageMaker Endpoint (light)"] = (
            hourly * _HOURS_PER_MONTH * sm.min_instances
        )
        avg_peak = (sm.min_instances + sm.max_instances) / 2
        costs["SageMaker Endpoint (peak)"] = (
            hourly * _HOURS_PER_MONTH * avg_peak
        )

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
    sm = manifest.sagemaker
    costs = _estimate_monthly_cost(manifest)

    light_total = sum(v for k, v in costs.items() if "(peak)" not in k)
    peak_total = sum(v for k, v in costs.items() if "(light)" not in k)

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  SageMaker Deployment Plan: {svc:<28} │
  │  Region: {manifest.region:<47} │
  ├──────────────────────────────────────────────────────────┤
  │                                                          │""")

    if sm.serverless:
        print(f"""  │  COMPUTE (SageMaker Serverless Inference)               │
  │    Memory:        {sm.serverless_memory_mb} MB{' ' * (35 - len(str(sm.serverless_memory_mb)))}│
  │    Max concurrency: {sm.serverless_max_concurrency:<36} │
  │    Scale-to-zero: yes (pay per request)                │
  │                                                          │""")
    else:
        print(f"""  │  COMPUTE (SageMaker Real-Time Endpoint)                 │
  │    Instance type:  {sm.instance_type:<37} │
  │    Min instances:  {sm.min_instances:<37} │
  │    Max instances:  {sm.max_instances:<37} │
  │    Auto-scaling:   target {sm.target_invocations_per_instance} invocations/instance{' ' * max(0, 14 - len(str(sm.target_invocations_per_instance)))}│
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

    print(f"  │  ESTIMATED MONTHLY COST                                  │")
    print(f"  │                                                          │")
    if sm.serverless:
        for k, v in costs.items():
            if v == 0:
                continue
            print(f"  │    {k:<30} ~${v:>6.0f}/mo{' ' * 8}│")
        print(f"  │    {'─' * 40}{' ' * 13}│")
        print(f"  │    {'Total':<30} ~${light_total:>6.0f}/mo{' ' * 8}│")
    else:
        print(f"  │  Light load ({sm.min_instances} instance{'s' if sm.min_instances > 1 else ''}):{' ' * (36 - len(str(sm.min_instances)))}│")
        for k, v in costs.items():
            if "(peak)" in k or v == 0:
                continue
            label = k.replace(" (light)", "")
            print(f"  │    {label:<30} ~${v:>6.0f}/mo{' ' * 8}│")
        print(f"  │    {'─' * 40}{' ' * 13}│")
        print(f"  │    {'Total':<30} ~${light_total:>6.0f}/mo{' ' * 8}│")
        print(f"  │                                                          │")
        print(f"  │  Peak load ({sm.max_instances} instances):{' ' * (35 - len(str(sm.max_instances)))}│")
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
# Approval
# ---------------------------------------------------------------------------


def _get_approval() -> str:
    print("\n  [y] Accept and deploy")
    print("  [n] Cancel")
    try:
        choice = input("\n  Your choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "n"
    return choice


# ---------------------------------------------------------------------------
# Docker build + ECR push
# ---------------------------------------------------------------------------


def _ensure_sagemaker_dockerfile() -> None:
    if Path("Dockerfile.sagemaker").exists():
        return

    from ml_platform.serving.sagemaker import create_sagemaker_dockerfile

    Path("Dockerfile.sagemaker").write_text(create_sagemaker_dockerfile())
    print("  ✓  Dockerfile.sagemaker created")


def _docker_build_and_push(
    manifest: ProjectManifest, account_id: str
) -> str:
    """Build SageMaker-compatible Docker image and push to ECR."""
    import boto3

    region = manifest.region
    repo_name = f"{manifest.service_name}-sagemaker"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"

    ecr = boto3.client("ecr", region_name=region)

    try:
        ecr.create_repository(repositoryName=repo_name)
        print(f"  ✓  Created ECR repository: {repo_name}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"  ✓  ECR repository exists: {repo_name}")

    print("  Building SageMaker Docker image...")
    _run_cmd(f"docker build -t {image_uri} -f Dockerfile.sagemaker .")

    print("  Authenticating with ECR...")
    token_resp = ecr.get_authorization_token()
    auth = token_resp["authorizationData"][0]
    registry = auth["proxyEndpoint"]
    decoded = base64.b64decode(auth["authorizationToken"]).decode()
    password = decoded.split(":")[1]
    _run_cmd(
        f"docker login --username AWS --password-stdin {registry}",
        stdin_data=password,
    )

    print("  Pushing image to ECR...")
    _run_cmd(f"docker push {image_uri}")
    print(f"  ✓  Image pushed: {image_uri}")

    return image_uri


def _run_cmd(cmd: str, stdin_data: str | None = None) -> None:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, input=stdin_data
    )
    if result.returncode != 0:
        print(f"  ✗  Command failed: {cmd}")
        print(f"      {result.stderr.strip()}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# IAM role for SageMaker
# ---------------------------------------------------------------------------


def _ensure_sagemaker_role(
    svc: str, region: str, manifest: ProjectManifest
) -> str:
    """Create or retrieve the IAM role for the SageMaker endpoint."""
    import boto3

    iam = boto3.client("iam", region_name=region)
    role_name = f"{svc}-sagemaker-role"

    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    try:
        resp = iam.get_role(RoleName=role_name)
        print(f"  ✓  IAM role exists: {role_name}")
        return resp["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=trust_policy,
        Description=f"ml-platform SageMaker role for {svc}",
    )
    role_arn = resp["Role"]["Arn"]

    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    )

    inline_statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": "cloudwatch:PutMetricData",
            "Resource": "*",
        },
        {
            "Effect": "Allow",
            "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": "*",
        },
    ]

    if manifest.features.conversation_store or manifest.features.context_store:
        tables = []
        if manifest.features.conversation_store:
            tables.append(f"arn:aws:dynamodb:*:*:table/{svc}-conversations")
        if manifest.features.context_store:
            tables.append(f"arn:aws:dynamodb:*:*:table/{svc}-context")
        inline_statements.append({
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem", "dynamodb:GetItem",
                "dynamodb:DeleteItem", "dynamodb:Query",
            ],
            "Resource": tables,
        })

    if manifest.features.checkpointing:
        inline_statements.append({
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::ml-platform-{svc}-ckpt",
                f"arn:aws:s3:::ml-platform-{svc}-ckpt/*",
            ],
        })

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{svc}-sagemaker-policy",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": inline_statements,
        }),
    )

    print(f"  ✓  Created IAM role: {role_name}")

    print("  Waiting for IAM role propagation (10s)...")
    time.sleep(10)

    return role_arn


# ---------------------------------------------------------------------------
# SageMaker endpoint creation
# ---------------------------------------------------------------------------


def _create_endpoint(
    manifest: ProjectManifest,
    ecr_image_uri: str,
    role_arn: str,
) -> str:
    """Create SageMaker Model, EndpointConfig, and Endpoint."""
    import boto3

    sm_client = boto3.client("sagemaker", region_name=manifest.region)
    svc = manifest.service_name
    sm_cfg = manifest.sagemaker

    model_name = f"{svc}-model"
    config_name = f"{svc}-config"
    endpoint_name = f"{svc}-endpoint"

    env_vars: dict[str, str] = {
        "ML_PLATFORM_SERVICE_NAME": svc,
        "ML_PLATFORM_AWS_REGION": manifest.region,
        "SAGEMAKER_BIND": "true",
    }

    print(f"  Creating SageMaker Model: {model_name}")
    try:
        sm_client.delete_model(ModelName=model_name)
    except sm_client.exceptions.ClientError:
        pass
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image_uri,
            "Environment": env_vars,
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"  ✓  Model created: {model_name}")

    print(f"  Creating EndpointConfig: {config_name}")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
    except sm_client.exceptions.ClientError:
        pass

    variant: dict[str, Any] = {
        "VariantName": "primary",
        "ModelName": model_name,
    }

    if sm_cfg.serverless:
        variant["ServerlessConfig"] = {
            "MemorySizeInMB": sm_cfg.serverless_memory_mb,
            "MaxConcurrency": sm_cfg.serverless_max_concurrency,
        }
    else:
        variant["InstanceType"] = sm_cfg.instance_type
        variant["InitialInstanceCount"] = sm_cfg.initial_instance_count
        variant["InitialVariantWeight"] = 1.0

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[variant],
    )
    print(f"  ✓  EndpointConfig created: {config_name}")

    print(f"  Creating Endpoint: {endpoint_name}")
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"  Updating existing endpoint...")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm_client.exceptions.ClientError:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    print("  Waiting for endpoint to be InService... (typically 5-10 minutes)")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 15, "MaxAttempts": 60},
    )
    print(f"  ✓  Endpoint InService: {endpoint_name}")

    return endpoint_name


def _setup_auto_scaling(manifest: ProjectManifest, endpoint_name: str) -> None:
    """Configure Application Auto Scaling for the SageMaker endpoint."""
    import boto3

    sm_cfg = manifest.sagemaker
    if sm_cfg.serverless or sm_cfg.min_instances == sm_cfg.max_instances:
        return

    aas = boto3.client("application-autoscaling", region_name=manifest.region)
    resource_id = f"endpoint/{endpoint_name}/variant/primary"

    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=sm_cfg.min_instances,
        MaxCapacity=sm_cfg.max_instances,
    )

    aas.put_scaling_policy(
        PolicyName=f"{manifest.service_name}-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "TargetValue": float(sm_cfg.target_invocations_per_instance),
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    print(
        f"  ✓  Auto-scaling configured: {sm_cfg.min_instances}-{sm_cfg.max_instances} "
        f"instances (target: {sm_cfg.target_invocations_per_instance} inv/instance)"
    )


# ---------------------------------------------------------------------------
# Resource manifest
# ---------------------------------------------------------------------------


def _write_resource_manifest(
    manifest: ProjectManifest, endpoint_name: str, ecr_image_uri: str, role_arn: str
) -> Path:
    """Write ``.ml-platform/SERVICE.sagemaker.resources.json``."""
    resource_dir = Path(_RESOURCE_DIR)
    resource_dir.mkdir(exist_ok=True)

    svc = manifest.service_name
    data = {
        "service_name": svc,
        "deploy_target": "sagemaker",
        "region": manifest.region,
        "endpoint_name": endpoint_name,
        "model_name": f"{svc}-model",
        "config_name": f"{svc}-config",
        "role_name": f"{svc}-sagemaker-role",
        "role_arn": role_arn,
        "ecr_repository": f"{svc}-sagemaker",
        "ecr_image_uri": ecr_image_uri,
        "serverless": manifest.sagemaker.serverless,
        "features": {
            "conversation_store": manifest.features.conversation_store,
            "context_store": manifest.features.context_store,
            "checkpointing": manifest.features.checkpointing,
        },
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    path = resource_dir / f"{svc}.sagemaker.resources.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_deploy_sagemaker(
    service_name: str = "",
    *,
    auto_approve: bool = False,
    manifest_path: str = "ml-platform.yaml",
) -> bool:
    """Execute the full SageMaker deploy flow.

    Args:
        service_name: Override the service name.
        auto_approve: Skip the approval prompt.
        manifest_path: Path to the YAML manifest.

    Returns:
        ``True`` on success.
    """
    try:
        manifest = load_manifest(manifest_path)
        if service_name:
            manifest.service_name = service_name
        manifest.deploy_target = "sagemaker"
        print(f"\n  Loaded {manifest_path}")
    except FileNotFoundError:
        manifest = interactive_create(service_name)
        manifest.deploy_target = "sagemaker"
        save_manifest(manifest, manifest_path)
        print(f"\n  ✓  Saved {manifest_path}")

    _print_plan(manifest)

    if not auto_approve:
        choice = _get_approval()
        if choice != "y":
            print("\n  Cancelled.")
            return False
    else:
        print("\n  --yes flag set, skipping approval.")

    print("\n  Checking AWS credentials...")
    import boto3

    try:
        sts = boto3.client("sts", region_name=manifest.region)
        identity = sts.get_caller_identity()
        account_id = identity["Account"]
        print(f"  ✓  Authenticated as {identity['Arn']}")
    except Exception as exc:
        print(f"  ✗  AWS credentials failed: {exc}")
        return False

    _ensure_sagemaker_dockerfile()
    ecr_image_uri = _docker_build_and_push(manifest, account_id)

    role_arn = _ensure_sagemaker_role(
        manifest.service_name, manifest.region, manifest
    )

    endpoint_name = _create_endpoint(manifest, ecr_image_uri, role_arn)

    _setup_auto_scaling(manifest, endpoint_name)

    res_path = _write_resource_manifest(
        manifest, endpoint_name, ecr_image_uri, role_arn
    )

    print(f"""
  ✓  SageMaker deployment complete.

  Endpoint name: {endpoint_name}
  Invoke via:
    import boto3
    sm = boto3.client("sagemaker-runtime", region_name="{manifest.region}")
    resp = sm.invoke_endpoint(
        EndpointName="{endpoint_name}",
        ContentType="application/json",
        Body='{{"messages": [{{"role": "user", "content": "hello"}}]}}',
    )
    print(resp["Body"].read().decode())

  Resource manifest: {res_path}

  Tear down: ml-platform destroy sagemaker --service-name {manifest.service_name}
""")
    return True
