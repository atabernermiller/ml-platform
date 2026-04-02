"""CloudFormation template generator for the full service stack.

Generates a single stack that contains everything declared in the
project manifest: VPC (optional, uses default), ECS Fargate service
with ALB, auto-scaling, IAM roles, CloudWatch log group, optional
DynamoDB tables, and a CloudWatch dashboard.

The template is returned as a Python dict suitable for
``json.dumps()`` and ``boto3 cloudformation.create_stack()``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ml_platform.cli.manifest import ProjectManifest


def generate_stack_template(
    manifest: ProjectManifest,
    *,
    ecr_image_uri: str,
    vpc_id: str = "",
    subnet_ids: list[str] | None = None,
    alert_rules: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a complete CloudFormation template from a project manifest.

    Args:
        manifest: The parsed project manifest.
        ecr_image_uri: Full ECR image URI (``account.dkr.ecr.region.amazonaws.com/repo:tag``).
        vpc_id: Existing VPC ID.  If empty, the template uses the default VPC.
        subnet_ids: Subnet IDs for the ALB and ECS tasks.  If empty, uses
            default subnets.
        alert_rules: Optional list of :class:`AlertRule` instances to
            translate into CloudWatch Alarms.

    Returns:
        CloudFormation template as a dict.
    """
    manifest._alert_rules = alert_rules or []  # type: ignore[attr-defined]
    svc = manifest.service_name
    resources: dict[str, Any] = {}
    outputs: dict[str, Any] = {}

    # -- IAM roles -----------------------------------------------------------
    resources["TaskExecutionRole"] = _ecs_execution_role(svc)
    resources["TaskRole"] = _ecs_task_role(svc, manifest)

    # -- CloudWatch log group ------------------------------------------------
    resources["LogGroup"] = {
        "Type": "AWS::Logs::LogGroup",
        "Properties": {
            "LogGroupName": f"/ecs/{svc}",
            "RetentionInDays": 30,
        },
    }

    # -- ECS cluster ---------------------------------------------------------
    resources["EcsCluster"] = {
        "Type": "AWS::ECS::Cluster",
        "Properties": {"ClusterName": f"{svc}-cluster"},
    }

    # -- Task definition -----------------------------------------------------
    resources["TaskDefinition"] = _task_definition(manifest, ecr_image_uri)

    # -- Security groups -----------------------------------------------------
    resources["AlbSecurityGroup"] = _alb_security_group(svc, vpc_id)
    resources["TaskSecurityGroup"] = _task_security_group(svc, vpc_id)

    # -- ALB -----------------------------------------------------------------
    alb_resources = _alb_resources(svc, subnet_ids)
    resources.update(alb_resources)

    # -- ECS service ---------------------------------------------------------
    resources["EcsService"] = _ecs_service(manifest)

    # -- Auto-scaling --------------------------------------------------------
    scaling_resources = _auto_scaling(manifest)
    resources.update(scaling_resources)

    # -- Optional DynamoDB tables --------------------------------------------
    if manifest.features.conversation_store:
        resources["ConversationTable"] = _dynamodb_table(
            f"{svc}-conversations", ttl_attr="ttl"
        )
    if manifest.features.context_store:
        resources["ContextTable"] = _dynamodb_table(
            f"{svc}-context", ttl_attr="ttl"
        )

    # -- Optional S3 bucket --------------------------------------------------
    if manifest.features.checkpointing:
        resources["CheckpointBucket"] = {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": f"ml-platform-{svc}-ckpt",
                "VersioningConfiguration": {"Status": "Enabled"},
            },
        }

    # -- CloudWatch dashboard ------------------------------------------------
    resources["Dashboard"] = _cloudwatch_dashboard(manifest)

    # -- CloudWatch alarms from alert rules ---------------------------------
    alarm_resources = _cloudwatch_alarms(manifest)
    resources.update(alarm_resources)

    # -- Outputs -------------------------------------------------------------
    outputs["ServiceUrl"] = {
        "Description": "ALB endpoint URL",
        "Value": {"Fn::GetAtt": ["Alb", "DNSName"]},
    }
    outputs["DashboardUrl"] = {
        "Description": "CloudWatch dashboard URL",
        "Value": {
            "Fn::Sub": (
                f"https://${{AWS::Region}}.console.aws.amazon.com/"
                f"cloudwatch/home?region=${{AWS::Region}}#dashboards:"
                f"name={svc}-dashboard"
            )
        },
    }

    template: dict[str, Any] = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"ml-platform stack for {svc}",
        "Resources": resources,
        "Outputs": outputs,
    }
    return template


def stack_name(manifest: ProjectManifest) -> str:
    """Canonical CloudFormation stack name for a service."""
    return f"{manifest.service_name}-stack"


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def _ecs_execution_role(svc: str) -> dict[str, Any]:
    return {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "RoleName": f"{svc}-exec-role",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
            "ManagedPolicyArns": [
                "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
            ],
        },
    }


def _ecs_task_role(svc: str, manifest: ProjectManifest) -> dict[str, Any]:
    statements: list[dict[str, Any]] = [
        {
            "Effect": "Allow",
            "Action": "cloudwatch:PutMetricData",
            "Resource": "*",
            "Condition": {
                "StringEquals": {"cloudwatch:namespace": "MLPlatform"}
            },
        },
        {
            "Effect": "Allow",
            "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": "*",
        },
    ]
    if manifest.features.checkpointing:
        statements.append(
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::ml-platform-{svc}-ckpt",
                    f"arn:aws:s3:::ml-platform-{svc}-ckpt/*",
                ],
            }
        )
    if manifest.features.conversation_store or manifest.features.context_store:
        tables = []
        if manifest.features.conversation_store:
            tables.append(f"arn:aws:dynamodb:*:*:table/{svc}-conversations")
        if manifest.features.context_store:
            tables.append(f"arn:aws:dynamodb:*:*:table/{svc}-context")
        statements.append(
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                ],
                "Resource": tables,
            }
        )
    return {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "RoleName": f"{svc}-task-role",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
            "Policies": [
                {
                    "PolicyName": f"{svc}-task-policy",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": statements,
                    },
                }
            ],
        },
    }


def _task_definition(
    manifest: ProjectManifest, ecr_image_uri: str
) -> dict[str, Any]:
    svc = manifest.service_name
    env = [
        {"Name": "ML_PLATFORM_SERVICE_NAME", "Value": svc},
        {"Name": "ML_PLATFORM_AWS_REGION", "Value": manifest.region},
    ]
    if manifest.features.checkpointing:
        env.append(
            {"Name": "ML_PLATFORM_S3_CHECKPOINT_BUCKET", "Value": f"ml-platform-{svc}-ckpt"}
        )
    return {
        "Type": "AWS::ECS::TaskDefinition",
        "Properties": {
            "Family": svc,
            "Cpu": str(manifest.cpu),
            "Memory": str(manifest.memory),
            "NetworkMode": "awsvpc",
            "RequiresCompatibilities": ["FARGATE"],
            "ExecutionRoleArn": {"Fn::GetAtt": ["TaskExecutionRole", "Arn"]},
            "TaskRoleArn": {"Fn::GetAtt": ["TaskRole", "Arn"]},
            "ContainerDefinitions": [
                {
                    "Name": svc,
                    "Image": ecr_image_uri,
                    "Essential": True,
                    "PortMappings": [{"ContainerPort": 8000, "Protocol": "tcp"}],
                    "Environment": env,
                    "LogConfiguration": {
                        "LogDriver": "awslogs",
                        "Options": {
                            "awslogs-group": f"/ecs/{svc}",
                            "awslogs-region": manifest.region,
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                    "HealthCheck": {
                        "Command": ["CMD-SHELL", "curl -f http://localhost:8000/health/live || exit 1"],
                        "Interval": 10,
                        "Timeout": 5,
                        "Retries": 3,
                    },
                }
            ],
        },
    }


def _alb_security_group(svc: str, vpc_id: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "GroupDescription": f"ALB for {svc}",
        "SecurityGroupIngress": [
            {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "CidrIp": "0.0.0.0/0"}
        ],
    }
    if vpc_id:
        props["VpcId"] = vpc_id
    return {"Type": "AWS::EC2::SecurityGroup", "Properties": props}


def _task_security_group(svc: str, vpc_id: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "GroupDescription": f"ECS tasks for {svc}",
        "SecurityGroupIngress": [
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "SourceSecurityGroupId": {"Ref": "AlbSecurityGroup"},
            }
        ],
    }
    if vpc_id:
        props["VpcId"] = vpc_id
    return {"Type": "AWS::EC2::SecurityGroup", "Properties": props}


def _alb_resources(
    svc: str, subnet_ids: list[str] | None
) -> dict[str, dict[str, Any]]:
    alb_props: dict[str, Any] = {
        "Name": f"{svc}-alb",
        "Scheme": "internet-facing",
        "Type": "application",
        "SecurityGroups": [{"Ref": "AlbSecurityGroup"}],
    }
    if subnet_ids:
        alb_props["Subnets"] = subnet_ids

    return {
        "Alb": {
            "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
            "Properties": alb_props,
        },
        "TargetGroup": {
            "Type": "AWS::ElasticLoadBalancingV2::TargetGroup",
            "Properties": {
                "Name": f"{svc}-tg",
                "Port": 8000,
                "Protocol": "HTTP",
                "TargetType": "ip",
                "HealthCheckPath": "/health/ready",
                "HealthCheckIntervalSeconds": 10,
                "HealthyThresholdCount": 2,
                "UnhealthyThresholdCount": 3,
            },
        },
        "Listener": {
            "Type": "AWS::ElasticLoadBalancingV2::Listener",
            "Properties": {
                "LoadBalancerArn": {"Ref": "Alb"},
                "Port": 80,
                "Protocol": "HTTP",
                "DefaultActions": [
                    {"Type": "forward", "TargetGroupArn": {"Ref": "TargetGroup"}}
                ],
            },
        },
    }


def _ecs_service(manifest: ProjectManifest) -> dict[str, Any]:
    return {
        "Type": "AWS::ECS::Service",
        "DependsOn": ["Listener"],
        "Properties": {
            "ServiceName": manifest.service_name,
            "Cluster": {"Ref": "EcsCluster"},
            "TaskDefinition": {"Ref": "TaskDefinition"},
            "DesiredCount": manifest.scaling.min_tasks,
            "LaunchType": "FARGATE",
            "HealthCheckGracePeriodSeconds": 30,
            "DeploymentConfiguration": {
                "MinimumHealthyPercent": 100,
                "MaximumPercent": 200,
            },
            "NetworkConfiguration": {
                "AwsvpcConfiguration": {
                    "AssignPublicIp": "ENABLED",
                    "SecurityGroups": [{"Ref": "TaskSecurityGroup"}],
                }
            },
            "LoadBalancers": [
                {
                    "ContainerName": manifest.service_name,
                    "ContainerPort": 8000,
                    "TargetGroupArn": {"Ref": "TargetGroup"},
                }
            ],
        },
    }


def _auto_scaling(manifest: ProjectManifest) -> dict[str, dict[str, Any]]:
    svc = manifest.service_name
    return {
        "ScalableTarget": {
            "Type": "AWS::ApplicationAutoScaling::ScalableTarget",
            "Properties": {
                "MaxCapacity": manifest.scaling.max_tasks,
                "MinCapacity": manifest.scaling.min_tasks,
                "ResourceId": {
                    "Fn::Sub": f"service/{svc}-cluster/{svc}",
                },
                "RoleARN": {
                    "Fn::Sub": (
                        "arn:aws:iam::${AWS::AccountId}:role/"
                        "aws-service-role/ecs.application-autoscaling.amazonaws.com/"
                        "AWSServiceRoleForApplicationAutoScaling_ECSService"
                    )
                },
                "ScalableDimension": "ecs:service:DesiredCount",
                "ServiceNamespace": "ecs",
            },
            "DependsOn": ["EcsService"],
        },
        "ScaleUpPolicy": {
            "Type": "AWS::ApplicationAutoScaling::ScalingPolicy",
            "Properties": {
                "PolicyName": f"{svc}-scale-up",
                "PolicyType": "TargetTrackingScaling",
                "ScalingTargetId": {"Ref": "ScalableTarget"},
                "TargetTrackingScalingPolicyConfiguration": {
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
                    },
                    "TargetValue": float(manifest.scaling.scale_up_cpu),
                    "ScaleInCooldown": 300,
                    "ScaleOutCooldown": 60,
                },
            },
        },
    }


def _dynamodb_table(table_name: str, ttl_attr: str = "ttl") -> dict[str, Any]:
    return {
        "Type": "AWS::DynamoDB::Table",
        "Properties": {
            "TableName": table_name,
            "BillingMode": "PAY_PER_REQUEST",
            "AttributeDefinitions": [
                {"AttributeName": "pk", "AttributeType": "S"}
            ],
            "KeySchema": [{"AttributeName": "pk", "KeyType": "HASH"}],
            "TimeToLiveSpecification": {
                "AttributeName": ttl_attr,
                "Enabled": True,
            },
        },
    }


_CONDITION_TO_CW: dict[str, str] = {
    ">": "GreaterThanThreshold",
    ">=": "GreaterThanOrEqualToThreshold",
    "<": "LessThanThreshold",
    "<=": "LessThanOrEqualToThreshold",
}


def _cloudwatch_alarms(manifest: ProjectManifest) -> dict[str, dict[str, Any]]:
    """Generate CloudWatch Alarm resources from alert rules on the manifest."""
    rules: list[Any] = getattr(manifest, "_alert_rules", [])
    if not rules:
        return {}

    svc = manifest.service_name
    ns = "MLPlatform"
    resources: dict[str, dict[str, Any]] = {}

    for i, rule in enumerate(rules):
        cw_op = _CONDITION_TO_CW.get(rule.condition)
        if cw_op is None:
            continue

        safe_name = re.sub(r"[^A-Za-z0-9]", "", rule.name or f"Alert{i}")
        logical_id = f"Alarm{safe_name}"

        resources[logical_id] = {
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": f"{svc}-{rule.name}",
                "AlarmDescription": rule.description or f"Alert: {rule.metric} {rule.condition} {rule.threshold}",
                "Namespace": ns,
                "MetricName": rule.metric,
                "Dimensions": [{"Name": "service", "Value": svc}],
                "Statistic": "Average",
                "Period": max(rule.window_s, 60),
                "EvaluationPeriods": 1,
                "Threshold": rule.threshold,
                "ComparisonOperator": cw_op,
                "TreatMissingData": "notBreaching",
            },
        }

    return resources


def _cloudwatch_dashboard(manifest: ProjectManifest) -> dict[str, Any]:
    svc = manifest.service_name
    ns = "MLPlatform"
    region = manifest.region

    def _metric(name: str, stat: str = "Average") -> list[Any]:
        return [ns, name, "service", svc, {"stat": stat, "period": 60}]

    widgets: list[dict[str, Any]] = [
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Request Rate",
                "region": region,
                "metrics": [_metric("requests_total", "Sum")],
                "view": "timeSeries",
                "period": 60,
            },
        },
        {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title": "Latency P95 (ms)",
                "region": region,
                "metrics": [_metric("latency_ms", "p95")],
                "view": "timeSeries",
                "period": 60,
            },
        },
    ]

    if manifest.service_type in ("llm", "agent"):
        widgets.extend(
            [
                {
                    "type": "metric",
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Tokens / min",
                        "region": region,
                        "metrics": [_metric("total_tokens", "Sum")],
                        "view": "timeSeries",
                        "period": 60,
                    },
                },
                {
                    "type": "metric",
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Cost (USD)",
                        "region": region,
                        "metrics": [_metric("total_cost_usd", "Sum")],
                        "view": "timeSeries",
                        "period": 60,
                    },
                },
            ]
        )

    return {
        "Type": "AWS::CloudWatch::Dashboard",
        "Properties": {
            "DashboardName": f"{svc}-dashboard",
            "DashboardBody": {"Fn::Sub": json.dumps({"widgets": widgets})},
        },
    }
