"""ECS Fargate service construct for containerised ML services.

Provisions a Fargate service behind an Application Load Balancer,
including cluster, task definition, service, ALB, target group, and
health-check configuration.

Usage::

    from ml_platform.infra.constructs import EcsServiceConstruct

    svc = EcsServiceConstruct(
        self, "Inference",
        vpc=network.vpc,
        service_name="inference-api",
        container_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/inference:latest",
        environment={"MODEL_NAME": "my-model"},
    )
    print(svc.service_url)
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
)
from constructs import Construct


class EcsServiceConstruct(Construct):
    """Deploy a containerised ML service on ECS Fargate behind an ALB.

    This construct creates the full serving stack:

    * ECS cluster (one per construct instance)
    * Fargate task definition with configurable CPU / memory
    * Fargate service with desired task count
    * Internet-facing ALB with a target group and HTTP listener
    * Health-check on the specified path

    The ALB DNS name is exported as a CloudFormation output and exposed
    via the :pyattr:`service_url` property.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID within the construct tree.
        vpc: VPC in which to place the service and ALB.
        service_name: Logical name used for the ECS cluster, service, and
            CloudFormation export.
        container_image: Fully-qualified ECR (or public) image URI.
        cpu: Fargate task CPU units (256 | 512 | 1024 | 2048 | 4096).
        memory_limit_mib: Fargate task memory in MiB.
        desired_count: Number of running tasks.
        environment: Environment variables injected into the container.
        health_check_path: HTTP path the ALB uses for target health checks.
        container_port: Port the container listens on.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        service_name: str,
        container_image: str,
        cpu: int = 1024,
        memory_limit_mib: int = 2048,
        desired_count: int = 1,
        environment: dict[str, str] | None = None,
        health_check_path: str = "/health",
        container_port: int = 8080,
    ) -> None:
        super().__init__(scope, construct_id)

        cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=vpc,
            cluster_name=f"{service_name}-cluster",
        )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            cpu=cpu,
            memory_limit_mib=memory_limit_mib,
            family=f"{service_name}-task",
        )

        task_definition.add_container(
            "AppContainer",
            image=ecs.ContainerImage.from_registry(container_image),
            environment=environment or {},
            logging=ecs.LogDrivers.aws_logs(stream_prefix=service_name),
            port_mappings=[
                ecs.PortMapping(container_port=container_port),
            ],
        )

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=desired_count,
            public_load_balancer=True,
            listener_port=80,
            assign_public_ip=False,
        )

        fargate_service.target_group.configure_health_check(
            path=health_check_path,
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        self._service_url = (
            f"http://{fargate_service.load_balancer.load_balancer_dns_name}"
        )

        CfnOutput(
            self,
            "ServiceUrl",
            value=self._service_url,
            description=f"URL for the {service_name} service",
            export_name=f"{service_name}-url",
        )

    @property
    def service_url(self) -> str:
        """HTTP URL of the Application Load Balancer fronting the service."""
        return self._service_url
