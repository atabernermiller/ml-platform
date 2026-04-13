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
        secrets={"DB_PASSWORD": ecs.Secret.from_secrets_manager(db_secret)},
        certificate_arn="arn:aws:acm:us-east-1:123456789012:certificate/abc-123",
    )
    svc.task_definition.task_role.add_to_policy(...)
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_certificatemanager as acm,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_s3 as s3,
)
from constructs import Construct


class EcsServiceConstruct(Construct):
    """Deploy a containerised ML service on ECS Fargate behind an ALB.

    This construct creates the full serving stack:

    * ECS cluster (one per construct instance)
    * Fargate task definition with configurable CPU / memory
    * Fargate service with desired task count
    * Internet-facing ALB with a target group and health check
    * Optional HTTPS listener with HTTP-to-HTTPS redirect
    * Optional ALB deletion protection and access logging

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
        environment: Plaintext environment variables injected into the
            container.
        secrets: Secrets Manager / SSM references injected into the
            container.  Values are resolved at task launch time.
        health_check_path: HTTP path the ALB uses for target health checks.
        container_port: Port the container listens on.
        certificate_arn: Optional ACM certificate ARN for HTTPS.  When
            provided the ALB listens on 443 and redirects HTTP 80 to HTTPS.
        domain_name: Optional custom domain (informational, used in outputs).
        enable_deletion_protection: Enable ALB deletion protection to
            prevent accidental removal.  Defaults to ``True`` for
            production safety.
        access_log_bucket: S3 bucket for ALB access logs.  When provided,
            access logging is enabled on the load balancer.
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
        secrets: dict[str, ecs.Secret] | None = None,
        health_check_path: str = "/health",
        container_port: int = 8080,
        certificate_arn: str = "",
        domain_name: str = "",
        enable_deletion_protection: bool = True,
        access_log_bucket: s3.IBucket | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=vpc,
            cluster_name=f"{service_name}-cluster",
        )
        self._cluster = cluster

        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            cpu=cpu,
            memory_limit_mib=memory_limit_mib,
            family=f"{service_name}-task",
        )
        self._task_definition = task_definition

        task_definition.add_container(
            "AppContainer",
            image=ecs.ContainerImage.from_registry(container_image),
            environment=environment or {},
            secrets=secrets or {},
            logging=ecs.LogDrivers.aws_logs(stream_prefix=service_name),
            port_mappings=[
                ecs.PortMapping(container_port=container_port),
            ],
        )

        use_https = bool(certificate_arn)
        listener_port = 443 if use_https else 80

        certificate = None
        if use_https:
            certificate = acm.Certificate.from_certificate_arn(
                self, "Cert", certificate_arn
            )

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=desired_count,
            public_load_balancer=True,
            listener_port=listener_port,
            assign_public_ip=False,
            certificate=certificate,
            redirect_http=use_https,
        )
        self._fargate_service = fargate_service

        fargate_service.load_balancer.set_attribute(
            "deletion_protection.enabled",
            "true" if enable_deletion_protection else "false",
        )
        if access_log_bucket is not None:
            fargate_service.load_balancer.log_access_logs(access_log_bucket)

        fargate_service.target_group.configure_health_check(
            path=health_check_path,
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        protocol = "https" if use_https else "http"
        dns = fargate_service.load_balancer.load_balancer_dns_name
        if domain_name:
            self._service_url = f"{protocol}://{domain_name}"
        else:
            self._service_url = f"{protocol}://{dns}"

        CfnOutput(
            self,
            "ServiceUrl",
            value=self._service_url,
            description=f"URL for the {service_name} service",
            export_name=f"{service_name}-url",
        )

    @property
    def service_url(self) -> str:
        """URL of the Application Load Balancer fronting the service."""
        return self._service_url

    @property
    def task_definition(self) -> ecs.FargateTaskDefinition:
        """The Fargate task definition (for granting IAM permissions)."""
        return self._task_definition

    @property
    def service(self) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        """The ALB-fronted Fargate service (for auto-scaling, etc.)."""
        return self._fargate_service

    @property
    def cluster(self) -> ecs.ICluster:
        """The ECS cluster (for service discovery, exec access)."""
        return self._cluster
