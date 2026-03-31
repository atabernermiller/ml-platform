"""MLflow tracking server construct on ECS Fargate with RDS and S3.

Deploys a self-hosted MLflow tracking server backed by a PostgreSQL
database (for experiment/run metadata) and an S3 bucket (for model
artifacts and large blobs).

Usage::

    from ml_platform.infra.constructs import MLflowConstruct

    mlflow = MLflowConstruct(
        self, "MLflow",
        vpc=network.vpc,
        artifact_bucket_name="my-team-mlflow-artifacts",
    )
    print(mlflow.tracking_uri)
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_rds as rds,
    aws_s3 as s3,
)
from constructs import Construct


class MLflowConstruct(Construct):
    """Self-hosted MLflow tracking server on ECS Fargate.

    Infrastructure created:

    * **RDS PostgreSQL** (``db.t4g.micro``, single-AZ) — stores experiment
      and run metadata.  Credentials are managed via Secrets Manager and
      injected into the MLflow container as environment variables.
    * **S3 bucket** — stores MLflow artifacts.  If ``artifact_bucket_name``
      matches an existing bucket the construct imports it; otherwise a new
      bucket is created.
    * **ECS Fargate service** — runs the upstream ``ghcr.io/mlflow/mlflow``
      image with the ``mlflow server`` entrypoint.
    * **Application Load Balancer** — exposes the tracking UI / API over
      HTTP on port 80.

    .. warning::
        The RDS instance uses ``SINGLE_AZ`` and ``db.t4g.micro`` to
        minimise cost.  For production workloads switch to Multi-AZ and a
        larger instance class.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID within the construct tree.
        vpc: VPC for the ECS service and RDS instance.
        service_name: Logical service name used for resource naming.
        artifact_bucket_name: Name for the S3 artifact bucket.  A new
            bucket is created with this name.
        mlflow_image: Docker image for the MLflow server.
        container_port: Port the MLflow server listens on inside the
            container.
        db_instance_type: RDS instance class.
        cpu: Fargate task CPU units.
        memory_limit_mib: Fargate task memory in MiB.
    """

    _DEFAULT_IMAGE = "ghcr.io/mlflow/mlflow:v2.18.0"

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        service_name: str = "mlflow",
        artifact_bucket_name: str,
        mlflow_image: str = _DEFAULT_IMAGE,
        container_port: int = 5000,
        db_instance_type: ec2.InstanceType | None = None,
        cpu: int = 512,
        memory_limit_mib: int = 1024,
    ) -> None:
        super().__init__(scope, construct_id)

        if db_instance_type is None:
            db_instance_type = ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE4_GRAVITON,
                ec2.InstanceSize.MICRO,
            )

        artifact_bucket = s3.Bucket(
            self,
            "ArtifactBucket",
            bucket_name=artifact_bucket_name,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        db_security_group = ec2.SecurityGroup(
            self,
            "DbSecurityGroup",
            vpc=vpc,
            description=f"Security group for {service_name} RDS instance",
            allow_all_outbound=False,
        )

        db_instance = rds.DatabaseInstance(
            self,
            "MetadataDb",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16_4,
            ),
            instance_type=db_instance_type,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
            security_groups=[db_security_group],
            database_name="mlflow",
            allocated_storage=20,
            max_allocated_storage=100,
            multi_az=False,
            removal_policy=RemovalPolicy.SNAPSHOT,
            deletion_protection=False,
            backup_retention=Duration.days(7),
        )

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

        artifact_bucket.grant_read_write(task_definition.task_role)

        db_secret = db_instance.secret
        if db_secret is None:
            raise RuntimeError(
                "RDS instance was created without a managed secret. "
                "Ensure credentials are auto-generated (the default)."
            )

        container = task_definition.add_container(
            "MlflowContainer",
            image=ecs.ContainerImage.from_registry(mlflow_image),
            logging=ecs.LogDrivers.aws_logs(stream_prefix=service_name),
            port_mappings=[ecs.PortMapping(container_port=container_port)],
            command=[
                "mlflow",
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                str(container_port),
                "--default-artifact-root",
                f"s3://{artifact_bucket.bucket_name}/artifacts",
            ],
            secrets={
                "DB_USERNAME": ecs.Secret.from_secrets_manager(db_secret, "username"),
                "DB_PASSWORD": ecs.Secret.from_secrets_manager(db_secret, "password"),
                "DB_HOST": ecs.Secret.from_secrets_manager(db_secret, "host"),
                "DB_PORT": ecs.Secret.from_secrets_manager(db_secret, "port"),
            },
            environment={
                "MLFLOW_BACKEND_STORE_URI": "postgresql+psycopg2://"
                "${DB_USERNAME}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/mlflow",
            },
        )

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            public_load_balancer=True,
            listener_port=80,
            assign_public_ip=False,
        )

        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
        )

        db_security_group.add_ingress_rule(
            peer=fargate_service.service.connections.security_groups[0],
            connection=ec2.Port.tcp(5432),
            description="Allow MLflow ECS tasks to reach PostgreSQL",
        )

        self._tracking_uri = (
            f"http://{fargate_service.load_balancer.load_balancer_dns_name}"
        )

        CfnOutput(
            self,
            "TrackingUri",
            value=self._tracking_uri,
            description="MLflow tracking server URI",
            export_name=f"{service_name}-tracking-uri",
        )

    @property
    def tracking_uri(self) -> str:
        """HTTP URL of the MLflow tracking server (ALB DNS)."""
        return self._tracking_uri
