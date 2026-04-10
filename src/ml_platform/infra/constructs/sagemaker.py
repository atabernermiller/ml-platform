"""SageMaker endpoint construct for real-time and serverless inference.

Provisions a SageMaker Model, EndpointConfig, and Endpoint using L1
CloudFormation resources, giving full control over instance types,
serverless inference configuration, and IAM roles.

Usage::

    from ml_platform.infra.constructs import SageMakerEndpointConstruct

    ep = SageMakerEndpointConstruct(
        self, "ModelEndpoint",
        endpoint_name="fraud-detector",
        model_data_url="s3://my-bucket/models/fraud/model.tar.gz",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/fraud:latest",
    )
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
)
from constructs import Construct


class SageMakerEndpointConstruct(Construct):
    """Deploy a SageMaker real-time or serverless inference endpoint.

    When ``serverless=False`` (the default) a standard real-time endpoint
    is created with the specified instance type and count.  When
    ``serverless=True`` the endpoint uses SageMaker Serverless Inference,
    scaling to zero when idle and charging per-invocation.

    The construct creates a minimal IAM execution role that grants the
    SageMaker service access to pull the model artifact from S3 and to
    pull the container image from ECR.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID within the construct tree.
        endpoint_name: Name for the SageMaker endpoint (must be unique
            within the account/region).
        model_data_url: S3 URI pointing to the model artifact
            (``model.tar.gz``).
        image_uri: Docker image URI for the SageMaker inference container.
        instance_type: Instance type for real-time endpoints.
        instance_count: Initial instance count for real-time endpoints.
        serverless: If ``True``, create a serverless inference endpoint
            instead of a real-time one.
        serverless_memory_mb: Memory size for serverless inference
            (1024 | 2048 | 3072 | 4096 | 5120 | 6144).
        serverless_max_concurrency: Maximum concurrent invocations for
            serverless inference (1–200).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        endpoint_name: str,
        model_data_url: str,
        image_uri: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        serverless: bool = False,
        serverless_memory_mb: int = 4096,
        serverless_max_concurrency: int = 10,
    ) -> None:
        super().__init__(scope, construct_id)

        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )

        s3_bucket_arn = self._extract_bucket_arn(model_data_url)
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[s3_bucket_arn, f"{s3_bucket_arn}/*"],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=["*"],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudwatch:PutMetricData",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=["*"],
            )
        )

        model = sagemaker.CfnModel(
            self,
            "Model",
            execution_role_arn=execution_role.role_arn,
            model_name=f"{endpoint_name}-model",
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                model_data_url=model_data_url,
            ),
        )

        production_variants: list[
            sagemaker.CfnEndpointConfig.ProductionVariantProperty
        ]

        if serverless:
            production_variants = [
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name or f"{endpoint_name}-model",
                    variant_name="AllTraffic",
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        max_concurrency=serverless_max_concurrency,
                        memory_size_in_mb=serverless_memory_mb,
                    ),
                ),
            ]
        else:
            production_variants = [
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name or f"{endpoint_name}-model",
                    variant_name="AllTraffic",
                    instance_type=instance_type,
                    initial_instance_count=instance_count,
                    initial_variant_weight=1.0,
                ),
            ]

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "EndpointConfig",
            endpoint_config_name=f"{endpoint_name}-config",
            production_variants=production_variants,
        )
        endpoint_config.add_dependency(model)

        endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config.endpoint_config_name
            or f"{endpoint_name}-config",
        )
        endpoint.add_dependency(endpoint_config)

        self._endpoint_name = endpoint_name

        CfnOutput(
            self,
            "EndpointName",
            value=self._endpoint_name,
            description=f"SageMaker endpoint: {endpoint_name}",
            export_name=f"{endpoint_name}-sagemaker-endpoint",
        )

    @property
    def endpoint_name(self) -> str:
        """Name of the deployed SageMaker endpoint."""
        return self._endpoint_name

    @staticmethod
    def _extract_bucket_arn(s3_uri: str) -> str:
        """Derive an S3 bucket ARN from an ``s3://bucket/key`` URI.

        Args:
            s3_uri: Full S3 URI (e.g. ``s3://my-bucket/path/model.tar.gz``).

        Returns:
            ARN of the S3 bucket (``arn:aws:s3:::my-bucket``).

        Raises:
            ValueError: If the URI does not start with ``s3://``.
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(
                f"Expected an S3 URI starting with 's3://', got: {s3_uri!r}"
            )
        bucket = s3_uri.replace("s3://", "").split("/")[0]
        return f"arn:aws:s3:::{bucket}"
