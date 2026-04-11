"""Lambda function construct for ml-platform services.

Provisions a Lambda function with optional API Gateway v2 (HTTP API)
integration and/or S3 event-notification triggers, suitable for
event-driven workloads, lightweight inference endpoints, and
upload-triggers-processing pipelines.

Usage — HTTP endpoint::

    from ml_platform.infra.constructs import LambdaServiceConstruct

    fn = LambdaServiceConstruct(
        self, "InferenceFunction",
        service_name="inference-api",
        code_path="./src",
        handler="app.handler",
        secrets={"DB_PASSWORD": "arn:aws:secretsmanager:us-east-1:123:secret:db-pw"},
    )
    print(fn.function_url)

Usage — S3 event trigger::

    from aws_cdk import aws_s3 as s3
    from ml_platform.infra.constructs import LambdaServiceConstruct

    uploads = s3.Bucket(self, "Uploads")

    processor = LambdaServiceConstruct(
        self, "ImageProcessor",
        service_name="image-processor",
        code_path="./processor",
        handler="app.handler",
        s3_trigger=LambdaServiceConstruct.S3Trigger(
            bucket=uploads,
            prefix="uploads/",
            suffix=".jpg",
            events=[s3.EventType.OBJECT_CREATED],
        ),
        create_function_url=False,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
)
from constructs import Construct


class LambdaServiceConstruct(Construct):
    """Lambda function with optional HTTP endpoint and S3 event trigger.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID.
        service_name: Service name for resource naming.
        code_path: Path to the Lambda deployment package directory.
        handler: Lambda handler (e.g. ``"app.handler"``).
        runtime: Lambda runtime.
        memory_size: Memory allocation in MB.
        timeout: Function timeout.
        environment: Plaintext environment variables.
        secrets: Mapping of environment variable names to Secrets Manager
            secret ARNs.  The construct injects each secret ARN as an env
            var and grants the function ``secretsmanager:GetSecretValue``
            on those ARNs so the runtime can resolve them via
            :class:`~ml_platform.secrets.AWSSecretResolver`.
        create_function_url: Create a Lambda Function URL.
        s3_trigger: Optional :class:`S3Trigger` that wires an S3 bucket
            to invoke this function on object events.  The construct
            grants ``s3:GetObject`` on the bucket and injects
            ``S3_TRIGGER_BUCKET`` into the function environment.
    """

    @dataclass(frozen=True)
    class S3Trigger:
        """Configuration for an S3 event notification trigger.

        Attributes:
            bucket: The S3 bucket to watch.
            prefix: Key prefix filter (e.g. ``"uploads/"``).
            suffix: Key suffix filter (e.g. ``".jpg"``).
            events: S3 event types that fire the notification.  Defaults
                to ``[s3.EventType.OBJECT_CREATED]``.
        """

        bucket: s3.IBucket
        prefix: str = ""
        suffix: str = ""
        events: list[s3.EventType] = field(
            default_factory=lambda: [s3.EventType.OBJECT_CREATED]
        )

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        service_name: str,
        code_path: str,
        handler: str = "app.handler",
        runtime: lambda_.Runtime = lambda_.Runtime.PYTHON_3_12,
        memory_size: int = 512,
        timeout: Duration = Duration.seconds(30),
        environment: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        create_function_url: bool = True,
        s3_trigger: S3Trigger | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        merged_env = dict(environment or {})
        secret_arns: list[str] = []
        for env_name, secret_arn in (secrets or {}).items():
            merged_env[env_name] = secret_arn
            secret_arns.append(secret_arn)

        if s3_trigger is not None:
            merged_env["S3_TRIGGER_BUCKET"] = s3_trigger.bucket.bucket_name

        fn = lambda_.Function(
            self,
            "Function",
            function_name=service_name,
            runtime=runtime,
            handler=handler,
            code=lambda_.Code.from_asset(code_path),
            memory_size=memory_size,
            timeout=timeout,
            environment=merged_env,
        )
        self._function = fn

        if secret_arns:
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["secretsmanager:GetSecretValue"],
                    resources=secret_arns,
                )
            )

        if s3_trigger is not None:
            s3_trigger.bucket.grant_read(fn)

            has_filter = bool(s3_trigger.prefix or s3_trigger.suffix)
            filter_args: list[s3.NotificationKeyFilter] = []
            if has_filter:
                filter_args.append(
                    s3.NotificationKeyFilter(
                        prefix=s3_trigger.prefix or None,
                        suffix=s3_trigger.suffix or None,
                    )
                )

            for event_type in s3_trigger.events:
                s3_trigger.bucket.add_event_notification(
                    event_type,
                    s3n.LambdaDestination(fn),
                    *filter_args,
                )

        self._function_url = ""
        if create_function_url:
            url = fn.add_function_url(
                auth_type=lambda_.FunctionUrlAuthType.NONE,
            )
            self._function_url = url.url

            CfnOutput(
                self,
                "FunctionUrl",
                value=url.url,
                description=f"Lambda Function URL for {service_name}",
            )

        CfnOutput(
            self,
            "FunctionArn",
            value=fn.function_arn,
            description=f"Lambda function ARN for {service_name}",
        )

    @property
    def function(self) -> lambda_.IFunction:
        """The Lambda function."""
        return self._function

    @property
    def function_url(self) -> str:
        """Lambda Function URL (empty if not created)."""
        return self._function_url
