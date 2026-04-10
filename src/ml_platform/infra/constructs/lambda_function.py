"""Lambda function construct for ml-platform services.

Provisions a Lambda function with API Gateway v2 (HTTP API) integration,
suitable for event-driven workloads and lightweight inference endpoints.

Usage::

    from ml_platform.infra.constructs import LambdaServiceConstruct

    fn = LambdaServiceConstruct(
        self, "InferenceFunction",
        service_name="inference-api",
        code_path="./src",
        handler="app.handler",
        secrets={"DB_PASSWORD": "arn:aws:secretsmanager:us-east-1:123:secret:db-pw"},
    )
    print(fn.function_url)
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_iam as iam,
    aws_lambda as lambda_,
)
from constructs import Construct


class LambdaServiceConstruct(Construct):
    """Lambda function with optional HTTP API (API Gateway v2).

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
    """

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
    ) -> None:
        super().__init__(scope, construct_id)

        merged_env = dict(environment or {})
        secret_arns: list[str] = []
        for env_name, secret_arn in (secrets or {}).items():
            merged_env[env_name] = secret_arn
            secret_arns.append(secret_arn)

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
