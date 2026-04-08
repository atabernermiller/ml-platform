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
    )
    print(fn.function_url)
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigwv2,
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
        environment: Environment variables.
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
        create_function_url: bool = True,
    ) -> None:
        super().__init__(scope, construct_id)

        fn = lambda_.Function(
            self,
            "Function",
            function_name=service_name,
            runtime=runtime,
            handler=handler,
            code=lambda_.Code.from_asset(code_path),
            memory_size=memory_size,
            timeout=timeout,
            environment=environment or {},
        )
        self._function = fn

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
