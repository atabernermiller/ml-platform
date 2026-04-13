"""Secrets Manager construct for provisioning application secrets.

Closes the loop between the runtime-side
:class:`~ml_platform.secrets.AWSSecretResolver` and the CDK infrastructure
by creating Secrets Manager resources with a template of expected keys.

Usage::

    from ml_platform.infra.constructs import SecretsConstruct

    secrets = SecretsConstruct(
        self, "AppSecrets",
        service_name="my-service",
        secret_keys=["DB_HOST", "DB_PASSWORD", "API_KEY"],
    )

    # Grant a Lambda or ECS task read access:
    secrets.grant_read(lambda_fn)

    # Use the ARN at deploy time:
    print(secrets.secret_arn)
"""

from __future__ import annotations

import json
from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_secretsmanager as sm,
)
from constructs import Construct


class SecretsConstruct(Construct):
    """Provision a Secrets Manager secret with a defined key template.

    The secret is initialised with a JSON object whose keys match
    ``secret_keys`` and whose values are placeholder strings (or
    the values in ``initial_values``).  This ensures the secret
    exists at deploy time with the expected schema so that
    :class:`~ml_platform.secrets.AWSSecretResolver` can resolve it
    immediately.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID.
        service_name: Service name for the secret's logical name.
        secret_keys: List of expected key names in the JSON secret.
        initial_values: Optional mapping of key -> initial value.
            Keys not present here default to ``"CHANGE_ME"``.
        description: Human-readable description of the secret.
        removal_policy: What happens to the secret on stack deletion.
            Defaults to ``RETAIN`` for safety.
        rotation_days: If set, attaches an automatic rotation schedule
            that triggers every ``rotation_days`` days.  Requires
            ``rotation_lambda`` to be provided as well.
        rotation_lambda: Lambda function that performs the actual secret
            rotation.  Required when ``rotation_days`` is set.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        service_name: str = "",
        secret_keys: list[str] | None = None,
        initial_values: dict[str, str] | None = None,
        description: str = "",
        removal_policy: RemovalPolicy = RemovalPolicy.RETAIN,
        rotation_days: int | None = None,
        rotation_lambda: _lambda.IFunction | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        if rotation_days is not None and rotation_lambda is None:
            raise ValueError(
                "rotation_lambda is required when rotation_days is set. "
                "Secrets Manager needs a Lambda function to perform rotation."
            )

        template: dict[str, str] = {}
        for key in secret_keys or []:
            template[key] = (initial_values or {}).get(key, "CHANGE_ME")

        secret_name = f"{service_name}/config" if service_name else None

        self._secret = sm.Secret(
            self,
            "Secret",
            secret_name=secret_name,
            description=description or f"Application secrets for {service_name}",
            generate_secret_string=sm.SecretStringGenerator(
                secret_string_template=json.dumps(template),
                generate_string_key="_generated_password",
            ),
            removal_policy=removal_policy,
        )

        if rotation_days is not None and rotation_lambda is not None:
            self._secret.add_rotation_schedule(
                "RotationSchedule",
                automatically_after=Duration.days(rotation_days),
                rotate_immediately_on_update=True,
                rotation_lambda=rotation_lambda,
            )

        CfnOutput(
            self,
            "SecretArn",
            value=self._secret.secret_arn,
            description=f"Secret ARN for {service_name}",
        )

        if secret_name:
            CfnOutput(
                self,
                "SecretName",
                value=secret_name,
                description=f"Secret name for {service_name}",
            )

    @property
    def secret(self) -> sm.ISecret:
        """The underlying Secrets Manager secret."""
        return self._secret

    @property
    def secret_arn(self) -> str:
        """ARN of the Secrets Manager secret."""
        return self._secret.secret_arn

    @property
    def secret_name(self) -> str:
        """Name of the Secrets Manager secret."""
        return self._secret.secret_name

    def grant_read(self, grantee: iam.IGrantable) -> iam.Grant:
        """Grant read access to the secret.

        Works with Lambda functions, ECS task roles, EC2 instance roles,
        or any other IAM grantable principal.

        Args:
            grantee: IAM principal to grant access to.

        Returns:
            The grant object.
        """
        return self._secret.grant_read(grantee)
