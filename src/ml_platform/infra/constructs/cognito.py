"""Cognito User Pool construct for admin-auth web applications.

Provisions a Cognito User Pool with a secure default configuration
suitable for admin/back-office authentication: self-signup disabled,
strong password policy, and an app client configured for the
``USER_PASSWORD_AUTH`` and ``ADMIN_USER_PASSWORD_AUTH`` flows.

Usage::

    from ml_platform.infra.constructs import CognitoConstruct

    auth = CognitoConstruct(
        self, "AdminAuth",
        service_name="studio-admin",
    )

    # Pass to your frontend / API config:
    auth.user_pool_id
    auth.app_client_id
"""

from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_cognito as cognito,
)
from constructs import Construct


class CognitoConstruct(Construct):
    """Cognito User Pool with admin-auth defaults.

    Creates a User Pool with:

    * Self-signup **disabled** — users are created by administrators.
    * Strong password policy (configurable length and character
      requirements).
    * Email-based sign-in with required email attribute.
    * An app client authorised for ``USER_PASSWORD_AUTH`` and
      ``ADMIN_USER_PASSWORD_AUTH`` flows by default (matching the
      ``AdminInitiateAuthCommand`` pattern used by Next.js backends).
    * Optional custom domain prefix for the hosted UI.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID.
        service_name: Service name used for the User Pool name and
            CloudFormation outputs.
        password_policy: Override the default password policy.
        removal_policy: CloudFormation removal policy for the pool.
            Defaults to ``RETAIN`` to avoid accidental deletion.
        domain_prefix: If set, creates a Cognito hosted-UI domain at
            ``https://<prefix>.auth.<region>.amazoncognito.com``.
        auth_flows: Override the default Cognito auth-flow configuration.
            When ``None``, the construct enables ``USER_PASSWORD_AUTH``
            and ``ADMIN_USER_PASSWORD_AUTH``.
    """

    @dataclass(frozen=True)
    class PasswordPolicy:
        """Cognito password-policy configuration.

        Attributes:
            min_length: Minimum password length.
            require_lowercase: Require at least one lowercase character.
            require_uppercase: Require at least one uppercase character.
            require_digits: Require at least one digit.
            require_symbols: Require at least one symbol.
            temp_valid_days: Validity period for temporary passwords.
        """

        min_length: int = 12
        require_lowercase: bool = True
        require_uppercase: bool = True
        require_digits: bool = True
        require_symbols: bool = True
        temp_password_validity_days: int = 7

    _DEFAULT_AUTH_FLOWS = cognito.AuthFlow(
        user_password=True,
        admin_user_password=True,
        custom=False,
        user_srp=False,
    )

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        service_name: str = "",
        password_policy: PasswordPolicy | None = None,
        removal_policy: RemovalPolicy = RemovalPolicy.RETAIN,
        domain_prefix: str = "",
        auth_flows: cognito.AuthFlow | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        pw = password_policy or self.PasswordPolicy()

        pool_name = f"{service_name}-users" if service_name else None
        pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=pool_name,
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=pw.min_length,
                require_lowercase=pw.require_lowercase,
                require_uppercase=pw.require_uppercase,
                require_digits=pw.require_digits,
                require_symbols=pw.require_symbols,
                temp_password_validity=Duration.days(pw.temp_password_validity_days),
            ),
            removal_policy=removal_policy,
        )
        self._user_pool = pool

        resolved_auth_flows = auth_flows if auth_flows is not None else self._DEFAULT_AUTH_FLOWS

        client = pool.add_client(
            "AppClient",
            user_pool_client_name=f"{service_name}-client" if service_name else None,
            auth_flows=resolved_auth_flows,
            prevent_user_existence_errors=True,
            access_token_validity=Duration.hours(1),
            id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.days(30),
        )
        self._app_client = client

        self._domain: cognito.UserPoolDomain | None = None
        if domain_prefix:
            self._domain = pool.add_domain(
                "Domain",
                cognito_domain=cognito.CognitoDomainOptions(
                    domain_prefix=domain_prefix,
                ),
            )

        label = service_name or construct_id
        CfnOutput(
            self,
            "UserPoolId",
            value=pool.user_pool_id,
            description=f"Cognito User Pool ID for {label}",
        )
        CfnOutput(
            self,
            "AppClientId",
            value=client.user_pool_client_id,
            description=f"Cognito App Client ID for {label}",
        )

    @property
    def user_pool(self) -> cognito.IUserPool:
        """The Cognito User Pool."""
        return self._user_pool

    @property
    def user_pool_id(self) -> str:
        """The User Pool ID (resolved at deploy time)."""
        return self._user_pool.user_pool_id

    @property
    def app_client(self) -> cognito.UserPoolClient:
        """The app client configured for ``USER_PASSWORD_AUTH``."""
        return self._app_client

    @property
    def app_client_id(self) -> str:
        """The app client ID (resolved at deploy time)."""
        return self._app_client.user_pool_client_id

    @property
    def domain(self) -> cognito.UserPoolDomain | None:
        """The hosted-UI domain, or ``None`` if not configured."""
        return self._domain
