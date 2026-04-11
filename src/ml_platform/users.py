"""Cognito User Pool management and an in-memory fallback.

Complements :mod:`ml_platform.auth` (JWT validation) with user
lifecycle operations: creation, deletion, password reset, and
attribute management.

Two backends are provided:

- ``CognitoUserPool`` -- wraps the AWS Cognito ``admin_*`` APIs.
- ``InMemoryUserPool`` -- stores users in a dict for local development.

Usage::

    from ml_platform.users import CognitoUserPool, InMemoryUserPool

    pool = CognitoUserPool(user_pool_id="us-east-1_Abc123")
    pool.create_user("alice", email="alice@example.com")
    user = pool.get_user("alice")
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import UserPool
from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_cognito_idp.client import CognitoIdentityProviderClient

logger = logging.getLogger(__name__)

__all__ = [
    "CognitoUserPool",
    "InMemoryUserPool",
]


class CognitoUserPool(UserPool):
    """AWS Cognito User Pool backend.

    Uses the ``admin_*`` Cognito APIs, which require IAM credentials
    (not end-user tokens).

    Required IAM permissions::

        cognito-idp:AdminCreateUser
        cognito-idp:AdminGetUser
        cognito-idp:AdminDeleteUser
        cognito-idp:AdminResetUserPassword
        cognito-idp:ListUsers

    Args:
        user_pool_id: Cognito User Pool ID (e.g. ``"us-east-1_Abc123"``).
        region: AWS region.
    """

    def __init__(self, user_pool_id: str, region: str | None = None) -> None:
        import boto3

        self._pool_id = user_pool_id
        self._client: CognitoIdentityProviderClient = boto3.client(
            "cognito-idp", region_name=resolve_region(region)
        )

    def create_user(
        self,
        username: str,
        *,
        email: str = "",
        phone: str = "",
        attributes: dict[str, str] | None = None,
        temporary_password: str = "",
    ) -> dict[str, Any]:
        user_attrs: list[dict[str, str]] = []
        if email:
            user_attrs.append({"Name": "email", "Value": email})
            user_attrs.append({"Name": "email_verified", "Value": "true"})
        if phone:
            user_attrs.append({"Name": "phone_number", "Value": phone})
        for k, v in (attributes or {}).items():
            user_attrs.append({"Name": k, "Value": v})

        kwargs: dict[str, Any] = {
            "UserPoolId": self._pool_id,
            "Username": username,
            "UserAttributes": user_attrs,
            "MessageAction": "SUPPRESS",
        }
        if temporary_password:
            kwargs["TemporaryPassword"] = temporary_password

        response = self._client.admin_create_user(**kwargs)
        user = response["User"]
        logger.info("Created Cognito user %s", username)
        return {
            "username": user["Username"],
            "status": user["UserStatus"],
            "attributes": {
                a["Name"]: a["Value"] for a in user.get("Attributes", [])
            },
            "created": str(user.get("UserCreateDate", "")),
        }

    def get_user(self, username: str) -> dict[str, Any] | None:
        try:
            response = self._client.admin_get_user(
                UserPoolId=self._pool_id,
                Username=username,
            )
        except self._client.exceptions.UserNotFoundException:
            return None
        except Exception:
            logger.exception("Failed to get user: %s", username)
            raise

        return {
            "username": response["Username"],
            "status": response["UserStatus"],
            "attributes": {
                a["Name"]: a["Value"]
                for a in response.get("UserAttributes", [])
            },
            "enabled": response.get("Enabled", True),
        }

    def delete_user(self, username: str) -> bool:
        try:
            self._client.admin_delete_user(
                UserPoolId=self._pool_id,
                Username=username,
            )
            logger.info("Deleted Cognito user %s", username)
            return True
        except self._client.exceptions.UserNotFoundException:
            return False
        except Exception:
            logger.exception("Failed to delete user: %s", username)
            raise

    def reset_password(self, username: str) -> bool:
        try:
            self._client.admin_reset_user_password(
                UserPoolId=self._pool_id,
                Username=username,
            )
            logger.info("Password reset initiated for %s", username)
            return True
        except Exception as exc:
            logger.warning("Password reset failed for %s: %s", username, exc)
            return False

    def list_users(self, *, limit: int = 60) -> list[dict[str, Any]]:
        response = self._client.list_users(
            UserPoolId=self._pool_id,
            Limit=min(limit, 60),
        )
        return [
            {
                "username": u["Username"],
                "status": u["UserStatus"],
                "attributes": {
                    a["Name"]: a["Value"] for a in u.get("Attributes", [])
                },
                "enabled": u.get("Enabled", True),
            }
            for u in response.get("Users", [])
        ]


class InMemoryUserPool(UserPool):
    """In-memory user pool for local development and testing.

    Stores users in a dict keyed by username.  Password reset is
    recorded as a status change.
    """

    def __init__(self) -> None:
        self._users: dict[str, dict[str, Any]] = {}

    def create_user(
        self,
        username: str,
        *,
        email: str = "",
        phone: str = "",
        attributes: dict[str, str] | None = None,
        temporary_password: str = "",
    ) -> dict[str, Any]:
        attrs: dict[str, str] = dict(attributes or {})
        if email:
            attrs["email"] = email
        if phone:
            attrs["phone_number"] = phone

        user: dict[str, Any] = {
            "username": username,
            "status": "FORCE_CHANGE_PASSWORD" if temporary_password else "CONFIRMED",
            "attributes": attrs,
            "enabled": True,
            "password": temporary_password or uuid.uuid4().hex,
        }
        self._users[username] = user
        return {
            "username": username,
            "status": user["status"],
            "attributes": dict(attrs),
        }

    def get_user(self, username: str) -> dict[str, Any] | None:
        user = self._users.get(username)
        if user is None:
            return None
        return {
            "username": user["username"],
            "status": user["status"],
            "attributes": dict(user["attributes"]),
            "enabled": user["enabled"],
        }

    def delete_user(self, username: str) -> bool:
        return self._users.pop(username, None) is not None

    def reset_password(self, username: str) -> bool:
        user = self._users.get(username)
        if user is None:
            return False
        user["status"] = "RESET_REQUIRED"
        return True

    def list_users(self, *, limit: int = 60) -> list[dict[str, Any]]:
        users = list(self._users.values())[:limit]
        return [
            {
                "username": u["username"],
                "status": u["status"],
                "attributes": dict(u["attributes"]),
                "enabled": u["enabled"],
            }
            for u in users
        ]
