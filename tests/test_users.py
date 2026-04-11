"""Tests for Cognito User Pool management."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.users import CognitoUserPool, InMemoryUserPool


class TestInMemoryUserPool:
    def test_create_and_get_user(self) -> None:
        pool = InMemoryUserPool()
        result = pool.create_user("alice", email="alice@example.com")
        assert result["username"] == "alice"
        assert result["status"] == "CONFIRMED"

        user = pool.get_user("alice")
        assert user is not None
        assert user["attributes"]["email"] == "alice@example.com"

    def test_get_missing_user(self) -> None:
        pool = InMemoryUserPool()
        assert pool.get_user("nobody") is None

    def test_delete_user(self) -> None:
        pool = InMemoryUserPool()
        pool.create_user("bob")
        assert pool.delete_user("bob") is True
        assert pool.get_user("bob") is None
        assert pool.delete_user("bob") is False

    def test_reset_password(self) -> None:
        pool = InMemoryUserPool()
        pool.create_user("carol")
        assert pool.reset_password("carol") is True
        user = pool.get_user("carol")
        assert user is not None
        assert user["status"] == "RESET_REQUIRED"

    def test_reset_password_missing_user(self) -> None:
        pool = InMemoryUserPool()
        assert pool.reset_password("ghost") is False

    def test_list_users(self) -> None:
        pool = InMemoryUserPool()
        pool.create_user("u1", email="u1@test.com")
        pool.create_user("u2", email="u2@test.com")
        pool.create_user("u3", email="u3@test.com")
        users = pool.list_users()
        assert len(users) == 3

    def test_list_users_with_limit(self) -> None:
        pool = InMemoryUserPool()
        for i in range(10):
            pool.create_user(f"user-{i}")
        assert len(pool.list_users(limit=3)) == 3

    def test_create_with_temporary_password(self) -> None:
        pool = InMemoryUserPool()
        result = pool.create_user("dave", temporary_password="TempPass1!")
        assert result["status"] == "FORCE_CHANGE_PASSWORD"

    def test_create_with_attributes(self) -> None:
        pool = InMemoryUserPool()
        result = pool.create_user(
            "eve",
            email="eve@test.com",
            phone="+1234567890",
            attributes={"custom:tier": "premium"},
        )
        user = pool.get_user("eve")
        assert user is not None
        assert user["attributes"]["custom:tier"] == "premium"
        assert user["attributes"]["phone_number"] == "+1234567890"


@mock_aws
class TestCognitoUserPool:
    @staticmethod
    def _create_pool() -> tuple[str, CognitoUserPool]:
        client = boto3.client("cognito-idp", region_name="us-east-1")
        response = client.create_user_pool(PoolName="test-pool")
        pool_id = response["UserPool"]["Id"]
        return pool_id, CognitoUserPool(user_pool_id=pool_id)

    def test_create_and_get_user(self) -> None:
        _, pool = self._create_pool()
        result = pool.create_user("alice", email="alice@example.com")
        assert result["username"] == "alice"

        user = pool.get_user("alice")
        assert user is not None
        assert user["username"] == "alice"

    def test_get_missing_user(self) -> None:
        _, pool = self._create_pool()
        assert pool.get_user("nobody") is None

    def test_delete_user(self) -> None:
        _, pool = self._create_pool()
        pool.create_user("bob")
        assert pool.delete_user("bob") is True
        assert pool.get_user("bob") is None

    def test_delete_missing_user(self) -> None:
        _, pool = self._create_pool()
        assert pool.delete_user("ghost") is False

    def test_list_users(self) -> None:
        _, pool = self._create_pool()
        pool.create_user("u1")
        pool.create_user("u2")
        users = pool.list_users()
        assert len(users) == 2
        usernames = {u["username"] for u in users}
        assert "u1" in usernames
        assert "u2" in usernames

    def test_reset_password(self) -> None:
        pool_id, pool = self._create_pool()
        pool.create_user("carol", email="carol@test.com", temporary_password="TempP@ss1")
        # Confirm the user so reset is allowed (moto rejects reset in FORCE_CHANGE_PASSWORD)
        client = boto3.client("cognito-idp", region_name="us-east-1")
        client.admin_set_user_password(
            UserPoolId=pool_id,
            Username="carol",
            Password="ConfirmedP@ss1",
            Permanent=True,
        )
        assert pool.reset_password("carol") is True
