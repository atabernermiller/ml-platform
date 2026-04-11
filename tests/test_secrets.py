"""Tests for secret resolution backends."""

from __future__ import annotations

import json
import os

import boto3
import pytest
from moto import mock_aws

from ml_platform.secrets import AWSSecretResolver, EnvSecretResolver


class TestEnvSecretResolver:
    def test_get_existing_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_PASSWORD", "supersecret")
        resolver = EnvSecretResolver()
        assert resolver.get("db-password") == "supersecret"

    def test_get_missing_secret_raises(self) -> None:
        resolver = EnvSecretResolver()
        with pytest.raises(KeyError, match="Secret not found in env"):
            resolver.get("nonexistent/secret")

    def test_get_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DB_PASSWORD", "secret123")
        resolver = EnvSecretResolver(prefix="APP_")
        assert resolver.get("db-password") == "secret123"

    def test_get_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_CONFIG", '{"host": "localhost", "port": 5432}')
        resolver = EnvSecretResolver()
        config = resolver.get_json("db-config")
        assert config["host"] == "localhost"
        assert config["port"] == 5432

    def test_env_key_conversion(self) -> None:
        resolver = EnvSecretResolver()
        assert resolver._env_key("prod/db-password") == "PROD_DB_PASSWORD"
        assert resolver._env_key("my/nested/secret") == "MY_NESTED_SECRET"


@mock_aws
class TestAWSSecretResolver:
    def test_get_secret(self) -> None:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        client.create_secret(Name="test/secret", SecretString="myvalue")

        resolver = AWSSecretResolver(cache_ttl_s=0)
        assert resolver.get("test/secret") == "myvalue"

    def test_get_missing_secret_raises(self) -> None:
        resolver = AWSSecretResolver(cache_ttl_s=0)
        with pytest.raises(KeyError, match="Secret not found"):
            resolver.get("nonexistent/secret")

    def test_get_json_secret(self) -> None:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        client.create_secret(
            Name="db/config",
            SecretString=json.dumps({"host": "db.example.com", "port": 5432}),
        )

        resolver = AWSSecretResolver()
        config = resolver.get_json("db/config")
        assert config["host"] == "db.example.com"

    def test_caching(self) -> None:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        client.create_secret(Name="cached/secret", SecretString="value1")

        resolver = AWSSecretResolver(cache_ttl_s=300)
        assert resolver.get("cached/secret") == "value1"

        # Second call should use cache
        client.update_secret(SecretId="cached/secret", SecretString="value2")
        assert resolver.get("cached/secret") == "value1"

        resolver.clear_cache()
        assert resolver.get("cached/secret") == "value2"
