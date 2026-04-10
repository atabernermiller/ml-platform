"""Secret resolution with AWS Secrets Manager and environment variable backends.

Provides a :class:`SecretResolver` that loads secrets at startup with
caching support, replacing the insecure pattern of passing raw credentials
via environment variables.

Usage::

    from ml_platform.secrets import AWSSecretResolver, EnvSecretResolver

    resolver = AWSSecretResolver(region="us-east-1")
    db_password = resolver.get("prod/db/password")
    db_config = resolver.get_json("prod/db/config")
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import SecretResolver

if TYPE_CHECKING:
    from mypy_boto3_secretsmanager.client import SecretsManagerClient

logger = logging.getLogger(__name__)

__all__ = [
    "AWSSecretResolver",
    "EnvSecretResolver",
]


class AWSSecretResolver(SecretResolver):
    """AWS Secrets Manager backend with in-memory caching.

    Secrets are cached for ``cache_ttl_s`` seconds to reduce API calls.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        secretsmanager:GetSecretValue – on the secret ARN(s)

    Args:
        region: AWS region.
        cache_ttl_s: Cache time-to-live in seconds (0 disables caching).
    """

    def __init__(self, region: str = "us-east-1", cache_ttl_s: int = 300) -> None:
        import boto3

        self._client: SecretsManagerClient = boto3.client("secretsmanager", region_name=region)
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[str, float]] = {}

    def get(self, secret_id: str) -> str:
        cached = self._cache.get(secret_id)
        if cached and (time.time() - cached[1]) < self._cache_ttl_s:
            return cached[0]

        try:
            response = self._client.get_secret_value(SecretId=secret_id)
        except self._client.exceptions.ResourceNotFoundException as exc:
            raise KeyError(f"Secret not found: {secret_id}") from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve secret '{secret_id}': {type(exc).__name__}: {exc}"
            ) from exc

        value: str = response["SecretString"]
        if self._cache_ttl_s > 0:
            self._cache[secret_id] = (value, time.time())
        logger.debug("Resolved secret %s (cached=%s)", secret_id, self._cache_ttl_s > 0)
        return value

    def get_json(self, secret_id: str) -> dict[str, Any]:
        raw = self.get(secret_id)
        return json.loads(raw)

    def clear_cache(self) -> None:
        """Manually clear the in-memory secret cache."""
        self._cache.clear()


class EnvSecretResolver(SecretResolver):
    """Environment-variable-backed secret resolver for local development.

    Reads secrets from environment variables. The secret ID is converted
    to an environment variable name by uppercasing and replacing ``/``
    and ``-`` with ``_``.

    For example, ``"prod/db-password"`` becomes ``PROD_DB_PASSWORD``.

    Args:
        prefix: Optional prefix added to the env var name.
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def _env_key(self, secret_id: str) -> str:
        key = secret_id.upper().replace("/", "_").replace("-", "_")
        if self._prefix:
            return f"{self._prefix}{key}"
        return key

    def get(self, secret_id: str) -> str:
        env_key = self._env_key(secret_id)
        value = os.environ.get(env_key)
        if value is None:
            raise KeyError(f"Secret not found in env: {env_key} (for {secret_id})")
        return value

    def get_json(self, secret_id: str) -> dict[str, Any]:
        raw = self.get(secret_id)
        return json.loads(raw)
