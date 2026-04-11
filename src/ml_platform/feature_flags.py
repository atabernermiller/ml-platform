"""Feature flags and A/B testing with DynamoDB and static backends.

Provides a :class:`FeatureGate` interface for checking feature flags and
gradual rollouts, with implementations for DynamoDB-backed flags and
a static config-based backend for local development.

Usage::

    from ml_platform.feature_flags import StaticFeatureGate, DynamoDBFeatureGate

    gate = StaticFeatureGate(flags={"new_ui": True, "beta_model": False})
    if gate.is_enabled("new_ui"):
        ...

    gate = DynamoDBFeatureGate(table_name="feature-flags")
    variant = gate.get_variant("checkout_flow", context={"user_id": "u-123"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import FeatureGate
from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import Table as DynamoDBTable_

logger = logging.getLogger(__name__)

__all__ = [
    "StaticFeatureGate",
    "DynamoDBFeatureGate",
]


class StaticFeatureGate(FeatureGate):
    """In-memory feature gate backed by a static dict.

    Useful for local development, testing, and config-file-based flags.

    Args:
        flags: Mapping of flag names to enabled states.
        variants: Optional mapping of flag names to variant names for
            A/B testing.
    """

    def __init__(
        self,
        flags: dict[str, bool] | None = None,
        variants: dict[str, str] | None = None,
    ) -> None:
        self._flags: dict[str, bool] = dict(flags or {})
        self._variants: dict[str, str] = dict(variants or {})

    def is_enabled(
        self, flag: str, *, context: dict[str, Any] | None = None
    ) -> bool:
        return self._flags.get(flag, False)

    def get_variant(
        self, flag: str, *, context: dict[str, Any] | None = None
    ) -> str:
        return self._variants.get(flag, "control")

    def all_flags(self) -> dict[str, bool]:
        return dict(self._flags)

    def set_flag(self, flag: str, enabled: bool) -> None:
        """Update a flag's state at runtime.

        Args:
            flag: Flag name.
            enabled: New enabled state.
        """
        self._flags[flag] = enabled

    def set_variant(self, flag: str, variant: str) -> None:
        """Update a flag's variant at runtime.

        Args:
            flag: Flag name.
            variant: New variant name.
        """
        self._variants[flag] = variant


class DynamoDBFeatureGate(FeatureGate):
    """DynamoDB-backed feature gate with caching and percentage rollouts.

    Table schema::

        Partition key: ``flag_name`` (String)

    Each item may contain:

    - ``enabled`` (Boolean): Whether the flag is on.
    - ``percentage`` (Number, 0-100): Percentage rollout. When set,
      the flag is enabled for a deterministic subset of users based
      on ``context["user_id"]``.
    - ``variant`` (String): Active variant name for A/B tests.
    - ``variants`` (Map): Variant weights for multi-armed experiments.

    AWS credentials are resolved via boto3's default credential chain.

    Args:
        table_name: DynamoDB table name.
        region: AWS region.
        cache_ttl_s: Cache time-to-live in seconds.
    """

    def __init__(
        self,
        table_name: str,
        region: str | None = None,
        cache_ttl_s: int = 60,
    ) -> None:
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name=resolve_region(region))
        self._table: DynamoDBTable_ = dynamodb.Table(table_name)
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    def _get_flag_item(self, flag: str) -> dict[str, Any] | None:
        cached = self._cache.get(flag)
        if cached and (time.time() - cached[1]) < self._cache_ttl_s:
            return cached[0]

        response = self._table.get_item(Key={"flag_name": flag})
        item = response.get("Item")
        if item and self._cache_ttl_s > 0:
            self._cache[flag] = (item, time.time())
        return item

    def is_enabled(
        self, flag: str, *, context: dict[str, Any] | None = None
    ) -> bool:
        item = self._get_flag_item(flag)
        if item is None:
            return False

        enabled = item.get("enabled", False)
        if not enabled:
            return False

        percentage = item.get("percentage")
        if percentage is not None and context:
            user_id = context.get("user_id", "")
            if user_id:
                bucket = _hash_to_percentage(f"{flag}:{user_id}")
                return bucket < float(percentage)

        return bool(enabled)

    def get_variant(
        self, flag: str, *, context: dict[str, Any] | None = None
    ) -> str:
        item = self._get_flag_item(flag)
        if item is None:
            return "control"
        return item.get("variant", "control")

    def all_flags(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        response = self._table.scan()
        for item in response.get("Items", []):
            name = item.get("flag_name", "")
            if name:
                results[name] = bool(item.get("enabled", False))
        return results

    def clear_cache(self) -> None:
        """Manually clear the flag cache."""
        self._cache.clear()


def _hash_to_percentage(value: str) -> float:
    """Deterministically map a string to a float in [0, 100)."""
    digest = hashlib.md5(value.encode()).hexdigest()
    return (int(digest[:8], 16) % 10000) / 100.0
