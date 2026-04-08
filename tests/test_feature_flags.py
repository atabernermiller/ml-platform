"""Tests for feature flags and A/B testing."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.feature_flags import DynamoDBFeatureGate, StaticFeatureGate


class TestStaticFeatureGate:
    def test_is_enabled(self) -> None:
        gate = StaticFeatureGate(flags={"new_ui": True, "beta": False})
        assert gate.is_enabled("new_ui") is True
        assert gate.is_enabled("beta") is False

    def test_unknown_flag_is_disabled(self) -> None:
        gate = StaticFeatureGate()
        assert gate.is_enabled("unknown") is False

    def test_get_variant(self) -> None:
        gate = StaticFeatureGate(variants={"checkout": "treatment_a"})
        assert gate.get_variant("checkout") == "treatment_a"

    def test_get_variant_default(self) -> None:
        gate = StaticFeatureGate()
        assert gate.get_variant("anything") == "control"

    def test_all_flags(self) -> None:
        flags = {"a": True, "b": False, "c": True}
        gate = StaticFeatureGate(flags=flags)
        assert gate.all_flags() == flags

    def test_set_flag(self) -> None:
        gate = StaticFeatureGate(flags={"x": False})
        assert gate.is_enabled("x") is False
        gate.set_flag("x", True)
        assert gate.is_enabled("x") is True

    def test_set_variant(self) -> None:
        gate = StaticFeatureGate()
        gate.set_variant("exp", "variant_b")
        assert gate.get_variant("exp") == "variant_b"


@mock_aws
class TestDynamoDBFeatureGate:
    @staticmethod
    def _create_gate() -> DynamoDBFeatureGate:
        dynamodb = boto3.client("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="feature-flags",
            KeySchema=[{"AttributeName": "flag_name", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "flag_name", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        return DynamoDBFeatureGate(
            table_name="feature-flags",
            region="us-east-1",
            cache_ttl_s=0,
        )

    def test_flag_not_found(self) -> None:
        gate = self._create_gate()
        assert gate.is_enabled("unknown") is False

    def test_enabled_flag(self) -> None:
        gate = self._create_gate()
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("feature-flags")
        table.put_item(Item={"flag_name": "new_feature", "enabled": True})
        assert gate.is_enabled("new_feature") is True

    def test_disabled_flag(self) -> None:
        gate = self._create_gate()
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("feature-flags")
        table.put_item(Item={"flag_name": "off_feature", "enabled": False})
        assert gate.is_enabled("off_feature") is False

    def test_percentage_rollout(self) -> None:
        gate = self._create_gate()
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("feature-flags")
        table.put_item(Item={"flag_name": "gradual", "enabled": True, "percentage": 50})

        enabled_count = 0
        total = 100
        for i in range(total):
            if gate.is_enabled("gradual", context={"user_id": f"user-{i}"}):
                enabled_count += 1

        assert 20 < enabled_count < 80

    def test_get_variant(self) -> None:
        gate = self._create_gate()
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("feature-flags")
        table.put_item(Item={"flag_name": "ab_test", "variant": "treatment_b"})
        assert gate.get_variant("ab_test") == "treatment_b"

    def test_all_flags(self) -> None:
        gate = self._create_gate()
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("feature-flags")
        table.put_item(Item={"flag_name": "a", "enabled": True})
        table.put_item(Item={"flag_name": "b", "enabled": False})
        flags = gate.all_flags()
        assert flags == {"a": True, "b": False}
