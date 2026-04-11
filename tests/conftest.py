"""Shared pytest fixtures for ml-platform tests."""

from __future__ import annotations

from typing import Generator

import boto3
import pytest
from moto import mock_aws

from ml_platform.config import ServiceConfig


@pytest.fixture()
def service_config() -> ServiceConfig:
    """Return a ``ServiceConfig`` populated with test-safe defaults."""
    return ServiceConfig(
        service_name="test-service",
        s3_checkpoint_bucket="test-checkpoint-bucket",
        s3_checkpoint_prefix="checkpoints/",
        checkpoint_interval_s=10,
        metrics_interval_s=5,
        state_backend="dynamodb",
        state_table_name="test-context",
        state_ttl_s=3600,
    )


@pytest.fixture()
def mock_s3() -> Generator[None, None, None]:
    """Provide a moto-mocked S3 environment with a pre-created test bucket."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-checkpoint-bucket")
        yield


@pytest.fixture()
def mock_dynamodb() -> Generator[None, None, None]:
    """Provide a moto-mocked DynamoDB environment with a test context table.

    The table mirrors the schema expected by ``DynamoDBContextStore``:
    partition key ``request_id`` (S) with TTL on the ``ttl`` attribute.
    """
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="test-context",
            KeySchema=[{"AttributeName": "request_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "request_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.update_time_to_live(
            TableName="test-context",
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        yield
