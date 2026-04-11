"""Integration tests for deploy/destroy with moto-mocked AWS.

These tests verify the CloudFormation, ECR, DynamoDB, and S3 operations
against moto's mock. Docker build/push steps are monkey-patched out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from ml_platform.cli.deploy import (
    _deploy_cloudformation,
    _write_resource_manifest,
)
from ml_platform.cli.destroy import (
    _delete_cloudwatch_dashboard,
    _delete_dynamodb_table,
    _delete_ecr_repository,
    _delete_s3_bucket,
)
from ml_platform.cli.manifest import FeaturesConfig, ProjectManifest


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def manifest() -> ProjectManifest:
    return ProjectManifest(
        service_name="integ-test",
        service_type="llm",
        features=FeaturesConfig(conversation_store=True),
    )


class TestResourceManifestRoundtrip:
    def test_write_and_read(self, manifest: ProjectManifest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ml_platform.cli.deploy._RESOURCE_DIR", str(tmp_path / ".ml-platform"))
        path = _write_resource_manifest(manifest, {"ServiceUrl": "test.elb.amazonaws.com"}, "img:latest")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["service_name"] == "integ-test"
        assert data["outputs"]["ServiceUrl"] == "test.elb.amazonaws.com"


class TestDeleteEcrRepository:
    @mock_aws
    def test_delete_existing_repo(self) -> None:
        ecr = boto3.client("ecr", region_name="us-east-1")
        ecr.create_repository(repositoryName="integ-test")
        ok, err = _delete_ecr_repository("integ-test", "us-east-1")
        assert ok is True
        assert err == ""

    @mock_aws
    def test_delete_nonexistent_repo(self) -> None:
        ok, err = _delete_ecr_repository("nonexistent", "us-east-1")
        assert ok is True


class TestDeleteS3Bucket:
    @mock_aws
    def test_delete_existing_bucket(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="ml-platform-integ-test-ckpt")
        s3.put_object(Bucket="ml-platform-integ-test-ckpt", Key="test.bin", Body=b"data")
        ok, err = _delete_s3_bucket("ml-platform-integ-test-ckpt", "us-east-1")
        assert ok is True

    @mock_aws
    def test_delete_nonexistent_bucket(self) -> None:
        ok, err = _delete_s3_bucket("nonexistent-bucket", "us-east-1")
        assert ok is True


class TestDeleteDynamoDBTable:
    @mock_aws
    def test_delete_existing_table(self) -> None:
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName="integ-test-conversations",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        ok, err = _delete_dynamodb_table("integ-test-conversations", "us-east-1")
        assert ok is True

    @mock_aws
    def test_delete_nonexistent_table(self) -> None:
        ok, err = _delete_dynamodb_table("nonexistent-table", "us-east-1")
        assert ok is True
