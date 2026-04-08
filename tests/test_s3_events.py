"""Tests for S3 event notification wiring."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.events import InMemoryEventBus
from ml_platform.s3_events import FileStoreEventEmitter, S3NotificationManager
from ml_platform.storage import LocalFileStore


class TestFileStoreEventEmitter:
    def test_put_emits_event(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        emitter.put("doc.txt", b"hello", content_type="text/plain")
        assert len(bus.published) == 1
        evt = bus.published[0]
        assert evt["detail_type"] == "FileUploaded"
        assert evt["detail"]["key"] == "doc.txt"
        assert evt["detail"]["content_type"] == "text/plain"
        assert evt["detail"]["size_bytes"] == 5

    def test_delete_emits_event(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        emitter.put("f.txt", b"data")
        bus.published.clear()
        assert emitter.delete("f.txt") is True
        assert len(bus.published) == 1
        assert bus.published[0]["detail_type"] == "FileDeleted"

    def test_delete_nonexistent_no_event(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        assert emitter.delete("missing.txt") is False
        assert len(bus.published) == 0

    def test_get_delegates_without_event(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        emitter.put("f.bin", b"bytes")
        bus.published.clear()
        assert emitter.get("f.bin") == b"bytes"
        assert len(bus.published) == 0

    def test_subscriber_receives_upload_events(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        uploads: list[dict[str, Any]] = []
        bus.subscribe("FileUploaded", uploads.append)
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        emitter.put("a.jpg", b"img1")
        emitter.put("b.jpg", b"img2")
        assert len(uploads) == 2

    def test_list_keys_delegates(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        bus = InMemoryEventBus()
        emitter = FileStoreEventEmitter(store=store, event_bus=bus)

        emitter.put("dir/a.txt", b"a")
        emitter.put("dir/b.txt", b"b")
        keys = emitter.list_keys("dir")
        assert len(keys) == 2


@mock_aws
class TestS3NotificationManager:
    def test_add_queue_notification(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="notify-bucket")

        mgr = S3NotificationManager(bucket="notify-bucket", region="us-east-1")
        mgr.add_queue_notification(
            queue_arn="arn:aws:sqs:us-east-1:123456789012:my-queue",
            prefix="uploads/",
            suffix=".jpg",
        )
        config = mgr.get_configuration()
        queues = config.get("QueueConfigurations", [])
        assert len(queues) == 1
        assert queues[0]["QueueArn"] == "arn:aws:sqs:us-east-1:123456789012:my-queue"

    def test_add_lambda_notification(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="lambda-bucket")

        mgr = S3NotificationManager(bucket="lambda-bucket", region="us-east-1")
        mgr.add_lambda_notification(
            function_arn="arn:aws:lambda:us-east-1:123456789012:function:process",
        )
        config = mgr.get_configuration()
        lambdas = config.get("LambdaFunctionConfigurations", [])
        assert len(lambdas) == 1

    def test_clear_notifications(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="clear-bucket")

        mgr = S3NotificationManager(bucket="clear-bucket", region="us-east-1")
        mgr.add_queue_notification(
            queue_arn="arn:aws:sqs:us-east-1:123456789012:q",
        )
        mgr.clear_notifications()
        config = mgr.get_configuration()
        assert not config.get("QueueConfigurations")
