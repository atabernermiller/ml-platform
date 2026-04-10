"""S3 event notification wiring for FileStore uploads.

Connects :class:`~ml_platform.storage.S3FileStore` uploads to downstream
consumers (SQS queues, Lambda functions, or the EventBridge bus) via
S3 bucket notification configuration.

Also provides a local-development ``FileStoreEventEmitter`` that wraps
a :class:`~ml_platform._interfaces.FileStore` and publishes events to
an :class:`~ml_platform._interfaces.EventBus` on every ``put`` / ``delete``.

Usage::

    from ml_platform.s3_events import S3NotificationManager, FileStoreEventEmitter

    # AWS: configure real S3 notifications
    mgr = S3NotificationManager(bucket="my-assets", region="us-east-1")
    mgr.add_queue_notification(queue_arn="arn:aws:sqs:...", events=["s3:ObjectCreated:*"])

    # Local: wrap a FileStore + EventBus for in-process event dispatch
    emitter = FileStoreEventEmitter(store=local_store, event_bus=local_bus)
    emitter.put("photos/1.jpg", data)  # triggers event on the bus
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import EventBus, FileStore

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

__all__ = [
    "S3NotificationManager",
    "FileStoreEventEmitter",
]


class S3NotificationManager:
    """Manage S3 bucket notification configurations.

    Wraps the ``put_bucket_notification_configuration`` API to wire S3
    object events to SQS queues, Lambda functions, or SNS topics.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        s3:PutBucketNotificationConfiguration – on the bucket
        s3:GetBucketNotificationConfiguration – on the bucket

    Args:
        bucket: S3 bucket name.
        region: AWS region.
    """

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        import boto3

        self._bucket = bucket
        self._s3: S3Client = boto3.client("s3", region_name=region)

    def get_configuration(self) -> dict[str, Any]:
        """Return the current notification configuration for the bucket.

        Returns:
            Notification configuration dict.
        """
        resp = self._s3.get_bucket_notification_configuration(Bucket=self._bucket)
        resp.pop("ResponseMetadata", None)
        return resp

    def add_queue_notification(
        self,
        queue_arn: str,
        events: list[str] | None = None,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """Add an SQS queue notification for object events.

        Args:
            queue_arn: ARN of the target SQS queue.
            events: S3 event types (default ``["s3:ObjectCreated:*"]``).
            prefix: Key prefix filter.
            suffix: Key suffix filter.
        """
        events = events or ["s3:ObjectCreated:*"]
        config = self.get_configuration()
        queue_configs: list[dict[str, Any]] = config.get("QueueConfigurations", [])

        rules: dict[str, Any] = {}
        filter_rules: list[dict[str, str]] = []
        if prefix:
            filter_rules.append({"Name": "prefix", "Value": prefix})
        if suffix:
            filter_rules.append({"Name": "suffix", "Value": suffix})
        if filter_rules:
            rules["Key"] = {"FilterRules": filter_rules}

        entry: dict[str, Any] = {
            "QueueArn": queue_arn,
            "Events": events,
        }
        if rules:
            entry["Filter"] = rules
        queue_configs.append(entry)
        config["QueueConfigurations"] = queue_configs

        self._s3.put_bucket_notification_configuration(
            Bucket=self._bucket,
            NotificationConfiguration=config,
        )
        logger.info("Added SQS notification %s -> %s", self._bucket, queue_arn)

    def add_lambda_notification(
        self,
        function_arn: str,
        events: list[str] | None = None,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """Add a Lambda function notification for object events.

        Args:
            function_arn: ARN of the target Lambda function.
            events: S3 event types (default ``["s3:ObjectCreated:*"]``).
            prefix: Key prefix filter.
            suffix: Key suffix filter.
        """
        events = events or ["s3:ObjectCreated:*"]
        config = self.get_configuration()
        lambda_configs: list[dict[str, Any]] = config.get("LambdaFunctionConfigurations", [])

        filter_rules: list[dict[str, str]] = []
        if prefix:
            filter_rules.append({"Name": "prefix", "Value": prefix})
        if suffix:
            filter_rules.append({"Name": "suffix", "Value": suffix})

        entry: dict[str, Any] = {
            "LambdaFunctionArn": function_arn,
            "Events": events,
        }
        if filter_rules:
            entry["Filter"] = {"Key": {"FilterRules": filter_rules}}
        lambda_configs.append(entry)
        config["LambdaFunctionConfigurations"] = lambda_configs

        self._s3.put_bucket_notification_configuration(
            Bucket=self._bucket,
            NotificationConfiguration=config,
        )
        logger.info("Added Lambda notification %s -> %s", self._bucket, function_arn)

    def clear_notifications(self) -> None:
        """Remove all notification configurations from the bucket."""
        self._s3.put_bucket_notification_configuration(
            Bucket=self._bucket,
            NotificationConfiguration={},
        )
        logger.info("Cleared all notifications for %s", self._bucket)


class FileStoreEventEmitter:
    """Wrapper that emits events on FileStore operations.

    Delegates storage to an underlying :class:`FileStore` and publishes
    ``FileUploaded`` / ``FileDeleted`` events to an :class:`EventBus`.

    Useful for local development where real S3 notifications are not
    available.

    Args:
        store: Underlying file store.
        event_bus: Event bus to publish to.
        source: Event source identifier.
    """

    def __init__(
        self,
        store: FileStore,
        event_bus: EventBus,
        source: str = "ml_platform.file_store",
    ) -> None:
        self._store = store
        self._bus = event_bus
        self._source = source

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file and emit a ``FileUploaded`` event."""
        result_key = self._store.put(key, data, content_type=content_type)
        self._bus.publish(
            source=self._source,
            detail_type="FileUploaded",
            detail={
                "key": result_key,
                "content_type": content_type,
                "size_bytes": len(data),
            },
        )
        return result_key

    def delete(self, key: str) -> bool:
        """Delete a file and emit a ``FileDeleted`` event if successful."""
        deleted = self._store.delete(key)
        if deleted:
            self._bus.publish(
                source=self._source,
                detail_type="FileDeleted",
                detail={"key": key},
            )
        return deleted

    def get(self, key: str) -> bytes | None:
        """Delegate to the underlying store (no event emitted)."""
        return self._store.get(key)

    def list_keys(self, prefix: str = "", *, max_keys: int | None = None) -> list[str]:
        """Delegate to the underlying store (no event emitted)."""
        return self._store.list_keys(prefix, max_keys=max_keys)
