"""Queue and event processing with SQS and in-memory backends.

Provides a :class:`QueueWorker` base class for consuming messages from
SQS (or an in-memory queue for local development), with automatic
message deletion on success and configurable error handling.

Usage::

    from ml_platform.queue import SQSQueueBackend, InMemoryQueueBackend

    # Producer
    queue = SQSQueueBackend(queue_url="https://sqs.us-east-1.amazonaws.com/123/my-queue")  # noqa: E501
    queue.send({"action": "send_email", "to": "user@example.com"})

    # Consumer
    messages = queue.receive(max_messages=5)
    for msg in messages:
        process(msg["body"])
        queue.delete(msg["receipt_handle"])
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import QueueBackend
from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

__all__ = [
    "SQSQueueBackend",
    "InMemoryQueueBackend",
    "QueueWorker",
]


class SQSQueueBackend(QueueBackend):
    """AWS SQS-backed queue.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        sqs:SendMessage    – on the queue ARN
        sqs:ReceiveMessage – on the queue ARN
        sqs:DeleteMessage  – on the queue ARN

    Args:
        queue_url: Full SQS queue URL.
        region: AWS region.
    """

    def __init__(self, queue_url: str, region: str | None = None) -> None:
        import boto3

        self._queue_url = queue_url
        self._sqs: SQSClient = boto3.client("sqs", region_name=resolve_region(region))

    def send(self, message: dict[str, Any], *, delay_s: int = 0) -> str:
        response = self._sqs.send_message(
            QueueUrl=self._queue_url,
            MessageBody=json.dumps(message, default=str),
            DelaySeconds=delay_s,
        )
        msg_id: str = response["MessageId"]
        logger.debug("Sent SQS message %s", msg_id)
        return msg_id

    def receive(
        self, *, max_messages: int = 1, wait_time_s: int = 20
    ) -> list[dict[str, Any]]:
        response = self._sqs.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=wait_time_s,
        )
        results: list[dict[str, Any]] = []
        for msg in response.get("Messages", []):
            results.append({
                "body": json.loads(msg["Body"]),
                "receipt_handle": msg["ReceiptHandle"],
                "message_id": msg["MessageId"],
            })
        return results

    def delete(self, receipt_handle: str) -> None:
        self._sqs.delete_message(
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
        )


class InMemoryQueueBackend(QueueBackend):
    """In-memory queue for local development and testing.

    Thread-safe via :class:`queue.Queue`.  Messages are stored in a
    FIFO queue with receipt handles for delete-after-process semantics.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._in_flight: dict[str, tuple[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def send(self, message: dict[str, Any], *, delay_s: int = 0) -> str:
        msg_id = uuid.uuid4().hex
        self._queue.put((msg_id, message))
        return msg_id

    def receive(
        self, *, max_messages: int = 1, wait_time_s: int = 20
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(max_messages):
            try:
                msg_id, body = self._queue.get_nowait()
            except queue.Empty:
                break
            receipt_handle = uuid.uuid4().hex
            with self._lock:
                self._in_flight[receipt_handle] = (msg_id, body)
            results.append({
                "body": body,
                "receipt_handle": receipt_handle,
                "message_id": msg_id,
            })
        return results

    def delete(self, receipt_handle: str) -> None:
        with self._lock:
            self._in_flight.pop(receipt_handle, None)

    @property
    def pending_count(self) -> int:
        """Number of messages waiting in the queue."""
        return self._queue.qsize()


class QueueWorker:
    """Base class for queue message consumers.

    Subclass and override :meth:`process` to implement message handling.
    Call :meth:`run_once` for single-batch processing or :meth:`run`
    for a continuous polling loop.

    Args:
        backend: Queue backend to consume from.
        max_messages: Messages to fetch per batch.
        wait_time_s: Long-poll wait time.
    """

    def __init__(
        self,
        backend: QueueBackend,
        *,
        max_messages: int = 5,
        wait_time_s: int = 20,
    ) -> None:
        self._backend = backend
        self._max_messages = max_messages
        self._wait_time_s = wait_time_s
        self._running = False

    def process(self, message: dict[str, Any]) -> None:
        """Process a single message. Override in subclasses.

        Args:
            message: Parsed message body.

        Raises:
            Exception: If processing fails, the message is not deleted
                and becomes available for retry after the visibility timeout.
        """
        raise NotImplementedError

    def run_once(self) -> int:
        """Fetch and process one batch of messages.

        Returns:
            Number of messages successfully processed.
        """
        messages = self._backend.receive(
            max_messages=self._max_messages,
            wait_time_s=self._wait_time_s,
        )
        processed = 0
        for msg in messages:
            try:
                self.process(msg["body"])
                self._backend.delete(msg["receipt_handle"])
                processed += 1
            except Exception:
                logger.exception("Failed to process message %s", msg.get("message_id"))
        return processed

    def run(self, *, max_iterations: int | None = None) -> None:
        """Run the consumer loop.

        Args:
            max_iterations: Stop after this many iterations (``None``
                for indefinite).
        """
        self._running = True
        iterations = 0
        while self._running:
            self.run_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

    def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._running = False
