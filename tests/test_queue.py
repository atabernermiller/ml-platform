"""Tests for queue and event processing backends."""

from __future__ import annotations

import json
from typing import Any

import boto3
import pytest
from moto import mock_aws

from ml_platform.queue import InMemoryQueueBackend, QueueWorker, SQSQueueBackend


class TestInMemoryQueueBackend:
    def test_send_and_receive(self) -> None:
        q = InMemoryQueueBackend()
        msg_id = q.send({"action": "test"})
        assert isinstance(msg_id, str)

        messages = q.receive(max_messages=1)
        assert len(messages) == 1
        assert messages[0]["body"] == {"action": "test"}

    def test_delete_removes_in_flight(self) -> None:
        q = InMemoryQueueBackend()
        q.send({"key": "value"})
        messages = q.receive(max_messages=1)
        q.delete(messages[0]["receipt_handle"])
        assert len(q._in_flight) == 0

    def test_receive_empty_queue(self) -> None:
        q = InMemoryQueueBackend()
        messages = q.receive(max_messages=5)
        assert messages == []

    def test_fifo_ordering(self) -> None:
        q = InMemoryQueueBackend()
        q.send({"order": 1})
        q.send({"order": 2})
        q.send({"order": 3})

        messages = q.receive(max_messages=3)
        assert [m["body"]["order"] for m in messages] == [1, 2, 3]

    def test_pending_count(self) -> None:
        q = InMemoryQueueBackend()
        q.send({"a": 1})
        q.send({"b": 2})
        assert q.pending_count == 2
        q.receive(max_messages=1)
        assert q.pending_count == 1


class TestQueueWorker:
    def test_run_once_processes_messages(self) -> None:
        q = InMemoryQueueBackend()
        q.send({"value": 10})
        q.send({"value": 20})

        processed: list[dict[str, Any]] = []

        class TestWorker(QueueWorker):
            def process(self, message: dict[str, Any]) -> None:
                processed.append(message)

        worker = TestWorker(q, max_messages=5, wait_time_s=0)
        count = worker.run_once()
        assert count == 2
        assert len(processed) == 2

    def test_run_with_max_iterations(self) -> None:
        q = InMemoryQueueBackend()
        iterations = 0

        class CountingWorker(QueueWorker):
            def process(self, message: dict[str, Any]) -> None:
                nonlocal iterations
                iterations += 1

        worker = CountingWorker(q, max_messages=1, wait_time_s=0)
        worker.run(max_iterations=3)
        # No messages, but should have run 3 iterations
        assert iterations == 0

    def test_failed_message_not_deleted(self) -> None:
        q = InMemoryQueueBackend()
        q.send({"fail": True})

        class FailingWorker(QueueWorker):
            def process(self, message: dict[str, Any]) -> None:
                raise ValueError("Processing failed")

        worker = FailingWorker(q, max_messages=1, wait_time_s=0)
        count = worker.run_once()
        assert count == 0
        # Message still in-flight
        assert len(q._in_flight) == 1


@mock_aws
class TestSQSQueueBackend:
    def test_send_and_receive(self) -> None:
        sqs = boto3.client("sqs", region_name="us-east-1")
        response = sqs.create_queue(QueueName="test-queue")
        queue_url = response["QueueUrl"]

        q = SQSQueueBackend(queue_url=queue_url, region="us-east-1")
        msg_id = q.send({"action": "process"})
        assert isinstance(msg_id, str)

        messages = q.receive(max_messages=1, wait_time_s=0)
        assert len(messages) == 1
        assert messages[0]["body"] == {"action": "process"}

    def test_delete_message(self) -> None:
        sqs = boto3.client("sqs", region_name="us-east-1")
        response = sqs.create_queue(QueueName="del-queue")
        queue_url = response["QueueUrl"]

        q = SQSQueueBackend(queue_url=queue_url, region="us-east-1")
        q.send({"key": "val"})

        messages = q.receive(max_messages=1, wait_time_s=0)
        q.delete(messages[0]["receipt_handle"])

        # Verify no messages left
        remaining = q.receive(max_messages=1, wait_time_s=0)
        assert len(remaining) == 0
