"""Tests for transactional email backends."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from ml_platform.email import ConsoleEmailBackend, SESEmailBackend, render_template


class TestRenderTemplate:
    def test_basic_substitution(self) -> None:
        template = "Hello {{name}}, your order #{{order_id}} is confirmed."
        result = render_template(template, {"name": "Alice", "order_id": "123"})
        assert result == "Hello Alice, your order #123 is confirmed."

    def test_no_placeholders(self) -> None:
        template = "No placeholders here."
        assert render_template(template, {"key": "val"}) == template

    def test_empty_variables(self) -> None:
        template = "Hello {{name}}"
        assert render_template(template, {}) == "Hello {{name}}"


class TestConsoleEmailBackend:
    def test_send_records_message(self) -> None:
        backend = ConsoleEmailBackend()
        msg_id = backend.send(
            to=["user@example.com"],
            subject="Test",
            body_text="Hello",
        )
        assert msg_id.startswith("console-")
        assert len(backend.sent_messages) == 1
        assert backend.sent_messages[0]["subject"] == "Test"
        assert backend.sent_messages[0]["to"] == ["user@example.com"]

    def test_send_multiple_messages(self) -> None:
        backend = ConsoleEmailBackend()
        backend.send(to=["a@test.com"], subject="First", body_text="1")
        backend.send(to=["b@test.com"], subject="Second", body_text="2")
        assert len(backend.sent_messages) == 2

    def test_send_with_html_and_reply_to(self) -> None:
        backend = ConsoleEmailBackend()
        backend.send(
            to=["user@test.com"],
            subject="HTML",
            body_text="plain",
            body_html="<h1>HTML</h1>",
            reply_to=["reply@test.com"],
        )
        msg = backend.sent_messages[0]
        assert msg["body_html"] == "<h1>HTML</h1>"
        assert msg["reply_to"] == ["reply@test.com"]


@mock_aws
class TestSESEmailBackend:
    def test_send_email(self) -> None:
        ses = boto3.client("ses", region_name="us-east-1")
        ses.verify_email_identity(EmailAddress="noreply@example.com")

        backend = SESEmailBackend(
            region="us-east-1",
            default_sender="noreply@example.com",
        )
        msg_id = backend.send(
            to=["user@example.com"],
            subject="Order Confirmed",
            body_text="Your order is confirmed.",
        )
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_send_without_sender_raises(self) -> None:
        backend = SESEmailBackend(region="us-east-1")
        with pytest.raises(ValueError, match="No sender address"):
            backend.send(
                to=["user@test.com"],
                subject="Test",
                body_text="Hello",
            )
