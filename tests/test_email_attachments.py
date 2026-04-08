"""Tests for email attachments (send_raw with MIME)."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from ml_platform.email import Attachment, ConsoleEmailBackend, SESEmailBackend


class TestConsoleEmailBackendRaw:
    def test_send_raw_with_attachments(self) -> None:
        backend = ConsoleEmailBackend()
        att = Attachment(
            filename="invoice.pdf",
            content=b"%PDF-fake-content",
            content_type="application/pdf",
        )
        msg_id = backend.send_raw(
            to=["user@example.com"],
            subject="Your Invoice",
            body_text="Please find your invoice attached.",
            attachments=[att],
        )
        assert msg_id.startswith("console-raw-")
        assert len(backend.sent_messages) == 1
        msg = backend.sent_messages[0]
        assert len(msg["attachments"]) == 1
        assert msg["attachments"][0]["filename"] == "invoice.pdf"
        assert msg["attachments"][0]["size_bytes"] == len(b"%PDF-fake-content")

    def test_send_raw_without_attachments(self) -> None:
        backend = ConsoleEmailBackend()
        msg_id = backend.send_raw(
            to=["user@test.com"],
            subject="No Attachments",
            body_text="Just text.",
        )
        assert msg_id.startswith("console-raw-")
        assert len(backend.sent_messages) == 1
        assert backend.sent_messages[0]["attachments"] == []

    def test_send_raw_multiple_attachments(self) -> None:
        backend = ConsoleEmailBackend()
        attachments = [
            Attachment("a.txt", b"aaa", "text/plain"),
            Attachment("b.png", b"\x89PNG", "image/png"),
        ]
        backend.send_raw(
            to=["u@x.com"],
            subject="Multi",
            body_text="Multiple attachments.",
            attachments=attachments,
        )
        msg = backend.sent_messages[0]
        assert len(msg["attachments"]) == 2


@mock_aws
class TestSESEmailBackendRaw:
    def test_send_raw_with_attachment(self) -> None:
        ses = boto3.client("ses", region_name="us-east-1")
        ses.verify_email_identity(EmailAddress="noreply@example.com")

        backend = SESEmailBackend(
            region="us-east-1",
            default_sender="noreply@example.com",
        )
        att = Attachment("report.csv", b"col1,col2\n1,2", "text/csv")
        msg_id = backend.send_raw(
            to=["user@example.com"],
            subject="Report",
            body_text="Attached.",
            body_html="<p>Attached.</p>",
            attachments=[att],
        )
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_send_raw_without_sender_raises(self) -> None:
        backend = SESEmailBackend(region="us-east-1")
        with pytest.raises(ValueError, match="No sender address"):
            backend.send_raw(
                to=["user@test.com"],
                subject="Fail",
                body_text="Should fail.",
            )


class TestAttachment:
    def test_creation(self) -> None:
        att = Attachment("file.pdf", b"data", "application/pdf")
        assert att.filename == "file.pdf"
        assert att.content == b"data"
        assert att.content_type == "application/pdf"

    def test_default_content_type(self) -> None:
        att = Attachment("blob.bin", b"\x00\x01")
        assert att.content_type == "application/octet-stream"
