"""Transactional email service with SES and console backends.

Provides an ``EmailService`` that supports template rendering, retry
logic, and send tracking, backed by AWS SES in production and a
console-logger fallback for local development.

Usage::

    from ml_platform.email import SESEmailBackend, ConsoleEmailBackend

    backend = SESEmailBackend(region="us-east-1", default_sender="noreply@example.com")
    backend.send(
        to=["user@example.com"],
        subject="Order Confirmed",
        body_text="Your order #123 has been confirmed.",
        body_html="<h1>Order Confirmed</h1><p>Order #123</p>",
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import EmailBackend

if TYPE_CHECKING:
    from mypy_boto3_ses.client import SESClient

logger = logging.getLogger(__name__)

__all__ = [
    "SESEmailBackend",
    "ConsoleEmailBackend",
    "render_template",
    "Attachment",
]


class Attachment:
    """An email attachment.

    Attributes:
        filename: Name shown in the email client.
        content: Raw file bytes.
        content_type: MIME type (e.g. ``"application/pdf"``).
    """

    __slots__ = ("filename", "content", "content_type")

    def __init__(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.filename = filename
        self.content = content
        self.content_type = content_type


def render_template(template: str, variables: dict[str, str]) -> str:
    """Render a simple template by substituting ``{{key}}`` placeholders.

    Args:
        template: Template string with ``{{key}}`` placeholders.
        variables: Mapping of placeholder names to values.

    Returns:
        Rendered string.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", value)
    return result


class SESEmailBackend(EmailBackend):
    """AWS SES-backed email sender.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        ses:SendEmail – on the verified identity/domain

    Args:
        region: AWS region for the SES client.
        default_sender: Default ``From`` address when none is specified
            in :meth:`send`.
        configuration_set: Optional SES configuration set name for
            send tracking and event publishing.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        default_sender: str = "",
        configuration_set: str = "",
    ) -> None:
        import boto3

        self._region = region
        self._default_sender = default_sender
        self._configuration_set = configuration_set
        self._ses: SESClient = boto3.client("ses", region_name=region)

    def send(
        self,
        *,
        to: list[str],
        subject: str,
        body_text: str,
        body_html: str = "",
        from_addr: str = "",
        reply_to: list[str] | None = None,
    ) -> str:
        sender = from_addr or self._default_sender
        if not sender:
            raise ValueError("No sender address: provide from_addr or set default_sender")

        body: dict[str, Any] = {"Text": {"Data": body_text, "Charset": "UTF-8"}}
        if body_html:
            body["Html"] = {"Data": body_html, "Charset": "UTF-8"}

        kwargs: dict[str, Any] = {
            "Source": sender,
            "Destination": {"ToAddresses": to},
            "Message": {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": body,
            },
        }
        if reply_to:
            kwargs["ReplyToAddresses"] = reply_to
        if self._configuration_set:
            kwargs["ConfigurationSetName"] = self._configuration_set

        response = self._ses.send_email(**kwargs)
        message_id: str = response["MessageId"]
        logger.info("Sent email to %s, message_id=%s", to, message_id)
        return message_id

    def send_raw(
        self,
        *,
        to: list[str],
        subject: str,
        body_text: str,
        body_html: str = "",
        from_addr: str = "",
        reply_to: list[str] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Send an email with MIME attachments via ``SendRawEmail``.

        Required for invoices, PDFs, or any binary attachment.

        Required IAM permissions::

            ses:SendRawEmail – on the verified identity/domain

        Args:
            to: Recipient addresses.
            subject: Subject line.
            body_text: Plain-text body.
            body_html: Optional HTML body.
            from_addr: Sender address (uses default if empty).
            reply_to: Optional reply-to addresses.
            attachments: List of :class:`Attachment` objects.

        Returns:
            SES message identifier.
        """
        sender = from_addr or self._default_sender
        if not sender:
            raise ValueError("No sender address: provide from_addr or set default_sender")

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(to)
        if reply_to:
            msg["Reply-To"] = ", ".join(reply_to)

        body_part = MIMEMultipart("alternative")
        body_part.attach(MIMEText(body_text, "plain", "utf-8"))
        if body_html:
            body_part.attach(MIMEText(body_html, "html", "utf-8"))
        msg.attach(body_part)

        for att in attachments or []:
            part = MIMEApplication(att.content)
            part.add_header(
                "Content-Disposition", "attachment", filename=att.filename
            )
            if att.content_type:
                part.set_type(att.content_type)
            msg.attach(part)

        response = self._ses.send_raw_email(
            Source=sender,
            Destinations=to,
            RawMessage={"Data": msg.as_string()},
        )
        message_id: str = response["MessageId"]
        logger.info("Sent raw email to %s, message_id=%s", to, message_id)
        return message_id


class ConsoleEmailBackend(EmailBackend):
    """Log-based email backend for local development.

    Logs email details at INFO level instead of sending. Useful for
    development and testing without SES credentials.

    Sent messages are also accumulated in :attr:`sent_messages` for
    test assertions.
    """

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    def send(
        self,
        *,
        to: list[str],
        subject: str,
        body_text: str,
        body_html: str = "",
        from_addr: str = "",
        reply_to: list[str] | None = None,
    ) -> str:
        message_id = f"console-{uuid.uuid4().hex[:12]}"
        record = {
            "message_id": message_id,
            "to": to,
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html,
            "from_addr": from_addr,
            "reply_to": reply_to or [],
        }
        self.sent_messages.append(record)
        logger.info(
            "Email (console): to=%s subject=%r\n%s",
            to,
            subject,
            body_text,
        )
        return message_id

    def send_raw(
        self,
        *,
        to: list[str],
        subject: str,
        body_text: str,
        body_html: str = "",
        from_addr: str = "",
        reply_to: list[str] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Log a raw email with attachments (local development).

        Args:
            to: Recipient addresses.
            subject: Subject line.
            body_text: Plain-text body.
            body_html: Optional HTML body.
            from_addr: Sender address.
            reply_to: Optional reply-to addresses.
            attachments: List of :class:`Attachment` objects.

        Returns:
            Console message identifier.
        """
        message_id = f"console-raw-{uuid.uuid4().hex[:12]}"
        att_info = [
            {"filename": a.filename, "size_bytes": len(a.content), "content_type": a.content_type}
            for a in (attachments or [])
        ]
        record = {
            "message_id": message_id,
            "to": to,
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html,
            "from_addr": from_addr,
            "reply_to": reply_to or [],
            "attachments": att_info,
        }
        self.sent_messages.append(record)
        logger.info(
            "Email (console raw): to=%s subject=%r attachments=%d",
            to,
            subject,
            len(att_info),
        )
        return message_id
