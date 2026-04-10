"""S3-backed state checkpoint manager.

Handles uploading local checkpoint directories to S3 and downloading the
most recent checkpoint for cold-start recovery.  Each checkpoint is stored
under a timestamped prefix so that rollbacks are straightforward.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)


class S3StateManager:
    """Manages model state checkpoints in S3.

    Directory layout in S3::

        s3://{bucket}/{prefix}{service_name}/
            2026-03-31T12:00:00Z/
                state.joblib
                pacer.json
            2026-03-31T12:05:00Z/
                state.joblib
                pacer.json

    AWS credentials are resolved via boto3's default credential chain
    (env vars, ``~/.aws/credentials``, ECS task role, EC2 instance
    profile).  No explicit keys are accepted.

    Required IAM permissions::

        s3:PutObject    – on arn:aws:s3:::{bucket}/*
        s3:GetObject    – on arn:aws:s3:::{bucket}/*
        s3:ListBucket   – on arn:aws:s3:::{bucket}

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix (should end with ``/``).
        region: AWS region for the S3 client.
    """

    def __init__(self, bucket: str, prefix: str, region: str = "us-east-1") -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._s3: S3Client = boto3.client("s3", region_name=region)

    def upload(self, local_dir: str) -> str:
        """Upload a local checkpoint directory to S3.

        Args:
            local_dir: Path to directory containing checkpoint files.

        Returns:
            The S3 key prefix where the checkpoint was stored.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        s3_prefix = f"{self._prefix}{timestamp}/"

        for root, _dirs, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}{relative}"
                self._s3.upload_file(local_path, self._bucket, s3_key)
                logger.debug("Uploaded %s -> s3://%s/%s", local_path, self._bucket, s3_key)

        logger.info("Checkpoint uploaded to s3://%s/%s", self._bucket, s3_prefix)
        return s3_prefix

    def download_latest(self) -> str | None:
        """Download the most recent checkpoint to a temporary directory.

        Returns:
            Path to a temporary directory containing the checkpoint files,
            or ``None`` if no checkpoints exist.  The caller owns the
            temporary directory and should clean it up when no longer needed.
        """
        response = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=self._prefix, Delimiter="/"
        )
        prefixes = [
            p["Prefix"] for p in response.get("CommonPrefixes", [])
        ]
        if not prefixes:
            return None

        latest_prefix = sorted(prefixes)[-1]

        objects = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=latest_prefix
        )
        if "Contents" not in objects:
            return None

        tmpdir = tempfile.mkdtemp(prefix="ml_platform_ckpt_")
        for obj in objects["Contents"]:
            key = obj["Key"]
            relative = key[len(latest_prefix):]
            if not relative:
                continue
            local_path = os.path.join(tmpdir, relative)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(self._bucket, key, local_path)
            logger.debug("Downloaded s3://%s/%s -> %s", self._bucket, key, local_path)

        logger.info("Restored checkpoint from s3://%s/%s", self._bucket, latest_prefix)
        return tmpdir

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoint prefixes, oldest first.

        Returns:
            Sorted list of S3 key prefixes (timestamps).
        """
        response = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=self._prefix, Delimiter="/"
        )
        prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]
        return sorted(prefixes)
