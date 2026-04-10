"""General-purpose file storage with S3 and local filesystem backends.

Unlike :class:`~ml_platform.serving.state_manager.S3StateManager`, which
is scoped to model checkpoint directories, ``FileStore`` handles individual
files with content-type awareness, presigned URLs for browser uploads, and
public URL generation.

Two backends are provided:

- ``S3FileStore`` -- production-ready, supports presigned URLs and optional
  CloudFront CDN integration.
- ``LocalFileStore`` -- stores files on the local filesystem for development
  and testing.

Usage::

    from ml_platform.storage import S3FileStore, LocalFileStore

    store = S3FileStore(bucket="my-assets", region="us-east-1")
    store.put("images/photo.jpg", data, content_type="image/jpeg")
    url = store.public_url("images/photo.jpg")
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Any
from pathlib import Path

from ml_platform._interfaces import FileStore

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

__all__ = [
    "S3FileStore",
    "LocalFileStore",
]


class S3FileStore(FileStore):
    """S3-backed file store with presigned URL and CDN support.

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        s3:PutObject       – on arn:aws:s3:::{bucket}/*
        s3:GetObject       – on arn:aws:s3:::{bucket}/*
        s3:DeleteObject    – on arn:aws:s3:::{bucket}/*
        s3:ListBucket      – on arn:aws:s3:::{bucket}

    Args:
        bucket: S3 bucket name.
        prefix: Optional key prefix for all operations.
        region: AWS region for the S3 client.
        cloudfront_domain: Optional CloudFront domain for public URLs.
            When set, :meth:`public_url` returns a CloudFront URL instead
            of an S3 URL.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        cloudfront_domain: str = "",
    ) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._cloudfront_domain = cloudfront_domain.rstrip("/")
        self._s3: S3Client = boto3.client("s3", region_name=region)

    def _full_key(self, key: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{key.lstrip('/')}"
        return key.lstrip("/")

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        full_key = self._full_key(key)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=full_key,
            Body=data,
            ContentType=content_type,
        )
        logger.debug("Uploaded %d bytes to s3://%s/%s", len(data), self._bucket, full_key)
        return full_key

    def get(self, key: str) -> bytes | None:
        full_key = self._full_key(key)
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=full_key)
            return response["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            if "NoSuchKey" in str(type(exc).__name__) or "404" in str(exc):
                logger.debug("Key not found: s3://%s/%s", self._bucket, full_key)
                return None
            raise

    def delete(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=full_key)
        except Exception:
            return False
        self._s3.delete_object(Bucket=self._bucket, Key=full_key)
        logger.debug("Deleted s3://%s/%s", self._bucket, full_key)
        return True

    def presigned_upload_url(
        self,
        key: str,
        *,
        expires_in_s: int = 3600,
        content_type: str = "application/octet-stream",
    ) -> str:
        full_key = self._full_key(key)
        return self._s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": full_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_in_s,
        )

    def presigned_download_url(self, key: str, *, expires_in_s: int = 3600) -> str:
        full_key = self._full_key(key)
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": full_key},
            ExpiresIn=expires_in_s,
        )

    def public_url(self, key: str) -> str:
        full_key = self._full_key(key)
        if self._cloudfront_domain:
            return f"https://{self._cloudfront_domain}/{full_key}"
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{full_key}"

    def list_keys(
        self, prefix: str = "", *, max_keys: int | None = None
    ) -> list[str]:
        search_prefix = self._full_key(prefix) if prefix else (self._prefix + "/" if self._prefix else "")
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._bucket,
                "Prefix": search_prefix,
            }
            if max_keys is not None:
                kwargs["MaxKeys"] = min(1000, max_keys - len(keys))
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._s3.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                keys.append(obj["Key"])
                if max_keys is not None and len(keys) >= max_keys:
                    return keys
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        return keys


class LocalFileStore(FileStore):
    """Filesystem-backed file store for local development.

    Files are stored under ``base_dir`` with the key as the relative
    path.  Presigned URLs return ``file://`` URIs.

    Args:
        base_dir: Root directory for file storage.
        base_url: Base URL for :meth:`public_url` (default ``file://``).
    """

    def __init__(self, base_dir: str, base_url: str = "") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url.rstrip("/")

    def _path(self, key: str) -> Path:
        return self._base_dir / key.lstrip("/")

    def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug("Wrote %d bytes to %s", len(data), path)
        return key

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def presigned_upload_url(
        self,
        key: str,
        *,
        expires_in_s: int = 3600,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self._path(key).as_uri()

    def presigned_download_url(self, key: str, *, expires_in_s: int = 3600) -> str:
        return self._path(key).as_uri()

    def public_url(self, key: str) -> str:
        if self._base_url:
            return f"{self._base_url}/{key.lstrip('/')}"
        return self._path(key).as_uri()

    def list_keys(
        self, prefix: str = "", *, max_keys: int | None = None
    ) -> list[str]:
        search_dir = self._base_dir / prefix if prefix else self._base_dir
        if not search_dir.exists():
            return []
        keys: list[str] = []
        for path in sorted(search_dir.rglob("*")):
            if path.is_file():
                keys.append(str(path.relative_to(self._base_dir)))
                if max_keys is not None and len(keys) >= max_keys:
                    break
        return keys
