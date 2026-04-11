"""Tests for general-purpose file storage backends."""

from __future__ import annotations

import tempfile

import boto3
import pytest
from moto import mock_aws

from ml_platform.storage import LocalFileStore, S3FileStore


# ---------------------------------------------------------------------------
# LocalFileStore
# ---------------------------------------------------------------------------


class TestLocalFileStore:
    def test_put_and_get(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        store.put("docs/readme.txt", b"hello", content_type="text/plain")
        assert store.get("docs/readme.txt") == b"hello"

    def test_get_missing_key(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        assert store.get("nonexistent.txt") is None

    def test_delete(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        store.put("f.txt", b"data")
        assert store.delete("f.txt") is True
        assert store.get("f.txt") is None
        assert store.delete("f.txt") is False

    def test_list_keys(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        store.put("a/1.txt", b"a1")
        store.put("a/2.txt", b"a2")
        store.put("b/3.txt", b"b3")

        all_keys = store.list_keys()
        assert len(all_keys) == 3

        a_keys = store.list_keys("a")
        assert len(a_keys) == 2

    def test_public_url_with_base_url(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path), base_url="http://cdn.example.com")
        url = store.public_url("images/photo.jpg")
        assert url == "http://cdn.example.com/images/photo.jpg"

    def test_presigned_urls(self, tmp_path: str) -> None:
        store = LocalFileStore(str(tmp_path))
        upload_url = store.presigned_upload_url("test.bin")
        download_url = store.presigned_download_url("test.bin")
        assert upload_url.startswith("file://")
        assert download_url.startswith("file://")


# ---------------------------------------------------------------------------
# S3FileStore
# ---------------------------------------------------------------------------


@mock_aws
class TestS3FileStore:
    @staticmethod
    def _make_store(prefix: str = "") -> S3FileStore:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-files")
        return S3FileStore(bucket="test-files", prefix=prefix)

    def test_put_and_get(self) -> None:
        store = self._make_store()
        store.put("test.txt", b"content", content_type="text/plain")
        assert store.get("test.txt") == b"content"

    def test_get_missing_key(self) -> None:
        store = self._make_store()
        assert store.get("missing.txt") is None

    def test_delete(self) -> None:
        store = self._make_store()
        store.put("del.txt", b"data")
        assert store.delete("del.txt") is True
        assert store.get("del.txt") is None

    def test_list_keys(self) -> None:
        store = self._make_store(prefix="assets")
        store.put("img/a.jpg", b"a")
        store.put("img/b.jpg", b"b")
        keys = store.list_keys("img")
        assert len(keys) == 2

    def test_presigned_urls(self) -> None:
        store = self._make_store()
        url = store.presigned_upload_url("upload.bin")
        assert "test-files" in url
        url = store.presigned_download_url("download.bin")
        assert "test-files" in url

    def test_public_url_s3(self) -> None:
        store = self._make_store()
        url = store.public_url("file.txt")
        assert "test-files" in url
        assert "file.txt" in url

    def test_public_url_cloudfront(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="cf-bucket")
        store = S3FileStore(
            bucket="cf-bucket",
            cloudfront_domain="d123.cloudfront.net",
        )
        url = store.public_url("img/photo.jpg")
        assert url == "https://d123.cloudfront.net/img/photo.jpg"

    def test_prefix_handling(self) -> None:
        store = self._make_store(prefix="uploads")
        key = store.put("photo.jpg", b"image-data")
        assert key.startswith("uploads/")
