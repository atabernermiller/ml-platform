"""Tests for S3-backed state checkpoint manager."""

from __future__ import annotations

import os
import tempfile
import time

from ml_platform.serving.state_manager import S3StateManager


def _write_checkpoint(directory: str, filename: str, content: str) -> None:
    """Write a single file into a checkpoint directory."""
    path = os.path.join(directory, filename)
    with open(path, "w") as fh:
        fh.write(content)


def test_upload_and_download(mock_s3: None) -> None:
    mgr = S3StateManager(
        bucket="test-checkpoint-bucket",
        prefix="checkpoints/",
    )

    with tempfile.TemporaryDirectory() as src:
        _write_checkpoint(src, "state.json", '{"weights": [1, 2, 3]}')
        mgr.upload(src)

    restored_dir = mgr.download_latest()
    assert restored_dir is not None

    restored_path = os.path.join(restored_dir, "state.json")
    with open(restored_path) as fh:
        assert fh.read() == '{"weights": [1, 2, 3]}'


def test_download_latest_empty(mock_s3: None) -> None:
    mgr = S3StateManager(
        bucket="test-checkpoint-bucket",
        prefix="empty-prefix/",
    )
    assert mgr.download_latest() is None


def test_list_checkpoints(mock_s3: None) -> None:
    mgr = S3StateManager(
        bucket="test-checkpoint-bucket",
        prefix="checkpoints/",
    )

    with tempfile.TemporaryDirectory() as src:
        _write_checkpoint(src, "a.bin", "first")
        mgr.upload(src)

    time.sleep(1.1)

    with tempfile.TemporaryDirectory() as src:
        _write_checkpoint(src, "b.bin", "second")
        mgr.upload(src)

    checkpoints = mgr.list_checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints == sorted(checkpoints)
