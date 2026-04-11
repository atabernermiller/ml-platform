"""CDN integration with CloudFront and a no-op local fallback.

Provides signed URL generation, cache invalidation, and public URL
construction for CloudFront distributions.

Usage::

    from ml_platform.cdn import CloudFrontCDN, NoOpCDN

    cdn = CloudFrontCDN(
        domain="d123.cloudfront.net",
        distribution_id="EDFDVBD6EXAMPLE",
    )
    url = cdn.public_url("/images/photo.jpg")
    cdn.invalidate(["/images/*"])
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ml_platform._interfaces import CDNBackend
from ml_platform.config import resolve_region

if TYPE_CHECKING:
    from mypy_boto3_cloudfront.client import CloudFrontClient

logger = logging.getLogger(__name__)

__all__ = [
    "CloudFrontCDN",
    "NoOpCDN",
]


class CloudFrontCDN(CDNBackend):
    """AWS CloudFront CDN backend.

    Supports cache invalidation via the CloudFront API and public URL
    construction.  Signed URL generation requires a CloudFront key pair
    (not implemented here -- use ``S3FileStore.presigned_download_url``
    or configure CloudFront Origin Access Control).

    AWS credentials are resolved via boto3's default credential chain.

    Required IAM permissions::

        cloudfront:CreateInvalidation – on the distribution

    Args:
        domain: CloudFront distribution domain (e.g. ``d123.cloudfront.net``).
        distribution_id: CloudFront distribution ID for invalidation calls.
        region: AWS region for the CloudFront client.
    """

    def __init__(
        self,
        domain: str,
        distribution_id: str = "",
        region: str | None = None,
    ) -> None:
        self._domain = domain.rstrip("/")
        self._distribution_id = distribution_id
        self._region = resolve_region(region)
        self._cf_client: CloudFrontClient | None = None

    def _get_client(self) -> CloudFrontClient:
        if self._cf_client is None:
            import boto3
            self._cf_client = boto3.client("cloudfront", region_name=self._region)
        return self._cf_client

    def signed_url(self, path: str, *, expires_in_s: int = 3600) -> str:
        """Return a time-limited public URL.

        .. warning::

            This appends a plaintext ``expires`` query parameter — it is
            **not** cryptographically signed.  For access-controlled
            content, use ``botocore.signers.CloudFrontSigner`` with a
            CloudFront key pair, or ``S3FileStore.presigned_download_url``.
        """
        import warnings

        warnings.warn(
            "CloudFrontCDN.signed_url() appends a plaintext expiry parameter, "
            "not a cryptographic signature. Use expiring_url() to silence this "
            "warning, or use botocore.signers.CloudFrontSigner for real signing.",
            stacklevel=2,
        )
        return self.expiring_url(path, expires_in_s=expires_in_s)

    def expiring_url(self, path: str, *, expires_in_s: int = 3600) -> str:
        """Return a URL with a plaintext expiry query parameter.

        This does **not** provide cryptographic access control.  It is
        useful for cache-busting or short-lived links where the origin
        enforces access separately.

        Args:
            path: Resource path.
            expires_in_s: Seconds until the URL is considered expired.

        Returns:
            URL with ``?expires=<unix_timestamp>`` appended.
        """
        expiry = int(time.time()) + expires_in_s
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"https://{self._domain}{clean_path}?expires={expiry}"

    def invalidate(self, paths: list[str]) -> str:
        if not self._distribution_id:
            logger.warning("No distribution_id configured; skipping invalidation")
            return "no-op"
        client = self._get_client()
        caller_ref = f"ml-platform-{uuid.uuid4().hex[:8]}"
        response = client.create_invalidation(
            DistributionId=self._distribution_id,
            InvalidationBatch={
                "Paths": {"Quantity": len(paths), "Items": paths},
                "CallerReference": caller_ref,
            },
        )
        inv_id: str = response["Invalidation"]["Id"]
        logger.info("Created invalidation %s for %d paths", inv_id, len(paths))
        return inv_id

    def public_url(self, path: str) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"https://{self._domain}{clean_path}"


class NoOpCDN(CDNBackend):
    """No-op CDN for local development.

    Returns direct file URLs and logs invalidation requests without
    making any network calls.

    Args:
        base_url: Base URL prefix for :meth:`public_url`.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        self.invalidations: list[list[str]] = []

    def signed_url(self, path: str, *, expires_in_s: int = 3600) -> str:
        return self.public_url(path)

    def invalidate(self, paths: list[str]) -> str:
        inv_id = f"local-{uuid.uuid4().hex[:8]}"
        self.invalidations.append(paths)
        logger.info("Local CDN invalidation %s: %s", inv_id, paths)
        return inv_id

    def public_url(self, path: str) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{clean_path}"
