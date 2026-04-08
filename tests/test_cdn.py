"""Tests for CDN abstraction."""

from __future__ import annotations

from ml_platform.cdn import CloudFrontCDN, NoOpCDN


class TestNoOpCDN:
    def test_public_url(self) -> None:
        cdn = NoOpCDN(base_url="http://localhost:3000")
        assert cdn.public_url("/images/photo.jpg") == "http://localhost:3000/images/photo.jpg"

    def test_public_url_no_leading_slash(self) -> None:
        cdn = NoOpCDN(base_url="http://localhost:3000")
        assert cdn.public_url("images/photo.jpg") == "http://localhost:3000/images/photo.jpg"

    def test_signed_url_equals_public_url(self) -> None:
        cdn = NoOpCDN()
        assert cdn.signed_url("/test.txt") == cdn.public_url("/test.txt")

    def test_invalidate(self) -> None:
        cdn = NoOpCDN()
        inv_id = cdn.invalidate(["/images/*", "/css/*"])
        assert inv_id.startswith("local-")
        assert len(cdn.invalidations) == 1
        assert cdn.invalidations[0] == ["/images/*", "/css/*"]


class TestCloudFrontCDN:
    def test_public_url(self) -> None:
        cdn = CloudFrontCDN(domain="d123.cloudfront.net")
        assert cdn.public_url("/img/photo.jpg") == "https://d123.cloudfront.net/img/photo.jpg"

    def test_signed_url_has_expiry(self) -> None:
        cdn = CloudFrontCDN(domain="d123.cloudfront.net")
        url = cdn.signed_url("/file.pdf", expires_in_s=3600)
        assert "expires=" in url
        assert url.startswith("https://d123.cloudfront.net/file.pdf")

    def test_invalidate_without_distribution_id(self) -> None:
        cdn = CloudFrontCDN(domain="d123.cloudfront.net")
        result = cdn.invalidate(["/path/*"])
        assert result == "no-op"
