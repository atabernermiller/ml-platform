"""CloudFront CDN construct for serving static and user-uploaded content.

Provisions a CloudFront distribution with an S3 origin, Origin Access
Control, and optional custom domain.

Usage::

    from ml_platform.infra.constructs import CDNConstruct

    cdn = CDNConstruct(
        self, "CDN",
        bucket=assets_bucket,
        domain_names=["assets.example.com"],
    )
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_cloudfront as cf,
    aws_cloudfront_origins as origins,
    aws_s3 as s3,
)
from constructs import Construct


class CDNConstruct(Construct):
    """CloudFront distribution backed by an S3 bucket.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID.
        bucket: S3 bucket to serve as the origin. If ``None``, a new
            bucket is created.
        service_name: Service name for resource naming.
        domain_names: Optional custom domain names for the distribution.
        default_ttl: Default cache TTL.
        max_ttl: Maximum cache TTL.
        price_class: CloudFront price class.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        bucket: s3.IBucket | None = None,
        service_name: str = "",
        domain_names: list[str] | None = None,
        default_ttl: Duration = Duration.hours(24),
        max_ttl: Duration = Duration.days(365),
        price_class: cf.PriceClass = cf.PriceClass.PRICE_CLASS_100,
    ) -> None:
        super().__init__(scope, construct_id)

        if bucket is None:
            bucket = s3.Bucket(
                self,
                "OriginBucket",
                bucket_name=f"{service_name}-cdn-origin" if service_name else None,
                removal_policy=RemovalPolicy.RETAIN,
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            )

        self._bucket = bucket

        cache_policy = cf.CachePolicy(
            self,
            "CachePolicy",
            default_ttl=default_ttl,
            max_ttl=max_ttl,
            enable_accept_encoding_gzip=True,
            enable_accept_encoding_brotli=True,
        )

        distribution = cf.Distribution(
            self,
            "Distribution",
            default_behavior=cf.BehaviorOptions(
                origin=origins.S3BucketV2Origin(bucket),
                viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cache_policy,
                allowed_methods=cf.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
            ),
            domain_names=domain_names or [],
            price_class=price_class,
        )

        self._distribution = distribution
        self._domain_name = distribution.distribution_domain_name

        CfnOutput(
            self,
            "DistributionDomain",
            value=self._domain_name,
            description="CloudFront distribution domain name",
        )

        CfnOutput(
            self,
            "DistributionId",
            value=distribution.distribution_id,
            description="CloudFront distribution ID",
        )

    @property
    def distribution(self) -> cf.IDistribution:
        """The CloudFront distribution."""
        return self._distribution

    @property
    def domain_name(self) -> str:
        """The CloudFront distribution domain name."""
        return self._domain_name

    @property
    def bucket(self) -> s3.IBucket:
        """The S3 origin bucket."""
        return self._bucket
