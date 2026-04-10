"""Multi-region deployment constructs for high availability.

Provides CDK constructs for active-passive multi-region deployments
with Route 53 health-check-based failover.

Usage::

    from ml_platform.infra.constructs import MultiRegionConstruct

    mr = MultiRegionConstruct(
        self, "MultiRegion",
        service_name="inference-api",
        primary_endpoint="primary-alb.us-east-1.elb.amazonaws.com",
        secondary_endpoint="secondary-alb.us-west-2.elb.amazonaws.com",
        domain_name="api.example.com",
        hosted_zone_id="Z1234567890",
        hosted_zone_name="example.com",
    )
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    aws_route53 as route53,
)
from constructs import Construct


class MultiRegionConstruct(Construct):
    """Route 53 failover routing for multi-region deployments.

    Creates a Route 53 health check on the primary endpoint and
    configures failover DNS records so traffic shifts to the secondary
    region when the primary is unhealthy.

    When ``hosted_zone_id`` and ``domain_name`` are provided, the
    construct creates PRIMARY and SECONDARY failover CNAME records.
    Without them, only the health check is created (useful when DNS
    is managed externally).

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID.
        service_name: Service name for resource naming.
        primary_endpoint: FQDN (hostname) of the primary service.
        secondary_endpoint: FQDN (hostname) of the secondary service.
        domain_name: Custom domain for the failover record set.
        hosted_zone_id: Route 53 hosted zone ID.
        hosted_zone_name: Route 53 hosted zone domain name (e.g.
            ``"example.com"``).  Required when ``hosted_zone_id`` is set.
        health_check_path: HTTP path for health checks.
        failover_threshold: Number of consecutive failures before failover.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        service_name: str,
        primary_endpoint: str,
        secondary_endpoint: str,
        domain_name: str = "",
        hosted_zone_id: str = "",
        hosted_zone_name: str = "",
        health_check_path: str = "/health",
        failover_threshold: int = 3,
    ) -> None:
        super().__init__(scope, construct_id)

        self._primary_endpoint = primary_endpoint
        self._secondary_endpoint = secondary_endpoint
        self._service_name = service_name

        health_check = route53.CfnHealthCheck(
            self,
            "PrimaryHealthCheck",
            health_check_config=route53.CfnHealthCheck.HealthCheckConfigProperty(
                fully_qualified_domain_name=primary_endpoint,
                port=443,
                type="HTTPS",
                resource_path=health_check_path,
                failure_threshold=failover_threshold,
                request_interval=30,
            ),
        )
        self._health_check = health_check

        if hosted_zone_id and domain_name:
            zone = route53.HostedZone.from_hosted_zone_attributes(
                self,
                "Zone",
                hosted_zone_id=hosted_zone_id,
                zone_name=hosted_zone_name or domain_name,
            )

            route53.CfnRecordSet(
                self,
                "PrimaryRecord",
                hosted_zone_id=hosted_zone_id,
                name=domain_name,
                type="CNAME",
                ttl="60",
                set_identifier=f"{service_name}-primary",
                failover="PRIMARY",
                health_check_id=health_check.attr_health_check_id,
                resource_records=[primary_endpoint],
            )

            route53.CfnRecordSet(
                self,
                "SecondaryRecord",
                hosted_zone_id=hosted_zone_id,
                name=domain_name,
                type="CNAME",
                ttl="60",
                set_identifier=f"{service_name}-secondary",
                failover="SECONDARY",
                resource_records=[secondary_endpoint],
            )

        CfnOutput(
            self,
            "PrimaryEndpoint",
            value=primary_endpoint,
            description=f"Primary endpoint for {service_name}",
        )
        CfnOutput(
            self,
            "SecondaryEndpoint",
            value=secondary_endpoint,
            description=f"Secondary endpoint for {service_name}",
        )
        CfnOutput(
            self,
            "HealthCheckId",
            value=health_check.attr_health_check_id,
            description=f"Route 53 health check ID for {service_name}",
        )

    @property
    def health_check_id(self) -> str:
        """Route 53 health check ID."""
        return self._health_check.attr_health_check_id
