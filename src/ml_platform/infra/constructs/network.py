"""VPC networking construct for ML platform services.

Provisions a VPC with public and private subnets across two availability
zones. A single NAT gateway in one public subnet keeps costs down while
still allowing private-subnet resources (ECS tasks, SageMaker endpoints,
RDS instances) to reach the internet for image pulls and API calls.

Usage::

    from ml_platform.infra.constructs import NetworkConstruct

    net = NetworkConstruct(self, "Network")
    ecs = EcsServiceConstruct(self, "Svc", vpc=net.vpc, ...)
"""

from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class NetworkConstruct(Construct):
    """VPC with public and private subnets suitable for ML services.

    The network layout is intentionally simple and cost-optimised for
    non-production or early-stage ML workloads:

    * **2 AZs** — sufficient for high availability without the cost of a
      third AZ.
    * **1 NAT gateway** — minimises cost; upgrade to one-per-AZ for
      production HA via ``nat_gateways``.
    * **Private subnets** — for ECS tasks, RDS, SageMaker VPC endpoints.
    * **Public subnets** — for ALBs and bastion hosts.

    Args:
        scope: CDK construct scope.
        construct_id: Logical ID within the construct tree.
        cidr: VPC CIDR block.  Defaults to ``"10.0.0.0/16"``.
        max_azs: Maximum number of availability zones.  Defaults to ``2``.
        nat_gateways: Number of NAT gateways.  Defaults to ``1``.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cidr: str = "10.0.0.0/16",
        max_azs: int = 2,
        nat_gateways: int = 1,
    ) -> None:
        super().__init__(scope, construct_id)

        self._vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(cidr),
            max_azs=max_azs,
            nat_gateways=nat_gateways,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

    @property
    def vpc(self) -> ec2.IVpc:
        """The provisioned VPC."""
        return self._vpc

    @property
    def private_subnets(self) -> list[ec2.ISubnet]:
        """Private subnets (one per AZ) with NAT egress."""
        return self._vpc.private_subnets

    @property
    def public_subnets(self) -> list[ec2.ISubnet]:
        """Public subnets (one per AZ) for internet-facing resources."""
        return self._vpc.public_subnets
