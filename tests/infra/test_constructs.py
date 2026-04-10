"""Tests for CDK constructs using cdk assertions (template-level checks).

Each test synthesises a minimal stack, then uses ``aws_cdk.assertions``
to verify the CloudFormation template contains the expected resources
and properties.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest
from aws_cdk import App, Duration, Stack, aws_ec2 as ec2, aws_ecs as ecs, aws_s3 as s3
from aws_cdk.assertions import Match, Template

from ml_platform.infra.constructs.cdn import CDNConstruct
from ml_platform.infra.constructs.ecs_service import EcsServiceConstruct
from ml_platform.infra.constructs.lambda_function import LambdaServiceConstruct
from ml_platform.infra.constructs.multi_region import MultiRegionConstruct
from ml_platform.infra.constructs.network import NetworkConstruct
from ml_platform.infra.constructs.sagemaker import SageMakerEndpointConstruct
from ml_platform.infra.constructs.secrets import SecretsConstruct


@pytest.fixture()
def _lambda_code_dir(tmp_path: Any) -> str:
    """Create a minimal Lambda code directory for CDK asset bundling."""
    code_dir = str(tmp_path / "lambda_code")
    os.makedirs(code_dir, exist_ok=True)
    (tmp_path / "lambda_code" / "app.py").write_text("def handler(event, ctx): pass")
    return code_dir


# ---------------------------------------------------------------------------
# CDNConstruct
# ---------------------------------------------------------------------------


class TestCDNConstruct:
    def test_creates_cloudfront_distribution(self) -> None:
        app = App()
        stack = Stack(app, "CdnStack")
        bucket = s3.Bucket(stack, "Bucket")
        CDNConstruct(stack, "CDN", bucket=bucket, service_name="test")
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudFront::Distribution", 1)

    def test_uses_oac_not_oai(self) -> None:
        """Verify the fix: S3BucketOrigin.with_origin_access_control creates
        an OAC resource, not an OAI."""
        app = App()
        stack = Stack(app, "OacStack")
        bucket = s3.Bucket(stack, "Bucket")
        CDNConstruct(stack, "CDN", bucket=bucket)
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)

    def test_creates_own_bucket_when_none_provided(self) -> None:
        app = App()
        stack = Stack(app, "NoBucketStack")
        cdn = CDNConstruct(stack, "CDN", service_name="assets")
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::S3::Bucket", 1)
        assert cdn.bucket is not None

    def test_exposes_properties(self) -> None:
        app = App()
        stack = Stack(app, "PropStack")
        bucket = s3.Bucket(stack, "Bucket")
        cdn = CDNConstruct(stack, "CDN", bucket=bucket)
        assert cdn.distribution is not None
        assert cdn.domain_name is not None
        assert cdn.bucket is bucket


# ---------------------------------------------------------------------------
# EcsServiceConstruct
# ---------------------------------------------------------------------------


class TestEcsServiceConstruct:
    @staticmethod
    def _make_stack_with_vpc() -> tuple[Stack, ec2.Vpc]:
        app = App()
        stack = Stack(app, "EcsStack")
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        return stack, vpc

    def test_creates_ecs_resources(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="test-svc",
            container_image="nginx:latest",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::ECS::Cluster", 1)
        template.resource_count_is("AWS::ECS::TaskDefinition", 1)
        template.resource_count_is("AWS::ECS::Service", 1)

    def test_secrets_injected_into_container(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        from aws_cdk import aws_secretsmanager as sm

        secret = sm.Secret(stack, "DbSecret")
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="secrets-svc",
            container_image="nginx:latest",
            secrets={"DB_PASSWORD": ecs.Secret.from_secrets_manager(secret)},
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ECS::TaskDefinition",
            {
                "ContainerDefinitions": Match.array_with([
                    Match.object_like({
                        "Secrets": Match.array_with([
                            Match.object_like({"Name": "DB_PASSWORD"}),
                        ]),
                    }),
                ]),
            },
        )

    def test_exposes_task_definition_service_cluster(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        svc = EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="prop-svc",
            container_image="nginx:latest",
        )
        assert svc.task_definition is not None
        assert svc.service is not None
        assert svc.cluster is not None
        assert svc.service_url.startswith("http")

    def test_https_creates_certificate_reference(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="https-svc",
            container_image="nginx:latest",
            certificate_arn="arn:aws:acm:us-east-1:123456789012:certificate/test-cert",
        )
        template = Template.from_stack(stack)
        # HTTPS listener should exist on port 443
        template.has_resource_properties(
            "AWS::ElasticLoadBalancingV2::Listener",
            Match.object_like({
                "Port": 443,
                "Protocol": "HTTPS",
            }),
        )

    def test_http_mode_listener_on_80(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="http-svc",
            container_image="nginx:latest",
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ElasticLoadBalancingV2::Listener",
            Match.object_like({
                "Port": 80,
                "Protocol": "HTTP",
            }),
        )

    def test_environment_vars_passed(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="env-svc",
            container_image="nginx:latest",
            environment={"MODEL_NAME": "bert-base"},
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ECS::TaskDefinition",
            {
                "ContainerDefinitions": Match.array_with([
                    Match.object_like({
                        "Environment": Match.array_with([
                            Match.object_like({
                                "Name": "MODEL_NAME",
                                "Value": "bert-base",
                            }),
                        ]),
                    }),
                ]),
            },
        )

    def test_custom_domain_in_service_url(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        svc = EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="domain-svc",
            container_image="nginx:latest",
            certificate_arn="arn:aws:acm:us-east-1:123456789012:certificate/x",
            domain_name="api.example.com",
        )
        assert svc.service_url == "https://api.example.com"


# ---------------------------------------------------------------------------
# LambdaServiceConstruct
# ---------------------------------------------------------------------------


class TestLambdaServiceConstruct:
    def test_creates_lambda_function(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "LambdaStack")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="test-fn",
            code_path=_lambda_code_dir,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Function", 1)

    def test_secrets_create_iam_policy(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "SecretLambdaStack")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="secret-fn",
            code_path=_lambda_code_dir,
            secrets={
                "DB_PASSWORD": "arn:aws:secretsmanager:us-east-1:123456789012:secret:db-pw",
            },
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Policy",
            Match.object_like({
                "PolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Action": "secretsmanager:GetSecretValue",
                        }),
                    ]),
                }),
            }),
        )

    def test_secrets_injected_as_env_vars(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "EnvLambdaStack")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="env-fn",
            code_path=_lambda_code_dir,
            environment={"APP_NAME": "test"},
            secrets={"SECRET_KEY": "arn:aws:secretsmanager:us-east-1:123:secret:key"},
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            Match.object_like({
                "Environment": Match.object_like({
                    "Variables": Match.object_like({
                        "APP_NAME": "test",
                        "SECRET_KEY": "arn:aws:secretsmanager:us-east-1:123:secret:key",
                    }),
                }),
            }),
        )

    def test_function_url_created_by_default(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "UrlStack")
        fn = LambdaServiceConstruct(
            stack, "Fn",
            service_name="url-fn",
            code_path=_lambda_code_dir,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Url", 1)

    def test_no_function_url_when_disabled(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "NoUrlStack")
        fn = LambdaServiceConstruct(
            stack, "Fn",
            service_name="nourl-fn",
            code_path=_lambda_code_dir,
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Url", 0)
        assert fn.function_url == ""


# ---------------------------------------------------------------------------
# NetworkConstruct
# ---------------------------------------------------------------------------


class TestNetworkConstruct:
    def test_creates_vpc(self) -> None:
        app = App()
        stack = Stack(app, "NetStack")
        NetworkConstruct(stack, "Net")
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::EC2::VPC", 1)

    def test_exposes_subnets(self) -> None:
        app = App()
        stack = Stack(app, "SubnetStack")
        net = NetworkConstruct(stack, "Net", max_azs=2)
        assert len(net.public_subnets) == 2
        assert len(net.private_subnets) == 2

    def test_custom_cidr(self) -> None:
        app = App()
        stack = Stack(app, "CidrStack")
        NetworkConstruct(stack, "Net", cidr="172.16.0.0/16")
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::EC2::VPC",
            {"CidrBlock": "172.16.0.0/16"},
        )


# ---------------------------------------------------------------------------
# SecretsConstruct
# ---------------------------------------------------------------------------


class TestSecretsConstruct:
    def test_creates_secret(self) -> None:
        app = App()
        stack = Stack(app, "SecStack")
        SecretsConstruct(
            stack, "Secrets",
            service_name="my-svc",
            secret_keys=["DB_HOST", "DB_PASSWORD"],
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::SecretsManager::Secret", 1)

    def test_secret_template_contains_keys(self) -> None:
        app = App()
        stack = Stack(app, "TplStack")
        SecretsConstruct(
            stack, "Secrets",
            service_name="tpl-svc",
            secret_keys=["API_KEY", "DB_URL"],
            initial_values={"DB_URL": "postgres://localhost/db"},
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SecretsManager::Secret",
            Match.object_like({
                "GenerateSecretString": Match.object_like({
                    "SecretStringTemplate": Match.string_like_regexp("API_KEY"),
                }),
            }),
        )

    def test_exposes_properties(self) -> None:
        app = App()
        stack = Stack(app, "PropSecStack")
        sc = SecretsConstruct(
            stack, "Secrets",
            service_name="prop-svc",
            secret_keys=["KEY"],
        )
        assert sc.secret is not None
        assert sc.secret_arn is not None
        assert sc.secret_name is not None

    def test_grant_read(self) -> None:
        app = App()
        stack = Stack(app, "GrantStack")
        from aws_cdk import aws_iam as iam

        role = iam.Role(
            stack, "Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        sc = SecretsConstruct(
            stack, "Secrets",
            service_name="grant-svc",
            secret_keys=["TOKEN"],
        )
        sc.grant_read(role)
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Policy",
            Match.object_like({
                "PolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Action": Match.array_with([
                                "secretsmanager:GetSecretValue",
                            ]),
                        }),
                    ]),
                }),
            }),
        )


# ---------------------------------------------------------------------------
# MultiRegionConstruct
# ---------------------------------------------------------------------------


class TestMultiRegionConstruct:
    def test_creates_health_check(self) -> None:
        app = App()
        stack = Stack(app, "MRStack")
        MultiRegionConstruct(
            stack, "MR",
            service_name="test-svc",
            primary_endpoint="primary.example.com",
            secondary_endpoint="secondary.example.com",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Route53::HealthCheck", 1)

    def test_failover_records_created_with_zone(self) -> None:
        app = App()
        stack = Stack(app, "MRZoneStack")
        MultiRegionConstruct(
            stack, "MR",
            service_name="zone-svc",
            primary_endpoint="primary.example.com",
            secondary_endpoint="secondary.example.com",
            domain_name="api.example.com",
            hosted_zone_id="Z12345",
            hosted_zone_name="example.com",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Route53::RecordSet", 2)

    def test_no_records_without_zone(self) -> None:
        app = App()
        stack = Stack(app, "MRNoZoneStack")
        MultiRegionConstruct(
            stack, "MR",
            service_name="nozone-svc",
            primary_endpoint="primary.example.com",
            secondary_endpoint="secondary.example.com",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Route53::RecordSet", 0)

    def test_exposes_health_check_id(self) -> None:
        app = App()
        stack = Stack(app, "MRPropStack")
        mr = MultiRegionConstruct(
            stack, "MR",
            service_name="prop-svc",
            primary_endpoint="primary.example.com",
            secondary_endpoint="secondary.example.com",
        )
        assert mr.health_check_id is not None


# ---------------------------------------------------------------------------
# SageMakerEndpointConstruct (IAM scoping)
# ---------------------------------------------------------------------------


class TestSageMakerEndpointConstruct:
    def test_no_full_access_policy(self) -> None:
        """Verify AmazonSageMakerFullAccess is NOT attached."""
        app = App()
        stack = Stack(app, "SMStack")
        SageMakerEndpointConstruct(
            stack, "Ep",
            endpoint_name="test-ep",
            model_data_url="s3://my-bucket/model.tar.gz",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-img:latest",
        )
        template = Template.from_stack(stack)
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "AmazonSageMakerFullAccess" not in tpl_str

    def test_creates_sagemaker_endpoint(self) -> None:
        app = App()
        stack = Stack(app, "SMEpStack")
        ep = SageMakerEndpointConstruct(
            stack, "Ep",
            endpoint_name="my-ep",
            model_data_url="s3://bucket/model.tar.gz",
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/img:1",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::SageMaker::Model", 1)
        template.resource_count_is("AWS::SageMaker::EndpointConfig", 1)
        template.resource_count_is("AWS::SageMaker::Endpoint", 1)
        assert ep.endpoint_name == "my-ep"
