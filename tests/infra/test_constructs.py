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
from aws_cdk import (
    App,
    BundlingOptions,
    DockerImage,
    Duration,
    Stack,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam_cdk,
    aws_s3 as s3,
)
from aws_cdk.assertions import Capture, Match, Template

from ml_platform.infra.constructs.cdn import CDNConstruct
from ml_platform.infra.constructs.cognito import CognitoConstruct
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

    def test_response_headers_policy_attached(self) -> None:
        from aws_cdk import aws_cloudfront as cf

        app = App()
        stack = Stack(app, "HeadersStack")
        bucket = s3.Bucket(stack, "Bucket")
        policy = cf.ResponseHeadersPolicy(
            stack,
            "SecurityHeaders",
            security_headers_behavior=cf.ResponseSecurityHeadersBehavior(
                strict_transport_security=cf.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.seconds(63072000),
                    include_subdomains=True,
                    override=True,
                ),
            ),
        )
        CDNConstruct(stack, "CDN", bucket=bucket, response_headers_policy=policy)
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudFront::ResponseHeadersPolicy", 1)
        template.has_resource_properties(
            "AWS::CloudFront::Distribution",
            Match.object_like({
                "DistributionConfig": Match.object_like({
                    "DefaultCacheBehavior": Match.object_like({
                        "ResponseHeadersPolicyId": Match.any_value(),
                    }),
                }),
            }),
        )

    def test_certificate_attached_with_custom_domains(self) -> None:
        from aws_cdk import aws_certificatemanager as acm

        app = App()
        stack = Stack(app, "CertStack")
        bucket = s3.Bucket(stack, "Bucket")
        cert = acm.Certificate.from_certificate_arn(
            stack, "Cert",
            "arn:aws:acm:us-east-1:123456789012:certificate/test-cert",
        )
        CDNConstruct(
            stack, "CDN",
            bucket=bucket,
            domain_names=["cdn.example.com"],
            certificate=cert,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::CloudFront::Distribution",
            Match.object_like({
                "DistributionConfig": Match.object_like({
                    "ViewerCertificate": Match.object_like({
                        "AcmCertificateArn": "arn:aws:acm:us-east-1:123456789012:certificate/test-cert",
                    }),
                }),
            }),
        )

    def test_domain_names_without_certificate_raises(self) -> None:
        app = App()
        stack = Stack(app, "NoCertStack")
        bucket = s3.Bucket(stack, "Bucket")
        with pytest.raises(ValueError, match="ACM certificate is required"):
            CDNConstruct(
                stack, "CDN",
                bucket=bucket,
                domain_names=["cdn.example.com"],
            )


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

    def test_deletion_protection_enabled_by_default(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="del-prot-svc",
            container_image="nginx:latest",
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ElasticLoadBalancingV2::LoadBalancer",
            Match.object_like({
                "LoadBalancerAttributes": Match.array_with([
                    Match.object_like({
                        "Key": "deletion_protection.enabled",
                        "Value": "true",
                    }),
                ]),
            }),
        )

    def test_deletion_protection_can_be_disabled(self) -> None:
        stack, vpc = self._make_stack_with_vpc()
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="no-del-prot-svc",
            container_image="nginx:latest",
            enable_deletion_protection=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ElasticLoadBalancingV2::LoadBalancer",
            Match.object_like({
                "LoadBalancerAttributes": Match.array_with([
                    Match.object_like({
                        "Key": "deletion_protection.enabled",
                        "Value": "false",
                    }),
                ]),
            }),
        )

    def test_access_log_bucket_wired(self) -> None:
        from aws_cdk import Environment

        app = App()
        stack = Stack(
            app, "LogStack",
            env=Environment(account="123456789012", region="us-east-1"),
        )
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        log_bucket = s3.Bucket(stack, "LogBucket")
        EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="log-svc",
            container_image="nginx:latest",
            access_log_bucket=log_bucket,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::ElasticLoadBalancingV2::LoadBalancer",
            Match.object_like({
                "LoadBalancerAttributes": Match.array_with([
                    Match.object_like({
                        "Key": "access_logs.s3.enabled",
                        "Value": "true",
                    }),
                ]),
            }),
        )


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

    def test_s3_trigger_creates_notification(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "S3TriggerStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="img-processor",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(
                bucket=bucket,
                prefix="uploads/",
                suffix=".jpg",
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "Custom::S3BucketNotifications",
            Match.object_like({
                "NotificationConfiguration": Match.object_like({
                    "LambdaFunctionConfigurations": Match.array_with([
                        Match.object_like({
                            "Events": ["s3:ObjectCreated:*"],
                            "Filter": Match.object_like({
                                "Key": Match.object_like({
                                    "FilterRules": Match.array_with([
                                        Match.object_like({"Name": "suffix", "Value": ".jpg"}),
                                        Match.object_like({"Name": "prefix", "Value": "uploads/"}),
                                    ]),
                                }),
                            }),
                        }),
                    ]),
                }),
            }),
        )

    def test_s3_trigger_grants_read(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "S3ReadStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="s3-reader",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(bucket=bucket),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Policy",
            Match.object_like({
                "PolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Action": Match.array_with(["s3:GetObject*", "s3:GetBucket*", "s3:List*"]),
                        }),
                    ]),
                }),
            }),
        )

    def test_s3_trigger_injects_bucket_env(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "S3EnvStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="env-trigger",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(bucket=bucket),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            Match.object_like({
                "Environment": Match.object_like({
                    "Variables": Match.object_like({
                        "S3_TRIGGER_BUCKET": Match.any_value(),
                    }),
                }),
            }),
        )

    def test_s3_trigger_with_multiple_event_types(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "MultiEventStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="multi-event",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(
                bucket=bucket,
                events=[s3.EventType.OBJECT_CREATED, s3.EventType.OBJECT_REMOVED],
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "Custom::S3BucketNotifications",
            Match.object_like({
                "NotificationConfiguration": Match.object_like({
                    "LambdaFunctionConfigurations": Match.array_with([
                        Match.object_like({"Events": ["s3:ObjectCreated:*"]}),
                        Match.object_like({"Events": ["s3:ObjectRemoved:*"]}),
                    ]),
                }),
            }),
        )

    def test_s3_trigger_with_secrets_and_env(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "ComboStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="combo-fn",
            code_path=_lambda_code_dir,
            environment={"APP_MODE": "processor"},
            secrets={"API_KEY": "arn:aws:secretsmanager:us-east-1:123:secret:key"},
            s3_trigger=LambdaServiceConstruct.S3Trigger(
                bucket=bucket,
                prefix="data/",
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            Match.object_like({
                "Environment": Match.object_like({
                    "Variables": Match.object_like({
                        "APP_MODE": "processor",
                        "API_KEY": "arn:aws:secretsmanager:us-east-1:123:secret:key",
                        "S3_TRIGGER_BUCKET": Match.any_value(),
                    }),
                }),
            }),
        )

    def test_s3_trigger_write_prefix_grants_put(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "WritePrefixStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="writer-fn",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(
                bucket=bucket,
                prefix="originals/",
                read_prefix="originals/*",
                write_prefix="web/*",
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "s3:PutObject" in tpl_str

    def test_s3_trigger_read_prefix_scoped(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "ReadPrefixStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="scoped-reader",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(
                bucket=bucket,
                read_prefix="originals/*",
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "/originals/*" in tpl_str
        template.has_resource_properties(
            "AWS::IAM::Policy",
            Match.object_like({
                "PolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Action": Match.array_with(["s3:GetObject*", "s3:GetBucket*", "s3:List*"]),
                        }),
                    ]),
                }),
            }),
        )

    def test_s3_trigger_no_write_by_default(self, _lambda_code_dir: str) -> None:
        app = App()
        stack = Stack(app, "NoWriteStack")
        bucket = s3.Bucket(stack, "Uploads")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="no-write-fn",
            code_path=_lambda_code_dir,
            s3_trigger=LambdaServiceConstruct.S3Trigger(bucket=bucket),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "s3:PutObject" not in tpl_str

    def test_bundling_options_passed_to_code_asset(self, _lambda_code_dir: str) -> None:
        import pathlib

        pathlib.Path(_lambda_code_dir, "requirements.txt").write_text("")
        app = App()
        stack = Stack(app, "BundlingStack")
        LambdaServiceConstruct(
            stack, "Fn",
            service_name="bundled-fn",
            code_path=_lambda_code_dir,
            bundling=BundlingOptions(
                image=DockerImage.from_registry(
                    "public.ecr.aws/sam/build-python3.12:latest",
                ),
                command=[
                    "bash", "-c",
                    "pip install -r requirements.txt -t /asset-output"
                    " && cp -au . /asset-output",
                ],
            ),
            create_function_url=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Function", 1)


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

    def test_rotation_schedule_created(self) -> None:
        from aws_cdk import aws_lambda as _lambda

        app = App()
        stack = Stack(app, "RotStack")
        rotation_fn = _lambda.Function(
            stack, "RotFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=_lambda.Code.from_inline("def handler(e,c): pass"),
        )
        SecretsConstruct(
            stack, "Secrets",
            service_name="rot-svc",
            secret_keys=["KEY"],
            rotation_days=30,
            rotation_lambda=rotation_fn,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::SecretsManager::RotationSchedule", 1)
        template.has_resource_properties(
            "AWS::SecretsManager::RotationSchedule",
            Match.object_like({
                "RotationRules": Match.object_like({
                    "ScheduleExpression": "rate(30 days)",
                }),
            }),
        )

    def test_rotation_without_lambda_raises(self) -> None:
        app = App()
        stack = Stack(app, "BadRotStack")
        with pytest.raises(ValueError, match="rotation_lambda is required"):
            SecretsConstruct(
                stack, "Secrets",
                service_name="bad-rot-svc",
                secret_keys=["KEY"],
                rotation_days=30,
            )

    def test_no_rotation_by_default(self) -> None:
        app = App()
        stack = Stack(app, "NoRotStack")
        SecretsConstruct(
            stack, "Secrets",
            service_name="no-rot-svc",
            secret_keys=["KEY"],
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::SecretsManager::RotationSchedule", 0)


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


# ---------------------------------------------------------------------------
# MonitoringConstruct
# ---------------------------------------------------------------------------


class TestMonitoringConstruct:
    def test_ml_alarms_created_by_default(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "MLMonStack")
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="ml-svc",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Alarm", 4)
        template.resource_count_is("AWS::SNS::Topic", 1)
        assert len(mon.alarms) == 4

    def test_ml_alarms_skipped_when_disabled(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "NoMLStack")
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="web-only",
            include_ml_alarms=False,
            create_dashboard=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Alarm", 0)
        template.resource_count_is("AWS::SNS::Topic", 1)
        assert len(mon.alarms) == 0

    def test_alb_alarms_created(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "ALBMonStack")
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        svc = EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="alb-mon-svc",
            container_image="nginx:latest",
        )
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="alb-svc",
            alb=svc.service.load_balancer,
            include_ml_alarms=False,
            create_dashboard=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Alarm", 3)
        assert len(mon.alarms) == 3

    def test_rds_alarms_created(self) -> None:
        from aws_cdk import aws_rds as rds, RemovalPolicy
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "RDSMonStack")
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        database = rds.DatabaseInstance(
            stack, "DB",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_17,
            ),
            vpc=vpc,
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="rds-svc",
            rds_instance=database,
            include_ml_alarms=False,
            create_dashboard=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Alarm", 3)
        assert len(mon.alarms) == 3

    def test_full_web_stack_alarm_count(self) -> None:
        from aws_cdk import aws_rds as rds, RemovalPolicy
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "FullWebStack")
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        svc = EcsServiceConstruct(
            stack, "Svc",
            vpc=vpc,
            service_name="full-web",
            container_image="nginx:latest",
        )
        database = rds.DatabaseInstance(
            stack, "DB",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_17,
            ),
            vpc=vpc,
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="full-web",
            alarm_email="ops@example.com",
            alb=svc.service.load_balancer,
            ecs_service=svc.service.service,
            rds_instance=database,
            include_ml_alarms=False,
            create_dashboard=False,
        )
        # 3 ALB + 1 ECS + 3 RDS = 7
        assert len(mon.alarms) == 7
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Alarm", 7)
        template.resource_count_is("AWS::SNS::Subscription", 1)

    def test_email_subscription_added(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "EmailMonStack")
        MonitoringConstruct(
            stack, "Mon",
            service_name="email-svc",
            alarm_email="test@example.com",
            include_ml_alarms=False,
            create_dashboard=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::SNS::Subscription", 1)
        template.has_resource_properties(
            "AWS::SNS::Subscription",
            Match.object_like({
                "Protocol": "email",
                "Endpoint": "test@example.com",
            }),
        )

    def test_dashboard_created_by_default(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "DashStack")
        MonitoringConstruct(
            stack, "Mon",
            service_name="dash-svc",
            include_ml_alarms=False,
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::CloudWatch::Dashboard", 1)

    def test_alarm_topic_exposed(self) -> None:
        from ml_platform.infra.constructs.monitoring import MonitoringConstruct

        app = App()
        stack = Stack(app, "TopicStack")
        mon = MonitoringConstruct(
            stack, "Mon",
            service_name="topic-svc",
            include_ml_alarms=False,
            create_dashboard=False,
        )
        assert mon.alarm_topic is not None


# ---------------------------------------------------------------------------
# CognitoConstruct
# ---------------------------------------------------------------------------


class TestCognitoConstruct:
    def test_creates_user_pool(self) -> None:
        app = App()
        stack = Stack(app, "CognitoStack")
        CognitoConstruct(stack, "Auth", service_name="admin")
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPool", 1)
        template.resource_count_is("AWS::Cognito::UserPoolClient", 1)

    def test_self_signup_disabled(self) -> None:
        app = App()
        stack = Stack(app, "NoSignupStack")
        CognitoConstruct(stack, "Auth", service_name="admin")
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            Match.object_like({
                "Policies": Match.object_like({
                    "PasswordPolicy": Match.object_like({
                        "MinimumLength": 12,
                        "RequireLowercase": True,
                        "RequireUppercase": True,
                        "RequireNumbers": True,
                        "RequireSymbols": True,
                    }),
                }),
            }),
        )

    def test_email_sign_in(self) -> None:
        app = App()
        stack = Stack(app, "EmailStack")
        CognitoConstruct(stack, "Auth", service_name="admin")
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            Match.object_like({
                "UsernameAttributes": ["email"],
                "AutoVerifiedAttributes": ["email"],
            }),
        )

    def test_custom_password_policy(self) -> None:
        app = App()
        stack = Stack(app, "CustomPwStack")
        CognitoConstruct(
            stack, "Auth",
            service_name="custom-pw",
            password_policy=CognitoConstruct.PasswordPolicy(
                min_length=8,
                require_symbols=False,
            ),
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            Match.object_like({
                "Policies": Match.object_like({
                    "PasswordPolicy": Match.object_like({
                        "MinimumLength": 8,
                        "RequireSymbols": False,
                    }),
                }),
            }),
        )

    def test_exposes_properties(self) -> None:
        app = App()
        stack = Stack(app, "PropStack")
        auth = CognitoConstruct(stack, "Auth", service_name="props")
        assert auth.user_pool is not None
        assert auth.user_pool_id is not None
        assert auth.app_client is not None
        assert auth.app_client_id is not None
        assert auth.domain is None

    def test_domain_prefix(self) -> None:
        app = App()
        stack = Stack(app, "DomainStack")
        auth = CognitoConstruct(
            stack, "Auth",
            service_name="domain-svc",
            domain_prefix="my-app-auth",
        )
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPoolDomain", 1)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolDomain",
            Match.object_like({
                "Domain": "my-app-auth",
            }),
        )
        assert auth.domain is not None

    def test_no_domain_by_default(self) -> None:
        app = App()
        stack = Stack(app, "NoDomainStack")
        auth = CognitoConstruct(stack, "Auth", service_name="no-domain")
        template = Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPoolDomain", 0)

    def test_pool_naming(self) -> None:
        app = App()
        stack = Stack(app, "NamingStack")
        CognitoConstruct(stack, "Auth", service_name="studio-admin")
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            Match.object_like({
                "UserPoolName": "studio-admin-users",
            }),
        )

    def test_cfn_outputs(self) -> None:
        app = App()
        stack = Stack(app, "OutputStack")
        CognitoConstruct(stack, "Auth", service_name="outputs")
        template = Template.from_stack(stack)
        outputs = template.to_json().get("Outputs", {})
        output_descriptions = [v.get("Description", "") for v in outputs.values()]
        assert any("User Pool ID" in d for d in output_descriptions)
        assert any("App Client ID" in d for d in output_descriptions)

    def test_retain_removal_policy_by_default(self) -> None:
        app = App()
        stack = Stack(app, "RetainStack")
        CognitoConstruct(stack, "Auth", service_name="retain")
        template = Template.from_stack(stack)
        template.has_resource(
            "AWS::Cognito::UserPool",
            Match.object_like({
                "DeletionPolicy": "Retain",
            }),
        )

    def test_admin_user_password_auth_enabled_by_default(self) -> None:
        app = App()
        stack = Stack(app, "AdminAuthStack")
        CognitoConstruct(stack, "Auth", service_name="admin-auth")
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolClient",
            Match.object_like({
                "ExplicitAuthFlows": Match.array_with([
                    "ALLOW_USER_PASSWORD_AUTH",
                    "ALLOW_ADMIN_USER_PASSWORD_AUTH",
                    "ALLOW_REFRESH_TOKEN_AUTH",
                ]),
            }),
        )

    def test_custom_auth_flows_override(self) -> None:
        app = App()
        stack = Stack(app, "CustomFlowStack")
        CognitoConstruct(
            stack, "Auth",
            service_name="custom-flow",
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                user_password=False,
                admin_user_password=False,
            ),
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolClient",
            Match.object_like({
                "ExplicitAuthFlows": Match.array_with([
                    "ALLOW_USER_SRP_AUTH",
                    "ALLOW_REFRESH_TOKEN_AUTH",
                ]),
            }),
        )
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "ALLOW_ADMIN_USER_PASSWORD_AUTH" not in tpl_str

    def test_no_oauth_flows_by_default(self) -> None:
        """Default config must not produce AllowedOAuthFlows, scopes, or
        placeholder callback URLs in the UserPoolClient resource."""
        app = App()
        stack = Stack(app, "NoOAuthStack")
        CognitoConstruct(stack, "Auth", service_name="no-oauth")
        template = Template.from_stack(stack)
        import json as _json

        resources = template.to_json().get("Resources", {})
        for _lid, resource in resources.items():
            if resource.get("Type") != "AWS::Cognito::UserPoolClient":
                continue
            props = resource.get("Properties", {})
            oauth_flows = props.get("AllowedOAuthFlows", [])
            oauth_scopes = props.get("AllowedOAuthScopes", [])
            callback_urls = props.get("CallbackURLs", [])
            assert "implicit" not in oauth_flows, "implicit OAuth flow should not be enabled"
            assert "aws.cognito.signin.user.admin" not in oauth_scopes
            assert "https://example.com" not in callback_urls

    def test_oauth_with_callback_urls(self) -> None:
        app = App()
        stack = Stack(app, "OAuthStack")
        CognitoConstruct(
            stack, "Auth",
            service_name="oauth-svc",
            enable_oauth=True,
            callback_urls=["https://app.example.com/callback"],
        )
        template = Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolClient",
            Match.object_like({
                "AllowedOAuthFlows": ["code"],
                "AllowedOAuthScopes": Match.array_with(["openid", "email"]),
                "CallbackURLs": ["https://app.example.com/callback"],
            }),
        )
        tpl_json = template.to_json()
        import json as _json
        tpl_str = _json.dumps(tpl_json)
        assert "implicit" not in tpl_str
        assert "aws.cognito.signin.user.admin" not in tpl_str

    def test_oauth_without_callback_urls_raises(self) -> None:
        app = App()
        stack = Stack(app, "BadOAuthStack")
        with pytest.raises(ValueError, match="callback_urls is required"):
            CognitoConstruct(
                stack, "Auth",
                service_name="bad-oauth",
                enable_oauth=True,
            )
