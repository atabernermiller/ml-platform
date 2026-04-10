"""Tests for CloudFormation template generation."""

from __future__ import annotations

import json

import pytest

from ml_platform.cli.cfn.template import generate_stack_template, stack_name
from ml_platform.cli.manifest import FeaturesConfig, ProjectManifest, ScalingConfig


@pytest.fixture()
def llm_manifest() -> ProjectManifest:
    return ProjectManifest(
        service_name="test-chatbot",
        service_type="llm",
        features=FeaturesConfig(conversation_store=True),
        compute_size="medium",
        scaling=ScalingConfig(min_tasks=1, max_tasks=4),
    )


@pytest.fixture()
def stateful_manifest() -> ProjectManifest:
    return ProjectManifest(
        service_name="test-bandit",
        service_type="stateful",
        features=FeaturesConfig(context_store=True, checkpointing=True),
    )


@pytest.fixture()
def minimal_manifest() -> ProjectManifest:
    return ProjectManifest(service_name="test-proxy", service_type="stateless")


class TestStackName:
    def test_format(self, llm_manifest: ProjectManifest) -> None:
        assert stack_name(llm_manifest) == "test-chatbot-stack"


class TestGenerateStackTemplate:
    def test_has_required_keys(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/test:latest")
        assert "AWSTemplateFormatVersion" in t
        assert "Resources" in t
        assert "Outputs" in t

    def test_includes_core_resources(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        r = t["Resources"]
        for key in ["TaskExecutionRole", "TaskRole", "LogGroup", "EcsCluster",
                     "TaskDefinition", "AlbSecurityGroup", "TaskSecurityGroup",
                     "Alb", "TargetGroup", "Listener", "EcsService",
                     "ScalableTarget", "ScaleUpPolicy", "Dashboard"]:
            assert key in r, f"Missing resource: {key}"

    def test_llm_includes_conversation_table(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        assert "ConversationTable" in t["Resources"]
        assert "ContextTable" not in t["Resources"]
        assert "CheckpointBucket" not in t["Resources"]

    def test_stateful_includes_context_and_s3(self, stateful_manifest: ProjectManifest) -> None:
        t = generate_stack_template(stateful_manifest, ecr_image_uri="img:latest")
        assert "ContextTable" in t["Resources"]
        assert "CheckpointBucket" in t["Resources"]
        assert "ConversationTable" not in t["Resources"]

    def test_minimal_no_optional_resources(self, minimal_manifest: ProjectManifest) -> None:
        t = generate_stack_template(minimal_manifest, ecr_image_uri="img:latest")
        assert "ConversationTable" not in t["Resources"]
        assert "ContextTable" not in t["Resources"]
        assert "CheckpointBucket" not in t["Resources"]

    def test_task_definition_uses_image_uri(self, llm_manifest: ProjectManifest) -> None:
        uri = "123456.dkr.ecr.us-east-1.amazonaws.com/test-chatbot:v1"
        t = generate_stack_template(llm_manifest, ecr_image_uri=uri)
        containers = t["Resources"]["TaskDefinition"]["Properties"]["ContainerDefinitions"]
        assert containers[0]["Image"] == uri

    def test_auto_scaling_matches_manifest(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        target = t["Resources"]["ScalableTarget"]["Properties"]
        assert target["MinCapacity"] == 1
        assert target["MaxCapacity"] == 4

    def test_dashboard_has_llm_widgets(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        body_raw = t["Resources"]["Dashboard"]["Properties"]["DashboardBody"]
        body = json.loads(body_raw["Fn::Sub"])
        titles = [w["properties"]["title"] for w in body["widgets"]]
        assert "Tokens / min" in titles
        assert "Cost (USD)" in titles

    def test_dashboard_stateful_no_llm_widgets(self, stateful_manifest: ProjectManifest) -> None:
        t = generate_stack_template(stateful_manifest, ecr_image_uri="img:latest")
        body_raw = t["Resources"]["Dashboard"]["Properties"]["DashboardBody"]
        body = json.loads(body_raw["Fn::Sub"])
        titles = [w["properties"]["title"] for w in body["widgets"]]
        assert "Tokens / min" not in titles

    def test_template_is_valid_json(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        serialized = json.dumps(t)
        assert json.loads(serialized) == t

    def test_context_table_uses_request_id_pk(
        self, stateful_manifest: ProjectManifest
    ) -> None:
        t = generate_stack_template(stateful_manifest, ecr_image_uri="img:latest")
        ctx_table = t["Resources"]["ContextTable"]
        key_schema = ctx_table["Properties"]["KeySchema"]
        assert key_schema[0]["AttributeName"] == "request_id"

    def test_conversation_table_uses_session_id_pk(
        self, llm_manifest: ProjectManifest
    ) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        conv_table = t["Resources"]["ConversationTable"]
        key_schema = conv_table["Properties"]["KeySchema"]
        assert key_schema[0]["AttributeName"] == "session_id"

    def test_logs_iam_scoped_to_log_group(
        self, llm_manifest: ProjectManifest
    ) -> None:
        t = generate_stack_template(llm_manifest, ecr_image_uri="img:latest")
        role = t["Resources"]["TaskRole"]
        statements = role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        logs_stmt = [s for s in statements if "logs:PutLogEvents" in s["Action"]]
        assert len(logs_stmt) == 1
        assert logs_stmt[0]["Resource"] != "*"

    def test_vpc_and_subnets_injected(self, llm_manifest: ProjectManifest) -> None:
        t = generate_stack_template(
            llm_manifest, ecr_image_uri="img:latest",
            vpc_id="vpc-abc123", subnet_ids=["subnet-1", "subnet-2"],
        )
        alb_sg = t["Resources"]["AlbSecurityGroup"]["Properties"]
        assert alb_sg["VpcId"] == "vpc-abc123"
        alb_props = t["Resources"]["Alb"]["Properties"]
        assert alb_props["Subnets"] == ["subnet-1", "subnet-2"]
