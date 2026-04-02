"""Tests for deploy plan generation and cost estimation.

These tests focus on the pure functions (plan display, cost estimation)
and do not call AWS.  Integration tests with moto are in test_deploy_aws.py.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from ml_platform.cli.deploy import _estimate_monthly_cost, _print_plan
from ml_platform.cli.manifest import FeaturesConfig, ProjectManifest, ScalingConfig


@pytest.fixture()
def chatbot_manifest() -> ProjectManifest:
    return ProjectManifest(
        service_name="my-chatbot",
        service_type="llm",
        features=FeaturesConfig(conversation_store=True),
        compute_size="medium",
        scaling=ScalingConfig(min_tasks=1, max_tasks=4),
    )


@pytest.fixture()
def bandit_manifest() -> ProjectManifest:
    return ProjectManifest(
        service_name="my-bandit",
        service_type="stateful",
        features=FeaturesConfig(context_store=True, checkpointing=True),
        compute_size="large",
        scaling=ScalingConfig(min_tasks=0, max_tasks=10),
    )


class TestCostEstimation:
    def test_chatbot_costs_positive(self, chatbot_manifest: ProjectManifest) -> None:
        costs = _estimate_monthly_cost(chatbot_manifest)
        assert all(v >= 0 for v in costs.values())
        assert costs["DynamoDB"] > 0
        assert costs["S3 (checkpoints)"] == 0

    def test_bandit_includes_s3(self, bandit_manifest: ProjectManifest) -> None:
        costs = _estimate_monthly_cost(bandit_manifest)
        assert costs["S3 (checkpoints)"] > 0
        assert costs["DynamoDB"] > 0

    def test_light_cheaper_than_peak(self, chatbot_manifest: ProjectManifest) -> None:
        costs = _estimate_monthly_cost(chatbot_manifest)
        assert costs["ECS Fargate (light)"] < costs["ECS Fargate (peak)"]

    def test_larger_compute_costs_more(self) -> None:
        small = ProjectManifest(service_name="s", compute_size="small")
        large = ProjectManifest(service_name="l", compute_size="xlarge")
        sc = _estimate_monthly_cost(small)
        lc = _estimate_monthly_cost(large)
        assert sc["ECS Fargate (light)"] < lc["ECS Fargate (light)"]

    def test_zero_min_tasks_still_estimates_one(self) -> None:
        m = ProjectManifest(
            service_name="test",
            scaling=ScalingConfig(min_tasks=0, max_tasks=4),
        )
        costs = _estimate_monthly_cost(m)
        assert costs["ECS Fargate (light)"] > 0


class TestPrintPlan:
    def test_contains_service_name(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "my-chatbot" in output

    def test_contains_cost_section(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "ESTIMATED MONTHLY COST" in output
        assert "/mo" in output

    def test_shows_storage_when_enabled(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "conversations" in output
        assert "STORAGE" in output

    def test_shows_skipped_when_disabled(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "Skipped" in output

    def test_contains_reliability_section(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "RELIABILITY" in output
        assert "Multi-AZ" in output

    def test_contains_scaling_info(self, chatbot_manifest: ProjectManifest) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_plan(chatbot_manifest)
        output = buf.getvalue()
        assert "Min tasks" in output
        assert "Max tasks" in output
        assert "Scale up" in output
