"""Tests for ml_platform.monitoring.dashboards -- composable dashboard generation."""

from __future__ import annotations

import pytest

from ml_platform.config import AgentConfig, LLMConfig, ServiceConfig, StatefulConfig
from ml_platform.monitoring.dashboards import generate_dashboard


class TestGenerateDashboard:
    def test_core_panels_always_present(self) -> None:
        config = ServiceConfig(service_name="svc")
        result = generate_dashboard(config, panel_sets=["core"])
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "Request Rate" in titles
        assert "Error Rate" in titles
        assert "Latency (ms)" in titles

    def test_agent_panels_included(self) -> None:
        config = ServiceConfig(
            service_name="agent-svc",
            agent=AgentConfig(),
        )
        result = generate_dashboard(config, panel_sets=["core", "agent"])
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "Steps per Run" in titles
        assert "Tool Usage Distribution" in titles
        assert "Multi-Model Cost Breakdown" in titles

    def test_auto_detect_agent(self) -> None:
        config = ServiceConfig(
            service_name="auto-agent",
            agent=AgentConfig(),
        )
        result = generate_dashboard(config)
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "Steps per Run" in titles
        assert "Tokens / min" in titles

    def test_auto_detect_stateful(self) -> None:
        config = ServiceConfig(
            service_name="stateful-svc",
            stateful=StatefulConfig(s3_checkpoint_bucket="b"),
        )
        result = generate_dashboard(config)
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "Cumulative Reward" in titles
        assert "Steps per Run" not in titles

    def test_core_only_omits_domain_panels(self) -> None:
        config = ServiceConfig(service_name="svc")
        result = generate_dashboard(config, panel_sets=["core"])
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "Steps per Run" not in titles
        assert "Cumulative Reward" not in titles
        assert "Tokens / min" not in titles

    def test_all_panel_sets_combinable(self) -> None:
        config = ServiceConfig(
            service_name="all",
            agent=AgentConfig(),
            llm=LLMConfig(),
        )
        result = generate_dashboard(
            config, panel_sets=["core", "llm", "agent"]
        )
        panels = result["dashboard"]["panels"]
        assert len(panels) > 5

    def test_dashboard_has_correct_title(self) -> None:
        config = ServiceConfig(service_name="my-svc")
        result = generate_dashboard(config, panel_sets=["core"])
        assert "my-svc" in result["dashboard"]["title"]

    def test_panels_are_valid_json_structure(self) -> None:
        config = ServiceConfig(service_name="svc", agent=AgentConfig())
        result = generate_dashboard(config, panel_sets=["core", "agent"])
        for panel in result["dashboard"]["panels"]:
            assert "title" in panel
            assert "type" in panel
            assert "gridPos" in panel
            assert "targets" in panel


class TestLegacyDashboards:
    def test_grafana_backward_compat(self) -> None:
        from ml_platform.monitoring.dashboards import generate_grafana_dashboard

        result = generate_grafana_dashboard("my-svc")
        assert "dashboard" in result
        panels = result["dashboard"]["panels"]
        assert len(panels) > 0

    def test_cloudwatch_backward_compat(self) -> None:
        from ml_platform.monitoring.dashboards import generate_cloudwatch_dashboard
        import json

        result = generate_cloudwatch_dashboard("my-svc")
        parsed = json.loads(result)
        assert "widgets" in parsed
