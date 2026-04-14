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


class TestWebPanelSet:
    def test_web_panels_with_all_resources(self) -> None:
        from ml_platform.monitoring.dashboards import WebResources

        config = ServiceConfig(service_name="web-app")
        resources = WebResources(
            alb_full_name="app/my-alb/abc123",
            ecs_cluster_name="my-cluster",
            ecs_service_name="my-service",
            rds_instance_id="my-db",
        )
        result = generate_dashboard(
            config, panel_sets=["web"], resources=resources,
        )
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "ALB Request Count" in titles
        assert "ALB 5xx / 4xx Errors" in titles
        assert "ALB Target Response Time" in titles
        assert "ECS CPU Utilisation" in titles
        assert "ECS Memory Utilisation" in titles
        assert "ECS Running Tasks" in titles
        assert "RDS CPU Utilisation" in titles
        assert "RDS Active Connections" in titles
        assert "RDS Free Storage (bytes)" in titles

    def test_web_panels_alb_only(self) -> None:
        from ml_platform.monitoring.dashboards import WebResources

        config = ServiceConfig(service_name="alb-only")
        resources = WebResources(alb_full_name="app/alb/123")
        result = generate_dashboard(
            config, panel_sets=["web"], resources=resources,
        )
        panels = result["dashboard"]["panels"]
        titles = {p["title"] for p in panels}
        assert "ALB Request Count" in titles
        assert "ECS CPU Utilisation" not in titles
        assert "RDS CPU Utilisation" not in titles

    def test_web_panels_no_resources_produces_empty(self) -> None:
        config = ServiceConfig(service_name="no-res")
        result = generate_dashboard(config, panel_sets=["web"])
        panels = result["dashboard"]["panels"]
        assert len(panels) == 0

    def test_web_skips_core_panels(self) -> None:
        from ml_platform.monitoring.dashboards import WebResources

        config = ServiceConfig(service_name="web-svc")
        resources = WebResources(alb_full_name="app/alb/123")
        result = generate_dashboard(
            config, panel_sets=["web"], resources=resources,
        )
        titles = {p["title"] for p in result["dashboard"]["panels"]}
        assert "Request Rate" not in titles
        assert "Error Rate" not in titles

    def test_web_panels_have_correct_namespaces(self) -> None:
        from ml_platform.monitoring.dashboards import WebResources

        config = ServiceConfig(service_name="ns-check")
        resources = WebResources(
            alb_full_name="app/alb/123",
            ecs_cluster_name="c",
            ecs_service_name="s",
            rds_instance_id="db",
        )
        result = generate_dashboard(
            config, panel_sets=["web"], resources=resources,
        )
        panels = result["dashboard"]["panels"]
        namespaces = set()
        for panel in panels:
            for target in panel.get("targets", []):
                namespaces.add(target.get("namespace", ""))
        assert "AWS/ApplicationELB" in namespaces
        assert "AWS/ECS" in namespaces
        assert "AWS/RDS" in namespaces
        assert "MLPlatform" not in namespaces


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
