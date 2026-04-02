"""Tests for ml_platform.health -- structured health checks."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ml_platform.health import CheckStatus, HealthCheck, HealthRegistry, HealthResult


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_defaults(self) -> None:
        hc = HealthCheck(name="test", check=lambda: True)
        assert hc.name == "test"
        assert hc.critical is True
        assert hc.description == ""

    def test_check_callable(self) -> None:
        hc = HealthCheck(name="ok", check=lambda: True)
        assert hc.check() is True

        hc2 = HealthCheck(name="fail", check=lambda: False)
        assert hc2.check() is False


# ---------------------------------------------------------------------------
# HealthResult
# ---------------------------------------------------------------------------


class TestHealthResult:
    def test_to_dict(self) -> None:
        r = HealthResult(
            status="healthy",
            service="svc",
            uptime_s=123.456,
            checks={"runtime": {"status": "ok"}},
        )
        d = r.to_dict()
        assert d["status"] == "healthy"
        assert d["service"] == "svc"
        assert d["uptime_s"] == 123.5
        assert d["checks"]["runtime"]["status"] == "ok"


# ---------------------------------------------------------------------------
# HealthRegistry -- liveness
# ---------------------------------------------------------------------------


class TestHealthRegistryLiveness:
    def test_healthy_by_default(self) -> None:
        reg = HealthRegistry(service_name="svc")
        result = reg.liveness()
        assert result.status == "healthy"
        assert result.service == "svc"
        assert result.uptime_s >= 0

    def test_unhealthy_when_marked(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.mark_unhealthy()
        assert reg.liveness().status == "unhealthy"

    def test_re_mark_healthy(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.mark_unhealthy()
        reg.mark_healthy()
        assert reg.liveness().status == "healthy"


# ---------------------------------------------------------------------------
# HealthRegistry -- readiness
# ---------------------------------------------------------------------------


class TestHealthRegistryReadiness:
    def test_healthy_all_pass(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="a", check=lambda: True))
        reg.register(HealthCheck(name="b", check=lambda: True))
        result = reg.readiness()
        assert result.status == "healthy"
        assert result.checks["a"]["status"] == "ok"
        assert result.checks["b"]["status"] == "ok"

    def test_unhealthy_critical_fail(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="critical", check=lambda: False, critical=True))
        result = reg.readiness()
        assert result.status == "unhealthy"
        assert result.checks["critical"]["status"] == "fail"

    def test_degraded_non_critical_fail(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="ok", check=lambda: True, critical=True))
        reg.register(HealthCheck(name="optional", check=lambda: False, critical=False))
        result = reg.readiness()
        assert result.status == "degraded"
        assert result.checks["ok"]["status"] == "ok"
        assert result.checks["optional"]["status"] == "degraded"

    def test_healthy_no_checks(self) -> None:
        reg = HealthRegistry(service_name="svc")
        result = reg.readiness()
        assert result.status == "healthy"

    def test_check_exception_treated_as_failure(self) -> None:
        def _boom() -> bool:
            raise RuntimeError("kaboom")

        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="broken", check=_boom, critical=True))
        result = reg.readiness()
        assert result.status == "unhealthy"
        assert result.checks["broken"]["status"] == "fail"

    def test_unhealthy_when_marked_dead(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="ok", check=lambda: True))
        reg.mark_unhealthy()
        result = reg.readiness()
        assert result.status == "unhealthy"

    def test_register_replaces_existing(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="a", check=lambda: False))
        reg.register(HealthCheck(name="a", check=lambda: True))
        result = reg.readiness()
        assert result.checks["a"]["status"] == "ok"

    def test_deregister(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="a", check=lambda: True))
        reg.deregister("a")
        assert "a" not in reg.readiness().checks

    def test_deregister_missing_is_noop(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.deregister("nonexistent")


# ---------------------------------------------------------------------------
# /health endpoints
# ---------------------------------------------------------------------------


def _make_app_with_registry(
    registry: HealthRegistry | None = None,
) -> Any:
    from ml_platform.config import ServiceConfig
    from ml_platform.serving._app_builder import build_base_app

    config = ServiceConfig(service_name="health-test")
    return build_base_app(
        config,
        readiness_check=lambda: True,
        metrics_source=lambda: {"ok": 1.0},
        health_registry=registry,
    )


@pytest.mark.asyncio
class TestHealthEndpoints:
    async def test_health_live_200(self) -> None:
        reg = HealthRegistry(service_name="svc")
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_s" in data

    async def test_health_live_unhealthy(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.mark_unhealthy()
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unhealthy"

    async def test_health_ready_200(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="ok", check=lambda: True))
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["checks"]["ok"]["status"] == "ok"

    async def test_health_ready_503_on_critical_fail(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="bad", check=lambda: False, critical=True))
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"

    async def test_health_ready_200_on_degraded(self) -> None:
        reg = HealthRegistry(service_name="svc")
        reg.register(HealthCheck(name="ok", check=lambda: True, critical=True))
        reg.register(HealthCheck(name="opt", check=lambda: False, critical=False))
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"

    async def test_legacy_health_endpoint(self) -> None:
        reg = HealthRegistry(service_name="svc")
        app = _make_app_with_registry(reg)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    async def test_fallback_without_registry(self) -> None:
        app = _make_app_with_registry(None)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            live = await c.get("/health/live")
            ready = await c.get("/health/ready")
        assert live.status_code == 200
        assert live.json()["status"] == "healthy"
        assert ready.status_code == 200
        assert ready.json()["status"] == "healthy"

    async def test_ready_fallback_503_when_not_ready(self) -> None:
        from ml_platform.config import ServiceConfig
        from ml_platform.serving._app_builder import build_base_app

        config = ServiceConfig(service_name="not-ready")
        app = build_base_app(
            config,
            readiness_check=lambda: False,
            metrics_source=lambda: {},
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"
