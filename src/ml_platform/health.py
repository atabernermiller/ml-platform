"""Structured health checks for ml-platform services.

Provides liveness/readiness probes with per-component status reporting,
automatic backend checks, and user-extensible custom checks.

Quick start::

    from ml_platform.health import HealthRegistry, HealthCheck, HealthStatus

    registry = HealthRegistry(service_name="my-chatbot")

    # Built-in checks are auto-registered by the runtime.
    # Add custom checks:
    registry.register(HealthCheck(
        name="model_loaded",
        check=lambda: model.is_loaded,
        critical=True,
    ))

    status = registry.readiness()
    # {
    #   "status": "healthy",
    #   "service": "my-chatbot",
    #   "uptime_s": 3642.1,
    #   "checks": {
    #     "runtime": {"status": "ok"},
    #     "model_loaded": {"status": "ok"},
    #     "dynamodb": {"status": "ok"},
    #   }
    # }

The registry distinguishes between:

- **Liveness** (``/health/live``): Is the process alive?  Always returns
  200 unless the service has explicitly marked itself as unhealthy.
  Used by orchestrators to decide whether to restart the container.

- **Readiness** (``/health/ready``): Can the service accept traffic?
  Returns 200 only when all **critical** checks pass.  Used by load
  balancers to decide whether to route requests.

The legacy ``/health`` endpoint is kept as an alias for ``/health/live``
for backward compatibility with existing ALB/ECS configurations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Result of a single health check."""

    OK = "ok"
    DEGRADED = "degraded"
    FAIL = "fail"


@dataclass
class HealthCheck:
    """A single named health check.

    Attributes:
        name: Unique identifier for this check.
        check: Callable returning ``True`` (healthy) or ``False``
            (unhealthy).  Must not raise exceptions.
        critical: If ``True``, a failure marks the service as not ready.
            Non-critical checks can fail without affecting readiness
            (the service is degraded but still serves traffic).
        description: Optional human-readable description.
    """

    name: str
    check: Callable[[], bool]
    critical: bool = True
    description: str = ""


@dataclass
class HealthResult:
    """Aggregate health check result.

    Attributes:
        status: Overall status (``healthy``, ``degraded``, ``unhealthy``).
        service: Service name.
        uptime_s: Seconds since the registry was created.
        checks: Per-component status.
    """

    status: str
    service: str
    uptime_s: float
    checks: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "status": self.status,
            "service": self.service,
            "uptime_s": round(self.uptime_s, 1),
            "checks": self.checks,
        }


class HealthRegistry:
    """Registry of health checks with liveness/readiness evaluation.

    The runtime creates one registry per service and auto-registers
    built-in checks for backends (DynamoDB, S3, model state).  Users
    can register additional custom checks at any time.

    Args:
        service_name: Included in all health responses.
    """

    def __init__(self, service_name: str = "") -> None:
        self._service_name = service_name
        self._checks: dict[str, HealthCheck] = {}
        self._start_time = time.monotonic()
        self._alive = True

    @property
    def service_name(self) -> str:
        """The service name included in health responses."""
        return self._service_name

    @property
    def checks(self) -> dict[str, HealthCheck]:
        """Registered checks keyed by name."""
        return dict(self._checks)

    def register(self, check: HealthCheck) -> None:
        """Register a health check.

        If a check with the same name already exists, it is replaced.

        Args:
            check: The health check to register.
        """
        self._checks[check.name] = check
        logger.debug("Health check registered: %s (critical=%s)", check.name, check.critical)

    def deregister(self, name: str) -> None:
        """Remove a health check by name.

        Args:
            name: The check name to remove.  No-op if not found.
        """
        self._checks.pop(name, None)

    def mark_unhealthy(self) -> None:
        """Mark the service as not alive (liveness probe will fail)."""
        self._alive = False
        logger.warning("Service marked unhealthy")

    def mark_healthy(self) -> None:
        """Mark the service as alive again."""
        self._alive = True

    def liveness(self) -> HealthResult:
        """Evaluate the liveness probe.

        Returns healthy if the process is alive and has not been
        explicitly marked unhealthy.

        Returns:
            Health result with overall status.
        """
        uptime = time.monotonic() - self._start_time
        status = "healthy" if self._alive else "unhealthy"
        return HealthResult(
            status=status,
            service=self._service_name,
            uptime_s=uptime,
        )

    def readiness(self) -> HealthResult:
        """Evaluate the readiness probe.

        Runs all registered checks and aggregates their results:

        - **healthy**: All checks pass.
        - **degraded**: All critical checks pass, but one or more
          non-critical checks fail.
        - **unhealthy**: At least one critical check fails.

        Returns:
            Health result with per-component status.
        """
        uptime = time.monotonic() - self._start_time
        check_results: dict[str, dict[str, str]] = {}
        any_critical_fail = False
        any_non_critical_fail = False

        for name, hc in self._checks.items():
            try:
                passed = hc.check()
            except Exception:
                logger.exception("Health check %s raised an exception", name)
                passed = False

            if passed:
                check_results[name] = {"status": CheckStatus.OK.value}
            else:
                if hc.critical:
                    check_results[name] = {"status": CheckStatus.FAIL.value}
                    any_critical_fail = True
                else:
                    check_results[name] = {"status": CheckStatus.DEGRADED.value}
                    any_non_critical_fail = True

        if any_critical_fail or not self._alive:
            overall = "unhealthy"
        elif any_non_critical_fail:
            overall = "degraded"
        else:
            overall = "healthy"

        return HealthResult(
            status=overall,
            service=self._service_name,
            uptime_s=uptime,
            checks=check_results,
        )
