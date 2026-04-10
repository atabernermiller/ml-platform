"""Shared FastAPI application builder for all serving paths.

This is an internal module (not part of the public API).  Each
``create_*_app()`` factory calls :func:`build_base_app` to get a FastAPI
instance with standard ``/health``, ``/metrics``, and ``/dashboard``
routes, then adds its service-specific routes on top.

Centralising common routes here means that a health-check format change,
a metrics schema update, or a dashboard bug fix is a single edit
regardless of how many serving paths exist.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ml_platform.config import ServiceConfig

_DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{service_name} – Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js" integrity="sha384-UUpbbMKPd3IjhDSEMBMPmVVMz8KQnWOMvdM4EXBcOG/FlnmpbNjr8TZ2mLFNccUb" crossorigin="anonymous"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif;
            background: #0f172a; color: #e2e8f0; padding: 1.5rem; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 1rem; color: #38bdf8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
             gap: 1rem; }}
    .card {{ background: #1e293b; border-radius: .75rem; padding: 1rem; }}
    .card h2 {{ font-size: .85rem; color: #94a3b8; margin-bottom: .75rem;
                text-transform: uppercase; letter-spacing: .05em; }}
    .metric {{ font-size: 2rem; font-weight: 700; }}
    .metric small {{ font-size: .9rem; color: #64748b; font-weight: 400; }}
    canvas {{ width: 100% !important; height: 180px !important; }}
    #error {{ color: #f87171; margin-bottom: 1rem; display: none; }}
  </style>
</head>
<body>
  <h1>{service_name}</h1>
  <div id="error"></div>
  <div class="grid" id="panels"></div>
  <script>
    const SERVICE = "{service_name}";
    const DASH_TYPE = "{dashboard_type}";
    async function refresh() {{
      try {{
        const r = await fetch("/dashboard/api/metrics");
        if (!r.ok) throw new Error(r.statusText);
        const d = await r.json();
        document.getElementById("error").style.display = "none";
        render(d);
      }} catch (e) {{
        const el = document.getElementById("error");
        el.textContent = "Failed to load metrics: " + e.message;
        el.style.display = "block";
      }}
    }}
    function esc(s) {{ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }}
    function render(d) {{
      const p = document.getElementById("panels");
      p.innerHTML = "";
      for (const [k, v] of Object.entries(d)) {{
        const card = document.createElement("div");
        card.className = "card";
        const fmt = typeof v === "number" ? v.toLocaleString(undefined,
          {{maximumFractionDigits: 4}}) : String(v);
        const h2 = document.createElement("h2");
        h2.textContent = k.replace(/_/g, " ");
        const metric = document.createElement("div");
        metric.className = "metric";
        metric.textContent = fmt;
        card.appendChild(h2);
        card.appendChild(metric);
        p.appendChild(card);
      }}
    }}
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class _RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject ``request_id`` into the log context for every request.

    Also reads ``X-Request-ID`` and ``X-Session-ID`` headers when
    present (for distributed tracing), and adds ``method`` and
    ``path`` for structured log correlation.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        from ml_platform.log import _log_context, bind

        _log_context.set({})

        request_id = (
            request.headers.get("x-request-id")
            or str(uuid.uuid4())
        )
        bind(request_id=request_id)

        session_id = request.headers.get("x-session-id", "")
        if session_id:
            bind(session_id=session_id)

        bind(method=request.method, path=request.url.path)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def build_base_app(
    config: ServiceConfig,
    *,
    readiness_check: Callable[[], bool],
    metrics_source: Callable[[], dict[str, float]],
    alert_status: Callable[[], list[dict[str, Any]]] | None = None,
    health_registry: Any | None = None,
    dashboard_type: Literal["stateful", "llm", "agent"] = "llm",
) -> FastAPI:
    """Create a FastAPI app pre-configured with shared platform routes.

    The returned app contains:

    - ``GET /health`` -- backward-compatible liveness probe (alias).
    - ``GET /health/live`` -- liveness probe (process alive?).
    - ``GET /health/ready`` -- readiness probe with per-component status.
    - ``GET /metrics`` -- current metric snapshot.
    - ``GET /alerts`` -- current alert rule states (if alerts configured).
    - ``GET /dashboard`` -- self-contained HTML dashboard.
    - ``GET /dashboard/api/metrics`` -- JSON endpoint consumed by the
      dashboard.
    - Request-context middleware that injects ``request_id`` into all
      log lines for the duration of each request.

    Callers add their service-specific routes (e.g. ``/predict``,
    ``/chat``, ``/run``) to the returned app.

    Args:
        config: Platform configuration.
        readiness_check: Callable returning ``True`` once the service
            can accept requests.
        metrics_source: Callable returning the current metric snapshot.
        alert_status: Optional callable returning current alert states.
        health_registry: Optional :class:`~ml_platform.health.HealthRegistry`
            for structured health checks.
        dashboard_type: Selects which dashboard panels to render.

    Returns:
        Pre-configured FastAPI application.
    """
    app = FastAPI(title=config.service_name)
    app.add_middleware(_RequestContextMiddleware)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        if health_registry is not None:
            return health_registry.liveness().to_dict()
        return {"status": "healthy", "service": config.service_name}

    @app.get("/health/live")
    async def health_live() -> dict[str, Any]:
        if health_registry is not None:
            return health_registry.liveness().to_dict()
        return {"status": "healthy", "service": config.service_name}

    @app.get("/health/ready")
    async def health_ready(response: Response) -> dict[str, Any]:
        if health_registry is not None:
            result = health_registry.readiness()
            if result.status == "unhealthy":
                response.status_code = 503
            return result.to_dict()
        if not readiness_check():
            response.status_code = 503
            return {"status": "unhealthy", "service": config.service_name}
        return {"status": "healthy", "service": config.service_name}

    @app.get("/metrics")
    async def metrics() -> dict[str, float]:
        if not readiness_check():
            raise HTTPException(status_code=503, detail="Service not initialized")
        return metrics_source()

    @app.get("/alerts")
    async def alerts() -> list[dict[str, Any]]:
        if alert_status is None:
            return []
        return alert_status()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return _DASHBOARD_TEMPLATE.format(
            service_name=config.service_name,
            dashboard_type=dashboard_type,
        )

    @app.get("/dashboard/api/metrics")
    async def dashboard_metrics() -> dict[str, Any]:
        if not readiness_check():
            return {"status": "initializing"}
        return metrics_source()

    return app
