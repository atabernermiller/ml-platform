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
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from ml_platform.config import ServiceConfig

_DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{service_name} – Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
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
    function render(d) {{
      const p = document.getElementById("panels");
      p.innerHTML = "";
      for (const [k, v] of Object.entries(d)) {{
        const card = document.createElement("div");
        card.className = "card";
        const fmt = typeof v === "number" ? v.toLocaleString(undefined,
          {{maximumFractionDigits: 4}}) : v;
        card.innerHTML = `<h2>${{k.replace(/_/g, " ")}}</h2>
          <div class="metric">${{fmt}}</div>`;
        p.appendChild(card);
      }}
    }}
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def build_base_app(
    config: ServiceConfig,
    *,
    readiness_check: Callable[[], bool],
    metrics_source: Callable[[], dict[str, float]],
    dashboard_type: Literal["stateful", "llm", "agent"] = "llm",
) -> FastAPI:
    """Create a FastAPI app pre-configured with shared platform routes.

    The returned app contains:

    - ``GET /health`` -- readiness probe.
    - ``GET /metrics`` -- current metric snapshot.
    - ``GET /dashboard`` -- self-contained HTML dashboard.
    - ``GET /dashboard/api/metrics`` -- JSON endpoint consumed by the
      dashboard.

    Callers add their service-specific routes (e.g. ``/predict``,
    ``/chat``, ``/run``) to the returned app.

    Args:
        config: Platform configuration.
        readiness_check: Callable returning ``True`` once the service
            can accept requests.
        metrics_source: Callable returning the current metric snapshot.
        dashboard_type: Selects which dashboard panels to render.

    Returns:
        Pre-configured FastAPI application.
    """
    app = FastAPI(title=config.service_name)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy", "service": config.service_name}

    @app.get("/metrics")
    async def metrics() -> dict[str, float]:
        if not readiness_check():
            raise HTTPException(status_code=503, detail="Service not initialized")
        return metrics_source()

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
