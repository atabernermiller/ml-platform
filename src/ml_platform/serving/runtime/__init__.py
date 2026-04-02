"""Runtimes for ml-platform services.

:class:`BaseRuntime` provides the shared lifecycle that all service types
need: metric backend initialisation, background metric-emission loops,
experiment-tracker wiring, and startup/shutdown hooks.

:class:`StatefulRuntime` extends the base with S3 checkpoint loops and
feedback handling for online-learning services.

:class:`AgentRuntime` extends the base with per-request
:class:`~ml_platform.llm.run_context.RunContext` creation and agent-level
metric aggregation.

The companion ``create_*_app()`` factories wire a runtime into FastAPI
routes and ASGI lifespan.
"""

from ml_platform.serving.runtime._agent import AgentRuntime
from ml_platform.serving.runtime._base import BaseRuntime
from ml_platform.serving.runtime._stateful import StatefulRuntime

__all__ = ["AgentRuntime", "BaseRuntime", "StatefulRuntime"]
