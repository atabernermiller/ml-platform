"""ml-platform: Shared infrastructure for production ML services.

Provides three service development paths:

- **Stateful services** (``StatefulServiceBase`` + FastAPI app factory) for
  online learning, contextual bandits, and feedback-loop services.
- **Agent services** (``AgentServiceBase`` + FastAPI app factory) for
  multi-step LLM workflows with tool use and multiple providers.
- **Stateless services** (BentoML integration via ``PlatformMonitor``) for
  inference-only endpoints with consistent monitoring.

All paths share:

- S3 / local state checkpointing (``StateManager``)
- DynamoDB / SQLite / in-memory context storage (``ContextStore``)
- Pluggable metric backends (``MetricsBackend``)
- Pluggable experiment tracking (``ExperimentTracker``)
- Cloud profiles for zero-config backend selection
- Built-in ``/dashboard`` route with live metrics
"""

from ml_platform._version import __version__
from ml_platform.config import AgentConfig, LLMConfig, ServiceConfig, StatefulConfig

__all__ = [
    "AgentConfig",
    "LLMConfig",
    "ServiceConfig",
    "StatefulConfig",
    "__version__",
]
