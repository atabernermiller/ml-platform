"""ml-platform: Shared infrastructure for production ML services.

Provides two service development paths:

- **Stateful services** (``StatefulServiceBase`` + FastAPI app factory) for
  online learning, contextual bandits, and feedback-loop services.
- **Stateless services** (BentoML integration via ``PlatformMonitor``) for
  inference-only endpoints with consistent monitoring.

Both paths share:

- S3 state checkpointing (``S3StateManager``)
- DynamoDB / Redis context storage (``ContextStore``)
- CloudWatch + OpenTelemetry metric emission (``MetricsEmitter``)
- MLflow experiment tracking (``ExperimentTracker``)
- AWS CDK infrastructure constructs (``ml_platform.infra``)
"""

from ml_platform._version import __version__
from ml_platform.config import ServiceConfig

__all__ = ["ServiceConfig", "__version__"]
