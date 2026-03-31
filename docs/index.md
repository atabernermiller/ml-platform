# ml-platform

Shared Python library for deploying ML services on AWS with consistent monitoring, experiment tracking, and infrastructure-as-code.

## Overview

ml-platform provides two service development paths:

- **Stateful services** (`StatefulServiceBase` + `StatefulRuntime` + FastAPI app factory) for online learning, contextual bandits, and feedback-loop services.
- **Stateless services** (BentoML integration via `PlatformMonitor`) for inference-only endpoints with consistent monitoring.

Both paths share:

- **S3 state checkpointing** via `S3StateManager`
- **DynamoDB context storage** via `ContextStore`
- **CloudWatch metric emission** via `MetricsEmitter` (EMF + direct put)
- **Pluggable experiment tracking** via `ExperimentTracker` ABC (ships with `MLflowTracker` and `NullTracker`)
- **AWS CDK constructs** for VPC, ECS Fargate, SageMaker, MLflow, and CloudWatch alarms

## Quick start

```bash
pip install "ml-platform[stateful]"

# Bootstrap AWS resources
ml-platform bootstrap --service-name my-bandit --s3-bucket my-ckpt --region us-east-1

# Validate setup
ml-platform check --service-name my-bandit --s3-bucket my-ckpt --region us-east-1
```

Then implement your service:

```python
from ml_platform.config import ServiceConfig
from ml_platform.serving.stateful import (
    StatefulServiceBase, PredictionResult, create_stateful_app,
)

class MyBandit(StatefulServiceBase):
    async def predict(self, payload: dict) -> PredictionResult: ...
    async def process_feedback(self, request_id: str, feedback: dict) -> None: ...
    def save_state(self, artifact_dir: str) -> None: ...
    def load_state(self, artifact_dir: str) -> None: ...
    def metrics_snapshot(self) -> dict[str, float]: ...

config = ServiceConfig(service_name="my-bandit", s3_checkpoint_bucket="my-ckpt")
app = create_stateful_app(MyBandit, config)
```

## Package layout

```
ml_platform/
├── config.py              # ServiceConfig dataclass
├── serving/
│   ├── stateful.py        # StatefulServiceBase + create_stateful_app()
│   ├── runtime.py         # StatefulRuntime (transport-agnostic)
│   ├── stateless.py       # PlatformMonitor (BentoML integration)
│   ├── state_manager.py   # S3StateManager
│   ├── context_store.py   # ContextStore ABC + DynamoDB/InMemory
│   └── schemas.py         # Pydantic request/response models
├── monitoring/
│   ├── metrics.py         # MetricsEmitter (CloudWatch EMF)
│   └── dashboards.py      # Grafana/CloudWatch dashboard generators
├── tracking/
│   ├── base.py            # ExperimentTracker ABC + NullTracker
│   └── mlflow.py          # MLflowTracker
├── infra/constructs/      # AWS CDK constructs
└── cli/                   # ml-platform check / bootstrap
```
