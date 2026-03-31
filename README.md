# ml-platform

Shared Python library for deploying ML services on AWS with consistent monitoring, experiment tracking, and infrastructure-as-code. The library provides two service development paths — **stateful** (FastAPI) for online-learning and feedback-loop services, and **stateless** (BentoML) for inference-only endpoints — both backed by CloudWatch metrics, pluggable experiment tracking, S3 state checkpointing, and CDK constructs.

## Architecture

```
                          ┌─────────────────────────────────┐
                          │         Your ML Service          │
                          └──────────┬──────────┬────────────┘
                                     │          │
                 ┌───────────────────┘          └───────────────────┐
                 ▼                                                  ▼
    ┌────────────────────────┐                        ┌────────────────────────┐
    │  Stateful Path         │                        │  Stateless Path        │
    │                        │                        │  (BentoML + monitor)   │
    │  StatefulServiceBase   │                        │                        │
    │  StatefulRuntime       │                        │  PlatformMonitor       │
    │  create_stateful_app() │                        │  with_platform_        │
    │  /predict  /feedback   │                        │    monitoring()        │
    │  /health   /metrics    │                        │                        │
    └──────────┬─────────────┘                        └──────────┬─────────────┘
               │                                                  │
               └──────────────────┬───────────────────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │          Shared Platform Layer        │
               │                                       │
               │  S3StateManager    ContextStore        │
               │  MetricsEmitter    ExperimentTracker   │
               │  (CloudWatch EMF)  (ABC: MLflow, etc.) │
               └──────────────────┬────────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │       CDK Infrastructure Constructs   │
               │                                       │
               │  NetworkConstruct                     │
               │  EcsServiceConstruct                  │
               │  SageMakerEndpointConstruct           │
               │  MLflowConstruct                      │
               │  MonitoringConstruct                  │
               └──────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Core library only
pip install ml-platform

# With stateful serving dependencies (FastAPI + boto3)
pip install "ml-platform[stateful]"

# Everything (stateful, stateless, monitoring, tracking, infra, dev tools)
pip install "ml-platform[all]"
```

### Bootstrap your environment

Before writing any service code, create the required AWS resources and validate your setup:

```bash
# Create S3 bucket, DynamoDB table, and print the IAM policy to attach
ml-platform bootstrap \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1

# Verify everything is configured correctly
ml-platform check \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1
```

See [CLI: Setup & Validation](#cli-setup--validation) for full details.

### Minimal Stateful Service

```python
from ml_platform.config import ServiceConfig
from ml_platform.serving.stateful import (
    StatefulServiceBase,
    PredictionResult,
    create_stateful_app,
)

class MyBanditService(StatefulServiceBase):
    def __init__(self) -> None:
        self._total_reward = 0.0

    async def predict(self, payload: dict) -> PredictionResult:
        return PredictionResult(
            request_id="req-001",
            prediction={"arm": "model-a"},
            metadata={"cost_usd": 0.003},
        )

    async def process_feedback(self, request_id: str, feedback: dict) -> None:
        self._total_reward += feedback.get("reward", 0.0)

    def save_state(self, artifact_dir: str) -> None: ...
    def load_state(self, artifact_dir: str) -> None: ...
    def metrics_snapshot(self) -> dict[str, float]:
        return {"total_reward": self._total_reward}

config = ServiceConfig(
    service_name="pareto-bandit",
    s3_checkpoint_bucket="my-checkpoints",
    mlflow_tracking_uri="http://mlflow.internal:5000",
)
app = create_stateful_app(MyBanditService, config)
# Run: uvicorn my_service:app --host 0.0.0.0 --port 8000
```

### Using StatefulRuntime directly (non-HTTP transports)

`StatefulRuntime` owns all lifecycle orchestration (S3 restore, checkpoint loops, metric emission) without depending on FastAPI. Use it to integrate stateful services into gRPC servers, AWS Lambda handlers, or CLI tools:

```python
from ml_platform.config import ServiceConfig
from ml_platform.serving.runtime import StatefulRuntime

runtime = StatefulRuntime(MyBanditService, config)

# In your async entrypoint:
await runtime.startup()
result = await runtime.predict({"context": [1, 2, 3]})
await runtime.process_feedback(result.request_id, {"reward": 1.0})
await runtime.shutdown()
```

`create_stateful_app()` is a thin adapter that wires a `StatefulRuntime` into FastAPI routes and lifespan.

## Stateless BentoML Service

```python
import bentoml
import numpy as np
from ml_platform.config import ServiceConfig
from ml_platform.serving.stateless import with_platform_monitoring

config = ServiceConfig(service_name="iris-classifier")

@bentoml.service(resources={"cpu": "2"})
class IrisClassifier:
    model = bentoml.models.get("iris_sklearn:latest")

    def __init__(self) -> None:
        self._monitor = with_platform_monitoring(self, config)

    @bentoml.api
    def predict(self, features: np.ndarray) -> np.ndarray:
        result = self.model.predict(features)
        self._monitor.record_prediction({"n_samples": len(features)})
        return result
```

## Experiment Tracking

`ExperimentTracker` is an abstract base class with a 5-method contract (`run_id`, `log_params`, `log_metrics`, `log_artifact`, `end_run`). The library ships with two implementations:

| Class | Backend | Install extra |
|---|---|---|
| `MLflowTracker` | MLflow tracking server | `ml-platform[tracking]` |
| `NullTracker` | Silent no-op (for dev/testing) | *(included in core)* |

```python
from ml_platform.tracking import MLflowTracker, NullTracker

# Production: log to MLflow
tracker = MLflowTracker(
    tracking_uri="http://mlflow.internal:5000",
    experiment_name="my-experiment",
)

# Development / testing: no-op
tracker = NullTracker()

# Both share the same interface
tracker.log_params({"learning_rate": 0.01})
tracker.log_metrics({"accuracy": 0.95})
tracker.end_run()
```

To integrate a different backend (Weights & Biases, Neptune, Comet, etc.), subclass `ExperimentTracker` and implement the five abstract methods.

## CDK Infrastructure

Compose the provided CDK constructs to deploy your service with consistent networking, compute, and monitoring:

```python
from aws_cdk import App, Stack
from ml_platform.infra.constructs import (
    NetworkConstruct,
    EcsServiceConstruct,
    MonitoringConstruct,
    MLflowConstruct,
)

class MyServiceStack(Stack):
    def __init__(self, scope, construct_id, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        network = NetworkConstruct(self, "Network")
        mlflow = MLflowConstruct(self, "MLflow", vpc=network.vpc)
        ecs = EcsServiceConstruct(
            self, "Service",
            vpc=network.vpc,
            service_name="pareto-bandit",
            container_image="123456789.dkr.ecr.us-east-1.amazonaws.com/bandit:latest",
        )
        MonitoringConstruct(
            self, "Monitoring",
            service_name="pareto-bandit",
        )

app = App()
MyServiceStack(app, "ParetoBanditStack")
app.synth()
```

## Monitoring Dashboards

The `ml_platform.monitoring.dashboards` module generates ready-to-import dashboard configurations for both **Grafana** and **CloudWatch**:

```python
from ml_platform.monitoring.dashboards import (
    generate_grafana_dashboard,
    generate_cloudwatch_dashboard,
)

grafana_json = generate_grafana_dashboard("my-service", "us-east-1")
cw_body = generate_cloudwatch_dashboard("my-service", "us-east-1")
```

Each dashboard includes panels for:

| Panel | Metric | Statistic |
|---|---|---|
| Cumulative Reward | `cumulative_reward` | Maximum |
| Cost per Request | `cost_usd` | Average |
| Exploration Rate | `exploration_rate` | Average |
| Budget Utilisation | `pacer_cost_ema` | Average |
| Prediction Latency | `latency_ms` | P50 / P95 / P99 |
| Feedback Delay | `feedback_delay_s` | Average |
| Prediction Error | `prediction_error` | Average |

All metrics are emitted under the `MLPlatform` CloudWatch namespace with a `service` dimension.

## CLI: Setup & Validation

The library ships with a CLI that automates environment setup and validates your AWS configuration.

### Preflight check

Verify that credentials, buckets, tables, and permissions are all in order:

```bash
ml-platform check \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --dynamodb-table my-bandit-context \
  --region us-east-1
```

Sample output:

```
ml-platform preflight check
  Service: my-bandit
  Region:  us-east-1

  ✓  Authenticated as arn:aws:iam::123456789:role/dev-role (account 123456789)
  ✓  Bucket s3://my-checkpoints exists (empty under checkpoints/)
  ✓  S3 write permission verified (PutObject + DeleteObject)
  ✓  Table my-bandit-context exists (status=ACTIVE, TTL enabled)
  ✓  CloudWatch PutMetricData permission verified

  ✓  5/5 checks passed.
```

### Bootstrap resources

Create the S3 bucket and DynamoDB table automatically, and get the minimum IAM policy to attach:

```bash
# Preview what will be created (no changes made)
ml-platform bootstrap \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1 \
  --dry-run

# Actually create resources
ml-platform bootstrap \
  --service-name my-bandit \
  --s3-bucket my-checkpoints \
  --region us-east-1
```

The bootstrap command will:
1. Create the S3 bucket (with versioning enabled) if it doesn't exist
2. Create the DynamoDB context table (with TTL) if it doesn't exist
3. Print the minimum IAM policy JSON tailored to your configuration

## AWS Credentials & IAM Permissions

The library uses [boto3's standard credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) and **never** accepts explicit access keys. Credentials are resolved automatically in this order:

| Priority | Source | Typical environment |
|---|---|---|
| 1 | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars | CI/CD pipelines |
| 2 | `~/.aws/credentials` (shared credentials file) | Local development |
| 3 | `AWS_PROFILE` / `~/.aws/config` | Local development with named profiles |
| 4 | ECS container credentials (`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`) | **ECS Fargate** (automatic via task role) |
| 5 | EC2/SageMaker instance metadata (IMDS) | EC2, SageMaker endpoints |

The only AWS-specific value you configure is `ServiceConfig.aws_region` (default `us-east-1`), which is passed to every boto3 client as `region_name`.

### Minimum IAM policy

The permissions below cover all optional features. Omit sections for features you don't use. You can also run `ml-platform bootstrap` to generate a policy tailored to your configuration.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Checkpoints",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR_CHECKPOINT_BUCKET",
        "arn:aws:s3:::YOUR_CHECKPOINT_BUCKET/*"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "cloudwatch:namespace": "MLPlatform"
        }
      }
    },
    {
      "Sid": "DynamoDBContextStore",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:DeleteItem"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/YOUR_CONTEXT_TABLE"
    }
  ]
}
```

| Statement | Used by | When needed |
|---|---|---|
| **S3Checkpoints** | `S3StateManager` | `ServiceConfig.s3_checkpoint_bucket` is set |
| **CloudWatchMetrics** | `MetricsEmitter.emit_direct()` | Only for direct `PutMetricData` calls; EMF-via-stdout needs no extra IAM beyond CloudWatch Logs (granted automatically to ECS tasks) |
| **DynamoDBContextStore** | `DynamoDBContextStore` | `ServiceConfig.state_table_name` is set and you use the DynamoDB backend |

> **Note:** When running on ECS Fargate, the `EcsServiceConstruct` CDK construct should attach these permissions to the task execution role. CloudWatch Logs permissions (`logs:CreateLogStream`, `logs:PutLogEvents`) are included by default in ECS task roles and are required for EMF metric ingestion.

## Development

```bash
# Clone and install in editable mode with dev dependencies
git clone <repo-url> && cd ml-platform
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```
