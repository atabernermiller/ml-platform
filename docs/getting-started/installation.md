# Installation

## Install extras

ml-platform uses optional dependency groups so you only install what you need:

```bash
# Core only (pydantic)
pip install ml-platform

# Stateful serving (FastAPI + boto3)
pip install "ml-platform[stateful]"

# Stateless serving (BentoML)
pip install "ml-platform[stateless]"

# Monitoring (boto3 + OpenTelemetry)
pip install "ml-platform[monitoring]"

# Experiment tracking (MLflow + boto3)
pip install "ml-platform[tracking]"

# CDK infrastructure constructs
pip install "ml-platform[infra]"

# Everything including dev tools
pip install "ml-platform[all]"
```

## Dependency summary

| Extra | Key packages | Purpose |
|---|---|---|
| *(core)* | `pydantic>=2` | Configuration, schemas |
| `stateful` | `fastapi`, `uvicorn`, `boto3` | FastAPI app factory, S3/DynamoDB |
| `stateless` | `bentoml` | BentoML monitoring integration |
| `monitoring` | `boto3`, `opentelemetry-api/sdk` | CloudWatch metrics |
| `tracking` | `mlflow`, `boto3` | MLflow experiment tracking |
| `infra` | `aws-cdk-lib`, `constructs` | CDK constructs |
| `dev` | `pytest`, `moto`, `httpx`, `ruff`, `mypy` | Testing and linting |
