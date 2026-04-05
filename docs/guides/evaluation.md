# Evaluation & Drift Detection

ml-platform provides serving, monitoring, and alerting infrastructure but
intentionally does **not** bundle a specific evaluation or drift-detection
library.  Instead, a thin bridge module (`ml_platform.evaluation`) lets you
plug in any evaluation tool and route its outputs into the platform's
existing monitoring pipeline: CloudWatch metrics, Grafana dashboards,
alert rules, and MLflow experiment tracking.

## Architecture

```
┌───────────────────────────────┐
│  Your evaluation code         │
│  (Evidently / alibi-detect /  │
│   whylogs / custom)           │
└──────────┬────────────────────┘
           │  EvaluationResult
           ▼
┌───────────────────────────────┐
│  EvaluationReporter           │
│  ┌────────────┐ ┌───────────┐ │
│  │MetricsEmit │ │ MLflow    │ │
│  │  (CW/EMF)  │ │ Tracker   │ │
│  └─────┬──────┘ └─────┬─────┘ │
└────────┼──────────────┼───────┘
         ▼              ▼
   CloudWatch /     MLflow
   Grafana          Experiments
         │
         ▼
   AlertEvaluator
   (threshold rules, hysteresis,
    Slack / PagerDuty webhooks)
```

## Quick start

### 1. Install your evaluation library

ml-platform has no opinion on which library you use.  Pick whatever fits
your data and task:

```bash
# Tabular drift + data quality + ML performance
pip install evidently

# Statistical drift tests
pip install alibi-detect

# Data profiling / schema monitoring
pip install whylogs
```

### 2. Create an `EvaluationResult`

```python
from ml_platform.evaluation import EvaluationResult, EvaluationStatus

result = EvaluationResult(
    name="feature-drift",
    status=EvaluationStatus.WARNING,
    metrics={"drift_share": 0.35, "drifted_features": 4},
    details={"per_feature": {"age": 0.9, "income": 0.1}},
    tags={"dataset": "production", "model_version": "v2.3"},
)
```

| Field     | Purpose |
|-----------|---------|
| `name`    | Stable identifier; becomes a CloudWatch dimension and log label. |
| `status`  | `PASSED` / `WARNING` / `FAILED` — controls log level. |
| `metrics` | Numeric values emitted to CloudWatch and eligible for alert rules. |
| `details` | Structured data for debugging; not emitted as metrics. |
| `tags`    | String dimensions added to the CloudWatch `emit_event` call. |

### 3. Report through the monitoring pipeline

```python
from ml_platform.evaluation import EvaluationReporter
from ml_platform.monitoring import MetricsEmitter

emitter = MetricsEmitter(service_name="my-service")
reporter = EvaluationReporter(emitter=emitter)

reporter.report(result)
```

This single call:

- **Emits metrics** to CloudWatch via EMF with both raw keys (`drift_share`)
  and namespaced keys (`eval.feature-drift.drift_share`).
- **Logs** the result at the appropriate severity level.
- **Tracks** metrics in MLflow if a tracker is attached.

### 4. Wire up alert rules

Use the namespaced metric keys in your existing `AlertRule` definitions:

```python
from ml_platform.alerting import AlertRule
from ml_platform.config import ServiceConfig

config = ServiceConfig(
    service_name="my-service",
    alerts=[
        AlertRule(
            metric="eval.feature-drift.drift_share",
            condition=">",
            threshold=0.3,
            window_s=0,
            name="high-drift",
            severity="critical",
            description="More than 30% of features drifted",
        ),
        AlertRule(
            metric="eval.data-quality.missing_rate",
            condition=">",
            threshold=0.05,
            name="high-missing-rate",
        ),
    ],
    alert_webhook_url="https://hooks.slack.com/services/...",
)
```

Feed the reporter's snapshot into the evaluator on each metric cycle:

```python
snapshot = reporter.metrics_snapshot()
await alert_evaluator.evaluate(snapshot)
```

### 5. Schedule periodic evaluations

Use `@scheduled` to run evaluations automatically:

```python
from ml_platform.evaluation import EvaluationReporter, EvaluationResult, EvaluationStatus
from ml_platform.scheduling import scheduled
from ml_platform.serving.stateful import StatefulServiceBase

class MyService(StatefulServiceBase):
    def __init__(self) -> None:
        self.reporter = EvaluationReporter(emitter=self.metrics)

    @scheduled(interval_s=3600, name="hourly-drift")
    async def check_drift(self) -> None:
        reference, current = self._get_data_windows()
        drift_share = self._compute_drift(reference, current)

        self.reporter.report(EvaluationResult(
            name="feature-drift",
            status=(EvaluationStatus.FAILED if drift_share > 0.3
                    else EvaluationStatus.WARNING if drift_share > 0.15
                    else EvaluationStatus.PASSED),
            metrics={"drift_share": drift_share},
        ))
```

---

## Framework integration examples

### Evidently

[Evidently](https://github.com/evidentlyai/evidently) provides 100+
built-in metrics for data drift, data quality, and model performance on
both tabular and text data.

#### Data drift

```python
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from ml_platform.evaluation import EvaluationReporter, EvaluationResult, EvaluationStatus
from ml_platform.scheduling import scheduled


@scheduled(interval_s=3600, name="evidently-drift")
async def check_drift(self) -> None:
    reference_df, current_df = self._get_dataframes()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    output = report.as_dict()

    drift_result = output["metrics"][0]["result"]
    drift_share = drift_result["share_of_drifted_columns"]
    n_drifted = drift_result["number_of_drifted_columns"]

    self.reporter.report(EvaluationResult(
        name="feature-drift",
        status=(EvaluationStatus.FAILED if drift_share > 0.3
                else EvaluationStatus.WARNING if drift_share > 0.1
                else EvaluationStatus.PASSED),
        metrics={
            "drift_share": drift_share,
            "drifted_columns": float(n_drifted),
        },
        details={"evidently_report": output},
    ))
```

#### Data quality

```python
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report

from ml_platform.evaluation import EvaluationResult, EvaluationStatus


def run_quality_check(self, current_df) -> EvaluationResult:
    report = Report(metrics=[DataQualityPreset()])
    report.run(current_data=current_df)
    output = report.as_dict()

    missing = output["metrics"][0]["result"].get(
        "current", {}
    ).get("share_of_missing_values", 0.0)

    return EvaluationResult(
        name="data-quality",
        status=(EvaluationStatus.FAILED if missing > 0.1
                else EvaluationStatus.WARNING if missing > 0.05
                else EvaluationStatus.PASSED),
        metrics={"missing_rate": missing},
        details={"evidently_report": output},
    )
```

#### Classification performance

```python
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

from ml_platform.evaluation import EvaluationResult, EvaluationStatus


def run_classification_eval(self, reference_df, current_df) -> EvaluationResult:
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    output = report.as_dict()

    accuracy = output["metrics"][0]["result"].get("current", {}).get("accuracy", 0.0)
    f1 = output["metrics"][0]["result"].get("current", {}).get("f1", 0.0)

    return EvaluationResult(
        name="classification-perf",
        status=(EvaluationStatus.FAILED if accuracy < 0.8
                else EvaluationStatus.WARNING if accuracy < 0.9
                else EvaluationStatus.PASSED),
        metrics={"accuracy": accuracy, "f1": f1},
        details={"evidently_report": output},
    )
```

### alibi-detect

[alibi-detect](https://github.com/SeldonIO/alibi-detect) provides
statistical drift tests (KS, MMD, Chi-squared, etc.) and outlier
detection.

```python
from alibi_detect.cd import KSDrift

from ml_platform.evaluation import EvaluationReporter, EvaluationResult, EvaluationStatus
from ml_platform.scheduling import scheduled


@scheduled(interval_s=3600, name="alibi-drift")
async def check_drift(self) -> None:
    reference = self._get_reference_data()   # numpy array
    current = self._get_current_data()       # numpy array

    detector = KSDrift(reference, p_val=0.05)
    result = detector.predict(current)

    is_drift = bool(result["data"]["is_drift"])
    p_val = float(result["data"]["p_val"].mean())

    self.reporter.report(EvaluationResult(
        name="ks-drift",
        status=EvaluationStatus.FAILED if is_drift else EvaluationStatus.PASSED,
        metrics={
            "is_drift": float(is_drift),
            "mean_p_value": p_val,
        },
        details={
            "per_feature_p_values": result["data"]["p_val"].tolist(),
            "threshold": result["data"]["threshold"],
        },
    ))
```

### whylogs

[whylogs](https://github.com/whylabs/whylogs) generates lightweight
statistical profiles of your data that can be compared across time
windows.

```python
import whylogs as why

from ml_platform.evaluation import EvaluationReporter, EvaluationResult, EvaluationStatus
from ml_platform.scheduling import scheduled


@scheduled(interval_s=3600, name="whylogs-profile")
async def profile_data(self) -> None:
    current_df = self._get_current_dataframe()

    profile = why.log(current_df).profile()
    view = profile.view()
    summary = view.to_pandas()

    null_fraction = summary["types/null"].sum() / max(summary["counts/n"].sum(), 1)
    type_mismatch = summary["types/fractional"].sum()

    self.reporter.report(EvaluationResult(
        name="data-profile",
        status=(EvaluationStatus.FAILED if null_fraction > 0.1
                else EvaluationStatus.WARNING if null_fraction > 0.05
                else EvaluationStatus.PASSED),
        metrics={
            "null_fraction": null_fraction,
            "type_mismatches": float(type_mismatch),
        },
        tags={"profiler": "whylogs"},
    ))
```

---

## API reference

::: ml_platform.evaluation.EvaluationStatus

::: ml_platform.evaluation.EvaluationResult

::: ml_platform.evaluation.EvaluationReporter
