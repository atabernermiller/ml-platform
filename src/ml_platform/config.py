"""Centralized configuration for ML platform services.

``ServiceConfig`` is the single source of truth passed to app factories,
CDK constructs, and monitoring utilities. Values can be set via constructor
arguments or overridden from environment variables at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ServiceConfig:
    """Shared configuration consumed by all ml-platform components.

    Attributes:
        service_name: Unique name for this service (used in metrics, logs, S3 paths,
            and AWS resource naming).
        aws_region: AWS region for SDK calls and resource provisioning.
        s3_checkpoint_bucket: S3 bucket for state checkpoints. Empty disables
            checkpointing.
        s3_checkpoint_prefix: Key prefix within the checkpoint bucket.
        checkpoint_interval_s: Seconds between automatic state snapshots.
        metrics_interval_s: Seconds between automatic metric emission cycles.
        mlflow_tracking_uri: MLflow tracking server URI. Empty disables MLflow.
        mlflow_experiment_name: MLflow experiment name; defaults to ``service_name``.
        state_backend: Context-store backend technology.
        state_table_name: DynamoDB table or Redis key prefix for context storage.
        state_ttl_s: TTL for stored prediction contexts (seconds). 0 disables expiry.
    """

    service_name: str
    aws_region: str = "us-east-1"
    s3_checkpoint_bucket: str = ""
    s3_checkpoint_prefix: str = "checkpoints/"
    checkpoint_interval_s: int = 300
    metrics_interval_s: int = 60
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = ""
    state_backend: Literal["dynamodb", "redis"] = "dynamodb"
    state_table_name: str = ""
    state_ttl_s: int = 86_400

    def __post_init__(self) -> None:
        if not self.mlflow_experiment_name:
            object.__setattr__(self, "mlflow_experiment_name", self.service_name)
        if not self.state_table_name:
            object.__setattr__(self, "state_table_name", f"{self.service_name}-context")

    @classmethod
    def from_env(cls, **overrides: object) -> ServiceConfig:
        """Construct from environment variables with optional overrides.

        Environment variables are uppercased versions of field names, prefixed
        with ``ML_PLATFORM_`` (e.g., ``ML_PLATFORM_SERVICE_NAME``).

        Args:
            **overrides: Explicit values that take precedence over env vars.

        Returns:
            Populated configuration instance.
        """
        env_map: dict[str, object] = {}
        for field_name in cls.__dataclass_fields__:
            env_key = f"ML_PLATFORM_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == "int":
                    env_map[field_name] = int(env_val)
                elif field_type == "bool":
                    env_map[field_name] = env_val.lower() in ("1", "true", "yes")
                else:
                    env_map[field_name] = env_val
        env_map.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env_map)  # type: ignore[arg-type]
