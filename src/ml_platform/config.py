"""Centralized configuration for ML platform services.

``ServiceConfig`` is the single source of truth passed to app factories,
CDK constructs, and monitoring utilities.  Values can be set via constructor
arguments or overridden from environment variables at runtime.

The configuration uses nested sub-objects to keep service-type-specific
fields separate:

- ``StatefulConfig`` for online-learning / feedback-loop services.
- ``LLMConfig`` for single-call LLM services.
- ``AgentConfig`` for multi-step agentic services.

Only the relevant sub-object needs to be populated; the rest default to
``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal

_FALLBACK_REGION = "us-east-1"


def resolve_region(explicit: str | None = None) -> str:
    """Resolve the AWS region from a priority chain.

    Resolution order:

    1. *explicit* -- caller-supplied value (always wins when truthy).
    2. ``AWS_REGION`` environment variable (set by ECS, Lambda, and the
       AWS SDK on managed compute).
    3. ``AWS_DEFAULT_REGION`` environment variable (conventional ``aws
       configure`` default).
    4. Hard-coded fallback (``us-east-1``).

    Args:
        explicit: An explicitly provided region string.  Empty strings
            and ``None`` are treated as "not specified".

    Returns:
        A non-empty AWS region string.
    """
    if explicit:
        return explicit
    if env := os.environ.get("AWS_REGION"):
        return env
    if env := os.environ.get("AWS_DEFAULT_REGION"):
        return env
    return _FALLBACK_REGION

if TYPE_CHECKING:
    from ml_platform._interfaces import Profile
    from ml_platform.alerting import AlertRule


# ---------------------------------------------------------------------------
# Service-type-specific configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatefulConfig:
    """Configuration for :class:`StatefulServiceBase` apps.

    Attributes:
        checkpoint_interval_s: Seconds between automatic state snapshots.
        s3_checkpoint_bucket: S3 bucket for state checkpoints.  Empty
            string disables checkpointing.
        s3_checkpoint_prefix: Key prefix within the checkpoint bucket.
    """

    checkpoint_interval_s: int = 300
    s3_checkpoint_bucket: str = ""
    s3_checkpoint_prefix: str = "checkpoints/"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for :class:`LLMServiceBase` apps.

    Attributes:
        default_model: Default model identifier used when the caller does
            not specify one.
        token_budget_daily: Daily token budget across all models.  ``0``
            disables budget enforcement.
        cost_alert_threshold_usd: Emit an alert when cumulative daily cost
            exceeds this value.  ``0.0`` disables alerts.
        conversation_table_name: DynamoDB table for conversation history.
            Empty string disables automatic conversation storage.
        conversation_ttl_s: TTL for conversation messages in seconds
            (default: 7 days).
    """

    default_model: str = ""
    token_budget_daily: int = 0
    cost_alert_threshold_usd: float = 0.0
    conversation_table_name: str = ""
    conversation_ttl_s: int = 604_800


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for :class:`AgentServiceBase` apps.

    Attributes:
        max_steps_per_run: Upper limit on the number of steps (LLM calls
            plus tool executions) per agent run.  Prevents runaway loops.
        tool_timeout_s: Maximum wall-clock time for a single tool execution.
        max_concurrent_tool_calls: Upper limit on tools running in parallel
            within a single run.
        conversation_table_name: DynamoDB table for conversation history.
            Empty string disables automatic conversation storage.
        conversation_ttl_s: TTL for conversation messages in seconds
            (default: 7 days).
    """

    max_steps_per_run: int = 20
    tool_timeout_s: float = 30.0
    max_concurrent_tool_calls: int = 5
    conversation_table_name: str = ""
    conversation_ttl_s: int = 604_800


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceConfig:
    """Shared configuration consumed by all ml-platform components.

    AWS credentials are **not** part of this configuration.  All boto3
    clients created by the library use the `default credential chain
    <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>`_
    (environment variables, ``~/.aws/credentials``, ECS task role, EC2
    instance profile).  Only ``aws_region`` is specified here.

    Attributes:
        service_name: Unique name for this service (used in metrics, logs,
            S3 paths, and AWS resource naming).
        aws_region: AWS region for SDK calls and resource provisioning.
        metrics_interval_s: Seconds between automatic metric emission cycles.
        mlflow_tracking_uri: MLflow tracking server URI.  Empty disables.
        mlflow_experiment_name: MLflow experiment name; defaults to
            ``service_name``.
        log_level: Log level for the ``ml_platform`` logger and root
            logger (e.g. ``"DEBUG"``, ``"INFO"``).
        log_format: Log output format: ``"json"`` for production,
            ``"text"`` for development, ``""`` (empty) to skip
            auto-configuration and let the user manage logging.
        alerts: List of :class:`~ml_platform.alerting.AlertRule`
            instances to evaluate on every metric cycle.
        alert_webhook_url: URL to POST JSON alert events to (e.g.
            Slack webhook, PagerDuty endpoint).  Empty disables
            webhook notifications.
        profile: Cloud profile that bundles backend implementations.
            ``None`` means auto-detect (local if no AWS credentials,
            AWS otherwise).
        stateful: Configuration for stateful (online-learning) services.
        llm: Configuration for single-call LLM services.
        agent: Configuration for multi-step agentic services.

        s3_checkpoint_bucket: **Deprecated** -- use ``stateful.s3_checkpoint_bucket``.
        s3_checkpoint_prefix: **Deprecated** -- use ``stateful.s3_checkpoint_prefix``.
        checkpoint_interval_s: **Deprecated** -- use ``stateful.checkpoint_interval_s``.
        state_backend: Context-store backend technology.
        state_table_name: DynamoDB table or Redis key prefix for context storage.
        state_ttl_s: TTL for stored prediction contexts (seconds).
    """

    service_name: str
    aws_region: str = ""
    metrics_interval_s: int = 60
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = ""
    log_level: str = "INFO"
    log_format: Literal["json", "text", ""] = ""
    alerts: list[AlertRule] = field(default_factory=list)
    alert_webhook_url: str = ""

    profile: Profile | None = None

    stateful: StatefulConfig | None = None
    llm: LLMConfig | None = None
    agent: AgentConfig | None = None

    # Legacy fields -- kept for backward compatibility.
    s3_checkpoint_bucket: str = ""
    s3_checkpoint_prefix: str = "checkpoints/"
    checkpoint_interval_s: int = 300
    state_backend: Literal["dynamodb", "redis"] = "dynamodb"
    state_table_name: str = ""
    state_ttl_s: int = 86_400

    _REDACTED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "alert_webhook_url",
        "mlflow_tracking_uri",
    })

    def __post_init__(self) -> None:
        if not self.aws_region:
            object.__setattr__(self, "aws_region", resolve_region())
        if not self.mlflow_experiment_name:
            object.__setattr__(self, "mlflow_experiment_name", self.service_name)
        if not self.state_table_name:
            object.__setattr__(
                self, "state_table_name", f"{self.service_name}-context"
            )

        # Migrate legacy top-level checkpoint fields into StatefulConfig
        # when the caller used the old-style flat API.
        if self.stateful is None and self.s3_checkpoint_bucket:
            object.__setattr__(
                self,
                "stateful",
                StatefulConfig(
                    checkpoint_interval_s=self.checkpoint_interval_s,
                    s3_checkpoint_bucket=self.s3_checkpoint_bucket,
                    s3_checkpoint_prefix=self.s3_checkpoint_prefix,
                ),
            )

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in self._REDACTED_FIELDS and val:
                val = "***"
            parts.append(f"{f.name}={val!r}")
        return f"ServiceConfig({', '.join(parts)})"

    @classmethod
    def from_env(cls, **overrides: object) -> ServiceConfig:
        """Construct from environment variables with optional overrides.

        Environment variables are uppercased versions of field names,
        prefixed with ``ML_PLATFORM_`` (e.g., ``ML_PLATFORM_SERVICE_NAME``).

        Nested config objects are populated from environment variables
        with an extra segment:

        - ``ML_PLATFORM_AGENT_MAX_STEPS_PER_RUN``
        - ``ML_PLATFORM_LLM_DEFAULT_MODEL``
        - ``ML_PLATFORM_STATEFUL_CHECKPOINT_INTERVAL_S``

        Args:
            **overrides: Explicit values that take precedence over env vars.

        Returns:
            Populated configuration instance.
        """
        env_map: dict[str, object] = {}
        for f in fields(cls):
            if f.name in ("profile", "stateful", "llm", "agent", "alerts"):
                continue
            env_key = f"ML_PLATFORM_{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                env_map[f.name] = _coerce(f.type, env_val)

        agent_env = _read_nested_env("ML_PLATFORM_AGENT_", AgentConfig)
        if agent_env:
            env_map["agent"] = AgentConfig(**agent_env)

        llm_env = _read_nested_env("ML_PLATFORM_LLM_", LLMConfig)
        if llm_env:
            env_map["llm"] = LLMConfig(**llm_env)

        stateful_env = _read_nested_env("ML_PLATFORM_STATEFUL_", StatefulConfig)
        if stateful_env:
            env_map["stateful"] = StatefulConfig(**stateful_env)

        env_map.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env_map)  # type: ignore[arg-type]


_INT_PATTERNS = {"int", "int | None", "None | int", "Optional[int]"}
_FLOAT_PATTERNS = {"float", "float | None", "None | float", "Optional[float]"}
_BOOL_PATTERNS = {"bool", "bool | None", "None | bool", "Optional[bool]"}


def _coerce(type_hint: str, raw: str) -> object:
    """Best-effort coercion from env-var string to the declared type.

    Uses exact pattern matching on the stringified type hint to avoid
    false positives from substring checks (e.g. a type containing
    ``"int"`` in its name).
    """
    hint = type_hint.strip()

    if hint in _BOOL_PATTERNS:
        return raw.lower() in ("1", "true", "yes")

    base = hint.replace("Optional[", "").replace("]", "").replace(" | None", "").replace("None | ", "").strip()

    if base == "int":
        try:
            return int(raw)
        except ValueError:
            return raw
    if base == "float":
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _read_nested_env(
    prefix: str, dataclass_cls: type[Any]
) -> dict[str, object]:
    """Read environment variables matching *prefix* + field name."""
    result: dict[str, object] = {}
    for f in fields(dataclass_cls):
        env_key = f"{prefix}{f.name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            result[f.name] = _coerce(f.type, env_val)
    return result
