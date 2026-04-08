"""Abstract interfaces and protocols for the ml-platform library.

Every pluggable component is defined here in Layer 1 so that concrete
implementations in Layer 2 (profiles, adapters, backends) and consumers
in Layer 3 (serving) program against stable contracts.

Adding a new backend (e.g. a GCP profile) means writing classes that
satisfy these interfaces -- no existing code needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from ml_platform._types import Completion, Message

# Re-export for convenience
__all__ = [
    "MetricsBackend",
    "StateManager",
    "ContextStore",
    "ConversationStore",
    "ExperimentTracker",
    "LLMProvider",
    "Tool",
    "Profile",
    "FileStore",
    "EmailBackend",
    "CDNBackend",
    "SecretResolver",
    "QueueBackend",
    "Table",
    "FeatureGate",
    "EventBus",
    "UserPool",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricsBackend(ABC):
    """Interface for emitting numeric metrics to an observability backend.

    Implementations exist for CloudWatch EMF, Prometheus, and console
    output.  The serving runtime delegates all metric emission to the
    backend provided by the active :class:`Profile`.
    """

    @abstractmethod
    def emit(self, metrics: dict[str, float]) -> None:
        """Emit a batch of named metrics.

        Args:
            metrics: Mapping of metric names to numeric values.
        """
        ...

    @abstractmethod
    def emit_event(
        self,
        event_name: str,
        dimensions: dict[str, str],
        values: dict[str, float],
    ) -> None:
        """Emit a single event with custom dimensions.

        Use for per-request metrics where you need dimensions beyond the
        service name (e.g. model, provider, step type).

        Args:
            event_name: Logical event name for log categorisation.
            dimensions: Key-value dimension pairs.
            values: Metric name-value pairs.
        """
        ...


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class StateManager(ABC):
    """Interface for uploading and downloading model-state checkpoints.

    Implementations exist for S3 and local filesystem.
    """

    @abstractmethod
    def upload(self, local_dir: str) -> str:
        """Upload a local checkpoint directory.

        Args:
            local_dir: Path to directory containing checkpoint files.

        Returns:
            Backend-specific identifier for the stored checkpoint.
        """
        ...

    @abstractmethod
    def download_latest(self) -> str | None:
        """Download the most recent checkpoint to a temporary directory.

        Returns:
            Path to the temporary directory, or ``None`` if no checkpoints
            exist.  The caller owns the directory.
        """
        ...

    @abstractmethod
    def list_checkpoints(self) -> list[str]:
        """List available checkpoint identifiers, oldest first.

        Returns:
            Sorted list of checkpoint identifiers.
        """
        ...


# ---------------------------------------------------------------------------
# Context store (predict-feedback correlation)
# ---------------------------------------------------------------------------


class ContextStore(ABC):
    """Interface for storing prediction context between predict and feedback.

    Consume-once semantics: ``get`` retrieves **and deletes** the entry.
    """

    @abstractmethod
    def put(self, request_id: str, context: dict[str, Any]) -> None:
        """Store context for a prediction request.

        Args:
            request_id: Unique prediction identifier.
            context: Serialisable context data.
        """
        ...

    @abstractmethod
    def get(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve and delete context for a request.

        Args:
            request_id: Unique prediction identifier.

        Returns:
            Stored context, or ``None`` if not found or expired.
        """
        ...


# ---------------------------------------------------------------------------
# Conversation store (multi-turn session history)
# ---------------------------------------------------------------------------


class ConversationStore(ABC):
    """Interface for append-only message history with windowing.

    Unlike :class:`ContextStore` (consume-once), conversation stores
    accumulate messages across turns and support windowing strategies
    to bound context length.
    """

    @abstractmethod
    def append(self, session_id: str, message: Message) -> None:
        """Append a message to a session's history.

        Args:
            session_id: Session identifier.
            message: Message to append.
        """
        ...

    @abstractmethod
    def get_history(
        self,
        session_id: str,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        """Retrieve conversation history for a session.

        Args:
            session_id: Session identifier.
            max_messages: Return at most this many recent messages.
            max_tokens: Approximate token budget; truncate oldest messages
                to fit.

        Returns:
            Ordered list of messages (oldest first).
        """
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all messages for a session.

        Args:
            session_id: Session identifier.
        """
        ...


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


class ExperimentTracker(ABC):
    """Interface for experiment tracking backends.

    Implementations manage a single logical *run* and expose methods for
    logging parameters, metrics, and artifacts.

    .. note::

        The canonical implementation lives in
        ``ml_platform.tracking.base.ExperimentTracker``.  This duplicate
        definition exists so that Layer 1 can reference the interface
        without importing Layer 2.  The two are kept in sync -- the
        ``tracking.base`` version is authoritative.
    """

    @property
    @abstractmethod
    def run_id(self) -> str:
        """Unique identifier of the active run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Record hyperparameters for the active run."""
        ...

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Record a batch of numeric metrics."""
        ...

    @abstractmethod
    def log_artifact(
        self, local_path: str, artifact_subdir: str = ""
    ) -> None:
        """Upload a local file or directory as a run artifact."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """Finalize the active run."""
        ...


# ---------------------------------------------------------------------------
# LLM provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that any LLM calling wrapper must satisfy.

    The library never calls a provider SDK directly.  Instead, users
    implement (or use a shipped adapter for) this two-method protocol
    and the library wraps it with instrumentation.

    Example::

        class MyOpenAIProvider:
            async def complete(
                self, messages: list[Message], *, model: str = "gpt-4o", **kwargs
            ) -> Completion:
                response = await openai_client.chat.completions.create(
                    model=model, messages=[m.model_dump() for m in messages],
                )
                return Completion(
                    content=response.choices[0].message.content,
                    model=model,
                    provider="openai",
                    usage=CompletionUsage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    ),
                )
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str = "",
        **kwargs: Any,
    ) -> Completion:
        """Execute a chat completion.

        Args:
            messages: Conversation history.
            model: Model identifier (provider-specific).
            **kwargs: Additional provider-specific parameters.

        Returns:
            Normalised completion result.
        """
        ...


# ---------------------------------------------------------------------------
# Tool protocol (for agentic workflows)
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that an agent can execute.

    Users implement this for each tool they want an agent to use.
    The library wraps tool execution with observability (OTel spans,
    metrics, error handling) but never implements the tool logic itself.

    Example::

        class SearchDocsTool:
            name = "search_docs"
            description = "Search the knowledge base for relevant documents"
            parameters_schema = {"query": {"type": "string"}}

            async def execute(self, **kwargs: Any) -> str:
                query = kwargs.get("query", "")
                results = await search_engine.search(query)
                return "\\n".join(r.title for r in results)
    """

    @property
    def name(self) -> str:
        """Unique name for this tool."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON-Schema-like dict describing accepted keyword arguments."""
        ...

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result.

        Args:
            **kwargs: Arguments matching :attr:`parameters_schema`.

        Returns:
            String output suitable for inclusion in an LLM prompt.

        Raises:
            Exception: Any exception is caught by the :class:`RunContext`,
                recorded as an error step, and does not crash the agent run.
        """
        ...


# ---------------------------------------------------------------------------
# Cloud profile protocol
# ---------------------------------------------------------------------------


class Profile(Protocol):
    """Protocol bundling backend implementations for a deployment target.

    A profile is a pre-configured set of backends for a specific
    environment (local development, AWS, GCP, etc.).  The user picks a
    profile and the library wires all components automatically.

    ``LocalProfile`` uses console metrics, SQLite stores, and file-based
    state.  ``AWSProfile`` uses CloudWatch, DynamoDB, S3, and X-Ray.
    Each is self-contained and depends only on Layer 1 interfaces.
    """

    def create_metrics_backend(
        self, service_name: str, region: str
    ) -> MetricsBackend:
        """Create a metrics backend for this environment."""
        ...

    def create_state_manager(
        self, bucket: str, prefix: str, region: str
    ) -> StateManager | None:
        """Create a state manager, or ``None`` if checkpointing is disabled."""
        ...

    def create_context_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ContextStore:
        """Create a context store for predict-feedback correlation."""
        ...

    def create_conversation_store(
        self, table_name: str, region: str, ttl_s: int
    ) -> ConversationStore:
        """Create a conversation store for multi-turn sessions."""
        ...


# ---------------------------------------------------------------------------
# File storage
# ---------------------------------------------------------------------------


class FileStore(ABC):
    """Interface for general-purpose file storage.

    Implementations exist for S3 (with optional CloudFront CDN) and
    local filesystem.  Unlike :class:`StateManager`, which is scoped to
    checkpoint directories, ``FileStore`` handles individual files with
    content-type awareness, presigned URLs, and public URL generation.
    """

    @abstractmethod
    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Upload a file and return its storage key.

        Args:
            key: Logical path / key for the file.
            data: Raw file bytes.
            content_type: MIME type of the file.

        Returns:
            The canonical storage key.
        """
        ...

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Download a file by key.

        Args:
            key: Storage key returned by :meth:`put`.

        Returns:
            File contents, or ``None`` if the key does not exist.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a file by key.

        Args:
            key: Storage key.

        Returns:
            ``True`` if the file was deleted, ``False`` if it did not exist.
        """
        ...

    @abstractmethod
    def presigned_upload_url(self, key: str, *, expires_in_s: int = 3600, content_type: str = "application/octet-stream") -> str:
        """Generate a presigned URL for direct browser upload.

        Args:
            key: Destination key.
            expires_in_s: URL expiry in seconds.
            content_type: Expected content type.

        Returns:
            Presigned PUT URL.
        """
        ...

    @abstractmethod
    def presigned_download_url(self, key: str, *, expires_in_s: int = 3600) -> str:
        """Generate a presigned URL for direct browser download.

        Args:
            key: File key.
            expires_in_s: URL expiry in seconds.

        Returns:
            Presigned GET URL.
        """
        ...

    @abstractmethod
    def public_url(self, key: str) -> str:
        """Return a public URL for the file.

        Args:
            key: File key.

        Returns:
            Publicly accessible URL (CDN or direct S3).
        """
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys under the given prefix.

        Args:
            prefix: Key prefix to filter by.

        Returns:
            List of matching keys.
        """
        ...


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


class EmailBackend(ABC):
    """Interface for transactional email delivery.

    Implementations exist for AWS SES and a console/log mock for
    local development.
    """

    @abstractmethod
    def send(
        self,
        *,
        to: list[str],
        subject: str,
        body_text: str,
        body_html: str = "",
        from_addr: str = "",
        reply_to: list[str] | None = None,
    ) -> str:
        """Send a transactional email.

        Args:
            to: Recipient email addresses.
            subject: Email subject line.
            body_text: Plain-text body.
            body_html: Optional HTML body.
            from_addr: Sender address (uses default if empty).
            reply_to: Optional reply-to addresses.

        Returns:
            Message identifier from the email backend.
        """
        ...


# ---------------------------------------------------------------------------
# CDN
# ---------------------------------------------------------------------------


class CDNBackend(ABC):
    """Interface for CDN operations (cache invalidation, signed URLs).

    The primary implementation wraps AWS CloudFront.
    """

    @abstractmethod
    def signed_url(self, path: str, *, expires_in_s: int = 3600) -> str:
        """Generate a signed CDN URL for a resource.

        Args:
            path: Resource path relative to the distribution origin.
            expires_in_s: URL expiry in seconds.

        Returns:
            Signed URL.
        """
        ...

    @abstractmethod
    def invalidate(self, paths: list[str]) -> str:
        """Invalidate cached objects at the given paths.

        Args:
            paths: List of CDN paths to invalidate (e.g. ``["/images/*"]``).

        Returns:
            Invalidation ID from the CDN provider.
        """
        ...

    @abstractmethod
    def public_url(self, path: str) -> str:
        """Return the public CDN URL for a path.

        Args:
            path: Resource path.

        Returns:
            Full CDN URL.
        """
        ...


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


class SecretResolver(ABC):
    """Interface for loading secrets at startup.

    Implementations exist for AWS Secrets Manager (with caching) and
    environment variables (for local development).
    """

    @abstractmethod
    def get(self, secret_id: str) -> str:
        """Retrieve a secret value.

        Args:
            secret_id: Secret identifier (name or ARN).

        Returns:
            Secret string value.

        Raises:
            KeyError: If the secret does not exist.
        """
        ...

    @abstractmethod
    def get_json(self, secret_id: str) -> dict[str, Any]:
        """Retrieve and parse a JSON secret.

        Args:
            secret_id: Secret identifier.

        Returns:
            Parsed JSON object.

        Raises:
            KeyError: If the secret does not exist.
        """
        ...


# ---------------------------------------------------------------------------
# Queue / event processing
# ---------------------------------------------------------------------------


class QueueBackend(ABC):
    """Interface for consuming messages from a queue.

    Implementations exist for AWS SQS and an in-memory queue for
    local development.
    """

    @abstractmethod
    def send(self, message: dict[str, Any], *, delay_s: int = 0) -> str:
        """Send a message to the queue.

        Args:
            message: JSON-serialisable message body.
            delay_s: Delivery delay in seconds.

        Returns:
            Message identifier.
        """
        ...

    @abstractmethod
    def receive(self, *, max_messages: int = 1, wait_time_s: int = 20) -> list[dict[str, Any]]:
        """Receive messages from the queue.

        Args:
            max_messages: Maximum number of messages to receive (1-10).
            wait_time_s: Long-poll wait time in seconds.

        Returns:
            List of message dicts with ``"body"`` and ``"receipt_handle"`` keys.
        """
        ...

    @abstractmethod
    def delete(self, receipt_handle: str) -> None:
        """Delete a processed message from the queue.

        Args:
            receipt_handle: Receipt handle from a received message.
        """
        ...


# ---------------------------------------------------------------------------
# General-purpose data table
# ---------------------------------------------------------------------------


class Table(ABC):
    """Interface for general-purpose key-value data access.

    Implementations exist for DynamoDB and an in-memory dict for
    local development.  Unlike :class:`ContextStore` (consume-once)
    or :class:`ConversationStore` (append-only), ``Table`` supports
    standard CRUD with optional secondary key queries.
    """

    @abstractmethod
    def put_item(self, item: dict[str, Any]) -> None:
        """Insert or replace an item.

        Args:
            item: Item data; must contain the partition key field.
        """
        ...

    @abstractmethod
    def get_item(self, key: dict[str, Any]) -> dict[str, Any] | None:
        """Retrieve an item by its key.

        Args:
            key: Key fields (partition key, and sort key if applicable).

        Returns:
            Item data, or ``None`` if not found.
        """
        ...

    @abstractmethod
    def delete_item(self, key: dict[str, Any]) -> bool:
        """Delete an item by its key.

        Args:
            key: Key fields.

        Returns:
            ``True`` if deleted, ``False`` if the item did not exist.
        """
        ...

    @abstractmethod
    def query(
        self,
        partition_key: str,
        partition_value: Any,
        *,
        sort_key_condition: tuple[str, str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query items by partition key with optional sort-key filtering.

        Args:
            partition_key: Name of the partition key attribute.
            partition_value: Value to match.
            sort_key_condition: Optional ``(attribute, operator, value)``
                tuple, where operator is ``"eq"``, ``"begins_with"``,
                ``"between"``, ``"lt"``, ``"lte"``, ``"gt"``, or ``"gte"``.
            limit: Maximum number of items to return.

        Returns:
            List of matching items.
        """
        ...

    @abstractmethod
    def scan(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Scan all items in the table.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of items.
        """
        ...


# ---------------------------------------------------------------------------
# Feature flags / A/B testing
# ---------------------------------------------------------------------------


class FeatureGate(ABC):
    """Interface for feature flags and gradual rollouts.

    Implementations exist for DynamoDB-backed flags and a static
    config-file backend for local development.
    """

    @abstractmethod
    def is_enabled(self, flag: str, *, context: dict[str, Any] | None = None) -> bool:
        """Check whether a feature flag is enabled.

        Args:
            flag: Flag identifier.
            context: Optional context for targeting rules (e.g. user_id,
                region, percentage bucket).

        Returns:
            ``True`` if the flag is enabled for the given context.
        """
        ...

    @abstractmethod
    def get_variant(self, flag: str, *, context: dict[str, Any] | None = None) -> str:
        """Return the active variant for an A/B test flag.

        Args:
            flag: Flag identifier.
            context: Optional targeting context.

        Returns:
            Variant name (e.g. ``"control"``, ``"treatment_a"``).
            Returns ``"control"`` if the flag is not found.
        """
        ...

    @abstractmethod
    def all_flags(self) -> dict[str, bool]:
        """Return all flags and their current enabled states.

        Returns:
            Mapping of flag names to boolean enabled states.
        """
        ...


# ---------------------------------------------------------------------------
# Event bus (pub/sub)
# ---------------------------------------------------------------------------


class EventBus(ABC):
    """Interface for publishing and subscribing to events.

    Implementations exist for AWS EventBridge and an in-memory bus for
    local development.  Unlike :class:`QueueBackend` (point-to-point),
    ``EventBus`` supports fan-out to multiple subscribers and
    pattern-based routing.
    """

    @abstractmethod
    def publish(
        self,
        source: str,
        detail_type: str,
        detail: dict[str, Any],
    ) -> str:
        """Publish an event.

        Args:
            source: Logical source of the event (e.g. ``"orders.service"``).
            detail_type: Event type identifier (e.g. ``"OrderCreated"``).
            detail: JSON-serialisable event payload.

        Returns:
            Event identifier.
        """
        ...

    @abstractmethod
    def publish_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        """Publish multiple events in a single call.

        Each entry must contain ``source``, ``detail_type``, and ``detail``
        keys matching the :meth:`publish` signature.

        Args:
            entries: List of event dicts.

        Returns:
            List of event identifiers (one per entry).
        """
        ...


# ---------------------------------------------------------------------------
# User pool / identity management
# ---------------------------------------------------------------------------


class UserPool(ABC):
    """Interface for user identity management (signup, password reset, MFA).

    The primary implementation wraps AWS Cognito User Pools.  A stub
    in-memory implementation is provided for local development.
    """

    @abstractmethod
    def create_user(
        self,
        username: str,
        *,
        email: str = "",
        phone: str = "",
        attributes: dict[str, str] | None = None,
        temporary_password: str = "",
    ) -> dict[str, Any]:
        """Create a new user in the pool.

        Args:
            username: Unique username.
            email: User email address.
            phone: User phone number (E.164 format).
            attributes: Additional user attributes.
            temporary_password: Optional temporary password. If empty,
                the backend generates one.

        Returns:
            Dict with at least ``"username"`` and ``"status"`` keys.
        """
        ...

    @abstractmethod
    def get_user(self, username: str) -> dict[str, Any] | None:
        """Retrieve a user's profile.

        Args:
            username: Username to look up.

        Returns:
            User profile dict, or ``None`` if not found.
        """
        ...

    @abstractmethod
    def delete_user(self, username: str) -> bool:
        """Delete a user from the pool.

        Args:
            username: Username to delete.

        Returns:
            ``True`` if deleted, ``False`` if the user did not exist.
        """
        ...

    @abstractmethod
    def reset_password(self, username: str) -> bool:
        """Initiate a password reset for a user.

        Args:
            username: Target username.

        Returns:
            ``True`` if the reset was initiated.
        """
        ...

    @abstractmethod
    def list_users(self, *, limit: int = 60) -> list[dict[str, Any]]:
        """List users in the pool.

        Args:
            limit: Maximum number of users to return.

        Returns:
            List of user profile dicts.
        """
        ...
