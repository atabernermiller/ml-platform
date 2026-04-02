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
