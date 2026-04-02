"""In-process scheduled task system for periodic ML operations.

Provides a ``@scheduled`` decorator for defining periodic tasks on
service classes, and a :class:`TaskRunner` that manages their
background execution within the existing serving process.

Typical uses:

- Batch retraining on accumulated feedback
- Periodic evaluation / drift detection
- Cost and usage reporting
- Data cleanup (expired contexts, stale sessions)
- Embedding or index refresh

Example::

    from ml_platform.scheduling import scheduled

    class MyBandit(StatefulServiceBase):

        @scheduled(interval_s=3600, name="hourly-retrain")
        async def retrain(self) -> None:
            \"\"\"Retrain the model on recent feedback every hour.\"\"\"
            ...

        @scheduled(interval_s=86400, name="daily-eval")
        async def evaluate(self) -> None:
            \"\"\"Run evaluation suite once a day.\"\"\"
            ...

Tasks are automatically discovered and started when the service boots.
They stop gracefully on shutdown, finishing the current execution before
exiting.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_SCHEDULE_ATTR = "_ml_platform_schedule"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScheduledTask:
    """Descriptor for a single periodic task.

    Attributes:
        name: Human-readable name used in logs and metrics.
        fn: The async callable to execute.
        interval_s: Seconds between executions (measured from the end of
            the previous execution, not from the start).
        run_on_startup: If ``True``, execute immediately on startup before
            entering the periodic loop.
        max_retries: Number of retries on failure before skipping to the
            next interval.  ``0`` means no retries.
    """

    name: str
    fn: Callable[..., Awaitable[None]]
    interval_s: float
    run_on_startup: bool = False
    max_retries: int = 0
    retry_backoff_s: float = 1.0


@dataclass
class TaskExecution:
    """Record of a single task execution for observability.

    Attributes:
        task_name: Name of the task.
        started_at: Unix timestamp when execution began.
        duration_s: Wall-clock duration in seconds.
        success: ``True`` if the task completed without error.
        error: Error message if the task failed.
        attempt: Which attempt this was (1-based).
    """

    task_name: str
    started_at: float = 0.0
    duration_s: float = 0.0
    success: bool = True
    error: str = ""
    attempt: int = 1


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def scheduled(
    interval_s: float,
    *,
    name: str = "",
    run_on_startup: bool = False,
    max_retries: int = 0,
) -> Callable[..., Any]:
    """Mark an async method as a scheduled periodic task.

    The runtime discovers decorated methods on startup and runs them in
    the background.  Tasks share the same process as the serving loop
    so they should be cooperative (yield to the event loop regularly).

    Args:
        interval_s: Seconds between executions.
        name: Task name for logs/metrics.  Defaults to the method name.
        run_on_startup: Execute immediately on startup.
        max_retries: Retry count on failure before skipping to next interval.

    Returns:
        Decorated method (unchanged behaviour, metadata attached).

    Example::

        @scheduled(interval_s=3600)
        async def retrain(self) -> None:
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        task_name = name or fn.__name__
        setattr(fn, _SCHEDULE_ATTR, {
            "name": task_name,
            "interval_s": interval_s,
            "run_on_startup": run_on_startup,
            "max_retries": max_retries,
        })
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def discover_tasks(service: Any) -> list[ScheduledTask]:
    """Scan a service instance for ``@scheduled``-decorated methods.

    Args:
        service: A service instance (e.g. ``StatefulServiceBase`` subclass).

    Returns:
        List of :class:`ScheduledTask` descriptors, one per decorated method.
    """
    tasks: list[ScheduledTask] = []
    for attr_name in dir(service):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(service, attr_name)
        except Exception:
            continue
        meta = getattr(attr, _SCHEDULE_ATTR, None)
        if meta is not None:
            if not (asyncio.iscoroutinefunction(attr) or inspect.iscoroutinefunction(attr)):
                logger.warning(
                    "Scheduled task %r is not async; skipping. "
                    "Use 'async def' for scheduled tasks.",
                    meta["name"],
                )
                continue
            tasks.append(ScheduledTask(
                name=meta["name"],
                fn=attr,
                interval_s=meta["interval_s"],
                run_on_startup=meta["run_on_startup"],
                max_retries=meta["max_retries"],
            ))
    return tasks


# ---------------------------------------------------------------------------
# TaskRunner
# ---------------------------------------------------------------------------


class TaskRunner:
    """Manages the lifecycle of scheduled background tasks.

    Create one per runtime.  Call :meth:`start` after the service is
    initialized and :meth:`stop` during shutdown.

    Args:
        emitter: Optional metrics emitter for task-level observability.
        service_name: Service name for metric dimensions.
    """

    def __init__(
        self,
        emitter: Any | None = None,
        service_name: str = "",
    ) -> None:
        self._emitter = emitter
        self._service_name = service_name
        self._tasks: list[ScheduledTask] = []
        self._handles: list[asyncio.Task[None]] = []
        self._executions: collections.deque[TaskExecution] = collections.deque(maxlen=1000)
        self._running: bool = False

    @property
    def tasks(self) -> list[ScheduledTask]:
        """Registered tasks."""
        return list(self._tasks)

    @property
    def recent_executions(self) -> list[TaskExecution]:
        """Recent task execution records (most recent first, capped at 100)."""
        return list(reversed(list(self._executions)[-100:]))

    def register(self, task: ScheduledTask) -> None:
        """Add a task to be managed.

        Args:
            task: Task descriptor.
        """
        self._tasks.append(task)
        logger.info(
            "Registered scheduled task: %s (every %ds%s)",
            task.name,
            task.interval_s,
            ", run_on_startup" if task.run_on_startup else "",
        )

    def register_all(self, tasks: list[ScheduledTask]) -> None:
        """Register multiple tasks at once."""
        for t in tasks:
            self.register(t)

    async def start(self) -> None:
        """Start all registered tasks as background coroutines."""
        self._running = True
        for task in self._tasks:
            handle = asyncio.create_task(
                self._run_loop(task), name=f"sched:{task.name}"
            )
            self._handles.append(handle)
        if self._tasks:
            logger.info(
                "Started %d scheduled task(s) for %s",
                len(self._tasks),
                self._service_name,
            )

    async def stop(self) -> None:
        """Cancel all running task loops and wait for them to finish."""
        self._running = False
        for handle in self._handles:
            handle.cancel()
        if self._handles:
            await asyncio.gather(*self._handles, return_exceptions=True)
        self._handles.clear()
        logger.info("Stopped all scheduled tasks")

    def metrics_snapshot(self) -> dict[str, float]:
        """Return aggregate metrics for all managed tasks."""
        snapshot: dict[str, float] = {}
        snapshot["scheduled_tasks_registered"] = float(len(self._tasks))

        total_runs = 0
        total_failures = 0
        for ex in self._executions:
            total_runs += 1
            if not ex.success:
                total_failures += 1
        snapshot["scheduled_tasks_total_runs"] = float(total_runs)
        snapshot["scheduled_tasks_total_failures"] = float(total_failures)
        return snapshot

    # -- internal loop -------------------------------------------------------

    async def _run_loop(self, task: ScheduledTask) -> None:
        """Execute a task periodically until cancelled."""
        if task.run_on_startup:
            await self._execute_with_retries(task)

        while self._running:
            try:
                await asyncio.sleep(task.interval_s)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            await self._execute_with_retries(task)

    async def _execute_with_retries(self, task: ScheduledTask) -> None:
        """Run the task, retrying on failure up to max_retries."""
        for attempt in range(1, task.max_retries + 2):
            execution = await self._execute_once(task, attempt)
            self._executions.append(execution)
            self._emit_task_metrics(execution)

            if execution.success:
                return
            if attempt <= task.max_retries:
                backoff = min(task.retry_backoff_s * (2 ** (attempt - 1)), 60)
                logger.warning(
                    "Task %s failed (attempt %d/%d), retrying in %ds",
                    task.name, attempt, task.max_retries + 1, backoff,
                )
                await asyncio.sleep(backoff)

    async def _execute_once(
        self, task: ScheduledTask, attempt: int
    ) -> TaskExecution:
        start = time.monotonic()
        started_at = time.time()
        try:
            await task.fn()
            duration = time.monotonic() - start
            logger.info(
                "Task %s completed in %.1fs", task.name, duration,
            )
            return TaskExecution(
                task_name=task.name,
                started_at=started_at,
                duration_s=duration,
                success=True,
                attempt=attempt,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            duration = time.monotonic() - start
            logger.exception("Task %s failed after %.1fs", task.name, duration)
            return TaskExecution(
                task_name=task.name,
                started_at=started_at,
                duration_s=duration,
                success=False,
                error=str(exc),
                attempt=attempt,
            )

    def _emit_task_metrics(self, execution: TaskExecution) -> None:
        if self._emitter is None:
            return
        try:
            self._emitter.emit_event(
                "scheduled_task",
                dimensions={
                    "service": self._service_name,
                    "task": execution.task_name,
                },
                values={
                    "duration_s": execution.duration_s,
                    "success": 1.0 if execution.success else 0.0,
                    "attempt": float(execution.attempt),
                },
            )
        except Exception:
            logger.debug("Failed to emit task metrics", exc_info=True)
