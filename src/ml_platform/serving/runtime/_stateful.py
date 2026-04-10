"""Stateful runtime for online-learning services with checkpointing."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from typing import Any, Type

from ml_platform._interfaces import ContextStore, StateManager
from ml_platform.config import ServiceConfig
from ml_platform.health import HealthCheck
from ml_platform.serving.runtime._base import BaseRuntime

logger = logging.getLogger(__name__)


class StatefulRuntime(BaseRuntime):
    """Runtime for :class:`~ml_platform.serving.stateful.StatefulServiceBase`.

    Extends :class:`BaseRuntime` with S3 state restoration, periodic
    checkpoint loops, and feedback delegation.

    Args:
        service_cls: Concrete :class:`StatefulServiceBase` subclass.
        config: Platform-wide configuration.
        service_kwargs: Extra keyword arguments forwarded to ``service_cls()``.
    """

    def __init__(
        self,
        service_cls: Type[Any],
        config: ServiceConfig,
        *,
        service_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._service_cls = service_cls
        self._service_kwargs = service_kwargs or {}
        self._service: Any | None = None
        self._state_mgr: StateManager | None = None
        self._context_store: ContextStore | None = None

    @property
    def service(self) -> Any:
        """The underlying service instance.

        Raises:
            RuntimeError: If accessed before :meth:`startup` completes.
        """
        if self._service is None:
            raise RuntimeError("StatefulRuntime has not been started")
        return self._service

    def _get_service_for_scheduling(self) -> Any | None:
        return self._service

    def _register_health_checks(self) -> None:
        if self._health_registry is None:
            return

        self._health_registry.register(HealthCheck(
            name="service_instance",
            check=lambda: self._service is not None,
            critical=True,
            description="Service class instantiated",
        ))
        if self._context_store is not None:
            store = self._context_store
            self._health_registry.register(HealthCheck(
                name="context_store",
                check=lambda: store is not None,
                critical=False,
                description="Context store available",
            ))
        if self._state_mgr is not None:
            self._health_registry.register(HealthCheck(
                name="s3_state_manager",
                check=lambda: self._state_mgr is not None,
                critical=False,
                description="S3 state manager connected",
            ))

    async def _on_startup(self) -> None:
        service = self._service_cls(**self._service_kwargs)
        self._service = service

        bucket = ""
        prefix = "checkpoints/"
        if self._config.stateful:
            bucket = self._config.stateful.s3_checkpoint_bucket
            prefix = self._config.stateful.s3_checkpoint_prefix
        elif self._config.s3_checkpoint_bucket:
            bucket = self._config.s3_checkpoint_bucket
            prefix = self._config.s3_checkpoint_prefix

        if bucket:
            from ml_platform.serving.state_manager import S3StateManager

            self._state_mgr = S3StateManager(
                bucket=bucket,
                prefix=prefix,
                region=self._config.aws_region,
            )
            restored_dir = self._state_mgr.download_latest()
            if restored_dir:
                service.load_state(restored_dir)
                logger.info("State restored from s3://%s", bucket)
            else:
                logger.info("No S3 checkpoint found; starting fresh")

        self._context_store = self._create_context_store()

        service.on_startup()

        if self._state_mgr:
            self._bg_tasks.append(asyncio.create_task(self._checkpoint_loop()))

    async def _on_shutdown(self) -> None:
        if self._service is not None:
            self._service.on_shutdown()
            if self._state_mgr:
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._service.save_state(tmpdir)
                    self._state_mgr.upload(tmpdir)
                logger.info("Final checkpoint saved")

    def _metrics_snapshot(self) -> dict[str, float]:
        if self._service is not None:
            return self._service.metrics_snapshot()
        return {}

    # -- request handling ----------------------------------------------------

    async def predict(self, payload: dict[str, Any]) -> Any:
        """Delegate to the service's ``predict``, store context, and emit metrics."""
        from ml_platform.serving.stateful import PredictionResult

        result: PredictionResult = await self.service.predict(payload)

        if self._context_store is not None:
            try:
                self._context_store.put(result.request_id, {
                    "payload": payload,
                    "prediction": result.prediction,
                    "metadata": result.metadata,
                })
            except Exception:
                logger.exception(
                    "Failed to store context for request_id=%s", result.request_id
                )

        if self._emitter:
            self._emitter.emit_event(
                "prediction",
                dimensions={"service": self._config.service_name},
                values={
                    k: v
                    for k, v in result.metadata.items()
                    if isinstance(v, (int, float))
                },
            )
        return result

    async def process_feedback(
        self, request_id: str, feedback: dict[str, Any]
    ) -> None:
        """Retrieve stored context and delegate to the service."""
        context: dict[str, Any] | None = None
        if self._context_store is not None:
            try:
                context = self._context_store.get(request_id)
            except Exception:
                logger.exception(
                    "Failed to retrieve context for request_id=%s", request_id
                )

        await self.service.process_feedback(
            request_id, feedback, context=context
        )

    def metrics_snapshot(self) -> dict[str, float]:
        """Return the latest business metrics from the service."""
        return self._metrics_snapshot()

    # -- context store -------------------------------------------------------

    def _create_context_store(self) -> ContextStore | None:
        """Create a context store based on config.

        Uses DynamoDB when a table name is configured and AWS credentials
        are available; falls back to in-memory for local development.
        """
        table_name = self._config.state_table_name
        if not table_name:
            return None

        try:
            from ml_platform.serving.context_store import DynamoDBContextStore

            store = DynamoDBContextStore(
                table_name=table_name,
                region=self._config.aws_region,
                ttl_s=self._config.state_ttl_s,
            )
            logger.info("Context store: DynamoDB table %s", table_name)
            return store
        except Exception:
            from ml_platform.serving.context_store import InMemoryContextStore

            logger.critical(
                "DEGRADED: DynamoDB unavailable for table '%s'; falling back to "
                "in-memory context store. Context will NOT be shared across "
                "replicas and will be LOST on restart. Set state_table_name='' "
                "to silence this warning, or fix DynamoDB connectivity.",
                table_name,
                exc_info=True,
            )
            return InMemoryContextStore()

    # -- background loops ----------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        interval = self._config.checkpoint_interval_s
        if self._config.stateful:
            interval = self._config.stateful.checkpoint_interval_s
        while True:
            await asyncio.sleep(interval)
            if self._service and self._state_mgr:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        self._service.save_state(tmpdir)
                        self._state_mgr.upload(tmpdir)
                    logger.info("Periodic checkpoint saved")
                except Exception:
                    logger.exception("Checkpoint failed")
