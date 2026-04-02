"""Tests for the in-process scheduled task system."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ml_platform.scheduling import (
    ScheduledTask,
    TaskExecution,
    TaskRunner,
    discover_tasks,
    scheduled,
)


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestScheduledDecorator:
    def test_attaches_metadata(self) -> None:
        @scheduled(interval_s=3600, name="retrain")
        async def retrain(self: Any) -> None:
            pass

        meta = getattr(retrain, "_ml_platform_schedule")
        assert meta["name"] == "retrain"
        assert meta["interval_s"] == 3600
        assert meta["run_on_startup"] is False
        assert meta["max_retries"] == 0

    def test_defaults_name_to_function_name(self) -> None:
        @scheduled(interval_s=60)
        async def my_custom_task(self: Any) -> None:
            pass

        meta = getattr(my_custom_task, "_ml_platform_schedule")
        assert meta["name"] == "my_custom_task"

    def test_run_on_startup_flag(self) -> None:
        @scheduled(interval_s=60, run_on_startup=True)
        async def warmup(self: Any) -> None:
            pass

        meta = getattr(warmup, "_ml_platform_schedule")
        assert meta["run_on_startup"] is True

    def test_max_retries(self) -> None:
        @scheduled(interval_s=60, max_retries=3)
        async def flaky(self: Any) -> None:
            pass

        meta = getattr(flaky, "_ml_platform_schedule")
        assert meta["max_retries"] == 3

    def test_preserves_function_behavior(self) -> None:
        @scheduled(interval_s=60)
        async def identity(self: Any) -> str:
            return "hello"

        assert asyncio.iscoroutinefunction(identity)


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverTasks:
    def test_finds_decorated_methods(self) -> None:
        class MyService:
            @scheduled(interval_s=100, name="task_a")
            async def task_a(self) -> None:
                pass

            @scheduled(interval_s=200, name="task_b")
            async def task_b(self) -> None:
                pass

            async def not_scheduled(self) -> None:
                pass

        svc = MyService()
        tasks = discover_tasks(svc)
        names = {t.name for t in tasks}
        assert names == {"task_a", "task_b"}

    def test_empty_when_no_decorators(self) -> None:
        class PlainService:
            async def predict(self) -> None:
                pass

        assert discover_tasks(PlainService()) == []

    def test_skips_private_methods(self) -> None:
        class MyService:
            @scheduled(interval_s=60, name="hidden")
            async def _private_task(self) -> None:
                pass

        assert discover_tasks(MyService()) == []

    def test_binds_to_instance(self) -> None:
        class MyService:
            def __init__(self) -> None:
                self.called = False

            @scheduled(interval_s=60)
            async def mark(self) -> None:
                self.called = True

        svc = MyService()
        tasks = discover_tasks(svc)
        assert len(tasks) == 1
        asyncio.get_event_loop().run_until_complete(tasks[0].fn())
        assert svc.called is True


# ---------------------------------------------------------------------------
# TaskRunner tests
# ---------------------------------------------------------------------------


class TestTaskRunner:
    @pytest.mark.asyncio
    async def test_register_and_start(self) -> None:
        call_count = 0

        async def job() -> None:
            nonlocal call_count
            call_count += 1

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(name="counter", fn=job, interval_s=0.05))
        await runner.start()
        await asyncio.sleep(0.15)
        await runner.stop()
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_run_on_startup(self) -> None:
        call_count = 0

        async def job() -> None:
            nonlocal call_count
            call_count += 1

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(
            name="eager", fn=job, interval_s=999, run_on_startup=True,
        ))
        await runner.start()
        await asyncio.sleep(0.05)
        await runner.stop()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_error_does_not_crash_loop(self) -> None:
        calls: list[int] = []

        async def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("first call fails")

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(name="flaky", fn=flaky, interval_s=0.05))
        await runner.start()
        await asyncio.sleep(0.2)
        await runner.stop()
        assert len(calls) >= 2

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        attempts: list[int] = []

        async def fragile() -> None:
            attempts.append(1)
            if len(attempts) <= 2:
                raise RuntimeError("fail")

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(
            name="fragile", fn=fragile, interval_s=999,
            run_on_startup=True, max_retries=2, retry_backoff_s=0.01,
        ))
        await runner.start()
        await asyncio.sleep(0.2)
        await runner.stop()
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_metrics_snapshot(self) -> None:
        call_count = 0

        async def job() -> None:
            nonlocal call_count
            call_count += 1

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(
            name="quick", fn=job, interval_s=0.05, run_on_startup=True,
        ))
        await runner.start()
        await asyncio.sleep(0.1)
        await runner.stop()

        snapshot = runner.metrics_snapshot()
        assert snapshot["scheduled_tasks_registered"] == 1.0
        assert snapshot["scheduled_tasks_total_runs"] >= 1.0

    @pytest.mark.asyncio
    async def test_recent_executions(self) -> None:
        async def job() -> None:
            pass

        runner = TaskRunner(service_name="test")
        runner.register(ScheduledTask(
            name="tracked", fn=job, interval_s=0.02, run_on_startup=True,
        ))
        await runner.start()
        await asyncio.sleep(0.1)
        await runner.stop()

        execs = runner.recent_executions
        assert len(execs) >= 1
        assert all(isinstance(e, TaskExecution) for e in execs)
        assert execs[0].success is True
        assert execs[0].task_name == "tracked"

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        runner = TaskRunner(service_name="test")
        await runner.stop()
        await runner.stop()

    @pytest.mark.asyncio
    async def test_register_all(self) -> None:
        async def a() -> None:
            pass

        async def b() -> None:
            pass

        runner = TaskRunner(service_name="test")
        runner.register_all([
            ScheduledTask(name="a", fn=a, interval_s=60),
            ScheduledTask(name="b", fn=b, interval_s=120),
        ])
        assert len(runner.tasks) == 2


# ---------------------------------------------------------------------------
# Integration: scheduling with runtime
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_stateful_service_with_scheduled_tasks(self) -> None:
        """Scheduled tasks on a StatefulServiceBase are auto-discovered."""
        from ml_platform.config import ServiceConfig
        from ml_platform.serving.runtime import StatefulRuntime
        from ml_platform.serving.stateful import PredictionResult, StatefulServiceBase

        class TrainingService(StatefulServiceBase):
            def __init__(self) -> None:
                self.retrain_count = 0

            @scheduled(interval_s=0.05, run_on_startup=True)
            async def retrain(self) -> None:
                self.retrain_count += 1

            async def predict(self, payload: dict[str, Any]) -> PredictionResult:
                return PredictionResult(request_id="r1", prediction={})

            async def process_feedback(
                self, request_id: str, feedback: dict[str, Any], *, context: Any = None,
            ) -> None:
                pass

            def save_state(self, d: str) -> None:
                pass

            def load_state(self, d: str) -> None:
                pass

            def metrics_snapshot(self) -> dict[str, float]:
                return {"retrain_count": float(self.retrain_count)}

        config = ServiceConfig(service_name="sched-test")
        rt = StatefulRuntime(TrainingService, config)
        await rt.startup()

        await asyncio.sleep(0.15)
        assert rt.service.retrain_count >= 1

        await rt.shutdown()

    @pytest.mark.asyncio
    async def test_agent_service_with_scheduled_tasks(self) -> None:
        """Scheduled tasks on an AgentServiceBase are auto-discovered."""
        from ml_platform._types import AgentResult, Completion, CompletionUsage, Message
        from ml_platform.config import AgentConfig, ServiceConfig
        from ml_platform.llm.run_context import RunContext
        from ml_platform.serving.agent import AgentServiceBase
        from ml_platform.serving.runtime import AgentRuntime

        class MockProvider:
            async def complete(
                self, messages: list[Message], *, model: str = "", **kwargs: Any
            ) -> Completion:
                return Completion(
                    content="ok", model="m", provider="p",
                    usage=CompletionUsage(input_tokens=1, output_tokens=1),
                )

        class EvalAgent(AgentServiceBase):
            def __init__(self) -> None:
                self.eval_count = 0

            @scheduled(interval_s=0.05, run_on_startup=True)
            async def evaluate(self) -> None:
                self.eval_count += 1

            async def run(
                self, messages: list[Message], *, run_context: RunContext, **kwargs: Any,
            ) -> AgentResult:
                c = await run_context.complete(self.providers["main"], messages)
                return AgentResult(content=c.content, steps=run_context.steps, messages=messages)

        config = ServiceConfig(service_name="agent-sched", agent=AgentConfig())
        rt = AgentRuntime(EvalAgent, config, providers={"main": MockProvider()})
        await rt.startup()

        await asyncio.sleep(0.15)
        assert rt.service.eval_count >= 1

        await rt.shutdown()
