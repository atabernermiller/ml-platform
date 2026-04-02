"""Tests for ml_platform.config -- service configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from ml_platform.config import AgentConfig, LLMConfig, ServiceConfig, StatefulConfig


class TestServiceConfig:
    def test_minimal(self) -> None:
        c = ServiceConfig(service_name="svc")
        assert c.service_name == "svc"
        assert c.mlflow_experiment_name == "svc"
        assert c.state_table_name == "svc-context"

    def test_nested_agent_config(self) -> None:
        c = ServiceConfig(
            service_name="agent-svc",
            agent=AgentConfig(max_steps_per_run=10, tool_timeout_s=15.0),
        )
        assert c.agent is not None
        assert c.agent.max_steps_per_run == 10
        assert c.agent.tool_timeout_s == 15.0

    def test_nested_llm_config(self) -> None:
        c = ServiceConfig(
            service_name="llm-svc",
            llm=LLMConfig(default_model="gpt-4o"),
        )
        assert c.llm is not None
        assert c.llm.default_model == "gpt-4o"
        assert c.agent is None

    def test_backward_compat_checkpoint_fields(self) -> None:
        c = ServiceConfig(
            service_name="old-svc",
            s3_checkpoint_bucket="my-bucket",
            s3_checkpoint_prefix="pfx/",
        )
        assert c.stateful is not None
        assert c.stateful.s3_checkpoint_bucket == "my-bucket"
        assert c.stateful.s3_checkpoint_prefix == "pfx/"

    def test_no_stateful_migration_when_no_bucket(self) -> None:
        c = ServiceConfig(service_name="svc")
        assert c.stateful is None

    def test_repr_redacts_sensitive(self) -> None:
        c = ServiceConfig(
            service_name="svc",
            alert_webhook_url="https://secret.slack.com/hook",
            mlflow_tracking_uri="http://mlflow:5000",
        )
        r = repr(c)
        assert "secret.slack.com" not in r
        assert "mlflow:5000" not in r
        assert "'***'" in r
        assert "service_name='svc'" in r


class TestAgentConfigDefaults:
    def test_defaults(self) -> None:
        ac = AgentConfig()
        assert ac.max_steps_per_run == 20
        assert ac.tool_timeout_s == 30.0
        assert ac.max_concurrent_tool_calls == 5


class TestServiceConfigFromEnv:
    def test_reads_agent_env(self) -> None:
        env = {
            "ML_PLATFORM_SERVICE_NAME": "env-svc",
            "ML_PLATFORM_AGENT_MAX_STEPS_PER_RUN": "50",
            "ML_PLATFORM_AGENT_TOOL_TIMEOUT_S": "10.0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            c = ServiceConfig.from_env()
        assert c.service_name == "env-svc"
        assert c.agent is not None
        assert c.agent.max_steps_per_run == 50
        assert c.agent.tool_timeout_s == 10.0

    def test_overrides_take_precedence(self) -> None:
        env = {"ML_PLATFORM_SERVICE_NAME": "from-env"}
        with mock.patch.dict(os.environ, env, clear=False):
            c = ServiceConfig.from_env(service_name="override")
        assert c.service_name == "override"

    def test_reads_llm_env(self) -> None:
        env = {
            "ML_PLATFORM_SERVICE_NAME": "llm-env",
            "ML_PLATFORM_LLM_DEFAULT_MODEL": "gpt-4o",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            c = ServiceConfig.from_env()
        assert c.llm is not None
        assert c.llm.default_model == "gpt-4o"
