"""Serving layer: base classes and utilities for ML service endpoints."""

from ml_platform.serving.agent import AgentServiceBase, create_agent_app
from ml_platform.serving.context_store import (
    ContextStore,
    DynamoDBContextStore,
    InMemoryContextStore,
)
from ml_platform.serving.conversation_store import (
    ConversationStore,
    DynamoDBConversationStore,
    InMemoryConversationStore,
)
from ml_platform.serving.lambda_adapter import create_lambda_handler, wrap_for_lambda
from ml_platform.serving.openapi_export import (
    export_openapi_schema,
    generate_typescript_client,
    generate_typescript_types,
)
from ml_platform.serving.rate_limit import TokenBucketLimiter, add_rate_limit_middleware
from ml_platform.serving.runtime import AgentRuntime, BaseRuntime, StatefulRuntime
from ml_platform.serving.websocket import WebSocketManager, add_websocket_routes
from ml_platform.serving.sagemaker import wrap_for_sagemaker
from ml_platform.serving.schemas import FeedbackRequest, PredictRequest, PredictResponse
from ml_platform.serving.state_manager import S3StateManager
from ml_platform.serving.stateful import (
    PredictionResult,
    StatefulServiceBase,
    create_stateful_app,
)

__all__ = [
    "AgentRuntime",
    "AgentServiceBase",
    "BaseRuntime",
    "ContextStore",
    "ConversationStore",
    "DynamoDBContextStore",
    "DynamoDBConversationStore",
    "FeedbackRequest",
    "InMemoryContextStore",
    "InMemoryConversationStore",
    "PredictRequest",
    "PredictResponse",
    "PredictionResult",
    "S3StateManager",
    "StatefulRuntime",
    "StatefulServiceBase",
    "TokenBucketLimiter",
    "WebSocketManager",
    "add_rate_limit_middleware",
    "add_websocket_routes",
    "create_agent_app",
    "create_lambda_handler",
    "create_stateful_app",
    "export_openapi_schema",
    "generate_typescript_client",
    "generate_typescript_types",
    "wrap_for_lambda",
    "wrap_for_sagemaker",
]
