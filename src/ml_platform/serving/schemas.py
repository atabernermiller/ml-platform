"""Shared Pydantic request/response models for ML platform HTTP APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Inbound prediction request.

    The ``payload`` dict is intentionally unstructured at the platform level;
    each project defines its own schema via documentation or a subclass.
    """

    payload: dict[str, Any] = Field(
        ..., description="Project-specific prediction input."
    )


class PredictResponse(BaseModel):
    """Outbound prediction response.

    Attributes:
        request_id: Unique identifier for correlating feedback to this prediction.
        prediction: Model output (project-specific schema).
        metadata: Auxiliary data such as cost, latency, or debug info.
    """

    request_id: str
    prediction: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    """Inbound feedback for a previous prediction.

    Attributes:
        request_id: Must match a ``request_id`` from a prior ``PredictResponse``.
        feedback: Project-specific feedback payload (e.g., reward, label, rating).
    """

    request_id: str
    feedback: dict[str, Any]
