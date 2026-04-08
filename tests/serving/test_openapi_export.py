"""Tests for OpenAPI schema export and TypeScript generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from ml_platform.serving.openapi_export import (
    export_openapi_schema,
    generate_typescript_client,
    generate_typescript_types,
)


class _InferRequest(BaseModel):
    features: list[float]
    model_name: str = "default"


class _InferResponse(BaseModel):
    prediction: float
    confidence: float


def _make_app() -> FastAPI:
    app = FastAPI(title="Test API", version="1.0.0")

    @app.post("/infer", response_model=_InferResponse)
    async def infer(req: _InferRequest) -> _InferResponse:
        return _InferResponse(prediction=0.5, confidence=0.9)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


class TestExportOpenAPISchema:
    def test_returns_dict(self) -> None:
        app = _make_app()
        schema = export_openapi_schema(app)
        assert isinstance(schema, dict)
        assert schema["info"]["title"] == "Test API"
        assert "paths" in schema

    def test_writes_to_file(self) -> None:
        app = _make_app()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        export_openapi_schema(app, output_path=path)
        content = Path(path).read_text()
        data = json.loads(content)
        assert data["info"]["title"] == "Test API"


class TestGenerateTypescriptTypes:
    def test_generates_interfaces(self) -> None:
        app = _make_app()
        schema = export_openapi_schema(app)
        ts = generate_typescript_types(schema)
        assert "export interface" in ts
        assert "_InferRequest" in ts or "Body_infer" in ts

    def test_auto_generated_header(self) -> None:
        schema = {"info": {"title": "My API", "version": "1.0"}, "components": {"schemas": {}}}
        ts = generate_typescript_types(schema)
        assert "Auto-generated" in ts


class TestGenerateTypescriptClient:
    def test_generates_client_class(self) -> None:
        app = _make_app()
        schema = export_openapi_schema(app)
        ts = generate_typescript_client(schema, base_url="http://api.example.com")
        assert "export class APIClient" in ts
        assert "http://api.example.com" in ts
        assert "async" in ts

    def test_has_methods_for_endpoints(self) -> None:
        app = _make_app()
        schema = export_openapi_schema(app)
        ts = generate_typescript_client(schema)
        assert "health" in ts.lower()
