"""Tests for WebSocket adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from ml_platform.serving.websocket import WebSocketManager, add_websocket_routes


@pytest.fixture()
def ws_app() -> tuple[FastAPI, WebSocketManager]:
    """Create a FastAPI app with WebSocket routes."""
    app = FastAPI()
    manager = WebSocketManager()
    add_websocket_routes(app, manager)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app, manager


class TestWebSocketManager:
    def test_initial_state(self) -> None:
        mgr = WebSocketManager()
        assert mgr.connection_count == 0
        assert mgr.room_count("any") == 0

    def test_on_message_registers_handler(self) -> None:
        mgr = WebSocketManager()
        received: list[tuple[str, dict[str, Any]]] = []

        @mgr.on_message
        async def handler(cid: str, data: dict[str, Any]) -> None:
            received.append((cid, data))

        assert len(mgr._handlers) == 1


class TestWebSocketRoutes:
    def test_websocket_connect_and_send(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, manager = ws_app
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            assert manager.connection_count == 1
            ws.send_json({"msg": "hello"})
            # No response expected without a handler sending back
        assert manager.connection_count == 0

    def test_websocket_room_connect(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, manager = ws_app
        client = TestClient(app)
        with client.websocket_connect("/ws/chat-room") as ws:
            assert manager.connection_count == 1
            assert manager.room_count("chat-room") == 1
        assert manager.connection_count == 0
        assert manager.room_count("chat-room") == 0

    def test_websocket_message_handler(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, manager = ws_app
        received: list[tuple[str, dict[str, Any]]] = []

        @manager.on_message
        async def echo(cid: str, data: dict[str, Any]) -> None:
            received.append((cid, data))
            await manager.send(cid, {"echo": data})

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"msg": "hello"})
            response = ws.receive_json()
            assert response["echo"]["msg"] == "hello"
        assert len(received) == 1

    def test_broadcast(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, manager = ws_app

        @manager.on_message
        async def on_msg(cid: str, data: dict[str, Any]) -> None:
            if data.get("action") == "broadcast":
                await manager.broadcast({"event": "update"})

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                assert manager.connection_count == 2
                ws1.send_json({"action": "broadcast"})
                r1 = ws1.receive_json()
                r2 = ws2.receive_json()
                assert r1["event"] == "update"
                assert r2["event"] == "update"

    def test_room_messaging(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, manager = ws_app

        @manager.on_message
        async def on_msg(cid: str, data: dict[str, Any]) -> None:
            room = data.get("room")
            if room and data.get("action") == "room_send":
                await manager.send_to_room(room, {"room_msg": data.get("text")})

        client = TestClient(app)
        with client.websocket_connect("/ws/room-a") as ws_a:
            with client.websocket_connect("/ws/room-b") as ws_b:
                # Send to room-a only
                ws_a.send_json({"action": "room_send", "room": "room-a", "text": "hi"})
                r = ws_a.receive_json()
                assert r["room_msg"] == "hi"

    def test_health_endpoint_still_works(
        self, ws_app: tuple[FastAPI, WebSocketManager]
    ) -> None:
        app, _ = ws_app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
