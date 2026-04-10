"""WebSocket adapter for real-time features in FastAPI services.

The serving layer is HTTP-only by default.  This module adds WebSocket
support for live features such as order tracking, chat, and streaming
inference results.

Provides:

- :class:`WebSocketManager` -- manages connected clients, broadcasts,
  and room-based messaging.
- :func:`add_websocket_routes` -- attaches ``/ws`` and ``/ws/{room}``
  endpoints to an existing FastAPI app.

Usage::

    from ml_platform.serving.websocket import WebSocketManager, add_websocket_routes

    manager = WebSocketManager()
    add_websocket_routes(app, manager)

    # From any endpoint or background task:
    await manager.broadcast({"event": "order_update", "order_id": "123"})
    await manager.send_to_room("chat-room-1", {"msg": "Hello!"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

__all__ = [
    "WebSocketManager",
    "add_websocket_routes",
]

MessageHandler = Callable[[str, dict[str, Any]], Awaitable[None]]


class WebSocketManager:
    """Manages WebSocket connections with room support.

    Each connected client is assigned a unique ``client_id`` and can
    optionally join a named room.  Supports broadcast (all clients),
    room-level messaging, and direct client messaging.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._rooms: dict[str, set[str]] = {}
        self._handlers: list[MessageHandler] = []
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._connections)

    def room_count(self, room: str) -> int:
        """Number of clients in a specific room."""
        return len(self._rooms.get(room, set()))

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        """Register a handler called for every incoming message.

        Args:
            handler: Async callable ``(client_id, data) -> None``.

        Returns:
            The same handler (for use as a decorator).
        """
        self._handlers.append(handler)
        return handler

    async def connect(
        self, websocket: WebSocket, room: str = ""
    ) -> str:
        """Accept a WebSocket connection and return the assigned client ID.

        Args:
            websocket: The incoming WebSocket.
            room: Optional room to join immediately.

        Returns:
            Unique client identifier.
        """
        await websocket.accept()
        client_id = uuid.uuid4().hex[:12]
        async with self._lock:
            self._connections[client_id] = websocket
            if room:
                self._rooms.setdefault(room, set()).add(client_id)
        logger.debug("WS connected: %s (room=%s)", client_id, room or "<none>")
        return client_id

    async def disconnect(self, client_id: str) -> None:
        """Remove a client from all rooms and close the connection.

        Args:
            client_id: Client identifier from :meth:`connect`.
        """
        async with self._lock:
            self._connections.pop(client_id, None)
            for members in self._rooms.values():
                members.discard(client_id)
            empty_rooms = [r for r, m in self._rooms.items() if not m]
            for r in empty_rooms:
                del self._rooms[r]
        logger.debug("WS disconnected: %s", client_id)

    async def send(self, client_id: str, data: dict[str, Any]) -> bool:
        """Send a JSON message to a specific client.

        Args:
            client_id: Target client.
            data: JSON-serialisable message.

        Returns:
            ``True`` if sent, ``False`` if the client is not connected.
        """
        ws = self._connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_json(data)
            return True
        except Exception:
            await self.disconnect(client_id)
            return False

    async def broadcast(self, data: dict[str, Any]) -> int:
        """Send a message to all connected clients.

        Args:
            data: JSON-serialisable message.

        Returns:
            Number of clients the message was sent to.
        """
        sent = 0
        stale: list[str] = []
        for cid, ws in list(self._connections.items()):
            try:
                await ws.send_json(data)
                sent += 1
            except Exception:
                stale.append(cid)
        for cid in stale:
            await self.disconnect(cid)
        return sent

    async def send_to_room(self, room: str, data: dict[str, Any]) -> int:
        """Send a message to all clients in a room.

        Args:
            room: Room name.
            data: JSON-serialisable message.

        Returns:
            Number of clients the message was sent to.
        """
        members = self._rooms.get(room, set())
        sent = 0
        stale: list[str] = []
        for cid in list(members):
            ws = self._connections.get(cid)
            if ws is None:
                stale.append(cid)
                continue
            try:
                await ws.send_json(data)
                sent += 1
            except Exception:
                stale.append(cid)
        for cid in stale:
            await self.disconnect(cid)
        return sent

    async def join_room(self, client_id: str, room: str) -> None:
        """Add a client to a room.

        Args:
            client_id: Client identifier.
            room: Room name.
        """
        async with self._lock:
            if client_id in self._connections:
                self._rooms.setdefault(room, set()).add(client_id)

    async def leave_room(self, client_id: str, room: str) -> None:
        """Remove a client from a room.

        Args:
            client_id: Client identifier.
            room: Room name.
        """
        async with self._lock:
            members = self._rooms.get(room)
            if members:
                members.discard(client_id)
                if not members:
                    del self._rooms[room]

    async def _handle_messages(
        self, client_id: str, websocket: WebSocket
    ) -> None:
        """Internal loop that reads messages and dispatches to handlers."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"text": raw}
                for handler in self._handlers:
                    try:
                        await handler(client_id, data)
                    except Exception:
                        logger.exception("WS handler error for %s", client_id)
        except WebSocketDisconnect:
            pass
        finally:
            await self.disconnect(client_id)


def add_websocket_routes(
    app: FastAPI,
    manager: WebSocketManager,
    *,
    path: str = "/ws",
    room_path: str = "/ws/{room}",
) -> FastAPI:
    """Attach WebSocket endpoints to a FastAPI application.

    Creates two routes:

    - ``ws://<host>/<path>`` -- general connection (no room).
    - ``ws://<host>/<room_path>`` -- room-based connection.

    Args:
        app: FastAPI application.
        manager: WebSocket manager instance.
        path: Base WebSocket path.
        room_path: Room-based WebSocket path.

    Returns:
        The same app with WebSocket routes added.
    """

    @app.websocket(path)
    async def ws_connect(websocket: WebSocket) -> None:
        client_id = await manager.connect(websocket)
        await manager._handle_messages(client_id, websocket)

    @app.websocket(room_path)
    async def ws_room_connect(websocket: WebSocket, room: str) -> None:
        client_id = await manager.connect(websocket, room=room)
        await manager._handle_messages(client_id, websocket)

    logger.info("WebSocket routes enabled: %s, %s", path, room_path)
    return app
