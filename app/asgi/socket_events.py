"""ASGI Socket.IO provenance handlers (full profile only)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

SOCKET_CORS_ORIGINS = [
    "http://localhost:8080",  # User study dev server
    "http://localhost:8081",  # standalone
    "http://localhost:5173",  # rebuild Vite dev server
    "https://vitality.mathcs.emory.edu",  # Production server
]


def create_async_server():
    """Create an ASGI-mode Socket.IO server (lazy import keeps papers clean)."""
    import socketio

    return socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins=SOCKET_CORS_ORIGINS,
    )


def register_socket_handlers(sio: Any, logger: Any) -> None:
    """Register connect / disconnect / log_event with stable client contracts."""

    @sio.event
    async def connect(sid, environ, auth=None):  # noqa: ARG001
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[%s] WebSocket Client connected: %s", timestamp, sid)
        await sio.emit(
            "connected",
            {"data": "Connected to Flask-SocketIO server"},
            to=sid,
        )

    @sio.event
    async def disconnect(sid, reason=None):  # noqa: ARG001
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if reason:
            logger.info(
                "[%s] WebSocket Client disconnected: %s (%s)",
                timestamp,
                sid,
                reason,
            )
        else:
            logger.info("[%s] WebSocket Client disconnected: %s", timestamp, sid)

    @sio.event
    async def log_event(sid, data):  # noqa: ARG001
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            if not isinstance(data, dict):
                raise ValueError("event must be an object")

            event_id = data.get("eventId")
            session_id = data.get("sessionId")
            action = data.get("action")
            event_data = data.get("eventData")
            if not all(
                isinstance(value, str) and value
                for value in (event_id, session_id, action)
            ):
                raise ValueError("eventId, sessionId, and action are required")
            if not isinstance(event_data, dict):
                raise ValueError("eventData must be an object")

            overview = (
                f"Socket Event - Actor Type: {data.get('actorType', 'unknown')} "
                f"| Action: {action}"
            )
            logger.info(
                {"message": overview, **data},
                extra={"provenance_event": True},
            )
            return {"status": "success", "timestamp": timestamp}
        except Exception as error:
            logger.error(
                "[%s] An error occured during logging event: %s", timestamp, error
            )
            logger.info("Raw data received: %s", data)
            return {"status": "error", "message": str(error)}
