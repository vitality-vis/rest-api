"""FastAPI + WSGIMiddleware (+ optional Chat / Socket.IO) composition."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from a2wsgi import WSGIMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.asgi.lifespan import (
    shutdown_bundle,
    startup_bundle,
)

if TYPE_CHECKING:
    from app.profiles import ApplicationBundle


def attach_asgi(
    bundle: ApplicationBundle,
    *,
    enable_chat: bool,
    enable_socketio: bool,
) -> None:
    """Build ``bundle.asgi_app`` (and optionally Socket.IO) in place.

    ``enable_chat`` / ``enable_socketio`` are profile wiring flags, not runtime
    capability probes (Azure/Supabase availability must not hide routes).
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        startup_bundle(bundle)
        try:
            yield
        finally:
            await shutdown_bundle(bundle)

    fastapi_app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    # Native ASGI routes no longer pass through Flask-CORS. Mirror the existing
    # public HTTP policy here so browser preflight works for /chat/v2 as well as
    # routes still served by the mounted Flask app.
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if enable_chat:
        # Full-only lazy import keeps the papers profile free of Agent modules.
        from app.asgi.chat_routes import register_chat_routes

        register_chat_routes(fastapi_app, bundle)
    # Mount after FastAPI-native routes. Flask owns remaining HTTP paths
    # (papers, user, chat import/history, SPA). Chat turns use ASGI ``/chat/v2``.
    fastapi_app.mount("/", WSGIMiddleware(bundle.flask_app))

    if not enable_socketio:
        bundle.asgi_app = fastapi_app
        bundle.socketio = None
        return

    import logging as logging_module

    import socketio

    from app.asgi.socket_events import create_async_server, register_socket_handlers

    sio = create_async_server()
    register_socket_handlers(sio, bundle.logger)
    logging_module.getLogger("socketio").setLevel(logging_module.WARNING)
    logging_module.getLogger("engineio").setLevel(logging_module.WARNING)
    bundle.socketio = sio
    bundle.asgi_app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
