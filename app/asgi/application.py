"""FastAPI + WSGIMiddleware (+ optional ASGI Socket.IO) composition."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from a2wsgi import WSGIMiddleware
from fastapi import FastAPI

from app.asgi.lifespan import (
    shutdown_bundle,
    startup_bundle,
)

if TYPE_CHECKING:
    from app.profiles import ApplicationBundle


def attach_asgi(bundle: ApplicationBundle, *, enable_socketio: bool) -> None:
    """Build ``bundle.asgi_app`` (and optionally ``bundle.socketio``) in place."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        startup_bundle(bundle)
        try:
            yield
        finally:
            await shutdown_bundle(bundle)

    fastapi_app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    # Mount after any future FastAPI-native routes. For now the Flask app owns
    # all HTTP paths including /chat*.
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
