"""FastAPI + WSGIMiddleware (+ optional Chat / Socket.IO) composition."""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from a2wsgi import WSGIMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Route

import config
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
    enable_mcp: bool,
    enable_socketio: bool,
) -> None:
    """Build ``bundle.asgi_app`` (and optionally Socket.IO) in place.

    ``enable_chat`` / ``enable_socketio`` are profile wiring flags, not runtime
    capability probes (Azure/Supabase availability must not hide routes).
    """

    mcp_endpoint = _McpEndpoint() if enable_mcp else None
    if enable_mcp:
        assert mcp_endpoint is not None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        startup_bundle(bundle)
        try:
            async with AsyncExitStack() as stack:
                if mcp_endpoint is not None:
                    # MCP session managers are deliberately single-use. Build a
                    # fresh runtime on every ASGI lifespan so local reloads and
                    # repeated TestClient lifecycles remain valid.
                    from app.mcp import create_public_mcp_server

                    mcp_server = create_public_mcp_server()
                    mcp_app = mcp_server.streamable_http_app(
                        streamable_http_path="/mcp",
                        stateless_http=True,
                        json_response=True,
                        transport_security=_mcp_transport_security(),
                    )
                    mcp_endpoint.target = mcp_app.routes[0].app
                    await stack.enter_async_context(mcp_server.session_manager.run())
                try:
                    yield
                finally:
                    if mcp_endpoint is not None:
                        mcp_endpoint.target = None
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
    if mcp_endpoint is not None:
        # An exact ASGI route avoids /mcp/ redirects on protocol POSTs.
        fastapi_app.router.routes.append(Route("/mcp", endpoint=mcp_endpoint))
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


def _mcp_transport_security():
    """Build fail-closed Host/Origin validation for the public MCP endpoint."""
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=config.mcp_allowed_hosts(),
        allowed_origins=config.mcp_allowed_origins(),
    )


class _McpEndpoint:
    """Stable ASGI route whose SDK runtime is replaced each lifespan."""

    def __init__(self) -> None:
        self.target = None

    async def __call__(self, scope, receive, send) -> None:
        if self.target is None:
            from starlette.responses import Response

            await Response(status_code=503)(scope, receive, send)
            return
        await self.target(scope, receive, send)
