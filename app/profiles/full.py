"""Full profile: papers + user/chat/Socket.IO/SPA."""

from __future__ import annotations

import asyncio
import logging as logging_module
import os
from datetime import datetime

from flask import Flask, Response, jsonify, request
from flask_cors import cross_origin
from flask_socketio import SocketIO, emit

from app.api.route_allowlist import load_full_blueprints
from app.profiles import AppProfile, ApplicationBundle, discover_capabilities
from app.wsgi import apply_profile_config, create_flask_app
from service.bootstrap import initialize_runtime
from service.static_cache import cached_data


def create_full_bundle() -> ApplicationBundle:
    """Build the full Flask + Socket.IO app and run full lifecycle."""
    logger = initialize_runtime(enable_gcp=True)
    app = create_flask_app(serve_frontend=True)
    for blueprint in load_full_blueprints():
        app.register_blueprint(blueprint)

    socketio = _create_socketio(app)
    _register_socket_handlers(socketio, logger)
    _register_full_only_http_routes(app)
    _register_spa_routes(app)

    from agents.agent_v1_legacy.runner import reset_all_sessions

    reset_all_sessions()
    print("[startup] Cleared all chat sessions (docs + memory).")

    cached_data.init()
    capabilities = discover_capabilities(
        AppProfile.FULL,
        zilliz_ready=cached_data.zilliz_ready,
        socket_io_enabled=True,
    )
    apply_profile_config(
        app,
        profile=AppProfile.FULL,
        capabilities=capabilities,
        socket_io_enabled=True,
    )
    _attach_logger(app, logger)
    logging_module.getLogger("socketio").setLevel(logging_module.WARNING)
    logging_module.getLogger("engineio").setLevel(logging_module.WARNING)

    logger.info(
        "Full profile ready (paperSearch=%s chat=%s userLibrary=%s vectorSearch=%s)",
        capabilities.paper_search,
        capabilities.chat,
        capabilities.user_library,
        capabilities.vector_search,
    )
    return ApplicationBundle(
        profile=AppProfile.FULL,
        flask_app=app,
        socketio=socketio,
        capabilities=capabilities,
        logger=logger,
    )


def _attach_logger(app: Flask, logger) -> None:
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)


def _create_socketio(app: Flask) -> SocketIO:
    return SocketIO(
        app,
        # Keep ``python main.py`` compatible with the synchronous/asyncio chat
        # bridge. Eventlet runs all greenlets on one OS thread, so a blocking chat
        # turn can otherwise prevent unrelated HTTP routes from being scheduled.
        async_mode="threading",
        cors_allowed_origins=[
            "http://localhost:8080",  # User study dev server
            "http://localhost:8081",  # standalone
            "http://localhost:5173",  # rebuild Vite dev server
            "https://vitality.mathcs.emory.edu",  # Production server
        ],
    )


def _register_socket_handlers(socketio: SocketIO, logger) -> None:
    @socketio.on("connect")
    def handle_connect(auth):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[%s] WebSocket Client connected: %s", timestamp, request.sid)
        emit("connected", {"data": "Connected to Flask-SocketIO server"})

    @socketio.on("disconnect")
    def handle_disconnect():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[%s] WebSocket Client disconnected: %s", timestamp, request.sid)

    @socketio.on("log_event")
    def handle_log_event(data):
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
            logger.error("[%s] An error occured during logging event: %s", timestamp, error)
            logger.info("Raw data received: %s", data)
            return {"status": "error", "message": str(error)}


def _register_full_only_http_routes(app: Flask) -> None:
    # Deprecated simple stream route (kept for compatibility; streaming_llm undefined).
    @app.route("/chat_stream_simple", methods=["POST"])
    @cross_origin()
    def chat_stream_simple():
        data = request.get_json(force=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return Response("Please Input Your Text", status=400)

        async def llm_stream():
            async for chunk in streaming_llm.astream(text):  # noqa: F821
                yield chunk.content or ""
            yield "[[STREAM_DONE]]"

        def sync_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                agen = llm_stream()
                while True:
                    part = loop.run_until_complete(agen.__anext__())
                    if not part:
                        continue
                    yield part
            except StopAsyncIteration:
                pass
            finally:
                loop.close()

        return Response(sync_stream(), mimetype="text/plain", status=200)

    @app.route("/resetMemory", methods=["POST"])
    @cross_origin()
    def reset_memory():
        from agents.agent_v1_legacy.runner import reset_all_sessions

        try:
            reset_all_sessions()
            print("[resetMemory] Cleared all sessions (docs + chat memory).")
            return jsonify({"status": "success", "message": "All sessions cleared."})
        except Exception as error:
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(error)}), 500


def _register_spa_routes(app: Flask) -> None:
    @app.route("/")
    @cross_origin()
    def index():
        return app.send_static_file("index.html")

    @app.errorhandler(404)
    def spa_fallback(error):
        """Serve the SPA shell for unknown GET paths (client-side routes)."""
        if request.method == "GET":
            static_path = os.path.join(app.static_folder or "", request.path.lstrip("/"))
            if request.path != "/" and os.path.isfile(static_path):
                return app.send_static_file(request.path.lstrip("/"))
            accept = request.accept_mimetypes
            if accept.accept_html or "text/html" in str(request.accept_mimetypes):
                return app.send_static_file("index.html")
        return jsonify({"message": "Not found"}), 404
