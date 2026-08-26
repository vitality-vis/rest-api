"""Full profile: papers + user/chat + ASGI Socket.IO + SPA."""

from __future__ import annotations

import asyncio
import os

from flask import Flask, Response, jsonify, request
from flask_cors import cross_origin

from app.asgi import attach_asgi
from app.api.route_allowlist import load_full_blueprints
from app.profiles import AppProfile, ApplicationBundle, discover_capabilities
from app.wsgi import apply_profile_config, create_flask_app
from service.bootstrap import initialize_runtime
from service.static_cache import cached_data


def create_full_bundle() -> ApplicationBundle:
    """Build the full Flask app and ASGI(+Socket.IO) shell.

    Cache init and chat session reset run in ASGI lifespan, not at import time.
    ``flask_app`` is HTTP-only (no Flask-SocketIO); provenance uses ASGI Socket.IO.
    """
    logger = initialize_runtime(enable_gcp=True)
    flask_app = create_flask_app(serve_frontend=True)
    for blueprint in load_full_blueprints():
        flask_app.register_blueprint(blueprint)

    _register_full_only_http_routes(flask_app)
    _register_spa_routes(flask_app)

    # Provisional snapshot; lifespan refreshes after cache init / session reset.
    capabilities = discover_capabilities(
        AppProfile.FULL,
        zilliz_ready=bool(getattr(cached_data, "zilliz_ready", False)),
        socket_io_enabled=True,
    )
    apply_profile_config(
        flask_app,
        profile=AppProfile.FULL,
        capabilities=capabilities,
        socket_io_enabled=True,
    )
    _attach_logger(flask_app, logger)

    bundle = ApplicationBundle(
        profile=AppProfile.FULL,
        flask_app=flask_app,
        asgi_app=None,
        socketio=None,
        capabilities=capabilities,
        logger=logger,
    )
    attach_asgi(bundle, enable_socketio=True)
    return bundle


def _attach_logger(app: Flask, logger) -> None:
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)


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
