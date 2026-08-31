"""Full profile: papers + user/chat + ASGI Socket.IO + SPA."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request
from flask_cors import cross_origin

from app.asgi import attach_asgi
from app.api.route_allowlist import load_full_blueprints
from app.profiles import AppProfile, ApplicationBundle, discover_capabilities
from app.wsgi import apply_profile_config, create_flask_app
from service.bootstrap import initialize_runtime
from service.static_cache import cached_data


def create_full_bundle() -> ApplicationBundle:
    """Build the full Flask app and ASGI(+Chat/Socket.IO) shell.

    Cache init, chat session reset, and the Agent executor run in ASGI lifespan.
    ``flask_app`` is an internal WSGI sub-app for middleware mounting only — not a
    production entrypoint. Provenance uses ASGI Socket.IO.
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
    # Public/read-only MCP can run on today's full deployment while the
    # standalone papers deployment is introduced.
    attach_asgi(bundle, enable_chat=True, enable_mcp=True, enable_socketio=True)
    return bundle


def _attach_logger(app: Flask, logger) -> None:
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)


def _register_full_only_http_routes(app: Flask) -> None:
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
