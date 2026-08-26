"""Papers profile: public paper API only (no user/chat/socket/agents)."""

from __future__ import annotations

from flask import Flask

from app.asgi import attach_asgi
from app.api.route_allowlist import load_papers_blueprints
from app.profiles import AppProfile, ApplicationBundle, discover_capabilities
from app.wsgi import apply_profile_config, create_flask_app
from service.bootstrap import initialize_runtime
from service.static_cache import cached_data


def create_papers_bundle() -> ApplicationBundle:
    """Build the papers Flask app and ASGI shell (lifecycle runs in lifespan)."""
    # GCP logging is a full-profile dependency; keep papers local-console only.
    logger = initialize_runtime(enable_gcp=False)
    flask_app = create_flask_app(serve_frontend=False)
    for blueprint in load_papers_blueprints():
        flask_app.register_blueprint(blueprint)

    # Provisional snapshot; lifespan refreshes after cache init.
    capabilities = discover_capabilities(
        AppProfile.PAPERS,
        zilliz_ready=bool(getattr(cached_data, "zilliz_ready", False)),
        socket_io_enabled=False,
    )
    apply_profile_config(
        flask_app,
        profile=AppProfile.PAPERS,
        capabilities=capabilities,
        socket_io_enabled=False,
    )
    _attach_logger(flask_app, logger)

    bundle = ApplicationBundle(
        profile=AppProfile.PAPERS,
        flask_app=flask_app,
        asgi_app=None,
        socketio=None,
        capabilities=capabilities,
        logger=logger,
    )
    attach_asgi(bundle, enable_socketio=False)
    return bundle


def _attach_logger(app: Flask, logger) -> None:
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
