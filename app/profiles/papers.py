"""Papers profile: public paper API only (no user/chat/socket/agents)."""

from __future__ import annotations

from flask import Flask

from app.api.route_allowlist import load_papers_blueprints
from app.profiles import AppProfile, ApplicationBundle, discover_capabilities
from app.wsgi import apply_profile_config, create_flask_app
from service.bootstrap import initialize_runtime
from service.static_cache import cached_data


def create_papers_bundle() -> ApplicationBundle:
    """Build the papers Flask app and run papers-only lifecycle."""
    # GCP logging is a full-profile dependency; keep papers local-console only.
    logger = initialize_runtime(enable_gcp=False)
    app = create_flask_app(serve_frontend=False)
    for blueprint in load_papers_blueprints():
        app.register_blueprint(blueprint)

    cached_data.init()
    capabilities = discover_capabilities(
        AppProfile.PAPERS,
        zilliz_ready=cached_data.zilliz_ready,
        socket_io_enabled=False,
    )
    apply_profile_config(
        app,
        profile=AppProfile.PAPERS,
        capabilities=capabilities,
        socket_io_enabled=False,
    )
    _attach_logger(app, logger)

    logger.info(
        "Papers profile ready (paperSearch=%s vectorSearch=%s)",
        capabilities.paper_search,
        capabilities.vector_search,
    )
    return ApplicationBundle(
        profile=AppProfile.PAPERS,
        flask_app=app,
        socketio=None,
        capabilities=capabilities,
        logger=logger,
    )


def _attach_logger(app: Flask, logger) -> None:
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
