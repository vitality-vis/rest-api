"""Shared Flask wiring used by profile factories."""

from __future__ import annotations

from flask import Flask
from flask_compress import Compress
from flask_cors import CORS

import config
from app.profiles import AppProfile, Capabilities


def create_flask_app(*, serve_frontend: bool) -> Flask:
    """Create a Flask app with shared CORS/compression defaults."""
    if serve_frontend:
        app = Flask(
            __name__,
            static_folder=config.FRONTEND_DIST_DIR,
            static_url_path="/",
        )
    else:
        app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config["CORS_HEADERS"] = "Content-Type, Authorization"
    Compress(app)
    return app


def apply_profile_config(
    app: Flask,
    *,
    profile: AppProfile,
    capabilities: Capabilities,
    socket_io_enabled: bool = False,
) -> None:
    app.config["VITALITY_APP_PROFILE"] = profile.value
    app.config["VITALITY_SOCKET_IO_ENABLED"] = socket_io_enabled
    app.config["VITALITY_CAPABILITIES"] = capabilities.as_dict()
