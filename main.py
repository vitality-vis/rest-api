"""Unique process entry: profile-aware Flask (and Socket.IO) composition root."""

from __future__ import annotations

import argparse
import os

import config

config.load_project_environment()

from app.application import create_application

_bundle = create_application()
app = _bundle.flask_app
socketio = _bundle.socketio
logger = _bundle.logger


def get_rag_agent():
    """Legacy helper retained for callers; chat sessions own the agent instance."""
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VitaLITy REST API")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 3000))
    debug_mode = args.debug
    profile = _bundle.profile.value
    print(f"Starting VitaLITy API profile={profile} on http://localhost:{port}")
    print(f"Debug mode: {debug_mode}")

    if socketio is not None:
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode,
            allow_unsafe_werkzeug=True,
        )
    else:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode,
        )
