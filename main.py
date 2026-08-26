"""Unique process entry: profile-aware ASGI composition root."""

from __future__ import annotations

import argparse
import os

import config

config.load_project_environment()

from app.application import create_application
from app.runtime import run_local

_bundle = create_application()
app = _bundle.asgi_app
socketio = _bundle.socketio  # ASGI AsyncServer for full; None for papers
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
        help="Set Uvicorn log level to debug (default: False)",
    )
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 3000))
    profile = _bundle.profile.value
    print(f"Starting VitaLITy API profile={profile} on http://localhost:{port}")
    print(f"Uvicorn log level: {'debug' if args.debug else 'info'}")
    # Pass the app object (not "main:app") to avoid a second import of this module.
    run_local(app, port=port, debug=args.debug)
