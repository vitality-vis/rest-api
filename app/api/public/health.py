"""Public health endpoint: profile + discovered capabilities."""

from flask import Blueprint, current_app, jsonify
from flask_cors import cross_origin


health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
@cross_origin()
def health():
    """Return the startup capability snapshot without external I/O."""
    profile_value = current_app.config.get("VITALITY_APP_PROFILE", "unknown")
    return jsonify(
        {
            "status": "ok",
            "profile": profile_value,
            "capabilities": current_app.config.get("VITALITY_CAPABILITIES") or {},
        }
    )
