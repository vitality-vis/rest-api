"""Public health endpoint: profile + discovered capabilities."""

from flask import Blueprint, current_app, jsonify
from flask_cors import cross_origin


health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
@cross_origin()
def health():
    """Return the startup capability snapshot without external I/O.

    Full profile may include ``agentRuntime`` (admission/metrics snapshot).
    Papers profile returns ``agentRuntime: null``. No user/run identifiers.
    """
    profile_value = current_app.config.get("VITALITY_APP_PROFILE", "unknown")
    snapshot_provider = current_app.config.get("VITALITY_AGENT_RUNTIME_SNAPSHOT")
    agent_runtime = None
    if callable(snapshot_provider):
        snapshot = snapshot_provider()
        agent_runtime = snapshot.as_dict() if snapshot is not None else None

    return jsonify(
        {
            "status": "ok",
            "profile": profile_value,
            "capabilities": current_app.config.get("VITALITY_CAPABILITIES") or {},
            "agentRuntime": agent_runtime,
        }
    )
