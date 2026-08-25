"""Full-app browser bootstrap config (not part of the papers profile)."""

from flask import Blueprint, jsonify
from flask_cors import cross_origin

import config


app_config_bp = Blueprint("app_config", __name__)


@app_config_bp.route("/getPublicConfig", methods=["GET"])
@cross_origin()
def get_public_config():
    """Return non-sensitive runtime settings needed by the browser."""
    return jsonify(
        {
            "libraryPdfMaxBytes": config.LIBRARY_PDF_MAX_BYTES,
            "availableModels": list(config.AZURE_OPENAI_AVAILABLE_MODELS.keys()),
            "defaultModel": config.AZURE_OPENAI_DEFAULT_MODEL or None,
        }
    )
