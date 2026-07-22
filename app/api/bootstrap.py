"""Endpoints that supply data needed to initialise the client application."""

from flask import Blueprint, current_app, jsonify
from flask_cors import cross_origin

import config
from service import zilliz
from service.static_cache import cached_data


bootstrap_bp = Blueprint("bootstrap", __name__)


@bootstrap_bp.route("/getPublicConfig", methods=["GET"])
@cross_origin()
def get_public_config():
    """Return non-sensitive runtime settings needed by the browser."""
    return jsonify({"libraryPdfMaxBytes": config.LIBRARY_PDF_MAX_BYTES})


@bootstrap_bp.route("/getUmapPoints", methods=["GET"])
@cross_origin()
def get_umap_points():
    """Return cached UMAP projection points used to initialise the visualization."""
    return jsonify(cached_data.get_umap_points())


@bootstrap_bp.route("/getMetaData", methods=["GET"])
@cross_origin()
def get_metadata():
    """Return cached filter facets, falling back to a live aggregation if needed."""
    cached_metadata = cached_data.get_aggregated_metadata()
    if cached_metadata:
        return jsonify(cached_metadata)

    current_app.logger.warning(
        "Metadata cache not available; computing filter facets in real time"
    )
    return jsonify({
        "authors_summary": zilliz.get_distinct_authors_with_counts(),
        "sources_summary": zilliz.get_distinct_sources_with_counts(),
        "keywords_summary": zilliz.get_distinct_keywords_with_counts(),
        "years_summary": zilliz.get_distinct_years_with_counts(),
        "citation_counts": zilliz.get_distinct_citation_counts(),
    })
