"""Standalone low-effort paper-search endpoint."""
from __future__ import annotations

from flask import Blueprint, Response, jsonify, request
from flask_cors import cross_origin
from pydantic import ValidationError

from agents.search_v2.models import MAX_SEARCH_QUERY_LENGTH, SearchV2Request
from agents.search_v2.runner import SearchCriteriaRequiredError, run_search
from agents.search_v2.logging import SearchV2Trace
from service.search import SearchUnavailableError


search_v2_bp = Blueprint("search_v2", __name__)


@search_v2_bp.route("/search/v2", methods=["POST"])
@cross_origin()
def search_v2():
    trace = SearchV2Trace.create()
    data = request.get_json(force=True) or {}
    try:
        payload = SearchV2Request.model_validate(data) if hasattr(SearchV2Request, "model_validate") else SearchV2Request.parse_obj(data)
    except ValidationError as error:
        query = data.get("query") if isinstance(data, dict) else None
        if isinstance(query, str) and len(query) > MAX_SEARCH_QUERY_LENGTH:
            response = jsonify({"error": f"Query is too long. Please keep it within {MAX_SEARCH_QUERY_LENGTH:,} characters."})
            response.headers["X-Trace-Id"] = trace.trace_id
            return response, 400
        response = jsonify({"error": "Invalid search request", "details": error.errors()})
        response.headers["X-Trace-Id"] = trace.trace_id
        return response, 400

    try:
        response = run_search(payload, trace=trace)
    except SearchCriteriaRequiredError as error:
        response = jsonify({"error": str(error)})
        response.headers["X-Trace-Id"] = trace.trace_id
        return response, 400
    except SearchUnavailableError as error:
        response = jsonify({"error": str(error)})
        response.headers["X-Trace-Id"] = trace.trace_id
        return response, 503
    body = response.model_dump(by_alias=True) if hasattr(response, "model_dump") else response.dict(by_alias=True)
    flask_response = jsonify(body)
    flask_response.headers["X-Trace-Id"] = trace.trace_id
    return flask_response
