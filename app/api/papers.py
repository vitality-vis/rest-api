"""Paper-list API endpoints."""

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

from model.query import QuerySchema
from repositories.zilliz.paper_repository import search_papers


MAX_PAPERS_PAGE_SIZE = 100
papers_bp = Blueprint("papers", __name__)


def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


@papers_bp.route("/getPapers", methods=["GET", "POST"])
@cross_origin()
def get_papers():
    """Fetch at most one bounded page of papers directly from Zilliz."""
    input_payload = request.args if request.method == "GET" else request.json or {}
    query = QuerySchema(
        title=input_payload.get("title"),
        abstract=input_payload.get("abstract"),
        author=input_payload.get("author"),
        source=input_payload.get("source"),
        keyword=input_payload.get("keyword"),
        min_year=input_payload.get("min_year"),
        max_year=input_payload.get("max_year"),
        min_citation_counts=input_payload.get("min_citation_counts"),
        max_citation_counts=input_payload.get("max_citation_counts"),
        id_list=input_payload.get("id_list"),
        offset=_bounded_int(input_payload.get("offset"), default=0, minimum=0, maximum=2**31 - 1),
        limit=_bounded_int(
            input_payload.get("limit"),
            default=MAX_PAPERS_PAGE_SIZE,
            minimum=1,
            maximum=MAX_PAPERS_PAGE_SIZE,
        ),
    )
    return jsonify(search_papers(query))
