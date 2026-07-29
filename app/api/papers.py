"""Paper-list API endpoints."""

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from pydantic import ValidationError

from config import is_supported_embedding_model
from model.paper import (
    GetPapersResponse,
    PaperCitationsRequest,
    PaperCitationsResponse,
    SearchRequest,
    SimilarPapersRequest,
)
from service.citations import (
    PaperCitationsNotFoundError,
    PaperCitationsProviderError,
    PaperCitationsUnavailableError,
    get_paper_citations,
)
from service.search import SearchUnavailableError, find_similar_by_papers, search


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
    try:
        query = SearchRequest(
            search_query=input_payload.get("search_query"),
            search_mode=input_payload.get("search_mode", "exact"),
            embedding_model=input_payload.get("embedding_model"),
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
    except ValidationError as error:
        return jsonify({"error": "Invalid getPapers request", "details": error.errors()}), 400
    if query.search_mode == "vector" and query.embedding_model and not is_supported_embedding_model(
        query.embedding_model
    ):
        return jsonify({"error": "Unsupported embedding_model"}), 400
    try:
        result = search(query)
    except SearchUnavailableError:
        return jsonify({"error": "Paper search is temporarily unavailable"}), 503
    response = GetPapersResponse(
        papers=result.papers,
        total=result.total,
        has_more=result.has_more,
    )
    if hasattr(response, "model_dump"):
        return jsonify(response.model_dump(by_alias=True, exclude_none=True))
    return jsonify(response.dict(by_alias=True, exclude_none=True))


@papers_bp.route("/getSimilarPapers", methods=["POST"])
@cross_origin()
def get_similar_papers():
    """Return papers related to one or more seed papers using RRF."""
    input_payload = request.json or {}
    if not isinstance(input_payload, dict):
        return jsonify({"error": "getSimilarPapers request body must be a JSON object"}), 400
    try:
        query = SimilarPapersRequest(
            seed_ids=input_payload.get("seed_ids") or [],
            limit=_bounded_int(
                input_payload.get("limit"),
                default=25,
                minimum=1,
                maximum=MAX_PAPERS_PAGE_SIZE,
            ),
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
        )
    except ValidationError as error:
        return jsonify({"error": "Invalid getSimilarPapers request", "details": error.errors()}), 400
    if not any(str(seed_id).strip() for seed_id in query.seed_ids):
        return jsonify({"error": "seed_ids must contain at least one paper ID"}), 400
    try:
        result = find_similar_by_papers(query)
    except SearchUnavailableError:
        return jsonify({"error": "Similar-paper search is temporarily unavailable"}), 503
    response = GetPapersResponse(papers=result.papers, has_more=result.has_more)
    if hasattr(response, "model_dump"):
        return jsonify(response.model_dump(by_alias=True, exclude_none=True))
    return jsonify(response.dict(by_alias=True, exclude_none=True))


@papers_bp.route("/getPaperCitations", methods=["POST"])
@cross_origin()
def get_paper_citations_endpoint():
    """Return one paper's references and cited-by works from OpenAlex."""
    input_payload = request.get_json(silent=True)
    if not isinstance(input_payload, dict):
        return jsonify(
            {"error": "getPaperCitations request body must be a JSON object"}
        ), 400

    raw_doi = input_payload.get("doi")
    try:
        query = PaperCitationsRequest(
            doi=raw_doi.strip() if isinstance(raw_doi, str) else raw_doi,
            limit=input_payload.get("limit", 50),
        )
    except ValidationError as error:
        return jsonify(
            {"error": "Invalid getPaperCitations request", "details": error.errors()}
        ), 400

    try:
        result = get_paper_citations(query.doi, query.limit)
    except PaperCitationsNotFoundError:
        return jsonify({"error": "Paper was not found in OpenAlex"}), 404
    except PaperCitationsUnavailableError:
        return jsonify({"error": "Paper citations are temporarily unavailable"}), 503
    except PaperCitationsProviderError:
        return jsonify({"error": "OpenAlex citation lookup failed"}), 502

    response = PaperCitationsResponse(**result)
    if hasattr(response, "model_dump"):
        return jsonify(response.model_dump(exclude_none=True))
    return jsonify(response.dict(exclude_none=True))
