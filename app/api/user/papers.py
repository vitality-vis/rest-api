"""User-facing paper resolution (public corpus + authenticated library)."""

from flask import Blueprint, Response, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from service.paper_registry import LibraryPaperResolutionError, resolve_papers


MAX_PAPERS_PAGE_SIZE = 100
user_papers_bp = Blueprint("user_papers", __name__)


def _optional_authenticated_user_id() -> str | None:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise SupabaseAuthenticationError("Malformed authorization header")
    return verify_access_token(access_token.strip())


@user_papers_bp.route("/papers/resolve", methods=["POST"])
@cross_origin()
def resolve_papers_endpoint():
    """Resolve public corpus IDs and the current user's library IDs."""
    payload = request.get_json(silent=True)
    paper_ids = payload.get("paper_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(paper_ids, list)
        or not paper_ids
        or len(paper_ids) > MAX_PAPERS_PAGE_SIZE
        or not all(isinstance(paper_id, str) and paper_id and len(paper_id) <= 1024 for paper_id in paper_ids)
    ):
        return jsonify({"error": "paper_ids must contain 1 to 100 IDs"}), 400
    try:
        user_id = _optional_authenticated_user_id()
    except SupabaseConfigurationError:
        return Response("Paper resolution is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return Response("Unauthorized", status=401, mimetype="text/plain")
    try:
        return jsonify({"papers": resolve_papers(user_id=user_id, paper_ids=paper_ids)})
    except LibraryPaperResolutionError:
        return Response("Paper metadata is unavailable", status=503, mimetype="text/plain")
