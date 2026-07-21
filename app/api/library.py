"""Authenticated API endpoints for a user's saved-paper library."""

from __future__ import annotations

import math

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.user_papers_repository import (
    UserPaperNotFoundError,
    UserPapersPersistenceError,
    delete_user_paper,
    import_user_papers,
    list_user_papers,
    save_user_paper,
)


library_bp = Blueprint("library", __name__)

MAX_PAPER_ID_LENGTH = 200
MAX_TITLE_LENGTH = 2_000
MAX_ABSTRACT_LENGTH = 100_000
MAX_SOURCE_LENGTH = 2_000
MAX_LIST_ITEMS = 500
MAX_LIST_ITEM_LENGTH = 2_000
MAX_IMPORT_PAPERS = 100


def _get_authenticated_user_id() -> str:
    authorization = request.headers.get("Authorization")
    if not authorization:
        raise SupabaseAuthenticationError("Missing authorization header")
    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise SupabaseAuthenticationError("Malformed authorization header")
    return verify_access_token(access_token.strip())


def _validate_paper_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PAPER_ID_LENGTH:
        raise ValueError("paper_id is invalid")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("paper_id is invalid")
    return value


def _bounded_text(value: object, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{field_name} is invalid")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise ValueError(f"{field_name} is invalid")
    return [_bounded_text(item, field_name, MAX_LIST_ITEM_LENGTH) for item in value]


def _nullable_integer(value: object, field_name: str) -> int | None:
    """Accept ints or whole-number floats (Zilliz/JSON may surface either)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} is invalid")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    raise ValueError(f"{field_name} is invalid")


def _metadata_paper_id(value: object) -> str:
    """Normalise body ID to the string form used in path params and filters."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Paper metadata ID must match paper_id")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return str(int(value))
    raise ValueError("Paper metadata ID must match paper_id")


def _nullable_coordinate(value: object, field_name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} is invalid")
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{field_name} is invalid")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError(f"{field_name} is invalid")
        coordinates.append(coordinate)
    return coordinates


def _validate_metadata_snapshot(value: object, paper_id: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Paper metadata must be an object")

    # Ignore response-only fields such as Sim / score from paper_to_api_response.
    if _metadata_paper_id(value.get("ID")) != paper_id:
        raise ValueError("Paper metadata ID must match paper_id")

    return {
        "ID": paper_id,
        "Title": _bounded_text(value.get("Title"), "Title", MAX_TITLE_LENGTH),
        "Abstract": _bounded_text(value.get("Abstract"), "Abstract", MAX_ABSTRACT_LENGTH),
        "Authors": _string_list(value.get("Authors"), "Authors"),
        "Keywords": _string_list(value.get("Keywords"), "Keywords"),
        "Source": _bounded_text(value.get("Source"), "Source", MAX_SOURCE_LENGTH),
        "Year": _nullable_integer(value.get("Year"), "Year"),
        "CitationCounts": _nullable_integer(value.get("CitationCounts"), "CitationCounts"),
        "ada_umap": _nullable_coordinate(value.get("ada_umap"), "ada_umap"),
        "specter_umap": _nullable_coordinate(value.get("specter_umap"), "specter_umap"),
    }


def _require_authenticated_user_id() -> tuple[str | None, Response | None]:
    try:
        return _get_authenticated_user_id(), None
    except SupabaseConfigurationError:
        current_app.logger.error("Supabase is not configured for the library")
        return None, Response("Library is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")


def _save_response(paper: dict[str, object]) -> dict[str, object]:
    """Return the compact representation defined by the PUT contract."""
    snapshot = paper.get("metadata_snapshot")
    title = snapshot.get("Title") if isinstance(snapshot, dict) else None
    return {
        "id": paper.get("id"),
        "paper_id": paper.get("paper_id"),
        "title": title,
        "azure_file_id": paper.get("azure_file_id"),
    }


def _validate_import_papers(value: object) -> list[tuple[str, dict[str, object]]]:
    if not isinstance(value, dict) or not isinstance(value.get("papers"), list):
        raise ValueError("papers must be an array")
    papers = value["papers"]
    if not papers or len(papers) > MAX_IMPORT_PAPERS:
        raise ValueError(f"papers must contain between 1 and {MAX_IMPORT_PAPERS} items")

    validated: list[tuple[str, dict[str, object]]] = []
    paper_ids: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            raise ValueError("Paper metadata must be an object")
        paper_id = _validate_paper_id(_metadata_paper_id(paper.get("ID")))
        if paper_id in paper_ids:
            raise ValueError("papers must not contain duplicate IDs")
        paper_ids.add(paper_id)
        validated.append((paper_id, _validate_metadata_snapshot(paper, paper_id)))
    return validated


@library_bp.route("/library/papers", methods=["GET"])
@cross_origin()
def get_library_papers():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        return jsonify({"papers": list_user_papers(user_id=user_id)})
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not load library papers: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")


@library_bp.route("/library/papers/import", methods=["POST"])
@cross_origin()
def import_library_papers():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        papers = _validate_import_papers(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        imported_papers = import_user_papers(user_id=user_id, papers=papers)
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not import library papers: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return jsonify({"papers": [_save_response(paper) for paper in imported_papers]})


@library_bp.route("/library/papers/<paper_id>", methods=["PUT"])
@cross_origin()
def put_library_paper(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
        metadata_snapshot = _validate_metadata_snapshot(request.get_json(silent=True), paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        paper, was_created = save_user_paper(
            user_id=user_id, paper_id=paper_id, metadata_snapshot=metadata_snapshot
        )
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not save library paper: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return jsonify(_save_response(paper)), 201 if was_created else 200


@library_bp.route("/library/papers/<paper_id>", methods=["DELETE"])
@cross_origin()
def remove_library_paper(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        delete_user_paper(user_id=user_id, paper_id=paper_id)
    except UserPaperNotFoundError:
        return Response("Not found", status=404, mimetype="text/plain")
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not delete library paper: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return Response(status=204)


__all__ = ["library_bp"]
