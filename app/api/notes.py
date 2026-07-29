"""Authenticated API endpoints for a user's single research-notes document."""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.user_notes_repository import (
    UserNotesPersistenceError,
    get_user_note,
    upsert_user_note,
)


notes_bp = Blueprint("notes", __name__)

# Large enough for research notes; still bounds accidental / hostile payloads.
MAX_NOTES_CONTENT_LENGTH = 500_000


def _get_authenticated_user_id() -> str:
    authorization = request.headers.get("Authorization")
    if not authorization:
        raise SupabaseAuthenticationError("Missing authorization header")
    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise SupabaseAuthenticationError("Malformed authorization header")
    return verify_access_token(access_token.strip())


def _require_authenticated_user_id() -> tuple[str | None, Response | None]:
    try:
        return _get_authenticated_user_id(), None
    except SupabaseConfigurationError:
        current_app.logger.error("Supabase is not configured for notes")
        return None, Response("Notes are unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")


def _validate_content(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("content is required")
    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if len(content) > MAX_NOTES_CONTENT_LENGTH:
        raise ValueError("content is too large")
    return content


def _note_response(note: dict[str, object] | None) -> dict[str, object]:
    """Always return a note-shaped payload; missing rows become empty content."""
    if note is None:
        return {
            "content": "",
            "created_at": None,
            "updated_at": None,
        }
    content = note.get("content")
    return {
        "content": content if isinstance(content, str) else "",
        "created_at": note.get("created_at"),
        "updated_at": note.get("updated_at"),
    }


@notes_bp.route("/notes", methods=["GET"])
@cross_origin()
def get_notes():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        note = get_user_note(user_id=user_id)
    except UserNotesPersistenceError as error:
        current_app.logger.error("Could not load user note: %s", error)
        return Response("Notes are unavailable", status=503, mimetype="text/plain")
    return jsonify(_note_response(note))


@notes_bp.route("/notes", methods=["PUT"])
@cross_origin()
def put_notes():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        content = _validate_content(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        note = upsert_user_note(user_id=user_id, content=content)
    except UserNotesPersistenceError as error:
        current_app.logger.error("Could not save user note: %s", error)
        return Response("Notes are unavailable", status=503, mimetype="text/plain")
    return jsonify(_note_response(note))
