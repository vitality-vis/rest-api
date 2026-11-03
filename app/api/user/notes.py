"""Authenticated project-scoped research-notes API."""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.projects_repository import (
    ProjectNotFoundError,
    ProjectsPersistenceError,
    get_project,
)
from repositories.supabase.user_notes_repository import (
    UserNotesPersistenceError,
    get_user_note,
    upsert_user_note,
)

notes_bp = Blueprint("notes", __name__)
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


def _note_response(note: dict[str, object] | None, project_id: str) -> dict[str, object]:
    if note is None:
        return {
            "project_id": project_id,
            "content": "",
            "created_at": None,
            "updated_at": None,
        }
    content = note.get("content")
    return {
        "project_id": project_id,
        "content": content if isinstance(content, str) else "",
        "created_at": note.get("created_at"),
        "updated_at": note.get("updated_at"),
    }


def _notes_error(error: Exception):
    if isinstance(error, ProjectNotFoundError):
        return jsonify({"error": "Project not found"}), 404
    current_app.logger.error("Notes operation failed: %s", error)
    return Response("Notes are unavailable", status=503, mimetype="text/plain")


def _handle_get(project_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    assert user_id is not None
    try:
        project = get_project(user_id=user_id, project_id=project_id)
        resolved_project_id = str(project["id"])
        note = get_user_note(user_id=user_id, project_id=resolved_project_id)
    except (ProjectsPersistenceError, UserNotesPersistenceError) as error:
        return _notes_error(error)
    return jsonify(_note_response(note, resolved_project_id))


def _handle_put(project_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    assert user_id is not None
    try:
        content = _validate_content(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        project = get_project(user_id=user_id, project_id=project_id)
        resolved_project_id = str(project["id"])
        note = upsert_user_note(
            user_id=user_id, project_id=resolved_project_id, content=content
        )
    except (ProjectsPersistenceError, UserNotesPersistenceError) as error:
        return _notes_error(error)
    return jsonify(_note_response(note, resolved_project_id))


@notes_bp.route("/projects/<uuid:project_id>/notes", methods=["GET"])
@cross_origin()
def get_project_notes(project_id):
    return _handle_get(str(project_id))


@notes_bp.route("/projects/<uuid:project_id>/notes", methods=["PUT"])
@cross_origin()
def put_project_notes(project_id):
    return _handle_put(str(project_id))
