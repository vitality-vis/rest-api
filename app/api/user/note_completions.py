"""Authenticated, project-scoped Notes completion API."""
from __future__ import annotations

import time
from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import SupabaseAuthenticationError, SupabaseConfigurationError, verify_access_token
from repositories.supabase.project_papers_repository import ProjectPapersPersistenceError, list_project_papers
from repositories.supabase.projects_repository import ProjectNotFoundError, ProjectsPersistenceError, get_project
from service.note_completion import NoteCompletionError, complete_note, sample_project_paper_titles

note_completions_bp = Blueprint("note_completions", __name__)
MAX_CONTEXT_BEFORE = 2_000
MAX_CONTEXT_AFTER = 500


def _authenticate() -> tuple[str | None, Response | None]:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None, Response("Unauthorized", status=401, mimetype="text/plain")
    try:
        return verify_access_token(token.strip()), None
    except SupabaseConfigurationError:
        current_app.logger.error("Supabase is not configured for note completion")
        return None, Response("Note completion is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")


def _payload(value: object) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object")
    before = value.get("context_before_cursor")
    after = value.get("context_after_cursor", "")
    request_id = value.get("client_request_id")
    if not isinstance(before, str) or not before.strip():
        raise ValueError("context_before_cursor is required")
    if not isinstance(after, str):
        raise ValueError("context_after_cursor must be a string")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("client_request_id is required")
    if len(before) > MAX_CONTEXT_BEFORE:
        raise ValueError(f"context_before_cursor exceeds {MAX_CONTEXT_BEFORE} characters")
    if len(after) > MAX_CONTEXT_AFTER:
        raise ValueError(f"context_after_cursor exceeds {MAX_CONTEXT_AFTER} characters")
    return before, after, request_id.strip()


@note_completions_bp.route("/projects/<uuid:project_id>/notes/completions", methods=["POST"])
@cross_origin()
def create_note_completion(project_id):
    request_started = time.perf_counter()
    user_id, error = _authenticate()
    if error is not None:
        return error
    assert user_id is not None
    auth_completed = time.perf_counter()
    try:
        before, after, client_request_id = _payload(request.get_json(silent=True))
    except ValueError as error_value:
        return jsonify({"error": str(error_value)}), 400

    project_id_text = str(project_id)
    try:
        project = get_project(user_id=user_id, project_id=project_id_text)
        papers = list_project_papers(user_id=user_id, project_id=str(project["id"]))
        library = sample_project_paper_titles(papers)
        context_completed = time.perf_counter()
        suggestion, model, model_timing = complete_note(
            context_before_cursor=before,
            context_after_cursor=after,
            library=library,
        )
    except ProjectNotFoundError:
        return jsonify({"error": "Project not found"}), 404
    except (ProjectsPersistenceError, ProjectPapersPersistenceError) as persistence_error:
        current_app.logger.error("Note completion context failed: %s", persistence_error)
        return Response("Note completion is unavailable", status=503, mimetype="text/plain")
    except NoteCompletionError as completion_error:
        current_app.logger.error("Note completion failed: %s", completion_error)
        return Response("Note completion is unavailable", status=503, mimetype="text/plain")

    response_completed = time.perf_counter()
    auth_ms = (auth_completed - request_started) * 1000
    context_ms = (context_completed - auth_completed) * 1000
    total_ms = (response_completed - request_started) * 1000
    current_app.logger.info(
        "Note completion timing request=%s kind=%s model=%s auth_ms=%.1f "
        "context_ms=%.1f ttft_ms=%.1f generation_ms=%.1f model_total_ms=%.1f total_ms=%.1f",
        client_request_id,
        suggestion["kind"],
        model,
        auth_ms,
        context_ms,
        model_timing["ttft_ms"],
        model_timing["generation_ms"],
        model_timing["model_total_ms"],
        total_ms,
    )
    response = jsonify({**suggestion, "client_request_id": client_request_id, "model": model})
    response.headers["Server-Timing"] = (
        f'auth;dur={auth_ms:.1f}, context;dur={context_ms:.1f}, '
        f'ttft;dur={model_timing["ttft_ms"]:.1f}, '
        f'generation;dur={model_timing["generation_ms"]:.1f}, '
        f'model;dur={model_timing["model_total_ms"]:.1f}, total;dur={total_ms:.1f}'
    )
    return response
