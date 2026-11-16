"""Authenticated, project-scoped Notes completion API."""
from __future__ import annotations

import time
from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from repositories.supabase.auth import SupabaseAuthenticationError, SupabaseConfigurationError, verify_access_token
from repositories.supabase.note_completion_repository import (
    NoteCompletionContextError, NoteCompletionProjectNotFoundError,
    get_note_completion_papers,
)
from service.note_completion import NoteCompletionError, complete_note, sample_project_paper_titles
from service.note_completion_rate_limit import note_completion_rate_limiter
from service.paper_registry import LibraryPaperResolutionError, resolve_papers

note_completions_bp = Blueprint("note_completions", __name__)
MAX_CONTEXT_BEFORE = 2_000
MAX_CONTEXT_AFTER = 500
MAX_CITED_PAPERS = 8
MAX_CITED_PAPER_ID_CHARS = 128


def _authenticate() -> tuple[str | None, str | None, Response | None]:
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None, None, Response("Unauthorized", status=401, mimetype="text/plain")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None, None, Response("Unauthorized", status=401, mimetype="text/plain")
    try:
        access_token = token.strip()
        return verify_access_token(access_token), access_token, None
    except SupabaseConfigurationError:
        current_app.logger.error("Supabase is not configured for note completion")
        return None, None, Response("Note completion is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, None, Response("Unauthorized", status=401, mimetype="text/plain")


def _cited_paper_ids(value: dict) -> list[str]:
    if "cited_paper_ids" not in value:
        return []
    raw = value.get("cited_paper_ids")
    if not isinstance(raw, list):
        raise ValueError("cited_paper_ids must be an array")
    ids: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("cited_paper_ids must contain strings")
        paper_id = item.strip()
        if not paper_id or paper_id in seen:
            continue
        if len(paper_id) > MAX_CITED_PAPER_ID_CHARS:
            raise ValueError(
                f"cited_paper_ids entries must be at most {MAX_CITED_PAPER_ID_CHARS} characters"
            )
        seen.add(paper_id)
        ids.append(paper_id)
        if len(ids) >= MAX_CITED_PAPERS:
            break
    return ids


def _payload(value: object) -> tuple[str, str, str, list[str]]:
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
    return before, after, request_id.strip(), _cited_paper_ids(value)


def _resolve_paragraph_citations(*, user_id: str, paper_ids: list[str]) -> list[dict[str, object]]:
    if not paper_ids:
        return []
    try:
        # Skip OpenAlex: note completion is latency-sensitive; missing IDs stay omitted.
        return resolve_papers(
            user_id=user_id, paper_ids=paper_ids, hydrate_openalex=False
        )
    except LibraryPaperResolutionError as resolution_error:
        current_app.logger.warning(
            "Note completion citation resolve failed: %s", resolution_error
        )
        return []
    except Exception as unexpected_error:
        current_app.logger.warning(
            "Note completion citation resolve unexpected failure: %s", unexpected_error
        )
        return []


@note_completions_bp.route("/projects/<uuid:project_id>/notes/completions", methods=["POST"])
@cross_origin()
def create_note_completion(project_id):
    request_started = time.perf_counter()
    user_id, access_token, error = _authenticate()
    if error is not None:
        return error
    assert user_id is not None
    assert access_token is not None
    auth_completed = time.perf_counter()
    project_id_text = str(project_id)
    try:
        before, after, client_request_id, cited_paper_ids = _payload(
            request.get_json(silent=True)
        )
    except ValueError as error_value:
        return jsonify({"error": str(error_value)}), 400

    allowed, retry_after = note_completion_rate_limiter.check(
        f"{user_id}:{project_id_text}"
    )
    if not allowed:
        response = jsonify({"error": "Too many note completion requests"})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    try:
        papers = get_note_completion_papers(
            user_id=user_id, access_token=access_token, project_id=project_id_text
        )
        library = sample_project_paper_titles(papers)
        paragraph_citations = _resolve_paragraph_citations(
            user_id=user_id, paper_ids=cited_paper_ids
        )
        context_completed = time.perf_counter()
        suggestion, model, model_timing = complete_note(
            context_before_cursor=before,
            context_after_cursor=after,
            library=library,
            paragraph_citations=paragraph_citations,
        )
    except NoteCompletionProjectNotFoundError:
        return jsonify({"error": "Project not found"}), 404
    except NoteCompletionContextError as persistence_error:
        current_app.logger.error("Note completion context failed: %s", persistence_error)
        return Response("Note completion is unavailable", status=503, mimetype="text/plain")
    except NoteCompletionError as completion_error:
        current_app.logger.error("Note completion failed: %s", completion_error)
        return Response("Note completion is unavailable", status=503, mimetype="text/plain")

    response_completed = time.perf_counter()
    auth_ms = (auth_completed - request_started) * 1000
    context_ms = (context_completed - auth_completed) * 1000
    total_ms = (response_completed - request_started) * 1000
    response = jsonify({**suggestion, "client_request_id": client_request_id, "model": model})
    response.headers["Server-Timing"] = (
        f'auth;dur={auth_ms:.1f}, context;dur={context_ms:.1f}, '
        f'ttft;dur={model_timing["ttft_ms"]:.1f}, '
        f'generation;dur={model_timing["generation_ms"]:.1f}, '
        f'model;dur={model_timing["model_total_ms"]:.1f}, total;dur={total_ms:.1f}'
    )
    return response
