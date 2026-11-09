"""Minimal Supabase RPC context for latency-sensitive note completions."""
from __future__ import annotations

import requests

from repositories.supabase.client import get_supabase_settings, pooled_request
from repositories.supabase.note_completion_cache import (
    cache_note_completion_papers, get_cached_note_completion_papers,
)

DATABASE_REQUEST_TIMEOUT_SECONDS = 10


class NoteCompletionContextError(RuntimeError):
    """Raised when note-completion context cannot be loaded."""


class NoteCompletionProjectNotFoundError(NoteCompletionContextError):
    """Raised when the project is absent or not owned by the caller."""


def get_note_completion_papers(
    *, user_id: str, access_token: str, project_id: str
) -> list[dict[str, object]]:
    """Return only ordered paper identifiers and titles from one authenticated RPC."""
    cached = get_cached_note_completion_papers(user_id=user_id, project_id=project_id)
    if cached is not None:
        return cached

    settings = get_supabase_settings()
    try:
        response = pooled_request(
            "POST",
            f"{settings.url}/rest/v1/rpc/get_note_completion_context",
            headers={
                "apikey": settings.service_role_key,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"p_project_id": project_id},
            timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise NoteCompletionContextError(
            "Could not reach note-completion context database"
        ) from error

    if response.status_code not in {200, 201}:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("code") == "P0002":
            raise NoteCompletionProjectNotFoundError("Project not found")
        raise NoteCompletionContextError("Could not load note-completion context")

    try:
        payload = response.json()
    except ValueError as error:
        raise NoteCompletionContextError(
            "Note-completion context returned invalid JSON"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise NoteCompletionContextError("Note-completion context returned invalid data")

    papers: list[dict[str, object]] = []
    for item in payload["papers"]:
        if not isinstance(item, dict):
            raise NoteCompletionContextError("Note-completion context returned invalid data")
        paper_id = item.get("paper_id")
        title = item.get("title")
        if not isinstance(paper_id, str) or not isinstance(title, str):
            raise NoteCompletionContextError("Note-completion context returned invalid data")
        papers.append({"paper_id": paper_id, "metadata_snapshot": {"Title": title}})
    cache_note_completion_papers(user_id=user_id, project_id=project_id, papers=papers)
    return papers
