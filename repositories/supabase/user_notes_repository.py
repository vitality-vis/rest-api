"""Persistence for the authenticated user's single research-notes document."""

from __future__ import annotations

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10
_NOTE_COLUMNS = "id,user_id,content,created_at,updated_at"


class UserNotesPersistenceError(RuntimeError):
    """Raised when the user-notes database cannot complete an operation."""


def get_user_note(*, user_id: str) -> dict[str, object] | None:
    """Return the verified user's note row, or None when they have none yet."""
    response = _request(
        "GET",
        "user_notes",
        params={
            "user_id": f"eq.{user_id}",
            "select": _NOTE_COLUMNS,
            "limit": "1",
        },
    )
    if response.status_code != 200:
        raise UserNotesPersistenceError("Could not load user note")
    records = _json_list(response, "User note returned an invalid response")
    return records[0] if records else None


def upsert_user_note(*, user_id: str, content: str) -> dict[str, object]:
    """Insert or replace the verified user's note content by user_id."""
    response = _request(
        "POST",
        "user_notes",
        params={"on_conflict": "user_id"},
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        json={
            "user_id": user_id,
            "content": content,
        },
    )
    if response.status_code not in {200, 201}:
        raise UserNotesPersistenceError("Could not save user note")

    records = _json_list(response, "User note save returned an invalid response")
    if len(records) != 1:
        raise UserNotesPersistenceError("User note save returned an invalid response")
    return records[0]


def _request(method: str, path: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    headers = service_role_headers(settings)
    headers.update(kwargs.pop("headers", {}))
    try:
        return requests.request(
            method,
            f"{settings.url}/rest/v1/{path}",
            headers=headers,
            timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as error:
        raise UserNotesPersistenceError("Could not reach the user-notes database") from error


def _json_list(response: requests.Response, error_message: str) -> list[dict[str, object]]:
    try:
        payload = response.json()
    except ValueError as error:
        raise UserNotesPersistenceError(error_message) from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise UserNotesPersistenceError(error_message)
    return payload


__all__ = [
    "UserNotesPersistenceError",
    "get_user_note",
    "upsert_user_note",
]
