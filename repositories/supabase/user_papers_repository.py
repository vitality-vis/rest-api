"""Persistence for the authenticated user's saved-paper shelf."""

from __future__ import annotations

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10
_SHELF_COLUMNS = (
    "id,user_id,paper_id,metadata_snapshot,azure_file_id,uploaded_filename,"
    "uploaded_bytes,uploaded_at,created_at,updated_at"
)


class UserPapersPersistenceError(RuntimeError):
    """Raised when the user-papers database cannot complete an operation."""


class UserPaperNotFoundError(UserPapersPersistenceError):
    """Raised when a paper is not on the verified user's shelf."""


def list_user_papers(*, user_id: str) -> list[dict[str, object]]:
    """Return saved papers for one verified user, newest first."""
    response = _request(
        "GET",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "select": _SHELF_COLUMNS,
            "order": "created_at.desc",
        },
    )
    if response.status_code != 200:
        raise UserPapersPersistenceError("Could not load user papers")
    return _json_list(response, "User papers returned an invalid response")


def save_user_paper(
    *, user_id: str, paper_id: str, metadata_snapshot: dict[str, object]
) -> tuple[dict[str, object], bool]:
    """Create or update one user-owned paper and report whether it was new."""
    response = _request(
        "POST",
        "user_papers",
        params={"on_conflict": "user_id,paper_id"},
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        json={
            "user_id": user_id,
            "paper_id": paper_id,
            "metadata_snapshot": metadata_snapshot,
        },
    )
    # PostgREST returns 201 on insert and 200 when merge-duplicates updates.
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not save user paper")

    records = _json_list(response, "User paper save returned an invalid response")
    if len(records) != 1:
        raise UserPapersPersistenceError("User paper save returned an invalid response")
    return records[0], response.status_code == 201


def import_user_papers(
    *, user_id: str, papers: list[tuple[str, dict[str, object]]]
) -> list[dict[str, object]]:
    """Idempotently create or update a batch of papers for one user."""
    response = _request(
        "POST",
        "user_papers",
        params={"on_conflict": "user_id,paper_id"},
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        json=[
            {
                "user_id": user_id,
                "paper_id": paper_id,
                "metadata_snapshot": metadata_snapshot,
            }
            for paper_id, metadata_snapshot in papers
        ],
    )
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not import user papers")
    return _json_list(response, "User paper import returned an invalid response")


def delete_user_paper(*, user_id: str, paper_id: str) -> None:
    """Delete one paper only when it belongs to the verified user."""
    response = _request(
        "DELETE",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "paper_id": f"eq.{paper_id}",
        },
        headers={"Prefer": "return=representation"},
    )
    if response.status_code not in {200, 204}:
        raise UserPapersPersistenceError("Could not delete user paper")
    if response.status_code == 204 or not response.content:
        raise UserPaperNotFoundError("User paper does not exist")
    if not _json_list(response, "User paper delete returned an invalid response"):
        raise UserPaperNotFoundError("User paper does not exist")


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
        raise UserPapersPersistenceError("Could not reach the user-papers database") from error


def _json_list(response: requests.Response, error_message: str) -> list[dict[str, object]]:
    try:
        payload = response.json()
    except ValueError as error:
        raise UserPapersPersistenceError(error_message) from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise UserPapersPersistenceError(error_message)
    return payload


__all__ = [
    "UserPaperNotFoundError",
    "UserPapersPersistenceError",
    "delete_user_paper",
    "import_user_papers",
    "list_user_papers",
    "save_user_paper",
]
