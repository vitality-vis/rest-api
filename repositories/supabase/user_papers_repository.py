"""Persistence for the authenticated user's personal paper library."""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10
_SHELF_COLUMNS = (
    "id,user_id,paper_id,metadata_snapshot,is_saved,azure_file_id,uploaded_filename,"
    "uploaded_bytes,uploaded_at,created_at,updated_at"
)


class UserPapersPersistenceError(RuntimeError):
    """Raised when the user-papers database cannot complete an operation."""


class UserPaperNotFoundError(UserPapersPersistenceError):
    """Raised when a paper is not on the verified user's shelf."""


def list_user_papers(*, user_id: str, saved_only: bool = False) -> list[dict[str, object]]:
    """Return library papers for one verified user, newest first."""
    params: dict[str, str] = {
        "user_id": f"eq.{user_id}",
        "select": _SHELF_COLUMNS,
        "order": "created_at.desc",
    }
    if saved_only:
        params["is_saved"] = "eq.true"

    response = _request("GET", "user_papers", params=params)
    if response.status_code != 200:
        raise UserPapersPersistenceError("Could not load user papers")
    return _json_list(response, "User papers returned an invalid response")


def get_user_paper(*, user_id: str, paper_id: str) -> dict[str, object] | None:
    """Return one library paper for the verified user, or None if missing."""
    response = _request(
        "GET",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "paper_id": f"eq.{paper_id}",
            "select": _SHELF_COLUMNS,
            "limit": "1",
        },
    )
    if response.status_code != 200:
        raise UserPapersPersistenceError("Could not load user paper")
    records = _json_list(response, "User paper returned an invalid response")
    return records[0] if records else None


def save_user_paper(
    *, user_id: str, paper_id: str, metadata_snapshot: dict[str, object]
) -> tuple[dict[str, object], bool]:
    """Upsert one paper and set is_saved = true; report whether the row was new."""
    response = _request(
        "POST",
        "user_papers",
        params={"on_conflict": "user_id,paper_id"},
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        json={
            "user_id": user_id,
            "paper_id": paper_id,
            "metadata_snapshot": metadata_snapshot,
            "is_saved": True,
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
    """Idempotently upsert a batch of papers as saved for one user."""
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
                "is_saved": True,
            }
            for paper_id, metadata_snapshot in papers
        ],
    )
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not import user papers")
    return _json_list(response, "User paper import returned an invalid response")


def unsave_user_paper(*, user_id: str, paper_id: str) -> None:
    """Clear saved state; delete the row only when it has no uploaded file."""
    paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    if paper is None:
        raise UserPaperNotFoundError("User paper does not exist")

    if paper.get("azure_file_id"):
        response = _request(
            "PATCH",
            "user_papers",
            params={
                "user_id": f"eq.{user_id}",
                "paper_id": f"eq.{paper_id}",
            },
            headers={"Prefer": "return=representation"},
            json={"is_saved": False},
        )
        if response.status_code not in {200, 204}:
            raise UserPapersPersistenceError("Could not unsave user paper")
        if response.status_code == 204 or not response.content:
            raise UserPaperNotFoundError("User paper does not exist")
        if not _json_list(response, "User paper unsave returned an invalid response"):
            raise UserPaperNotFoundError("User paper does not exist")
        return

    delete_user_paper(user_id=user_id, paper_id=paper_id)


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


def upsert_user_paper_file(
    *,
    user_id: str,
    paper_id: str,
    metadata_snapshot: dict[str, object],
    azure_file_id: str,
    uploaded_filename: str,
    uploaded_bytes: int,
    create_if_missing: bool,
) -> dict[str, object]:
    """Write file metadata onto an existing row, or create an unsaved upload-only row."""
    existing = get_user_paper(user_id=user_id, paper_id=paper_id)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    file_fields = {
        "metadata_snapshot": metadata_snapshot,
        "azure_file_id": azure_file_id,
        "uploaded_filename": uploaded_filename,
        "uploaded_bytes": uploaded_bytes,
        "uploaded_at": uploaded_at,
    }

    if existing is None:
        if not create_if_missing:
            raise UserPaperNotFoundError("User paper does not exist")
        response = _request(
            "POST",
            "user_papers",
            headers={"Prefer": "return=representation"},
            json={
                "user_id": user_id,
                "paper_id": paper_id,
                "is_saved": False,
                **file_fields,
            },
        )
        if response.status_code not in {200, 201}:
            raise UserPapersPersistenceError("Could not create user paper file record")
        records = _json_list(response, "User paper file create returned an invalid response")
        if len(records) != 1:
            raise UserPapersPersistenceError("User paper file create returned an invalid response")
        return records[0]

    response = _request(
        "PATCH",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "paper_id": f"eq.{paper_id}",
        },
        headers={"Prefer": "return=representation"},
        json=file_fields,
    )
    if response.status_code not in {200, 204}:
        raise UserPapersPersistenceError("Could not update user paper file metadata")
    if response.status_code == 204 or not response.content:
        raise UserPaperNotFoundError("User paper does not exist")
    records = _json_list(response, "User paper file update returned an invalid response")
    if len(records) != 1:
        raise UserPapersPersistenceError("User paper file update returned an invalid response")
    return records[0]


def clear_user_paper_file(*, user_id: str, paper_id: str) -> None:
    """Clear upload fields; delete the row when it is also unsaved."""
    paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    if paper is None:
        raise UserPaperNotFoundError("User paper does not exist")

    if not paper.get("is_saved"):
        delete_user_paper(user_id=user_id, paper_id=paper_id)
        return

    response = _request(
        "PATCH",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "paper_id": f"eq.{paper_id}",
        },
        headers={"Prefer": "return=representation"},
        json={
            "azure_file_id": None,
            "uploaded_filename": None,
            "uploaded_bytes": None,
            "uploaded_at": None,
        },
    )
    if response.status_code not in {200, 204}:
        raise UserPapersPersistenceError("Could not clear user paper file metadata")
    if response.status_code == 204 or not response.content:
        raise UserPaperNotFoundError("User paper does not exist")
    if not _json_list(response, "User paper file clear returned an invalid response"):
        raise UserPaperNotFoundError("User paper does not exist")


def delete_empty_user_paper(*, user_id: str, paper_id: str) -> None:
    """Delete a row that has neither saved nor file state."""
    paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    if paper is None:
        return
    if paper.get("is_saved") or paper.get("azure_file_id"):
        return
    delete_user_paper(user_id=user_id, paper_id=paper_id)


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
    "clear_user_paper_file",
    "delete_empty_user_paper",
    "delete_user_paper",
    "get_user_paper",
    "import_user_papers",
    "list_user_papers",
    "save_user_paper",
    "unsave_user_paper",
    "upsert_user_paper_file",
]
