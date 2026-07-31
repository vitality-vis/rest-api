"""Persistence for the authenticated user's personal paper library."""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers


DATABASE_REQUEST_TIMEOUT_SECONDS = 10
_SHELF_COLUMNS = (
    "id,user_id,paper_id,metadata_snapshot,is_saved,origin,azure_file_id,uploaded_filename,"
    "uploaded_bytes,uploaded_at,vs_file_status,vs_file_id,"
    "vs_indexed_at,vs_last_error,created_at,updated_at"
)


class UserPapersPersistenceError(RuntimeError):
    """Raised when the user-papers database cannot complete an operation."""


class UserPaperNotFoundError(UserPapersPersistenceError):
    """Raised when a paper is not on the verified user's shelf."""


def list_user_papers(*, user_id: str, saved_only: bool = False) -> list[dict[str, object]]:
    """Return library papers for one verified user, oldest first."""
    params: dict[str, str] = {
        "user_id": f"eq.{user_id}",
        "select": _SHELF_COLUMNS,
        "order": "created_at.asc",
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


def get_user_papers_by_ids(*, user_id: str, paper_ids: list[str]) -> list[dict[str, object]]:
    """Return library rows matching the supplied IDs for one verified user."""
    unique_ids = list(dict.fromkeys(paper_ids))
    if not unique_ids:
        return []
    escaped_ids = [paper_id.replace("\\", "\\\\").replace('"', '\\"') for paper_id in unique_ids]
    paper_id_filter = "in.(" + ",".join(f'"{paper_id}"' for paper_id in escaped_ids) + ")"
    response = _request(
        "GET",
        "user_papers",
        params={
            "user_id": f"eq.{user_id}",
            "paper_id": paper_id_filter,
            "select": _SHELF_COLUMNS,
        },
    )
    if response.status_code != 200:
        raise UserPapersPersistenceError("Could not load user papers")
    return _json_list(response, "User papers returned an invalid response")


def save_user_paper(
    *, user_id: str, paper_id: str, metadata_snapshot: dict[str, object] | None
) -> tuple[dict[str, object], bool]:
    """Set a paper as saved without changing an existing imported row's origin."""
    existing = get_user_paper(user_id=user_id, paper_id=paper_id)
    if existing is not None and existing.get("origin") == "user":
        response = _request(
            "PATCH",
            "user_papers",
            params={"user_id": f"eq.{user_id}", "paper_id": f"eq.{paper_id}"},
            headers={"Prefer": "return=representation"},
            json={"is_saved": True},
        )
        if response.status_code not in {200, 204}:
            raise UserPapersPersistenceError("Could not save imported paper")
        records = _json_list(response, "Imported paper save returned an invalid response")
        if len(records) != 1:
            raise UserPapersPersistenceError("Imported paper save returned an invalid response")
        return records[0], False

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
            "origin": "corpus",
        },
    )
    # PostgREST returns 201 on insert and 200 when merge-duplicates updates.
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not save user paper")

    records = _json_list(response, "User paper save returned an invalid response")
    if len(records) != 1:
        raise UserPapersPersistenceError("User paper save returned an invalid response")
    return records[0], response.status_code == 201


def save_user_papers(
    *, user_id: str, papers: list[tuple[str, dict[str, object] | None]]
) -> list[dict[str, object]]:
    """Idempotently save papers while preserving imported-row identity."""
    existing = {
        str(paper.get("paper_id")): paper
        for paper in get_user_papers_by_ids(
            user_id=user_id, paper_ids=[paper_id for paper_id, _ in papers]
        )
    }
    imported = [
        (paper_id, snapshot)
        for paper_id, snapshot in papers
        if existing.get(paper_id, {}).get("origin") == "user"
    ]
    corpus = [
        (paper_id, snapshot)
        for paper_id, snapshot in papers
        if existing.get(paper_id, {}).get("origin") != "user"
    ]
    saved_imported = [
        save_user_paper(user_id=user_id, paper_id=paper_id, metadata_snapshot=snapshot)[0]
        for paper_id, snapshot in imported
    ]
    if not corpus:
        return saved_imported

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
                "origin": "corpus",
            }
            for paper_id, metadata_snapshot in corpus
        ],
    )
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not save user papers")
    return saved_imported + _json_list(response, "User paper bulk save returned an invalid response")


def import_user_papers(
    *, user_id: str, papers: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Upsert user-supplied papers with snapshots that always remain available.

    ``paper_id`` is either assigned by the API or is a guest-generated user UUID.
    The conflict target intentionally provides only retry/idempotency semantics; it
    does not deduplicate papers based on their metadata.
    """
    response = _request(
        "POST",
        "user_papers",
        params={"on_conflict": "user_id,paper_id"},
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        json=[
            {
                "user_id": user_id,
                "paper_id": paper["paper_id"],
                "metadata_snapshot": paper["metadata_snapshot"],
                "metadata_raw": paper.get("metadata_raw"),
                "is_saved": True,
                "origin": "user",
            }
            for paper in papers
        ],
    )
    if response.status_code not in {200, 201}:
        raise UserPapersPersistenceError("Could not import user papers")
    return _json_list(response, "User paper import returned an invalid response")


def unsave_user_paper(*, user_id: str, paper_id: str) -> None:
    """Clear saved state, retaining imported rows and corpus rows with a file."""
    paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    if paper is None:
        raise UserPaperNotFoundError("User paper does not exist")

    if paper.get("origin") == "user" or paper.get("azure_file_id"):
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


def unsave_user_papers(*, user_id: str, paper_ids: list[str]) -> None:
    """Apply the per-paper unsave rules in one authenticated API operation."""
    for paper_id in paper_ids:
        unsave_user_paper(user_id=user_id, paper_id=paper_id)


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
    metadata_snapshot: dict[str, object] | None,
    azure_file_id: str,
    uploaded_filename: str,
    uploaded_bytes: int,
    create_if_missing: bool,
    preserve_metadata_snapshot: bool = False,
) -> dict[str, object]:
    """Write file metadata onto an existing row, or create an unsaved upload-only row.

    Imported papers own an immutable canonical snapshot. Their file upload path
    sets ``preserve_metadata_snapshot`` so this generic file update cannot
    replace it with the corpus DOI fallback value.
    """
    existing = get_user_paper(user_id=user_id, paper_id=paper_id)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    file_fields: dict[str, object] = {
        "azure_file_id": azure_file_id,
        "uploaded_filename": uploaded_filename,
        "uploaded_bytes": uploaded_bytes,
        "uploaded_at": uploaded_at,
        "vs_file_status": "pending",
        "vs_file_id": None,
        "vs_indexed_at": None,
        "vs_last_error": None,
    }
    if not preserve_metadata_snapshot:
        file_fields["metadata_snapshot"] = metadata_snapshot

    if existing is None:
        if not create_if_missing or preserve_metadata_snapshot:
            raise UserPaperNotFoundError("User paper does not exist")
        response = _request(
            "POST",
            "user_papers",
            headers={"Prefer": "return=representation"},
            json={
                "user_id": user_id,
                "paper_id": paper_id,
                "is_saved": False,
                "origin": "corpus",
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
            "vs_file_status": "not_indexed",
            "vs_file_id": None,
            "vs_indexed_at": None,
            "vs_last_error": None,
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


def update_user_paper_index_state(
    *, user_id: str, paper_id: str, azure_file_id: str, status: str,
    vs_file_id: str | None = None, error: str | None = None,
) -> dict[str, object] | None:
    """Conditionally update only the current upload's indexing state."""
    payload: dict[str, object] = {"vs_file_status": status, "vs_last_error": error}
    if vs_file_id is not None:
        payload["vs_file_id"] = vs_file_id
    if status == "completed":
        payload["vs_indexed_at"] = datetime.now(timezone.utc).isoformat()
    response = _request(
        "PATCH", "user_papers",
        params={"user_id": f"eq.{user_id}", "paper_id": f"eq.{paper_id}",
                "azure_file_id": f"eq.{azure_file_id}"},
        headers={"Prefer": "return=representation"}, json=payload,
    )
    if response.status_code not in {200, 204}:
        raise UserPapersPersistenceError("Could not update file indexing status")
    if response.status_code == 204 or not response.content:
        return None
    records = _json_list(response, "User paper indexing update returned an invalid response")
    return records[0] if records else None


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
    "get_user_papers_by_ids",
    "import_user_papers",
    "list_user_papers",
    "save_user_paper",
    "save_user_papers",
    "unsave_user_paper",
    "unsave_user_papers",
    "update_user_paper_index_state",
    "upsert_user_paper_file",
]
