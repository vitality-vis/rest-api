"""Service-role persistence for project-scoped paper shelf memberships.

Callers must validate that ``project_id`` is an active project owned by ``user_id``
before invoking mutations. Every request still constrains both tenant columns.
"""

from __future__ import annotations

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers
from repositories.supabase.note_completion_cache import invalidate_note_completion_context

DATABASE_REQUEST_TIMEOUT_SECONDS = 10
_MEMBERSHIP_COLUMNS = "user_id,project_id,user_paper_id,tags,created_at,updated_at"
_ASSET_COLUMNS = (
    "id,user_id,paper_id,metadata_snapshot,metadata_raw,origin,azure_file_id,"
    "uploaded_filename,uploaded_bytes,uploaded_at,vs_file_status,vs_file_id,"
    "vs_indexed_at,vs_last_error,created_at,updated_at"
)
_JOIN_COLUMNS = (
    f"{_MEMBERSHIP_COLUMNS},"
    f"user_papers!project_papers_user_paper_owner_fkey({_ASSET_COLUMNS})"
)


class ProjectPapersPersistenceError(RuntimeError):
    """Raised when project-paper persistence cannot complete an operation."""


class ProjectPaperNotFoundError(ProjectPapersPersistenceError):
    """Raised when a project membership is absent."""


def list_project_papers(*, user_id: str, project_id: str) -> list[dict[str, object]]:
    """Return user-paper assets belonging to one project, with project tags."""
    response = _request(
        "GET",
        "project_papers",
        params={
            "user_id": f"eq.{user_id}",
            "project_id": f"eq.{project_id}",
            "select": _JOIN_COLUMNS,
            "order": "created_at.asc",
        },
    )
    if response.status_code != 200:
        raise ProjectPapersPersistenceError("Could not load project papers")
    memberships = _json_list(response, "Project papers returned an invalid response")
    papers: list[dict[str, object]] = []
    for membership in memberships:
        paper = membership.get("user_papers")
        if not isinstance(paper, dict):
            raise ProjectPapersPersistenceError("Project papers returned an invalid response")
        paper = dict(paper)
        paper["tags"] = membership.get("tags") or []
        paper["is_saved"] = True
        paper["project_id"] = project_id
        paper["membership_created_at"] = membership.get("created_at")
        paper["membership_updated_at"] = membership.get("updated_at")
        papers.append(paper)
    return papers


def add_project_paper(
    *, user_id: str, project_id: str, user_paper_id: str, tags: list[str] | None = None
) -> dict[str, object]:
    """Idempotently add an asset to a project without replacing existing tags."""
    response = _request(
        "POST",
        "project_papers",
        params={"on_conflict": "project_id,user_paper_id"},
        headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
        json={
            "user_id": user_id,
            "project_id": project_id,
            "user_paper_id": user_paper_id,
            "tags": tags or [],
        },
    )
    if response.status_code not in {200, 201}:
        raise ProjectPapersPersistenceError("Could not add project paper")
    records = _json_list(response, "Project paper add returned an invalid response")
    if records:
        invalidate_note_completion_context(user_id=user_id, project_id=project_id)
        return records[0]
    membership = get_project_paper(
        user_id=user_id, project_id=project_id, user_paper_id=user_paper_id
    )
    if membership is None:
        raise ProjectPapersPersistenceError("Could not verify project paper")
    invalidate_note_completion_context(user_id=user_id, project_id=project_id)
    return membership


def get_project_paper(
    *, user_id: str, project_id: str, user_paper_id: str
) -> dict[str, object] | None:
    response = _request(
        "GET",
        "project_papers",
        params={
            "user_id": f"eq.{user_id}",
            "project_id": f"eq.{project_id}",
            "user_paper_id": f"eq.{user_paper_id}",
            "select": _MEMBERSHIP_COLUMNS,
            "limit": "1",
        },
    )
    if response.status_code != 200:
        raise ProjectPapersPersistenceError("Could not load project paper")
    records = _json_list(response, "Project paper returned an invalid response")
    return records[0] if records else None


def has_project_paper(*, user_id: str, project_id: str, user_paper_id: str) -> bool:
    return get_project_paper(
        user_id=user_id, project_id=project_id, user_paper_id=user_paper_id
    ) is not None


def has_any_membership(*, user_id: str, user_paper_id: str) -> bool:
    response = _request(
        "GET",
        "project_papers",
        params={
            "user_id": f"eq.{user_id}",
            "user_paper_id": f"eq.{user_paper_id}",
            "select": "project_id",
            "limit": "1",
        },
    )
    if response.status_code != 200:
        raise ProjectPapersPersistenceError("Could not check project paper membership")
    return bool(_json_list(response, "Project paper check returned an invalid response"))


def update_project_paper_tags(
    *, user_id: str, project_id: str, user_paper_id: str, tags: list[str]
) -> dict[str, object]:
    response = _request(
        "PATCH",
        "project_papers",
        params={
            "user_id": f"eq.{user_id}",
            "project_id": f"eq.{project_id}",
            "user_paper_id": f"eq.{user_paper_id}",
        },
        headers={"Prefer": "return=representation"},
        json={"tags": tags},
    )
    if response.status_code not in {200, 204}:
        raise ProjectPapersPersistenceError("Could not update project paper tags")
    records = _json_list(response, "Project paper tags returned an invalid response")
    if len(records) != 1:
        raise ProjectPaperNotFoundError("Project paper does not exist")
    return records[0]


def remove_project_paper(*, user_id: str, project_id: str, user_paper_id: str) -> bool:
    """Remove only membership; the user-level asset is never deleted."""
    response = _request(
        "DELETE",
        "project_papers",
        params={
            "user_id": f"eq.{user_id}",
            "project_id": f"eq.{project_id}",
            "user_paper_id": f"eq.{user_paper_id}",
        },
        headers={"Prefer": "return=representation"},
    )
    if response.status_code not in {200, 204}:
        raise ProjectPapersPersistenceError("Could not remove project paper")
    removed = bool(_json_list(response, "Project paper remove returned an invalid response"))
    if removed:
        invalidate_note_completion_context(user_id=user_id, project_id=project_id)
    return removed



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
        raise ProjectPapersPersistenceError("Could not reach project-papers database") from error


def _json_list(response: requests.Response, message: str) -> list[dict[str, object]]:
    if response.status_code == 204 or not response.content:
        return []
    try:
        payload = response.json()
    except ValueError as error:
        raise ProjectPapersPersistenceError(message) from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ProjectPapersPersistenceError(message)
    return payload
