"""Supabase persistence for project listing and authenticated lifecycle RPCs."""

from __future__ import annotations

import logging

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers

DATABASE_REQUEST_TIMEOUT_SECONDS = 10
DATABASE_REQUEST_ATTEMPTS = 2
PROJECT_COLUMNS = "id,user_id,name,description,is_default,created_at,updated_at"

logger = logging.getLogger(__name__)


class ProjectsPersistenceError(RuntimeError):
    """Raised when project persistence is unavailable or returns invalid data."""


class ProjectNotFoundError(ProjectsPersistenceError):
    """Raised when a project is absent or not owned by the authenticated user."""


class ProjectConflictError(ProjectsPersistenceError):
    """Raised when a project lifecycle invariant rejects an operation."""


class ProjectValidationError(ProjectsPersistenceError):
    """Raised when Supabase rejects invalid project input."""


def list_projects(*, user_id: str, access_token: str) -> list[dict[str, object]]:
    """Ensure and return projects owned by the verified end user."""
    _ensure_default_project(access_token)
    response = _service_request(
        "GET",
        "projects",
        params={
            "user_id": f"eq.{user_id}",
            "select": PROJECT_COLUMNS,
            "order": "is_default.desc,updated_at.desc",
        },
    )
    if response.status_code != 200:
        raise ProjectsPersistenceError("Could not load projects")
    records = _json_list(response, "Project list returned an invalid response")
    if any(record.get("user_id") != user_id for record in records):
        raise ProjectsPersistenceError("Project list returned another user's data")
    return records


def get_project(*, user_id: str, project_id: str) -> dict[str, object]:
    """Return one owned project."""
    response = _service_request(
        "GET",
        "projects",
        params={
            "id": f"eq.{project_id}",
            "user_id": f"eq.{user_id}",
            "select": PROJECT_COLUMNS,
            "limit": "2",
        },
    )
    if response.status_code != 200:
        raise ProjectsPersistenceError("Could not load project")
    records = _json_list(response, "Project lookup returned an invalid response")
    if not records:
        raise ProjectNotFoundError("Project not found")
    if len(records) != 1:
        raise ProjectsPersistenceError("Project lookup returned an invalid response")
    project = records[0]
    if project.get("id") != project_id or project.get("user_id") != user_id:
        raise ProjectsPersistenceError("Project lookup returned unexpected data")
    return project


def create_project(*, access_token: str, name: str, description: str | None) -> dict[str, object]:
    return _rpc("create_project", access_token, {"p_name": name, "p_description": description})


def rename_project(*, access_token: str, project_id: str, name: str) -> dict[str, object]:
    return _rpc("rename_project", access_token, {"p_project_id": project_id, "p_name": name})


def _ensure_default_project(access_token: str) -> None:
    """Invoke the auth-derived, idempotent default-project RPC."""
    settings = get_supabase_settings()
    response = _request(
        "POST",
        "rpc/get_or_create_default_project",
        headers={
            "apikey": settings.service_role_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={},
    )
    if response.status_code not in {200, 201}:
        _raise_rpc_error(response)
    records = _json_list(response, "Default project RPC returned an invalid response")
    if len(records) != 1:
        raise ProjectsPersistenceError("Default project RPC returned an invalid response")


def _service_request(method: str, path: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    headers = service_role_headers(settings)
    headers.update(kwargs.pop("headers", {}))
    return _request(method, path, headers=headers, **kwargs)


def _rpc(name: str, access_token: str, payload: dict[str, object]) -> dict[str, object]:
    settings = get_supabase_settings()
    response = _request(
        "POST",
        f"rpc/{name}",
        headers={
            "apikey": settings.service_role_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if response.status_code not in {200, 201}:
        _raise_rpc_error(response)
    try:
        result = response.json()
    except ValueError as error:
        raise ProjectsPersistenceError("Project RPC returned an invalid response") from error
    if not isinstance(result, dict):
        raise ProjectsPersistenceError("Project RPC returned an invalid response")
    return result


def _request(method: str, path: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    url = f"{settings.url}/rest/v1/{path}"
    last_error: requests.RequestException | None = None
    for attempt in range(1, DATABASE_REQUEST_ATTEMPTS + 1):
        try:
            return requests.request(
                method,
                url,
                timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except requests.RequestException as error:
            last_error = error
            logger.warning(
                "Projects database %s %s failed (attempt %s/%s): %s",
                method,
                path,
                attempt,
                DATABASE_REQUEST_ATTEMPTS,
                error,
            )
    assert last_error is not None
    raise ProjectsPersistenceError("Could not reach the projects database") from last_error


def _raise_rpc_error(response: requests.Response) -> None:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    code = payload.get("code") if isinstance(payload, dict) else None
    if code == "P0002":
        raise ProjectNotFoundError("Project not found")
    if code == "22023":
        raise ProjectValidationError(str(payload.get("message") or "Invalid project"))
    if code == "55000":
        raise ProjectConflictError(str(payload.get("message") or "Project conflict"))
    raise ProjectsPersistenceError("Project operation failed")


def _json_list(response: requests.Response, message: str) -> list[dict[str, object]]:
    try:
        payload = response.json()
    except ValueError as error:
        raise ProjectsPersistenceError(message) from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ProjectsPersistenceError(message)
    return payload
