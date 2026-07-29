"""Persistence for the private Vector Store assigned to each user."""

from __future__ import annotations

import requests

from repositories.supabase.client import get_supabase_settings, service_role_headers

DATABASE_REQUEST_TIMEOUT_SECONDS = 10


class UserVectorStoresPersistenceError(RuntimeError):
    pass


def get_user_vector_store(*, user_id: str) -> dict[str, object] | None:
    response = _request(
        "GET", params={"user_id": f"eq.{user_id}", "select": "*", "limit": "1"}
    )
    records = _records(response)
    return records[0] if records else None


def create_user_vector_store(*, user_id: str, azure_vector_store_id: str) -> dict[str, object]:
    response = _request(
        "POST",
        headers={"Prefer": "return=representation"},
        json={"user_id": user_id, "azure_vector_store_id": azure_vector_store_id, "status": "ready"},
    )
    records = _records(response)
    if len(records) != 1:
        raise UserVectorStoresPersistenceError("Invalid user Vector Store response")
    return records[0]


def _request(method: str, **kwargs) -> requests.Response:
    settings = get_supabase_settings()
    headers = service_role_headers(settings)
    headers.update(kwargs.pop("headers", {}))
    try:
        response = requests.request(
            method,
            f"{settings.url}/rest/v1/user_vector_stores",
            headers=headers,
            timeout=DATABASE_REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as error:
        raise UserVectorStoresPersistenceError("Could not reach the Vector Store database") from error
    if response.status_code not in {200, 201}:
        raise UserVectorStoresPersistenceError("Could not update the Vector Store database")
    return response


def _records(response: requests.Response) -> list[dict[str, object]]:
    try:
        payload = response.json()
    except ValueError as error:
        raise UserVectorStoresPersistenceError("Invalid user Vector Store response") from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise UserVectorStoresPersistenceError("Invalid user Vector Store response")
    return payload


__all__ = ["UserVectorStoresPersistenceError", "create_user_vector_store", "get_user_vector_store"]
