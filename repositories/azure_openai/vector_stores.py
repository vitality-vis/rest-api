"""Minimal Azure OpenAI Vector Store and Responses adapter.

This adapter deliberately uses a File Search-specific API version instead of
the application's existing chat/files version.  Azure support for Vector
Stores, Responses, and File Search filters must be validated together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import sleep
from typing import Any

import requests


DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
TERMINAL_FILE_STATUSES = {"completed", "failed", "cancelled"}


class AzureVectorStoresError(RuntimeError):
    """Base error for Azure Vector Store and File Search failures."""


class AzureVectorStoresConfigurationError(AzureVectorStoresError):
    """Raised when the dedicated File Search settings are missing or invalid."""


class AzureVectorStoresTransientError(AzureVectorStoresError):
    """Raised for retryable Azure/network failures."""


@dataclass(frozen=True)
class AzureVectorStoresSettings:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    timeout_seconds: float


def get_azure_vector_stores_settings() -> AzureVectorStoresSettings:
    """Load settings specific to the File Search capability.

    The normal chat deployment is always reused. A dedicated File Search API
    version remains an optional override when Azure requires it.
    """
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    api_version = (
        os.getenv("AZURE_OPENAI_FILE_SEARCH_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or ""
    ).strip()
    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
    if not endpoint or not api_key or not api_version or not deployment:
        raise AzureVectorStoresConfigurationError("Azure OpenAI File Search is not configured")

    timeout_raw = os.getenv("AZURE_OPENAI_FILE_SEARCH_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else DEFAULT_REQUEST_TIMEOUT_SECONDS
    except ValueError as error:
        raise AzureVectorStoresConfigurationError("Azure OpenAI File Search timeout is invalid") from error
    if timeout_seconds <= 0:
        raise AzureVectorStoresConfigurationError("Azure OpenAI File Search timeout is invalid")
    return AzureVectorStoresSettings(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
        timeout_seconds=timeout_seconds,
    )


def create_vector_store(*, name: str, settings: AzureVectorStoresSettings | None = None) -> dict[str, Any]:
    """Create a Vector Store and return Azure's complete response payload."""
    if not name:
        raise AzureVectorStoresError("Vector Store name is required")
    return _request_json("POST", "/openai/vector_stores", json={"name": name}, settings=settings)


def attach_file(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: dict[str, str | int | float | bool] | None = None,
    settings: AzureVectorStoresSettings | None = None,
) -> dict[str, Any]:
    """Attach a File to a Vector Store, optionally with controlled filter attributes."""
    if not vector_store_id or not file_id:
        raise AzureVectorStoresError("Vector Store id and file id are required")
    payload: dict[str, Any] = {"file_id": file_id}
    if attributes:
        payload["attributes"] = attributes
    return _request_json(
        "POST",
        f"/openai/vector_stores/{vector_store_id}/files",
        json=payload,
        settings=settings,
    )


def get_vector_store_file(
    *, vector_store_id: str, vector_store_file_id: str, settings: AzureVectorStoresSettings | None = None
) -> dict[str, Any]:
    """Return the status of one Vector Store file association."""
    if not vector_store_id or not vector_store_file_id:
        raise AzureVectorStoresError("Vector Store id and file id are required")
    return _request_json(
        "GET",
        f"/openai/vector_stores/{vector_store_id}/files/{vector_store_file_id}",
        settings=settings,
    )


def detach_file(*, vector_store_id: str, vector_store_file_id: str, settings: AzureVectorStoresSettings | None = None) -> None:
    """Remove one file association; Azure 404 is treated as already detached."""
    try:
        _request_json("DELETE", f"/openai/vector_stores/{vector_store_id}/files/{vector_store_file_id}", settings=settings)
    except AzureVectorStoresError as error:
        if "(404)" not in str(error):
            raise


def poll_file_until_terminal(
    *,
    vector_store_id: str,
    vector_store_file_id: str,
    max_attempts: int,
    interval_seconds: float,
    settings: AzureVectorStoresSettings | None = None,
) -> dict[str, Any]:
    """Poll a Vector Store file until Azure reports a terminal status or timeout."""
    if max_attempts <= 0 or interval_seconds < 0:
        raise AzureVectorStoresError("Polling limits are invalid")
    latest: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        latest = get_vector_store_file(
            vector_store_id=vector_store_id,
            vector_store_file_id=vector_store_file_id,
            settings=settings,
        )
        status = latest.get("status")
        if status in TERMINAL_FILE_STATUSES:
            return latest
        if attempt + 1 < max_attempts:
            sleep(interval_seconds)
    raise AzureVectorStoresTransientError("Vector Store file indexing did not finish in time")


def create_file_search_response(
    *,
    input_text: str,
    vector_store_id: str,
    filters: dict[str, Any] | None = None,
    settings: AzureVectorStoresSettings | None = None,
) -> dict[str, Any]:
    """Call Responses with one Vector Store and an optional File Search filter.

    The exact request shape is intentionally concentrated here so the Phase 0
    smoke test can prove or reject it before any product endpoint depends on it.
    """
    if not input_text or not vector_store_id:
        raise AzureVectorStoresError("Input text and Vector Store id are required")
    tool: dict[str, Any] = {"type": "file_search", "vector_store_ids": [vector_store_id]}
    if filters is not None:
        tool["filters"] = filters
    resolved = settings or get_azure_vector_stores_settings()
    return _request_json(
        "POST",
        "/openai/responses",
        json={"model": resolved.deployment, "input": input_text, "tools": [tool]},
        settings=resolved,
    )


def _request_json(
    method: str,
    path: str,
    *,
    settings: AzureVectorStoresSettings | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    resolved = settings or get_azure_vector_stores_settings()
    try:
        response = requests.request(
            method,
            f"{resolved.endpoint}{path}",
            params={"api-version": resolved.api_version},
            headers={"api-key": resolved.api_key, "Content-Type": "application/json"},
            timeout=resolved.timeout_seconds,
            **kwargs,
        )
    except requests.Timeout as error:
        raise AzureVectorStoresTransientError("Azure File Search timed out") from error
    except requests.RequestException as error:
        raise AzureVectorStoresTransientError("Could not reach Azure File Search") from error

    if response.status_code in TRANSIENT_STATUS_CODES:
        raise AzureVectorStoresTransientError("Azure File Search is temporarily unavailable")
    if response.status_code >= 400:
        raise AzureVectorStoresError(f"Azure File Search request failed ({response.status_code})")
    try:
        payload = response.json()
    except ValueError as error:
        raise AzureVectorStoresError("Azure File Search returned an invalid response") from error
    if not isinstance(payload, dict):
        raise AzureVectorStoresError("Azure File Search returned an invalid response")
    return payload


__all__ = [
    "AzureVectorStoresConfigurationError",
    "AzureVectorStoresError",
    "AzureVectorStoresSettings",
    "AzureVectorStoresTransientError",
    "attach_file",
    "create_file_search_response",
    "create_vector_store",
    "detach_file",
    "get_azure_vector_stores_settings",
    "get_vector_store_file",
    "poll_file_until_terminal",
]
