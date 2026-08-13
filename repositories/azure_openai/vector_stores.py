"""Minimal Azure OpenAI Vector Store and Responses adapter.

Vector Stores, Responses, and File Search live on the Azure AI Foundry
``/openai/v1`` contract, which is versioned as ``v1`` or ``preview`` rather
than by the date-versioned API used for chat and Files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import sleep
from typing import Any

import requests


# /openai/v1 only accepts "v1" or "preview" — not the date-versioned chat API.
API_VERSION = "preview"
REQUEST_TIMEOUT_SECONDS = 60
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
TERMINAL_FILE_STATUSES = {"completed", "failed", "cancelled"}


class AzureVectorStoresError(RuntimeError):
    """Base error for Azure Vector Store and File Search failures."""


class AzureVectorStoresConfigurationError(AzureVectorStoresError):
    """Raised when Azure OpenAI credentials for File Search are missing."""


class AzureVectorStoresTransientError(AzureVectorStoresError):
    """Raised for retryable Azure/network failures."""


@dataclass(frozen=True)
class AzureVectorStoresSettings:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    timeout_seconds: float


def get_azure_vector_stores_settings(
    *, model: str | None = None
) -> AzureVectorStoresSettings:
    """Load settings for the Azure File Search capability.

    Reuses the chat endpoint and key. The Responses ``model`` field is the
    Azure deployment for the selected logical chat model (request override or
    ``AZURE_OPENAI_DEFAULT_MODEL``). API version and timeout are fixed for the
    ``/openai/v1`` Vector Store / File Search contract.
    """
    import config as app_config

    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    try:
        deployment = app_config.resolve_chat_deployment(model)
    except ValueError as error:
        raise AzureVectorStoresConfigurationError(str(error)) from error
    if not endpoint or not api_key or not deployment:
        raise AzureVectorStoresConfigurationError("Azure OpenAI File Search is not configured")
    return AzureVectorStoresSettings(
        endpoint=endpoint,
        api_key=api_key,
        api_version=API_VERSION,
        deployment=deployment,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def create_vector_store(*, name: str, settings: AzureVectorStoresSettings | None = None) -> dict[str, Any]:
    """Create a Vector Store and return Azure's complete response payload."""
    if not name:
        raise AzureVectorStoresError("Vector Store name is required")
    return _request_json("POST", "/openai/v1/vector_stores", json={"name": name}, settings=settings)


def attach_file(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: dict[str, str | int | float | bool] | None = None,
    settings: AzureVectorStoresSettings | None = None,
) -> dict[str, Any]:
    """Attach a File to a Vector Store, optionally with controlled filter attributes.

    Azure accepts ``attributes`` but does not currently persist or return them,
    so callers must not treat them as readable state.
    """
    if not vector_store_id or not file_id:
        raise AzureVectorStoresError("Vector Store id and file id are required")
    payload: dict[str, Any] = {"file_id": file_id}
    if attributes:
        payload["attributes"] = attributes
    return _request_json(
        "POST",
        f"/openai/v1/vector_stores/{vector_store_id}/files",
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
        f"/openai/v1/vector_stores/{vector_store_id}/files/{vector_store_file_id}",
        settings=settings,
    )


def detach_file(*, vector_store_id: str, vector_store_file_id: str, settings: AzureVectorStoresSettings | None = None) -> None:
    """Remove one file association; Azure 404 is treated as already detached."""
    try:
        _request_json("DELETE", f"/openai/v1/vector_stores/{vector_store_id}/files/{vector_store_file_id}", settings=settings)
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
        "/openai/v1/responses",
        json={"model": resolved.deployment, "input": input_text, "tools": [tool]},
        settings=resolved,
    )


def create_text_response(*, input_text: str, settings: AzureVectorStoresSettings | None = None) -> dict[str, Any]:
    """Call Responses without retrieval tools, for metadata-only synthesis."""
    if not input_text:
        raise AzureVectorStoresError("Input text is required")
    resolved = settings or get_azure_vector_stores_settings()
    return _request_json(
        "POST", "/openai/v1/responses",
        json={"model": resolved.deployment, "input": input_text}, settings=resolved,
    )


def response_output_text(response: dict[str, Any]) -> str | None:
    """Extract assistant text from the raw REST Responses payload."""
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts) or None


def response_file_citation_annotations(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return File Search annotations with their source text-block context.

    Annotation fields are preserved for the grounding layer, which decides how
    to map or render them.  This adapter deliberately does not apply any
    selected-paper policy.
    """
    citations: list[dict[str, Any]] = []
    output = response.get("output")
    if not isinstance(output, list):
        return citations
    for output_index, item in enumerate(output):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            output_text = block.get("text")
            annotations = block.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict) or annotation.get("type") != "file_citation":
                    continue
                file_id = annotation.get("file_id")
                if not isinstance(file_id, str) or not file_id:
                    continue
                citation = dict(annotation)
                citation["output_index"] = output_index
                citation["content_index"] = content_index
                if isinstance(output_text, str):
                    citation["output_text"] = output_text
                citations.append(citation)
    return citations


def response_file_citations(response: dict[str, Any]) -> list[dict[str, str]]:
    """Return the legacy compact projection of File Search annotations."""
    citations: list[dict[str, str]] = []
    for annotation in response_file_citation_annotations(response):
        file_id = annotation["file_id"]
        citation = {"file_id": file_id}
        filename = annotation.get("filename")
        if isinstance(filename, str) and filename:
            citation["filename"] = filename
        citations.append(citation)
    return citations


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
        raise AzureVectorStoresError(
            f"Azure File Search request failed ({response.status_code}): {_error_summary(response)}"
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise AzureVectorStoresError("Azure File Search returned an invalid response") from error
    if not isinstance(payload, dict):
        raise AzureVectorStoresError("Azure File Search returned an invalid response")
    return payload


def _error_summary(response: requests.Response) -> str:
    """Return Azure's non-sensitive error code/message for server logs."""
    try:
        payload = response.json()
    except ValueError:
        return "no error details returned"
    if not isinstance(payload, dict):
        return "no error details returned"
    error = payload.get("error")
    if not isinstance(error, dict):
        return "no error details returned"
    code = error.get("code")
    message = error.get("message")
    details = ": ".join(value for value in (code, message) if isinstance(value, str) and value.strip())
    return details[:500] if details else "no error details returned"


__all__ = [
    "AzureVectorStoresConfigurationError",
    "AzureVectorStoresError",
    "AzureVectorStoresSettings",
    "AzureVectorStoresTransientError",
    "attach_file",
    "create_file_search_response",
    "create_text_response",
    "create_vector_store",
    "detach_file",
    "get_azure_vector_stores_settings",
    "get_vector_store_file",
    "poll_file_until_terminal",
    "response_file_citation_annotations",
    "response_file_citations",
    "response_output_text",
]
