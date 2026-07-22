"""Azure OpenAI Files API adapter for library PDF uploads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO

import requests


DEFAULT_API_VERSION = "2024-10-21"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
PDF_PURPOSE = "assistants"


class AzureFilesError(RuntimeError):
    """Base error for Azure Files API failures."""


class AzureFilesConfigurationError(AzureFilesError):
    """Raised when Azure OpenAI file settings are missing."""


class AzureFilesTransientError(AzureFilesError):
    """Raised for timeouts, rate limits, and Azure 5xx responses."""


@dataclass(frozen=True)
class AzureUploadedFile:
    id: str
    filename: str
    bytes: int


@dataclass(frozen=True)
class AzureFilesSettings:
    endpoint: str
    api_key: str
    api_version: str
    timeout_seconds: float


def get_azure_files_settings() -> AzureFilesSettings:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
    if not endpoint or not api_key:
        raise AzureFilesConfigurationError("Azure OpenAI files are not configured")
    timeout_raw = os.getenv("AZURE_OPENAI_FILES_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else DEFAULT_REQUEST_TIMEOUT_SECONDS
    except ValueError as error:
        raise AzureFilesConfigurationError("Azure OpenAI files timeout is invalid") from error
    return AzureFilesSettings(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout_seconds=timeout_seconds,
    )


def upload_pdf_file(
    *,
    filename: str,
    file_obj: BinaryIO,
    settings: AzureFilesSettings | None = None,
) -> AzureUploadedFile:
    """Upload a PDF to Azure OpenAI Files with purpose=assistants."""
    resolved = settings or get_azure_files_settings()
    url = f"{resolved.endpoint}/openai/files"
    try:
        response = requests.post(
            url,
            params={"api-version": resolved.api_version},
            headers={"api-key": resolved.api_key},
            files={"file": (filename, file_obj, "application/pdf")},
            data={"purpose": PDF_PURPOSE},
            timeout=resolved.timeout_seconds,
        )
    except requests.Timeout as error:
        raise AzureFilesTransientError("Azure Files upload timed out") from error
    except requests.RequestException as error:
        raise AzureFilesTransientError("Could not reach Azure Files") from error

    if response.status_code in {429, 500, 502, 503, 504}:
        raise AzureFilesTransientError("Azure Files upload is temporarily unavailable")
    if response.status_code >= 400:
        raise AzureFilesError("Azure Files upload failed")

    try:
        payload = response.json()
    except ValueError as error:
        raise AzureFilesError("Azure Files upload returned an invalid response") from error

    file_id = payload.get("id")
    returned_filename = payload.get("filename") or filename
    size = payload.get("bytes")
    if not isinstance(file_id, str) or not file_id:
        raise AzureFilesError("Azure Files upload returned an invalid file id")
    if not isinstance(size, int) or size < 0:
        # Some Azure responses omit bytes; fall back after a successful upload.
        size = 0
    return AzureUploadedFile(id=file_id, filename=str(returned_filename), bytes=size)


def delete_file(*, file_id: str, settings: AzureFilesSettings | None = None) -> None:
    """Delete one Azure OpenAI file by id."""
    if not file_id:
        raise AzureFilesError("Azure file id is required")
    resolved = settings or get_azure_files_settings()
    url = f"{resolved.endpoint}/openai/files/{file_id}"
    try:
        response = requests.delete(
            url,
            params={"api-version": resolved.api_version},
            headers={"api-key": resolved.api_key},
            timeout=resolved.timeout_seconds,
        )
    except requests.Timeout as error:
        raise AzureFilesTransientError("Azure Files delete timed out") from error
    except requests.RequestException as error:
        raise AzureFilesTransientError("Could not reach Azure Files") from error

    if response.status_code in {404, 200, 204}:
        return
    if response.status_code in {429, 500, 502, 503, 504}:
        raise AzureFilesTransientError("Azure Files delete is temporarily unavailable")
    raise AzureFilesError("Azure Files delete failed")


__all__ = [
    "AzureFilesConfigurationError",
    "AzureFilesError",
    "AzureFilesTransientError",
    "AzureUploadedFile",
    "delete_file",
    "get_azure_files_settings",
    "upload_pdf_file",
]
