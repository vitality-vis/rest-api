"""OpenAlex repository package."""

from .client import (
    OpenAlexConfigurationError,
    OpenAlexError,
    OpenAlexTransientError,
    get_openalex_api_key,
    list_citing_works,
    list_referenced_works,
    normalize_doi,
    normalize_openalex_id,
    resolve_work_by_doi,
    summarize_work,
)

__all__ = [
    "OpenAlexConfigurationError",
    "OpenAlexError",
    "OpenAlexTransientError",
    "get_openalex_api_key",
    "list_citing_works",
    "list_referenced_works",
    "normalize_doi",
    "normalize_openalex_id",
    "resolve_work_by_doi",
    "summarize_work",
]
