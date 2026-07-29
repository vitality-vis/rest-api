"""Application service for OpenAlex paper citation lookup."""

from __future__ import annotations

from typing import Any, Dict

from repositories.openalex import (
    OpenAlexConfigurationError,
    OpenAlexError,
    OpenAlexTransientError,
    list_citing_works,
    list_referenced_works,
    normalize_doi,
    resolve_work_by_doi,
)


class PaperCitationsNotFoundError(RuntimeError):
    """Raised when OpenAlex cannot resolve the requested DOI."""


class PaperCitationsUnavailableError(RuntimeError):
    """Raised when citation lookup is not configured or temporarily unavailable."""


class PaperCitationsProviderError(RuntimeError):
    """Raised when OpenAlex rejects or returns an invalid request."""


def get_paper_citations(doi: str, limit: int = 50) -> Dict[str, Any]:
    """Return references and citing works for one DOI from OpenAlex."""
    normalized_doi = normalize_doi(doi)
    try:
        source = resolve_work_by_doi(normalized_doi)
        if source is None or not source.get("openalex_id"):
            raise PaperCitationsNotFoundError(
                "No OpenAlex work was found for the requested DOI"
            )

        openalex_id = str(source["openalex_id"])
        referenced_work_ids = source.get("referenced_works") or []
        references = list_referenced_works(
            openalex_id,
            limit,
            referenced_work_ids=referenced_work_ids,
        )
        cited_by = list_citing_works(openalex_id, limit)
    except (OpenAlexConfigurationError, OpenAlexTransientError) as error:
        raise PaperCitationsUnavailableError(str(error)) from error
    except OpenAlexError as error:
        raise PaperCitationsProviderError(str(error)) from error

    return {
        "source": {
            "doi": normalized_doi,
            "openalex_id": openalex_id,
        },
        "references": {
            "total_hint": len(referenced_work_ids),
            "items": references,
        },
        "cited_by": {
            "total_hint": max(0, int(source.get("cited_by_count") or 0)),
            "items": cited_by,
        },
    }
