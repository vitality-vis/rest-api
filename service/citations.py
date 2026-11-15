"""Application service for OpenAlex paper citation lookup."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from repositories.openalex import (
    OpenAlexConfigurationError,
    OpenAlexError,
    OpenAlexTransientError,
    list_citing_works,
    list_referenced_works,
    normalize_doi,
    resolve_work_by_doi,
)
from service.openalex_papers import annotate_corpus_membership


class PaperCitationsNotFoundError(RuntimeError):
    """Raised when OpenAlex cannot resolve the requested DOI."""


class PaperCitationsUnavailableError(RuntimeError):
    """Raised when citation lookup is not configured or temporarily unavailable."""


class PaperCitationsProviderError(RuntimeError):
    """Raised when OpenAlex rejects or returns an invalid request."""


def get_paper_citations(
    doi: str,
    limit: int = 50,
    offset: int = 0,
    direction: Optional[Literal["references", "cited_by"]] = None,
) -> Dict[str, Any]:
    """Return references and citing works for one DOI from OpenAlex.

    Each neighbor is annotated with ``in_corpus`` after a Zilliz DOI gate.
    Corpus matches also receive the Vitality ``paper_id`` (``ID``).
    When ``direction`` is specified, only that citation direction is fetched.
    """
    normalized_doi = normalize_doi(doi)
    try:
        source = resolve_work_by_doi(normalized_doi)
        if source is None or not source.get("openalex_id"):
            raise PaperCitationsNotFoundError(
                "No OpenAlex work was found for the requested DOI"
            )

        openalex_id = str(source["openalex_id"])
        referenced_work_ids = source.get("referenced_works") or []
        references_total = len(referenced_work_ids)
        cited_by_total = max(0, int(source.get("citation_count") or 0))
        references = (
            list_referenced_works(
                openalex_id,
                limit,
                offset=offset,
                referenced_work_ids=referenced_work_ids,
            )
            if direction in (None, "references")
            else []
        )
        cited_by = (
            list_citing_works(openalex_id, limit, offset=offset)
            if direction in (None, "cited_by")
            else []
        )
    except (OpenAlexConfigurationError, OpenAlexTransientError) as error:
        raise PaperCitationsUnavailableError(str(error)) from error
    except OpenAlexError as error:
        raise PaperCitationsProviderError(str(error)) from error

    references = annotate_corpus_membership(references)
    cited_by = annotate_corpus_membership(cited_by)

    return {
        "source": {
            "doi": normalized_doi,
            "openalex_id": openalex_id,
        },
        "references": {
            "total_hint": references_total,
            "has_more": (
                offset + len(references) < references_total
                if direction in (None, "references")
                else False
            ),
            "papers": references,
        },
        "cited_by": {
            "total_hint": cited_by_total,
            "has_more": (
                offset + len(cited_by) < cited_by_total
                if direction in (None, "cited_by")
                else False
            ),
            "papers": cited_by,
        },
    }
