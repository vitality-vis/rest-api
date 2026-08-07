"""Application service for OpenAlex paper citation lookup."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from logger_config import get_logger
from repositories.openalex import (
    OpenAlexConfigurationError,
    OpenAlexError,
    OpenAlexTransientError,
    list_citing_works,
    list_referenced_works,
    normalize_doi,
    resolve_work_by_doi,
)
from repositories.zilliz.paper_repository import (
    RepositoryUnavailableError,
    find_corpus_papers_by_dois,
)

logging = get_logger()


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

    references = _annotate_corpus_membership(references)
    cited_by = _annotate_corpus_membership(cited_by)

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


def _annotate_corpus_membership(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set ``in_corpus`` / ``paper_id`` on each citation via Zilliz DOI membership.

    Papers without a DOI are marked ``in_corpus=False`` for now.
    """
    dois: List[str] = []
    for paper in papers:
        doi = paper.get("doi")
        if isinstance(doi, str) and doi.strip():
            dois.append(doi.strip())
        else:
            # TODO: Match citation neighbors without a DOI against the Zilliz
            # corpus (e.g. by normalized title/year or OpenAlex ID once stored).
            pass

    corpus_by_doi: Dict[str, str] = {}
    if dois:
        try:
            corpus_by_doi = find_corpus_papers_by_dois(dois)
        except RepositoryUnavailableError as error:
            # Citations should still return when Zilliz is down; treat as unknown
            # / not in corpus rather than failing the whole OpenAlex response.
            logging.warning(
                "Skipping corpus DOI gate for citations: %s",
                error,
            )
            corpus_by_doi = {}

    annotated: List[Dict[str, Any]] = []
    for paper in papers:
        item = dict(paper)
        doi = item.get("doi")
        if isinstance(doi, str) and doi.strip():
            paper_uid = corpus_by_doi.get(doi.strip().casefold())
            if paper_uid:
                item["in_corpus"] = True
                item["paper_id"] = paper_uid
            else:
                item["in_corpus"] = False
        else:
            item["in_corpus"] = False
        annotated.append(item)
    return annotated
