"""OpenAlex discovery helpers shared by chat tools and paper resolution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from logger_config import get_logger
from repositories.openalex import (
    OpenAlexConfigurationError,
    OpenAlexError,
    OpenAlexTransientError,
    get_works_by_ids,
    openalex_paper_id,
    parse_openalex_paper_id,
    resolve_source_id,
    search_works,
)
from repositories.zilliz.paper_repository import (
    RepositoryUnavailableError,
    find_corpus_papers_by_dois,
)

logging = get_logger()


class OpenAlexPapersUnavailableError(RuntimeError):
    """Raised when OpenAlex search/hydration is not configured or temporarily down."""


class OpenAlexPapersProviderError(RuntimeError):
    """Raised when OpenAlex rejects or returns an invalid search request."""


def annotate_corpus_membership(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set ``in_corpus`` / corpus ``paper_id`` via Zilliz DOI membership.

    Papers without a DOI stay ``in_corpus=False``. OpenAlex-only works keep an
    ``openalex:`` paper id so they can still be presented and resolved.
    """
    dois: List[str] = []
    for paper in papers:
        doi = paper.get("doi")
        if isinstance(doi, str) and doi.strip():
            dois.append(doi.strip())

    corpus_by_doi: Dict[str, str] = {}
    if dois:
        try:
            corpus_by_doi = find_corpus_papers_by_dois(dois)
        except RepositoryUnavailableError as error:
            logging.warning(
                "Skipping corpus DOI gate for OpenAlex papers: %s",
                error,
            )
            corpus_by_doi = {}

    annotated: List[Dict[str, Any]] = []
    for paper in papers:
        item = dict(paper)
        openalex_id = str(item.get("openalex_id") or "").strip()
        doi = item.get("doi")
        corpus_id = None
        if isinstance(doi, str) and doi.strip():
            corpus_id = corpus_by_doi.get(doi.strip().casefold())
        if corpus_id:
            item["in_corpus"] = True
            item["paper_id"] = corpus_id
        else:
            item["in_corpus"] = False
            if openalex_id:
                item["paper_id"] = openalex_paper_id(openalex_id)
        annotated.append(item)
    return annotated


def work_summary_to_api_paper(
    summary: Dict[str, Any], *, paper_id: str
) -> Dict[str, Any]:
    """Map an OpenAlex summary onto the frontend ``Paper`` wire shape."""
    return {
        "ID": paper_id,
        "Title": summary.get("title") or "",
        "Abstract": summary.get("abstract") or "",
        "Authors": list(summary.get("authors") or []),
        "Keywords": list(summary.get("keywords") or []),
        "Source": summary.get("source") or "",
        "Year": summary.get("year"),
        "CitationCounts": summary.get("citation_count"),
        "doi": summary.get("doi"),
        "umap": None,
    }


def search_openalex_papers(
    *,
    query: str = "",
    venue: str = "",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search OpenAlex and annotate corpus membership, preserving relevance order."""
    trimmed_query = (query or "").strip()
    trimmed_venue = (venue or "").strip()
    if not trimmed_query and not trimmed_venue and year_min is None and year_max is None:
        raise OpenAlexPapersProviderError(
            "Provide a search query, venue, or year filter."
        )

    try:
        source_id = (
            resolve_source_id(trimmed_venue) if trimmed_venue else None
        )
        if trimmed_venue and not source_id:
            raise OpenAlexPapersProviderError(
                f"No OpenAlex venue matched {trimmed_venue!r}."
            )
        papers = search_works(
            trimmed_query,
            year_min=year_min,
            year_max=year_max,
            source_id=source_id,
            limit=limit,
        )
    except (OpenAlexConfigurationError, OpenAlexTransientError) as error:
        raise OpenAlexPapersUnavailableError(str(error)) from error
    except OpenAlexError as error:
        raise OpenAlexPapersProviderError(str(error)) from error

    annotated = annotate_corpus_membership(papers)
    return {
        "query": trimmed_query,
        "venue": trimmed_venue,
        "venue_source_id": source_id,
        "year_min": year_min,
        "year_max": year_max,
        "papers": annotated,
    }


def resolve_openalex_api_papers(paper_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Hydrate ``openalex:`` paper IDs into API paper payloads keyed by those IDs."""
    requested: List[tuple[str, str]] = []
    for paper_id in paper_ids:
        openalex_id = parse_openalex_paper_id(paper_id)
        if openalex_id:
            requested.append((paper_id, openalex_id))
    if not requested:
        return {}

    try:
        works = get_works_by_ids([openalex_id for _, openalex_id in requested])
    except (OpenAlexConfigurationError, OpenAlexTransientError) as error:
        raise OpenAlexPapersUnavailableError(str(error)) from error
    except OpenAlexError as error:
        raise OpenAlexPapersProviderError(str(error)) from error

    resolved: Dict[str, Dict[str, Any]] = {}
    for paper_id, openalex_id in requested:
        summary = works.get(openalex_id)
        if summary is None:
            continue
        resolved[paper_id] = work_summary_to_api_paper(summary, paper_id=paper_id)
    return resolved
