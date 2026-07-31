"""Registry for trusted resolution of papers from a user's personal library.

This is deliberately a backend boundary: clients may choose paper IDs, but
only a matching ``user_papers`` row authorizes access to its metadata or file
state.  Imported papers are never looked up in the global catalog.
"""
from __future__ import annotations

from dataclasses import dataclass

from repositories.supabase.user_papers_repository import (
    UserPapersPersistenceError,
    get_user_paper,
    get_user_papers_by_ids,
)
from repositories.zilliz.mappers import paper_to_api_response
from repositories.zilliz.paper_repository import (
    RepositoryUnavailableError,
    get_papers_by_ids,
)


class LibraryPaperResolutionError(RuntimeError):
    """Raised when a library paper cannot be authorized or hydrated."""


@dataclass(frozen=True)
class ResolvedLibraryPaper:
    """Canonical metadata and library state for one authorized library paper."""

    paper_id: str
    origin: str
    metadata: dict[str, object]
    library_paper: dict[str, object]


def _snapshot_for_user_paper(
    library_paper: dict[str, object], paper_id: str
) -> dict[str, object] | None:
    """Validate the canonical snapshot required for an imported paper."""
    snapshot = library_paper.get("metadata_snapshot")
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("ID") != paper_id:
        return None
    if not isinstance(snapshot.get("Title"), str) or not snapshot["Title"].strip():
        return None
    if not isinstance(snapshot.get("Abstract"), str) or not snapshot["Abstract"].strip():
        return None
    return snapshot


def _snapshot_for_corpus_paper(library_paper: dict[str, object]) -> dict[str, object] | None:
    """Return a usable corpus fallback snapshot, when one was persisted."""
    snapshot = library_paper.get("metadata_snapshot")
    if not isinstance(snapshot, dict):
        return None
    if not isinstance(snapshot.get("Title"), str) or not snapshot["Title"].strip():
        return None
    return snapshot


def resolve_library_papers(
    *, user_id: str, paper_ids: list[str]
) -> list[ResolvedLibraryPaper]:
    """Resolve selected library papers in input order.

    User-origin papers exclusively use their canonical snapshot.  Corpus-origin
    papers use the catalog when available, with their snapshot as a resilience
    fallback.  ``metadata_raw`` is intentionally never consulted.
    """
    library_by_id: dict[str, dict[str, object]] = {}
    try:
        for paper_id in paper_ids:
            if paper_id in library_by_id:
                continue
            library_paper = get_user_paper(user_id=user_id, paper_id=paper_id)
            if library_paper is None:
                raise LibraryPaperResolutionError(
                    "One or more selected papers are unavailable."
                )
            library_by_id[paper_id] = library_paper
    except UserPapersPersistenceError as error:
        raise LibraryPaperResolutionError("Selected papers are temporarily unavailable.") from error

    return resolve_library_rows(
        library_papers=[library_by_id[paper_id] for paper_id in paper_ids]
    )


def resolve_library_rows(
    *, library_papers: list[dict[str, object]]
) -> list[ResolvedLibraryPaper]:
    """Resolve already-authorized library rows in their supplied order.

    Callers must obtain these rows through a user-scoped repository query.  This
    lets the library list endpoint resolve a whole library without re-querying
    each paper individually.
    """
    corpus_ids = [
        str(library_paper["paper_id"])
        for library_paper in library_papers
        if library_paper.get("origin") != "user"
    ]
    catalog_by_id: dict[str, dict[str, object]] = {}
    catalog_error: RepositoryUnavailableError | None = None
    if corpus_ids:
        try:
            catalog_by_id = {
                str(record.get("paper_uid")): paper_to_api_response(record)
                for record in get_papers_by_ids(corpus_ids)
                if record.get("paper_uid") is not None
            }
        except RepositoryUnavailableError as error:
            catalog_error = error

    resolved: list[ResolvedLibraryPaper] = []
    for library_paper in library_papers:
        paper_id = str(library_paper["paper_id"])
        origin = str(library_paper.get("origin") or "corpus")
        if origin == "user":
            metadata = _snapshot_for_user_paper(library_paper, paper_id)
            if metadata is None:
                raise LibraryPaperResolutionError(
                    "Imported paper metadata is unavailable."
                )
        else:
            metadata = catalog_by_id.get(paper_id) or _snapshot_for_corpus_paper(library_paper)
            if metadata is None:
                if catalog_error is not None:
                    raise LibraryPaperResolutionError(
                        "Selected paper metadata is temporarily unavailable."
                    ) from catalog_error
                raise LibraryPaperResolutionError(
                    "One or more selected papers are unavailable in the paper catalog."
                )
        resolved.append(
            ResolvedLibraryPaper(
                paper_id=paper_id,
                origin=origin,
                metadata=metadata,
                library_paper=library_paper,
            )
        )
    return resolved


def resolve_papers(
    *, user_id: str | None, paper_ids: list[str]
) -> list[dict[str, object]]:
    """Resolve public corpus papers and, when authorized, user-library papers.

    Imported ``user:`` IDs are only resolved from the current user's library;
    they are never sent to the corpus.  Other IDs resolve from the corpus when
    they are absent from that library.
    """
    requested_ids = list(dict.fromkeys(paper_ids))
    library_by_id: dict[str, dict[str, object]] = {}
    if user_id and requested_ids:
        try:
            library_by_id = {
                str(paper["paper_id"]): paper
                for paper in get_user_papers_by_ids(user_id=user_id, paper_ids=requested_ids)
            }
        except UserPapersPersistenceError as error:
            raise LibraryPaperResolutionError("User library is temporarily unavailable.") from error

    library_metadata_by_id = {
        paper.paper_id: paper.metadata
        for paper in resolve_library_rows(library_papers=list(library_by_id.values()))
    }
    corpus_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id not in library_metadata_by_id and not paper_id.startswith("user:")
    ]
    try:
        corpus_by_id = {
            str(record.get("paper_uid")): paper_to_api_response(record)
            for record in get_papers_by_ids(corpus_ids)
            if record.get("paper_uid") is not None
        }
    except RepositoryUnavailableError as error:
        raise LibraryPaperResolutionError("Paper metadata is temporarily unavailable.") from error

    return [
        library_metadata_by_id.get(paper_id) or corpus_by_id.get(paper_id)
        for paper_id in requested_ids
        if library_metadata_by_id.get(paper_id) or corpus_by_id.get(paper_id)
    ]


__all__ = [
    "LibraryPaperResolutionError",
    "ResolvedLibraryPaper",
    "resolve_library_papers",
    "resolve_library_rows",
    "resolve_papers",
]
