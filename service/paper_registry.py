"""Registry for trusted, library-aware paper resolution.

Clients may select public corpus IDs without saving them first. A matching
``user_papers`` row is still required to access personal-library state such as
an uploaded full text. Imported papers are never looked up in the global
catalog.
"""
from __future__ import annotations

from dataclasses import dataclass

from repositories.supabase.user_papers_repository import (
    UserPapersPersistenceError,
    get_user_papers_by_ids,
)
from repositories.zilliz.mappers import paper_to_api_response
from repositories.zilliz.paper_repository import (
    RepositoryUnavailableError,
    get_papers_by_ids,
)


class LibraryPaperResolutionError(RuntimeError):
    """Raised when requested paper metadata cannot be authorized or hydrated."""


@dataclass(frozen=True)
class ResolvedLibraryPaper:
    """Canonical metadata and library state for one authorized library paper."""

    paper_id: str
    origin: str
    metadata: dict[str, object]
    # Absent for a public corpus paper selected without first saving it.
    library_paper: dict[str, object] | None


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


def _unique_ids(paper_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(paper_ids))


def _catalog_metadata_by_id(paper_ids: list[str]) -> dict[str, dict[str, object]]:
    """Load public-corpus metadata once and index it by paper ID."""
    if not paper_ids:
        return {}
    return {
        str(record.get("paper_uid")): paper_to_api_response(record)
        for record in get_papers_by_ids(_unique_ids(paper_ids))
        if record.get("paper_uid") is not None
    }


def _library_rows_by_id(
    *, user_id: str, paper_ids: list[str], unavailable_message: str
) -> dict[str, dict[str, object]]:
    """Batch-load current-user library rows for the supplied IDs."""
    if not paper_ids:
        return {}
    try:
        return {
            str(paper["paper_id"]): paper
            for paper in get_user_papers_by_ids(
                user_id=user_id, paper_ids=_unique_ids(paper_ids)
            )
        }
    except UserPapersPersistenceError as error:
        raise LibraryPaperResolutionError(unavailable_message) from error


def _resolved_sources_by_id(
    *, user_id: str | None, paper_ids: list[str], library_unavailable_message: str
) -> tuple[dict[str, ResolvedLibraryPaper], dict[str, dict[str, object]]]:
    """Resolve library-backed papers first, then remaining public corpus IDs."""
    requested_ids = _unique_ids(paper_ids)
    library_rows_by_id = (
        _library_rows_by_id(
            user_id=user_id,
            paper_ids=requested_ids,
            unavailable_message=library_unavailable_message,
        )
        if user_id
        else {}
    )
    resolved_library_by_id = {
        paper.paper_id: paper
        for paper in resolve_library_rows(
            library_papers=list(library_rows_by_id.values())
        )
    }
    corpus_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id not in resolved_library_by_id and not paper_id.startswith("user:")
    ]
    try:
        corpus_by_id = _catalog_metadata_by_id(corpus_ids)
    except RepositoryUnavailableError as error:
        raise LibraryPaperResolutionError("Paper metadata is temporarily unavailable.") from error
    return resolved_library_by_id, corpus_by_id


def resolve_library_papers(
    *, user_id: str, paper_ids: list[str]
) -> list[ResolvedLibraryPaper]:
    """Resolve selected library papers in input order.

    User-origin papers exclusively use their canonical snapshot.  Corpus-origin
    papers use the catalog when available, with their snapshot as a resilience
    fallback.  ``metadata_raw`` is intentionally never consulted.
    """
    library_by_id = _library_rows_by_id(
        user_id=user_id,
        paper_ids=paper_ids,
        unavailable_message="Selected papers are temporarily unavailable.",
    )
    if any(paper_id not in library_by_id for paper_id in paper_ids):
        raise LibraryPaperResolutionError("One or more selected papers are unavailable.")

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
            catalog_by_id = _catalog_metadata_by_id(corpus_ids)
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
    requested_ids = _unique_ids(paper_ids)
    resolved_library_by_id, corpus_by_id = _resolved_sources_by_id(
        user_id=user_id,
        paper_ids=requested_ids,
        library_unavailable_message="User library is temporarily unavailable.",
    )

    return [
        (
            resolved_library_by_id[paper_id].metadata
            if paper_id in resolved_library_by_id
            else corpus_by_id.get(paper_id)
        )
        for paper_id in requested_ids
        if paper_id in resolved_library_by_id or paper_id in corpus_by_id
    ]


def resolve_selected_papers(
    *, user_id: str, paper_ids: list[str]
) -> list[ResolvedLibraryPaper]:
    """Strictly resolve selected papers from the library, then the corpus.

    A selected corpus paper does not need to have been saved first.  Only
    library-backed papers can contribute a full-text file; public corpus papers
    return ``library_paper=None`` and are therefore metadata-only evidence.
    Unlike :func:`resolve_papers`, this is strict: every requested ID must be
    resolved so the QA evidence order always matches the user's selection.
    """
    requested_ids = list(paper_ids)
    if not requested_ids:
        return []
    resolved_library_by_id, corpus_by_id = _resolved_sources_by_id(
        user_id=user_id,
        paper_ids=requested_ids,
        library_unavailable_message="Selected papers are temporarily unavailable.",
    )
    missing_user_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id.startswith("user:") and paper_id not in resolved_library_by_id
    ]
    if missing_user_ids:
        raise LibraryPaperResolutionError("One or more selected imported papers are unavailable.")

    missing_corpus_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id not in resolved_library_by_id
        and not paper_id.startswith("user:")
        and paper_id not in corpus_by_id
    ]
    if missing_corpus_ids:
        raise LibraryPaperResolutionError("One or more selected papers are unavailable in the paper catalog.")

    return [
        resolved_library_by_id.get(paper_id)
        or ResolvedLibraryPaper(
            paper_id=paper_id,
            origin="corpus",
            metadata=corpus_by_id[paper_id],
            library_paper=None,
        )
        for paper_id in requested_ids
    ]


__all__ = [
    "LibraryPaperResolutionError",
    "ResolvedLibraryPaper",
    "resolve_library_papers",
    "resolve_library_rows",
    "resolve_papers",
    "resolve_selected_papers",
]
