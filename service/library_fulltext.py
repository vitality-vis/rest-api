"""Codex evidence tools for user-uploaded library full text.

Azure Vector Store attribute filters are unreliable on our resources, so search
runs against the user's library store and results are post-filtered by the
trusted ``azure_file_id`` allowlist for the requested papers.
"""

from __future__ import annotations

from dataclasses import dataclass

from service.paper_registry import (
    LibraryPaperResolutionError,
    ResolvedLibraryPaper,
    resolve_selected_papers,
)

MAX_SEARCH_CHUNKS = 20
MAX_SEARCH_FETCH = 50
MAX_SEARCHABLE_PAPERS = 40
MAX_CONTENT_SEGMENTS = 40
DEFAULT_CONTENT_SEGMENTS = 12
MAX_CONTENT_CHARS = 24_000
DEFAULT_CONTENT_CHARS = 12_000
MAX_CHUNK_CHARS = 1_500


class LibraryFulltextError(RuntimeError):
    """User-safe library full-text tool failure."""


@dataclass(frozen=True)
class LibrarySearchRequest:
    query: str
    paper_ids: list[str] | None = None
    group: str = "all"
    limit: int | None = None


@dataclass(frozen=True)
class LibraryContentRequest:
    paper_id: str
    offset: int = 0
    limit: int | None = None
    max_chars: int | None = None


def search_library_full_text(
    *,
    user_id: str | None,
    project_id: str | None = None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None = None,
    request: LibrarySearchRequest,
) -> dict[str, object]:
    """Search uploaded library full text and return allowlisted chunks."""
    query = request.query.strip()
    if not query:
        raise LibraryFulltextError("A full-text search query is required.")
    if not user_id:
        raise LibraryFulltextError("Sign in to search uploaded full text.")

    targets, skipped = _resolve_searchable_targets(
        user_id=user_id,
        project_id=project_id,
        selected_paper_ids=selected_paper_ids,
        selected_paper_groups=selected_paper_groups,
        paper_ids=request.paper_ids,
        group=request.group,
    )
    if not targets:
        return {
            "query": query,
            "searched_paper_ids": [],
            "skipped_papers": skipped,
            "chunk_count": 0,
            "omitted_outside_scope": 0,
            "chunks": [],
            "instruction": (
                "No searchable uploaded full text is available for the requested "
                "papers. Use get_selected_paper_evidence for abstracts, or ask the "
                "user to upload/attach a PDF first."
            ),
        }

    from repositories.azure_openai.vector_stores import (
        AzureVectorStoresError,
        search_vector_store,
    )
    from repositories.supabase.user_vector_stores_repository import get_user_vector_store

    store = get_user_vector_store(user_id=user_id)
    store_id = store.get("azure_vector_store_id") if store else None
    if not isinstance(store_id, str) or not store_id:
        raise LibraryFulltextError(
            "Uploaded full texts are not ready. Retry after indexing completes."
        )

    limit = _bounded_int(request.limit, default=8, minimum=1, maximum=MAX_SEARCH_CHUNKS)
    fetch = min(MAX_SEARCH_FETCH, max(limit * 3, limit))
    file_to_paper = {paper.azure_file_id: paper.paper_id for paper in targets}
    try:
        raw = search_vector_store(
            vector_store_id=store_id,
            query=query,
            max_num_results=fetch,
        )
    except AzureVectorStoresError as error:
        raise LibraryFulltextError(
            "Library full-text search failed. Please try again."
        ) from error

    chunks: list[dict[str, object]] = []
    omitted_outside_scope = 0
    for item in raw.get("data") or []:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            continue
        paper_id = file_to_paper.get(file_id)
        if paper_id is None:
            omitted_outside_scope += 1
            continue
        text = _chunk_text(item)
        if not text:
            continue
        chunk: dict[str, object] = {
            "paper_id": paper_id,
            "text": text[:MAX_CHUNK_CHARS],
            "truncated": len(text) > MAX_CHUNK_CHARS,
        }
        score = item.get("score")
        if isinstance(score, (int, float)):
            chunk["score"] = float(score)
        chunks.append(chunk)
        if len(chunks) >= limit:
            break

    return {
        "query": query,
        "group": request.group,
        "searched_paper_ids": [paper.paper_id for paper in targets],
        "skipped_papers": skipped,
        "chunk_count": len(chunks),
        "omitted_outside_scope": omitted_outside_scope,
        "chunks": chunks,
        "instruction": (
            "Write the answer yourself from these full-text chunks. "
            "Cite factual claims as [[ID:paper_id]] using only supplied IDs. "
            "Disclose when chunks are thin, truncated, or when searchable papers "
            "were skipped because full text was unavailable."
        ),
    }


def get_selected_paper_full_text(
    *,
    user_id: str | None,
    project_id: str | None = None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None = None,
    request: LibraryContentRequest,
) -> dict[str, object]:
    """Return bounded parsed full text for one selected paper with uploaded PDF."""
    if not user_id:
        raise LibraryFulltextError("Sign in to read uploaded full text.")
    paper_id = request.paper_id.strip()
    if not paper_id:
        raise LibraryFulltextError("paper_id is required.")

    selected = _normalize_ids(selected_paper_ids)
    if paper_id not in set(selected):
        raise LibraryFulltextError(
            "Full text can only be retrieved for papers in the current selection."
        )
    # Groups are accepted for symmetry with other selected-paper tools; content
    # still requires the paper to be in the overall selection above.
    _ = _normalize_groups(selected, selected_paper_groups)

    try:
        resolved = resolve_selected_papers(
            user_id=user_id, project_id=project_id, paper_ids=[paper_id]
        )
    except LibraryPaperResolutionError as error:
        raise LibraryFulltextError(str(error)) from error
    if not resolved:
        raise LibraryFulltextError("Selected paper is unavailable.")
    paper = resolved[0]
    searchable = _searchable_file(paper)
    if searchable is None:
        return {
            "paper_id": paper_id,
            "full_text_available": False,
            "offset": max(request.offset, 0),
            "next_offset": None,
            "has_more": False,
            "chars_returned": 0,
            "truncated": False,
            "segments": [],
            "instruction": (
                "No uploaded full text is available for this paper. "
                "Use get_selected_paper_evidence for metadata/abstracts, or ask "
                "the user to upload/attach a PDF."
            ),
        }

    from repositories.supabase.user_vector_stores_repository import get_user_vector_store

    store = get_user_vector_store(user_id=user_id)
    store_id = store.get("azure_vector_store_id") if store else None
    if not isinstance(store_id, str) or not store_id:
        raise LibraryFulltextError(
            "Uploaded full texts are not ready. Retry after indexing completes."
        )

    offset = max(int(request.offset or 0), 0)
    segment_limit = _bounded_int(
        request.limit,
        default=DEFAULT_CONTENT_SEGMENTS,
        minimum=1,
        maximum=MAX_CONTENT_SEGMENTS,
    )
    max_chars = _bounded_int(
        request.max_chars,
        default=DEFAULT_CONTENT_CHARS,
        minimum=500,
        maximum=MAX_CONTENT_CHARS,
    )

    from repositories.azure_openai.vector_stores import AzureVectorStoresError

    needed_end = offset + segment_limit
    try:
        all_segments, azure_has_more = _load_content_segments(
            vector_store_id=store_id,
            file_id=searchable,
            minimum_segments=needed_end,
        )
    except AzureVectorStoresError as error:
        raise LibraryFulltextError(
            "Could not read the uploaded full text. Please try again."
        ) from error

    if offset >= len(all_segments) and not azure_has_more:
        return {
            "paper_id": paper_id,
            "full_text_available": True,
            "offset": offset,
            "next_offset": None,
            "has_more": False,
            "chars_returned": 0,
            "truncated": False,
            "segments": [],
            "instruction": (
                "No more full-text segments are available at this offset."
            ),
        }

    window = all_segments[offset:needed_end]
    segments: list[dict[str, object]] = []
    chars = 0
    truncated = False
    for index, text in enumerate(window):
        remaining = max_chars - chars
        if remaining <= 0:
            truncated = True
            break
        piece = text if len(text) <= remaining else text[:remaining]
        segments.append({"index": offset + index, "text": piece})
        chars += len(piece)
        if len(piece) < len(text):
            truncated = True
            break

    next_offset = offset + len(segments)
    has_more = (
        truncated
        or next_offset < len(all_segments)
        or (next_offset >= len(all_segments) and azure_has_more)
    )
    return {
        "paper_id": paper_id,
        "full_text_available": True,
        "offset": offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "chars_returned": chars,
        "truncated": truncated or has_more,
        "segments": segments,
        "instruction": (
            "This is partial parsed full text from the uploaded PDF, not the raw "
            "file. Write the answer yourself and cite [[ID:paper_id]]. "
            "Paginate with offset=next_offset when has_more is true."
        ),
    }


@dataclass(frozen=True)
class _SearchablePaper:
    paper_id: str
    azure_file_id: str


def _resolve_searchable_targets(
    *,
    user_id: str,
    project_id: str | None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None,
    paper_ids: list[str] | None,
    group: str,
) -> tuple[list[_SearchablePaper], list[dict[str, str]]]:
    selected = _normalize_ids(selected_paper_ids)
    if not selected:
        raise LibraryFulltextError("Select at least one paper first.")
    groups = _normalize_groups(selected, selected_paper_groups)
    if group == "all":
        candidates = selected
    elif group in groups:
        candidates = groups[group]
    else:
        raise LibraryFulltextError(
            f"Unknown paper group: {group}. Available groups: "
            f"{', '.join(groups) or 'none'}."
        )
    requested = _requested_subset(candidates, paper_ids)
    if len(requested) > MAX_SEARCHABLE_PAPERS:
        requested = requested[:MAX_SEARCHABLE_PAPERS]

    try:
        resolved = resolve_selected_papers(
            user_id=user_id, project_id=project_id, paper_ids=requested
        )
    except LibraryPaperResolutionError as error:
        raise LibraryFulltextError(str(error)) from error

    targets: list[_SearchablePaper] = []
    skipped: list[dict[str, str]] = []
    for paper in resolved:
        file_id = _searchable_file(paper)
        if file_id is None:
            skipped.append(
                {
                    "paper_id": paper.paper_id,
                    "reason": "full_text_unavailable",
                }
            )
            continue
        targets.append(_SearchablePaper(paper_id=paper.paper_id, azure_file_id=file_id))
    return targets, skipped


def _searchable_file(paper: ResolvedLibraryPaper) -> str | None:
    library = paper.library_paper or {}
    azure_file_id = library.get("azure_file_id")
    if (
        library.get("vs_file_status") == "completed"
        and isinstance(azure_file_id, str)
        and azure_file_id
    ):
        return azure_file_id
    return None


def _normalize_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        raise LibraryFulltextError("paper IDs must be an array of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LibraryFulltextError("paper IDs must be an array of strings.")
        paper_id = item.strip()
        if paper_id not in seen:
            seen.add(paper_id)
            normalized.append(paper_id)
    return normalized


def _normalize_groups(
    selected: list[str], groups: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    if not groups:
        return {}
    selected_set = set(selected)
    normalized: dict[str, list[str]] = {}
    for raw_name, raw_ids in groups.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise LibraryFulltextError("Paper group names must be non-empty strings.")
        ids = _normalize_ids(raw_ids)
        if any(paper_id not in selected_set for paper_id in ids):
            raise LibraryFulltextError(
                "Paper groups can only contain papers in the current selection."
            )
        normalized[raw_name.strip()] = ids
    return normalized


def _requested_subset(selected: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return selected
    normalized = _normalize_ids(requested)
    if not normalized:
        raise LibraryFulltextError("paper_ids cannot be empty when supplied.")
    selected_set = set(selected)
    outside = [paper_id for paper_id in normalized if paper_id not in selected_set]
    if outside:
        raise LibraryFulltextError(
            "Full text can only be retrieved for papers in the current selection."
        )
    return normalized


def _bounded_int(
    value: object, *, default: int, minimum: int, maximum: int
) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _chunk_text(item: dict[str, object]) -> str:
    content = item.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    text = item.get("text")
    return text.strip() if isinstance(text, str) else ""


def _load_content_segments(
    *,
    vector_store_id: str,
    file_id: str,
    minimum_segments: int,
) -> tuple[list[str], bool]:
    """Fetch enough parsed content pages to satisfy the requested window."""
    from repositories.azure_openai.vector_stores import get_vector_store_file_content

    segments: list[str] = []
    after: str | None = None
    has_more = False
    for _ in range(8):
        page = get_vector_store_file_content(
            vector_store_id=vector_store_id,
            file_id=file_id,
            after=after,
        )
        segments.extend(_content_segments(page))
        has_more = bool(page.get("has_more"))
        next_page = page.get("next_page") or page.get("after")
        if not has_more or not isinstance(next_page, str) or not next_page:
            has_more = False
            break
        after = next_page
        if len(segments) >= minimum_segments:
            break
    return segments, has_more


def _content_segments(page: dict[str, object]) -> list[str]:
    data = page.get("data")
    if not isinstance(data, list):
        return []
    segments: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            segments.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            segments.append(text.strip())
    return segments


__all__ = [
    "LibraryContentRequest",
    "LibraryFulltextError",
    "LibrarySearchRequest",
    "get_selected_paper_full_text",
    "search_library_full_text",
]
