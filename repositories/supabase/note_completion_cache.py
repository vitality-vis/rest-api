"""Small process-local cache for project note-completion paper context."""
from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock

CACHE_TTL_SECONDS = 30.0
CACHE_MAX_PROJECTS = 256

_CacheKey = tuple[str, str]
_CacheValue = tuple[float, tuple[tuple[str, str], ...]]
_cache: OrderedDict[_CacheKey, _CacheValue] = OrderedDict()
_cache_lock = Lock()


def get_cached_note_completion_papers(
    *, user_id: str, project_id: str
) -> list[dict[str, object]] | None:
    """Return a defensive copy when a non-expired project entry exists."""
    key = (user_id, project_id)
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, papers = entry
        if expires_at <= now:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return _to_papers(papers)


def cache_note_completion_papers(
    *, user_id: str, project_id: str, papers: list[dict[str, object]]
) -> None:
    """Store immutable identifiers and titles with bounded TTL/LRU retention."""
    compact: list[tuple[str, str]] = []
    for paper in papers:
        metadata = paper.get("metadata_snapshot")
        paper_id = paper.get("paper_id")
        title = metadata.get("Title") if isinstance(metadata, dict) else None
        if isinstance(paper_id, str) and isinstance(title, str):
            compact.append((paper_id, title))

    key = (user_id, project_id)
    with _cache_lock:
        _cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, tuple(compact))
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_PROJECTS:
            _cache.popitem(last=False)


def invalidate_note_completion_context(*, user_id: str, project_id: str) -> None:
    """Drop one project after a successful membership mutation."""
    with _cache_lock:
        _cache.pop((user_id, project_id), None)


def clear_note_completion_cache() -> None:
    """Clear all entries; intended for tests and process lifecycle hooks."""
    with _cache_lock:
        _cache.clear()


def _to_papers(papers: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    return [
        {"paper_id": paper_id, "metadata_snapshot": {"Title": title}}
        for paper_id, title in papers
    ]
