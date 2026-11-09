"""Bounded TTL cache behavior for note-completion context."""
from unittest.mock import patch

from repositories.supabase.note_completion_cache import (
    cache_note_completion_papers, clear_note_completion_cache,
    get_cached_note_completion_papers, invalidate_note_completion_context,
)


def paper(title: str = "Title") -> list[dict[str, object]]:
    return [{"paper_id": "paper-1", "metadata_snapshot": {"Title": title}}]


def setup_function():
    clear_note_completion_cache()


def teardown_function():
    clear_note_completion_cache()


def test_cache_is_scoped_by_user_and_project():
    cache_note_completion_papers(
        user_id="user-1", project_id="project-1", papers=paper()
    )
    assert get_cached_note_completion_papers(
        user_id="user-1", project_id="project-1"
    ) == paper()
    assert get_cached_note_completion_papers(
        user_id="user-2", project_id="project-1"
    ) is None


def test_cache_expires_and_can_be_invalidated():
    with patch(
        "repositories.supabase.note_completion_cache.time.monotonic", return_value=10.0
    ):
        cache_note_completion_papers(
            user_id="user-1", project_id="project-1", papers=paper()
        )
    with patch(
        "repositories.supabase.note_completion_cache.time.monotonic", return_value=41.0
    ):
        assert get_cached_note_completion_papers(
            user_id="user-1", project_id="project-1"
        ) is None

    cache_note_completion_papers(
        user_id="user-1", project_id="project-1", papers=paper()
    )
    invalidate_note_completion_context(user_id="user-1", project_id="project-1")
    assert get_cached_note_completion_papers(
        user_id="user-1", project_id="project-1"
    ) is None
