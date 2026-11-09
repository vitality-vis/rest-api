"""Tests for the dedicated note-completion context RPC client."""
from unittest.mock import Mock, patch

import pytest

from repositories.supabase.client import SupabaseSettings
from repositories.supabase.note_completion_cache import clear_note_completion_cache
from repositories.supabase.note_completion_repository import (
    NoteCompletionContextError, NoteCompletionProjectNotFoundError,
    get_note_completion_papers,
)

SETTINGS = SupabaseSettings(
    url="https://example.supabase.co", service_role_key="service-key"
)



@pytest.fixture(autouse=True)
def isolated_cache():
    clear_note_completion_cache()
    yield
    clear_note_completion_cache()


def response(status: int, payload: object) -> Mock:
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


def test_context_rpc_uses_user_token_and_returns_minimal_papers():
    payload = {
        "saved_total": 1,
        "papers": [{"paper_id": "paper-1", "title": "Title 1"}],
    }
    with (
        patch(
            "repositories.supabase.note_completion_repository.get_supabase_settings",
            return_value=SETTINGS,
        ),
        patch(
            "repositories.supabase.note_completion_repository.pooled_request",
            return_value=response(200, payload),
        ) as request,
    ):
        papers = get_note_completion_papers(
            user_id="user-1", access_token="user-token", project_id="project-1"
        )

    assert request.call_args.args[1].endswith(
        "/rest/v1/rpc/get_note_completion_context"
    )
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer user-token"
    assert request.call_args.kwargs["json"] == {"p_project_id": "project-1"}
    assert papers == [
        {"paper_id": "paper-1", "metadata_snapshot": {"Title": "Title 1"}}
    ]


def test_context_rpc_maps_missing_project():
    with (
        patch(
            "repositories.supabase.note_completion_repository.get_supabase_settings",
            return_value=SETTINGS,
        ),
        patch(
            "repositories.supabase.note_completion_repository.pooled_request",
            return_value=response(404, {"code": "P0002"}),
        ),
        pytest.raises(NoteCompletionProjectNotFoundError),
    ):
        get_note_completion_papers(
            user_id="user-1", access_token="token", project_id="missing"
        )


def test_context_rpc_rejects_malformed_payload():
    with (
        patch(
            "repositories.supabase.note_completion_repository.get_supabase_settings",
            return_value=SETTINGS,
        ),
        patch(
            "repositories.supabase.note_completion_repository.pooled_request",
            return_value=response(200, {"papers": [{"paper_id": "paper-1"}]}),
        ),
        pytest.raises(NoteCompletionContextError),
    ):
        get_note_completion_papers(
            user_id="user-1", access_token="token", project_id="project-1"
        )


def test_context_rpc_reuses_cached_project_context():
    payload = {
        "saved_total": 1,
        "papers": [{"paper_id": "paper-1", "title": "Title 1"}],
    }
    with (
        patch(
            "repositories.supabase.note_completion_repository.get_supabase_settings",
            return_value=SETTINGS,
        ),
        patch(
            "repositories.supabase.note_completion_repository.pooled_request",
            return_value=response(200, payload),
        ) as request,
    ):
        first = get_note_completion_papers(
            user_id="user-1", access_token="token", project_id="project-1"
        )
        first[0]["metadata_snapshot"]["Title"] = "mutated"
        second = get_note_completion_papers(
            user_id="user-1", access_token="token", project_id="project-1"
        )

    request.assert_called_once()
    assert second[0]["metadata_snapshot"]["Title"] == "Title 1"
