"""HTTP boundary tests for project-scoped note persistence."""
from unittest.mock import Mock, patch

from repositories.supabase.client import SupabaseSettings
from repositories.supabase.user_notes_repository import get_user_note, upsert_user_note

SETTINGS = SupabaseSettings(url="https://example.supabase.co", service_role_key="service-key")

def response(status, payload):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result

def request_patch(result):
    return (
        patch("repositories.supabase.user_notes_repository.get_supabase_settings", return_value=SETTINGS),
        patch("repositories.supabase.user_notes_repository.requests.request", return_value=result),
    )

def test_get_filters_user_and_project():
    settings, requesting = request_patch(response(200, []))
    with settings, requesting as request:
        get_user_note(user_id="user-1", project_id="project-1")
    assert request.call_args.kwargs["params"]["user_id"] == "eq.user-1"
    assert request.call_args.kwargs["params"]["project_id"] == "eq.project-1"

def test_upsert_uses_composite_conflict_target_and_project_id():
    settings, requesting = request_patch(response(201, [{"project_id": "project-1"}]))
    with settings, requesting as request:
        upsert_user_note(user_id="user-1", project_id="project-1", content="themes")
    assert request.call_args.kwargs["params"] == {"on_conflict": "user_id,project_id"}
    assert request.call_args.kwargs["json"] == {"user_id": "user-1", "project_id": "project-1", "content": "themes"}
