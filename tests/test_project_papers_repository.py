"""Tenant boundary tests for project paper persistence."""
from unittest.mock import Mock, patch

from repositories.supabase.client import SupabaseSettings
from repositories.supabase.project_papers_repository import (
    add_project_paper, list_project_papers, remove_project_paper,
    update_project_paper_tags,
)

SETTINGS = SupabaseSettings(url="https://example.supabase.co", service_role_key="service-key")


def response(status, payload):
    result = Mock(status_code=status, content=b"x")
    result.json.return_value = payload
    return result


def test_list_constrains_user_project_and_uses_membership_tags():
    membership = {"tags": ["project-tag"], "created_at": "now", "updated_at": "now", "user_papers": {"id": "asset-1"}}
    with patch("repositories.supabase.project_papers_repository.get_supabase_settings", return_value=SETTINGS), patch("repositories.supabase.project_papers_repository.requests.request", return_value=response(200, [membership])) as request:
        papers = list_project_papers(user_id="user-1", project_id="project-1")
    params = request.call_args.kwargs["params"]
    assert params["user_id"] == "eq.user-1"
    assert params["project_id"] == "eq.project-1"
    assert papers[0]["tags"] == ["project-tag"]


def test_add_membership_is_idempotent_and_tenant_scoped():
    row = {"user_id": "user-1", "project_id": "project-1", "user_paper_id": "asset-1", "tags": []}
    with patch("repositories.supabase.project_papers_repository.get_supabase_settings", return_value=SETTINGS), patch("repositories.supabase.project_papers_repository.requests.request", return_value=response(201, [row])) as request:
        assert add_project_paper(user_id="user-1", project_id="project-1", user_paper_id="asset-1") == row
    assert request.call_args.kwargs["params"]["on_conflict"] == "project_id,user_paper_id"
    assert "resolution=ignore-duplicates" in request.call_args.kwargs["headers"]["Prefer"]
    assert request.call_args.kwargs["json"]["user_id"] == "user-1"


def test_tag_update_constrains_all_tenant_keys():
    with patch("repositories.supabase.project_papers_repository.get_supabase_settings", return_value=SETTINGS), patch("repositories.supabase.project_papers_repository.requests.request", return_value=response(200, [{"tags": ["x"]}])) as request:
        update_project_paper_tags(user_id="user-1", project_id="project-1", user_paper_id="asset-1", tags=["x"])
    assert request.call_args.kwargs["params"] == {"user_id": "eq.user-1", "project_id": "eq.project-1", "user_paper_id": "eq.asset-1"}


def test_remove_deletes_only_tenant_scoped_membership():
    deleted = [{"user_id": "user-1", "project_id": "project-1", "user_paper_id": "asset-1"}]
    with patch("repositories.supabase.project_papers_repository.get_supabase_settings", return_value=SETTINGS), patch("repositories.supabase.project_papers_repository.requests.request", return_value=response(200, deleted)) as request:
        assert remove_project_paper(user_id="user-1", project_id="project-1", user_paper_id="asset-1") is True
    assert request.call_args.args[0] == "DELETE"
    assert request.call_args.args[1].endswith("/project_papers")
    assert request.call_args.kwargs["params"] == {"user_id": "eq.user-1", "project_id": "eq.project-1", "user_paper_id": "eq.asset-1"}
    assert request.call_args.kwargs["headers"]["Prefer"] == "return=representation"


def test_remove_returns_false_when_membership_is_absent():
    with patch("repositories.supabase.project_papers_repository.get_supabase_settings", return_value=SETTINGS), patch("repositories.supabase.project_papers_repository.requests.request", return_value=response(200, [])):
        assert remove_project_paper(user_id="user-1", project_id="project-1", user_paper_id="asset-1") is False
