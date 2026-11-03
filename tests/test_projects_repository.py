"""HTTP boundary tests for project persistence."""

from unittest.mock import Mock, patch

import pytest

from repositories.supabase.client import SupabaseSettings
from repositories.supabase.projects_repository import (
    ProjectNotFoundError,
    ProjectsPersistenceError,
    create_project,
    get_project,
    list_projects,
)

SETTINGS = SupabaseSettings(url="https://example.supabase.co", service_role_key="service-key")


def response(status, payload):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


def test_rpc_uses_user_authorization_and_service_apikey():
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch("repositories.supabase.projects_repository.requests.request", return_value=response(200, {"id": "p1"})) as request,
    ):
        create_project(access_token="user-token", name="Name", description=None)
    headers = request.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer user-token"
    assert headers["apikey"] == "service-key"
    assert request.call_args.kwargs["json"] == {"p_name": "Name", "p_description": None}


def test_list_ensures_default_with_user_token_then_lists_verified_user():
    default = {"id": "default-1", "user_id": "user-1", "is_default": True}
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch(
            "repositories.supabase.projects_repository.requests.request",
            side_effect=[response(200, [default]), response(200, [default])],
        ) as request,
    ):
        assert list_projects(user_id="user-1", access_token="user-token") == [default]
    rpc_call, list_call = request.call_args_list
    assert rpc_call.args[1].endswith("/rpc/get_or_create_default_project")
    assert rpc_call.kwargs["headers"]["Authorization"] == "Bearer user-token"
    assert rpc_call.kwargs["headers"]["apikey"] == "service-key"
    assert rpc_call.kwargs["json"] == {}
    assert "user_id" not in rpc_call.kwargs["json"]
    assert list_call.kwargs["params"]["user_id"] == "eq.user-1"
    assert "archived_at" not in list_call.kwargs["params"]


def test_empty_new_user_rpc_result_is_then_returned_by_list():
    default = {"id": "default-1", "user_id": "user-1", "is_default": True}
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch(
            "repositories.supabase.projects_repository.requests.request",
            side_effect=[response(200, [default]), response(200, [default])],
        ),
    ):
        projects = list_projects(user_id="user-1", access_token="new-user-token")
    assert projects == [default]


def test_list_rejects_cross_user_response():
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch(
            "repositories.supabase.projects_repository.requests.request",
            side_effect=[
                response(200, [{"user_id": "user-1", "is_default": True}]),
                response(200, [{"user_id": "user-2"}]),
            ],
        ),
        pytest.raises(ProjectsPersistenceError),
    ):
        list_projects(user_id="user-1", access_token="token")


def test_get_project_filters_id_and_user():
    project = {
        "id": "project-1", "user_id": "user-1", "is_default": False
    }
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch("repositories.supabase.projects_repository.requests.request", return_value=response(200, [project])) as request,
    ):
        assert get_project(user_id="user-1", project_id="project-1") == project
    params = request.call_args.kwargs["params"]
    assert params["id"] == "eq.project-1"
    assert params["user_id"] == "eq.user-1"
    assert "archived_at" not in params


def test_get_project_absent_is_not_found():
    with (
        patch("repositories.supabase.projects_repository.get_supabase_settings", return_value=SETTINGS),
        patch("repositories.supabase.projects_repository.requests.request", return_value=response(200, [])),
        pytest.raises(ProjectNotFoundError),
    ):
        get_project(user_id="user-1", project_id="project-1")
