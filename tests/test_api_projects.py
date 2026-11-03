"""Project API authentication, validation, and error mapping tests."""

from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.projects import projects_bp
from repositories.supabase.projects_repository import ProjectNotFoundError

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
PROJECT = {
    "id": PROJECT_ID,
    "user_id": "user-1",
    "name": "Research",
    "description": None,
    "is_default": False,
    "created_at": "2026-11-03T00:00:00Z",
    "updated_at": "2026-11-03T00:00:00Z",
}


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(projects_bp)
    return app.test_client()


def auth():
    return {"Authorization": "Bearer user-token"}


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/projects"), ("post", "/projects"), ("patch", f"/projects/{PROJECT_ID}")],
)
def test_project_routes_require_auth(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_list_filters_by_verified_user(client):
    with (
        patch("app.api.user.projects.verify_access_token", return_value="user-1"),
        patch("app.api.user.projects.list_projects", return_value=[PROJECT]) as listing,
    ):
        response = client.get("/projects", headers=auth())
    assert response.status_code == 200
    listing.assert_called_once_with(user_id="user-1", access_token="user-token")
    assert "user_id" not in response.get_json()["projects"][0]


def test_create_does_not_forward_user_id(client):
    with (
        patch("app.api.user.projects.verify_access_token", return_value="user-1"),
        patch("app.api.user.projects.create_project", return_value=PROJECT) as create,
    ):
        response = client.post(
            "/projects",
            headers=auth(),
            json={"name": " Research ", "description": None, "user_id": "attacker"},
        )
    assert response.status_code == 201
    create.assert_called_once_with(access_token="user-token", name="Research", description=None)


@pytest.mark.parametrize("name", ["", "   ", "x" * 121, None])
def test_create_rejects_invalid_name(client, name):
    with patch("app.api.user.projects.verify_access_token", return_value="user-1"):
        response = client.post("/projects", headers=auth(), json={"name": name})
    assert response.status_code == 400


def test_rename_maps_not_found_without_forwarding_user(client):
    with (
        patch("app.api.user.projects.verify_access_token", return_value="user-1"),
        patch("app.api.user.projects.rename_project", side_effect=ProjectNotFoundError()) as rename,
    ):
        response = client.patch(f"/projects/{PROJECT_ID}", headers=auth(), json={"name": "New"})
    assert response.status_code == 404
    rename.assert_called_once_with(access_token="user-token", project_id=PROJECT_ID, name="New")


def test_rename_rejects_non_name_fields(client):
    with patch("app.api.user.projects.verify_access_token", return_value="user-1"):
        response = client.patch(
            f"/projects/{PROJECT_ID}", headers=auth(), json={"name": "New", "description": "no"}
        )
    assert response.status_code == 400
