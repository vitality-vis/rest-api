"""Authentication, ownership, and validation checks for scoped notes."""
from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.notes import MAX_NOTES_CONTENT_LENGTH, notes_bp
from repositories.supabase.auth import SupabaseAuthenticationError, SupabaseConfigurationError
from repositories.supabase.projects_repository import ProjectNotFoundError

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
PROJECT = {"id": PROJECT_ID, "user_id": "user-1"}

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(notes_bp)
    return app.test_client()

def auth(): return {"Authorization": "Bearer token"}

def test_unscoped_notes_routes_do_not_exist(client):
    assert client.get("/notes", headers=auth()).status_code == 404
    assert client.put("/notes", headers=auth(), json={"content": "x"}).status_code == 404

@pytest.mark.parametrize("method", ["get", "put"])
def test_scoped_notes_require_auth(client, method):
    assert getattr(client, method)(f"/projects/{PROJECT_ID}/notes", json={"content": "hello"}).status_code == 401

def test_scoped_get_checks_project_and_reads_note(client):
    with patch("app.api.user.notes.verify_access_token", return_value="user-1"), patch("app.api.user.notes.get_project", return_value=PROJECT) as project, patch("app.api.user.notes.get_user_note", return_value=None) as get_note:
        response = client.get(f"/projects/{PROJECT_ID}/notes", headers=auth())
    assert response.status_code == 200
    project.assert_called_once_with(user_id="user-1", project_id=PROJECT_ID)
    get_note.assert_called_once_with(user_id="user-1", project_id=PROJECT_ID)
    assert response.get_json()["project_id"] == PROJECT_ID

def test_scoped_put_upserts_composite_scope(client):
    saved = {"project_id": PROJECT_ID, "content": "themes", "created_at": "c", "updated_at": "u"}
    with patch("app.api.user.notes.verify_access_token", return_value="user-1"), patch("app.api.user.notes.get_project", return_value=PROJECT), patch("app.api.user.notes.upsert_user_note", return_value=saved) as upsert:
        response = client.put(f"/projects/{PROJECT_ID}/notes", headers=auth(), json={"content": "themes"})
    assert response.status_code == 200
    upsert.assert_called_once_with(user_id="user-1", project_id=PROJECT_ID, content="themes")

def test_cross_owner_project_is_404(client):
    with patch("app.api.user.notes.verify_access_token", return_value="user-1"), patch("app.api.user.notes.get_project", side_effect=ProjectNotFoundError()):
        assert client.get(f"/projects/{PROJECT_ID}/notes", headers=auth()).status_code == 404

def test_invalid_project_uuid_is_404(client):
    assert client.get("/projects/not-a-uuid/notes", headers=auth()).status_code == 404

def test_put_notes_rejects_oversized_content(client):
    with patch("app.api.user.notes.verify_access_token", return_value="user-1"):
        response = client.put(f"/projects/{PROJECT_ID}/notes", headers=auth(), json={"content": "x" * (MAX_NOTES_CONTENT_LENGTH + 1)})
    assert response.status_code == 400

def test_notes_map_auth_and_configuration_errors(client):
    with patch("app.api.user.notes.verify_access_token", side_effect=SupabaseConfigurationError("missing")):
        assert client.get(f"/projects/{PROJECT_ID}/notes", headers=auth()).status_code == 503
    with patch("app.api.user.notes.verify_access_token", side_effect=SupabaseAuthenticationError("bad")):
        assert client.get(f"/projects/{PROJECT_ID}/notes", headers=auth()).status_code == 401
