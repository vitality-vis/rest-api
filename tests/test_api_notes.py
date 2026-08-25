"""Local auth and validation checks for the notes API."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.notes import MAX_NOTES_CONTENT_LENGTH, notes_bp
from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
)


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(notes_bp)
    return app.test_client()


def test_get_notes_requires_auth(client):
    response = client.get("/notes")
    assert response.status_code == 401


def test_put_notes_requires_auth(client):
    response = client.put("/notes", json={"content": "hello"})
    assert response.status_code == 401


def test_get_notes_returns_empty_when_missing(client):
    with (
        patch("app.api.user.notes.verify_access_token", return_value="user-1"),
        patch("app.api.user.notes.get_user_note", return_value=None),
    ):
        response = client.get(
            "/notes",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "content": "",
        "created_at": None,
        "updated_at": None,
    }


def test_put_notes_upserts_content(client):
    with (
        patch("app.api.user.notes.verify_access_token", return_value="user-1"),
        patch(
            "app.api.user.notes.upsert_user_note",
            return_value={
                "content": "themes",
                "created_at": "2026-07-29T00:00:00Z",
                "updated_at": "2026-07-29T01:00:00Z",
            },
        ) as upsert,
    ):
        response = client.put(
            "/notes",
            headers={"Authorization": "Bearer token"},
            json={"content": "themes"},
        )

    upsert.assert_called_once_with(user_id="user-1", content="themes")
    assert response.status_code == 200
    assert response.get_json()["content"] == "themes"


def test_put_notes_rejects_oversized_content(client):
    with patch("app.api.user.notes.verify_access_token", return_value="user-1"):
        response = client.put(
            "/notes",
            headers={"Authorization": "Bearer token"},
            json={"content": "x" * (MAX_NOTES_CONTENT_LENGTH + 1)},
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "content is too large"


def test_get_notes_maps_configuration_error(client):
    with patch(
        "app.api.user.notes.verify_access_token",
        side_effect=SupabaseConfigurationError("missing"),
    ):
        response = client.get(
            "/notes",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 503


def test_get_notes_maps_authentication_error(client):
    with patch(
        "app.api.user.notes.verify_access_token",
        side_effect=SupabaseAuthenticationError("bad token"),
    ):
        response = client.get(
            "/notes",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 401
