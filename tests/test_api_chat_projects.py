"""Explicit project boundary tests for chat history operations."""
from unittest.mock import patch
import pytest
from flask import Flask
from app.api.chat import chat_bp

PROJECT_ID = "11111111-1111-4111-8111-111111111111"
PROJECT = {"id": PROJECT_ID, "user_id": "user-1"}

@pytest.fixture
def client():
    app = Flask(__name__); app.register_blueprint(chat_bp); return app.test_client()

def auth(): return {"Authorization": "Bearer token"}

@pytest.mark.parametrize(("method", "path", "payload"), [
    ("post", "/chat/import", {"conversations": []}),
    ("get", "/chat/conversations", None),
    ("put", f"/chat/conversations/{PROJECT_ID}/closed", {"is_closed": True}),
])
def test_chat_history_apis_require_project_id(client, method, path, payload):
    with patch("app.api.chat.verify_access_token", return_value="user-1"):
        kwargs = {"headers": auth()}
        if payload is not None: kwargs["json"] = payload
        response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 400
    assert "project_id is required" in response.get_data(as_text=True)

def test_chat_list_is_project_scoped(client):
    with patch("app.api.chat.verify_access_token", return_value="user-1"), patch("app.api.chat.get_project", return_value=PROJECT), patch("app.api.chat.load_user_conversations", return_value=[]) as load:
        response = client.get(f"/chat/conversations?project_id={PROJECT_ID}", headers=auth())
    assert response.status_code == 200
    load.assert_called_once_with(user_id="user-1", project_id=PROJECT_ID)
