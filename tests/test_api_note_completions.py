"""Note completion auth, context sampling, and response contract."""
from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.note_completions import note_completions_bp
from service.note_completion import _parse_suggestion, sample_project_paper_titles

PROJECT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(note_completions_bp)
    return app.test_client()


def auth():
    return {"Authorization": "Bearer token"}


def payload():
    return {
        "context_before_cursor": "The evidence remains unclear",
        "context_after_cursor": "",
        "client_request_id": "request-1",
    }


def paper(index: int):
    return {
        "paper_id": f"paper-{index}",
        "metadata_snapshot": {"Title": f"Title {index}"},
    }


def test_completion_requires_auth(client):
    assert client.post(f"/projects/{PROJECT_ID}/notes/completions", json=payload()).status_code == 401


def test_completion_rejects_bad_payload(client):
    with patch("app.api.user.note_completions.verify_access_token", return_value="user-1"):
        response = client.post(
            f"/projects/{PROJECT_ID}/notes/completions",
            headers=auth(),
            json={"context_before_cursor": "x"},
        )
    assert response.status_code == 400


def test_completion_checks_project_and_returns_typed_suggestion(client):
    with (
        patch("app.api.user.note_completions.verify_access_token", return_value="user-1"),
        patch(
            "app.api.user.note_completions.get_note_completion_papers",
            return_value=[paper(1)],
        ) as get_context,
        patch(
            "app.api.user.note_completions.complete_note",
            return_value=(
                {"kind": "agent_action", "action": "search", "prompt": "Find evidence."},
                "gpt-6-luna",
                {"ttft_ms": 100.0, "generation_ms": 50.0, "model_total_ms": 150.0},
            ),
        ) as complete,
    ):
        response = client.post(f"/projects/{PROJECT_ID}/notes/completions", headers=auth(), json=payload())
    assert response.status_code == 200
    assert "ttft;dur=100.0" in response.headers["Server-Timing"]
    assert response.get_json() == {
        "kind": "agent_action",
        "action": "search",
        "prompt": "Find evidence.",
        "client_request_id": "request-1",
        "model": "gpt-6-luna",
    }
    get_context.assert_called_once_with(
        user_id="user-1", access_token="token", project_id=PROJECT_ID
    )
    assert complete.call_args.kwargs["library"]["titles"] == [{"paper_id": "paper-1", "title": "Title 1"}]


def test_stratified_sampling_is_deterministic():
    papers = [paper(index) for index in range(100)]
    first = sample_project_paper_titles(papers)
    second = sample_project_paper_titles(papers)
    assert first == second
    assert first["saved_total"] == 100
    assert len(first["titles"]) == 50
    assert first["omitted"] == 50
    assert [item["paper_id"] for item in first["titles"][-20:]] == [f"paper-{index}" for index in range(80, 100)]
    assert first["titles"][0]["paper_id"] == "paper-0"
    assert first["titles"][29]["paper_id"] == "paper-79"


def test_model_response_parser_enforces_schema():
    assert _parse_suggestion('{"kind":"text_completion","text":" continue"}') == {
        "kind": "text_completion",
        "text": " continue",
    }
    assert _parse_suggestion('{"kind":"none"}') == {"kind": "none"}
