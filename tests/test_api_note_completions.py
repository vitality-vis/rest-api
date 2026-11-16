"""Note completion auth, context sampling, and response contract."""
from unittest.mock import patch

import pytest
from flask import Flask

from app.api.user.note_completions import note_completions_bp
from service.note_completion import (
    _parse_suggestion,
    _user_context,
    format_paragraph_citations,
    sample_project_paper_titles,
)
from service.note_completion_rate_limit import note_completion_rate_limiter

PROJECT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def client():
    note_completion_rate_limiter.reset()
    app = Flask(__name__)
    app.register_blueprint(note_completions_bp)
    return app.test_client()


def auth():
    return {"Authorization": "Bearer token"}


def payload(**extra):
    return {
        "context_before_cursor": "The evidence remains unclear",
        "context_after_cursor": "",
        "client_request_id": "request-1",
        **extra,
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
    assert complete.call_args.kwargs["paragraph_citations"] == []


def test_completion_resolves_cited_papers_for_current_paragraph(client):
    resolved = [
        {
            "ID": "p1",
            "Title": "Deconstructing Human-AI Collaboration",
            "Abstract": "Scholars use AI during literature research.",
            "Authors": ["A", "B"],
            "Year": 2024,
            "Source": "CHI",
        }
    ]
    with (
        patch("app.api.user.note_completions.verify_access_token", return_value="user-1"),
        patch("app.api.user.note_completions.get_note_completion_papers", return_value=[]),
        patch(
            "app.api.user.note_completions.resolve_papers",
            return_value=resolved,
        ) as resolve,
        patch(
            "app.api.user.note_completions.complete_note",
            return_value=(
                {"kind": "text_completion", "text": " during literature research."},
                "gpt-6-luna",
                {"ttft_ms": 10.0, "generation_ms": 5.0, "model_total_ms": 15.0},
            ),
        ) as complete,
    ):
        response = client.post(
            f"/projects/{PROJECT_ID}/notes/completions",
            headers=auth(),
            json=payload(cited_paper_ids=["p1", "p1", "  "]),
        )
    assert response.status_code == 200
    resolve.assert_called_once_with(
        user_id="user-1", paper_ids=["p1"], hydrate_openalex=False
    )
    assert complete.call_args.kwargs["paragraph_citations"] == resolved


def test_completion_rejects_invalid_cited_paper_ids(client):
    with patch("app.api.user.note_completions.verify_access_token", return_value="user-1"):
        response = client.post(
            f"/projects/{PROJECT_ID}/notes/completions",
            headers=auth(),
            json=payload(cited_paper_ids="p1"),
        )
    assert response.status_code == 400


def test_completion_rate_limits_by_user_and_project(client):
    completion_result = (
        {"kind": "none"},
        "gpt-6-luna",
        {"ttft_ms": 10.0, "generation_ms": 5.0, "model_total_ms": 15.0},
    )
    with (
        patch("app.api.user.note_completions.verify_access_token", return_value="user-1"),
        patch("app.api.user.note_completions.get_note_completion_papers", return_value=[]),
        patch("app.api.user.note_completions.complete_note", return_value=completion_result),
    ):
        first = client.post(f"/projects/{PROJECT_ID}/notes/completions", headers=auth(), json=payload())
        second = client.post(f"/projects/{PROJECT_ID}/notes/completions", headers=auth(), json=payload())
        limited = client.post(f"/projects/{PROJECT_ID}/notes/completions", headers=auth(), json=payload())

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


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


def test_paragraph_citation_formatting_includes_metadata_and_abstract():
    formatted = format_paragraph_citations(
        [
            {
                "ID": "p1",
                "Title": "Deconstructing Human-AI Collaboration",
                "Abstract": "A" * 1_300,
                "Authors": ["Ada", "Bob", "Cara", "Dan", "Eve", "Fay", "Gus", "Hank", "Ivy"],
                "Year": 2024,
                "Source": "CHI",
            }
        ]
    )
    assert formatted.startswith("[[ID:p1]](")
    assert "title=Deconstructing Human-AI Collaboration" in formatted
    assert "year=2024" in formatted
    assert "venue=CHI" in formatted
    assert "authors=Ada, Bob, Cara, Dan, Eve, Fay, Gus, Hank (+1 more)" in formatted
    assert 'abstract truncated="true"=' in formatted
    assert "A" * 1_200 in formatted
    assert "A" * 1_201 not in formatted


def test_user_context_includes_paragraph_citations_only_when_present():
    library = {"saved_total": 0, "titles": [], "omitted": 0}
    without = _user_context(before="hello", after="", library=library)
    assert "PARAGRAPH_CITATIONS" not in without
    with_citations = _user_context(
        before="[[ID:p1]] ",
        after="",
        library=library,
        paragraph_citations=[
            {
                "ID": "p1",
                "Title": "Paper",
                "Abstract": "Abs",
                "Year": 2024,
                "Source": "CHI",
                "Authors": ["Ada"],
            }
        ],
    )
    assert "<PARAGRAPH_CITATIONS count=\"1\">" in with_citations
    assert "[[ID:p1]](title=Paper; year=2024; venue=CHI; authors=Ada; abstract=Abs)" in with_citations
