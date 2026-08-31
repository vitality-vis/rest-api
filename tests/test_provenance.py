"""Tests for shared provenance envelope helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.provenance.emit import (
    log_mcp_tool_event,
    sanitize_mcp_args,
    summarize_mcp_result,
    validate_ui_event,
)


def test_validate_ui_event_defaults_source():
    event = validate_ui_event(
        {
            "eventId": "evt-1",
            "sessionId": "sess-1",
            "actorType": "user",
            "action": "search.submit",
            "eventData": {"searchMode": "keyword.bm25"},
        }
    )
    assert event["source"] == "ui"


def test_validate_ui_event_rejects_non_ui_source():
    with pytest.raises(ValueError, match="must use source 'ui'"):
        validate_ui_event(
            {
                "source": "mcp",
                "eventId": "evt-1",
                "sessionId": "sess-1",
                "actorType": "user",
                "action": "search.submit",
                "eventData": {},
            }
        )


def test_sanitize_mcp_args_truncates_long_strings():
    sanitized = sanitize_mcp_args({"query": "x" * 600, "limit": 10})
    assert sanitized["query"].endswith("...")
    assert sanitized["limit"] == 10
    assert "authors" not in sanitized


@patch("app.provenance.emit.emit_provenance_event")
def test_log_mcp_tool_event_shape(emit_mock):
    log_mcp_tool_event(
        tool="search_papers_bm25",
        args={"query": "uncertainty visualization", "limit": 5},
        status="ok",
        latency_ms=42,
        result_summary={"resultTotal": 12, "resultCount": 5},
    )

    event = emit_mock.call_args.args[0]
    assert event["source"] == "mcp"
    assert event["action"] == "tool.search_papers_bm25"
    assert "sessionId" not in event
    assert "actorType" not in event
    assert event["eventData"]["status"] == "ok"
    assert event["eventData"]["latencyMs"] == 42


def test_summarize_mcp_result_search():
    summary = summarize_mcp_result(
        SimpleNamespace(
            papers=[SimpleNamespace(paper_id="p1")],
            total=99,
            has_more=True,
        )
    )
    assert summary == {
        "resultCount": 1,
        "resultTotal": 99,
        "resultHasMore": True,
    }
