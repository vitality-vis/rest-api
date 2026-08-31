"""Emit provenance envelopes to the shared logger sink."""

from __future__ import annotations

import time
from typing import Any, Literal
from uuid import uuid4

from logger_config import get_logger

PROVENANCE_SCHEMA_VERSION = 1
ProvenanceSource = Literal["ui", "mcp", "agent"]
_MAX_STRING_LENGTH = 500
_MAX_LIST_ITEMS = 20


def emit_provenance_event(event: dict[str, Any]) -> None:
    """Write one provenance envelope (UI, MCP, or future agent events)."""
    logger = get_logger()
    payload = dict(event)
    overview = payload.pop("message", None) or _format_overview(payload)
    # String message -> textPayload in GCP; structured fields -> jsonPayload.
    logger.info(
        overview,
        extra={
            "provenance_event": True,
            "json_fields": payload,
        },
    )


def validate_ui_event(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a browser Socket.IO provenance payload."""
    if not isinstance(data, dict):
        raise ValueError("event must be an object")

    event_id = data.get("eventId")
    session_id = data.get("sessionId")
    action = data.get("action")
    event_data = data.get("eventData")
    actor_type = data.get("actorType")

    if not all(isinstance(value, str) and value for value in (event_id, session_id, action)):
        raise ValueError("eventId, sessionId, and action are required")
    if not isinstance(event_data, dict):
        raise ValueError("eventData must be an object")
    if not isinstance(actor_type, str) or not actor_type:
        raise ValueError("actorType is required for ui provenance events")

    source = data.get("source") or "ui"
    if source != "ui":
        raise ValueError("Socket.IO provenance events must use source 'ui'")

    return {**data, "source": source}


def log_mcp_tool_event(
    *,
    tool: str,
    args: dict[str, Any],
    status: Literal["ok", "error"],
    latency_ms: int,
    result_summary: dict[str, Any] | None = None,
    error_message: str | None = None,
    mcp_session_id: str | None = None,
    agent_run_id: str | None = None,
    client: str | None = None,
) -> None:
    """Record one public MCP tool invocation."""
    event_data: dict[str, Any] = {
        "tool": tool,
        "args": sanitize_mcp_args(args),
        "status": status,
        "latencyMs": latency_ms,
    }
    if result_summary:
        event_data.update(result_summary)
    if error_message:
        event_data["errorMessage"] = _truncate(error_message)

    event: dict[str, Any] = {
        "schemaVersion": PROVENANCE_SCHEMA_VERSION,
        "source": "mcp",
        "eventId": uuid4().hex,
        "timestamp": int(time.time() * 1000),
        "action": f"tool.{tool}",
        "eventData": event_data,
    }
    if mcp_session_id:
        event["mcpSessionId"] = mcp_session_id
    if agent_run_id:
        event["agentRunId"] = agent_run_id
    if client:
        event["client"] = client

    emit_provenance_event(event)


def sanitize_mcp_args(args: dict[str, Any]) -> dict[str, Any]:
    """Keep MCP args bounded for logs; omit empty values."""
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        if value is None or value == "" or value == []:
            continue
        sanitized[key] = _sanitize_value(value)
    return sanitized


def summarize_mcp_result(result: Any) -> dict[str, Any]:
    """Derive a compact result summary without logging full payloads."""
    total = getattr(result, "total", None)
    has_more = getattr(result, "has_more", None)
    papers = getattr(result, "papers", None)
    if papers is not None:
        summary: dict[str, Any] = {
            "resultCount": len(papers),
        }
        if total is not None:
            summary["resultTotal"] = total
        if has_more is not None:
            summary["resultHasMore"] = has_more
        return summary

    paper_id = getattr(result, "paper_id", None) or getattr(result, "id", None)
    if paper_id is not None:
        return {"paperId": paper_id}

    doi = getattr(result, "doi", None)
    if doi is not None:
        summary = {"doi": doi}
        references = getattr(result, "references", None)
        cited_by = getattr(result, "cited_by", None)
        if references is not None:
            summary["referencesCount"] = len(getattr(references, "papers", []) or [])
        if cited_by is not None:
            summary["citedByCount"] = len(getattr(cited_by, "papers", []) or [])
        return summary

    return {}


def _format_overview(event: dict[str, Any]) -> str:
    source = event.get("source", "ui")
    action = event.get("action", "unknown")
    if source == "mcp":
        event_data = event.get("eventData") or {}
        status = event_data.get("status", "unknown")
        latency_ms = event_data.get("latencyMs")
        latency = f" | {latency_ms}ms" if latency_ms is not None else ""
        total = event_data.get("resultTotal")
        total_suffix = f" | total={total}" if total is not None else ""
        return f"MCP {action} | {status}{latency}{total_suffix}"

    actor_type = event.get("actorType", "unknown")
    return f"Provenance {source} | {actor_type} | {action}"


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        trimmed = [_sanitize_value(item) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            trimmed.append(f"...(+{len(value) - _MAX_LIST_ITEMS} more)")
        return trimmed
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def _truncate(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return value[:_MAX_STRING_LENGTH] + "..."
