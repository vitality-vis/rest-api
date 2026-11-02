"""Vitality tools executed locally when Codex requests them."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.chat.events import AgentAction, PapersResult

Emit = Callable[[Any], None]

DYNAMIC_TOOLS = [
    {
        "type": "function",
        "name": "search_papers",
        "description": (
            "Search the VitaLITy paper corpus. Use bm25 for exact terminology, "
            "vector for conceptual questions, and bool for exact phrases or "
            "Boolean expressions. Results are private working evidence and are not "
            "shown as papers in the user interface; use present_papers to show a "
            "curated subset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string", "enum": ["bm25", "vector", "bool"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "present_papers",
        "description": (
            "Present a curated subset of papers returned by search_papers during "
            "this turn. Call this only after judging relevance. The application "
            "shows these papers to the user in the supplied order."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 20,
                }
            },
            "required": ["paper_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "inspect_selected_papers",
        "description": (
            "Inspect the current paper selection without loading all abstracts. "
            "Returns the selected count and a small deterministic metadata sample. "
            "Use this first when the selection is large or its scope is unclear."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_selected_paper_evidence",
        "description": (
            "Retrieve bounded metadata evidence for papers in the current selection. "
            "You generate the answer yourself from the returned evidence. Use auto for "
            "normal questions, titles for broad overviews, or abstracts for detailed "
            "analysis. You may request a selected subset by ID. The server enforces "
            "selection scope and evidence budgets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "detail_level": {
                    "type": "string",
                    "enum": ["auto", "titles", "abstracts"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "sampling": {
                    "type": "string",
                    "enum": ["representative", "selected_order"],
                },
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "additionalProperties": False,
        },
    },
]


def tool_result(text: str, *, success: bool) -> dict[str, object]:
    """Return the dynamic-tool response Codex expects."""
    return {
        "contentItems": [{"type": "inputText", "text": text}],
        "success": success,
    }


def execute_tool(
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
    effort: str,
    call_id: str,
    tool: str,
    arguments: object,
    emit: Emit,
    candidates: dict[str, dict[str, object]] | None = None,
    presented_paper_ids: list[str] | None = None,
) -> dict[str, object]:
    """Run one Codex tool call inside the trusted Vitality process."""
    payload = _arguments(arguments)
    emit(
        AgentAction(
            action_id=call_id,
            action=_public_action(tool),
            status="started",
            data=_started_action_data(tool, payload, selected_paper_ids),
        )
    )
    try:
        if tool == "search_papers":
            result = _search_papers(payload, candidates=candidates)
        elif tool == "present_papers":
            result = _present_papers(
                payload,
                effort=effort,
                emit=emit,
                candidates=candidates,
                presented_paper_ids=presented_paper_ids,
            )
        elif tool == "inspect_selected_papers":
            result = _inspect_selected_papers(
                user_id=user_id,
                selected_paper_ids=selected_paper_ids,
            )
        elif tool == "get_selected_paper_evidence":
            result = _get_selected_paper_evidence(
                payload,
                user_id=user_id,
                selected_paper_ids=selected_paper_ids,
            )
        else:
            result = tool_result(f"{tool} is not available.", success=False)
    except Exception as error:
        emit(
            AgentAction(
                action_id=call_id,
                action=_public_action(tool),
                status="failed",
                data={"message": _public_error(error)},
            )
        )
        return tool_result(_public_error(error), success=False)
    emit(
        AgentAction(
            action_id=call_id,
            action=_public_action(tool),
            status="completed" if result["success"] else "failed",
            data=_completed_action_data(tool, result),
        )
    )
    return result


def _search_papers(
    payload: dict[str, Any],
    *,
    candidates: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    from model.paper import SearchRequest
    from service.search import SearchUnavailableError, search

    query = str(payload.get("query") or "").strip()
    if not query:
        return tool_result("A search query is required.", success=False)
    mode = payload.get("mode") or "bm25"
    if mode not in {"bm25", "vector", "bool"}:
        mode = "bm25"
    try:
        limit = int(payload.get("limit") or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = min(max(limit, 1), 20)
    try:
        found = search(SearchRequest(search_query=query, search_mode=mode, limit=limit))
    except SearchUnavailableError as error:
        return tool_result(str(error), success=False)

    papers: list[dict[str, object]] = []
    for paper in found.papers:
        paper_id = paper.get("ID") or paper.get("id")
        if not paper_id:
            continue
        paper_id = str(paper_id)
        item: dict[str, object] = {
            "paper_id": paper_id,
            "title": paper.get("Title") or "",
            "abstract": str(paper.get("Abstract") or "")[:2500],
            "authors": list(paper.get("Authors") or [])[:12],
            "keywords": list(paper.get("Keywords") or [])[:20],
            "year": paper.get("Year"),
            "venue": paper.get("Source") or "",
            "citation_count": paper.get("CitationCounts"),
        }
        papers.append(item)
        if candidates is not None:
            candidates[paper_id] = item
    return tool_result(
        json.dumps(
            {
                "query": query,
                "mode": mode,
                "papers": papers,
                "instruction": (
                    "Judge relevance from this metadata. Search again only if it "
                    "would materially improve coverage; call present_papers when "
                    "you have a useful set to show."
                ),
            },
            ensure_ascii=False,
        ),
        success=True,
    )


def _present_papers(
    payload: dict[str, Any],
    *,
    effort: str,
    emit: Emit,
    candidates: dict[str, dict[str, object]] | None,
    presented_paper_ids: list[str] | None,
) -> dict[str, object]:
    requested = payload.get("paper_ids")
    ids = (
        list(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))
        if isinstance(requested, list)
        else []
    )
    available = candidates or {}
    accepted = [paper_id for paper_id in ids if paper_id in available]
    rejected = [paper_id for paper_id in ids if paper_id not in available]
    if not accepted:
        return tool_result(
            "No supplied paper IDs came from search results in this turn.",
            success=False,
        )
    cumulative = list(presented_paper_ids or [])
    for paper_id in accepted:
        if paper_id not in cumulative:
            cumulative.append(paper_id)
    if presented_paper_ids is not None:
        presented_paper_ids[:] = cumulative
    emit(
        PapersResult(
            ids=cumulative,
            ranked_ids=cumulative,
            policy="agent_curated",
            effort=effort if effort in {"low", "medium", "high"} else "low",
            count_known=True,
        )
    )
    return tool_result(
        json.dumps(
            {
                "presented_paper_ids": accepted,
                "all_presented_paper_ids": cumulative,
                "rejected_unknown_ids": rejected,
            },
            ensure_ascii=False,
        ),
        success=True,
    )


def _inspect_selected_papers(
    *, user_id: str | None, selected_paper_ids: list[str]
) -> dict[str, object]:
    from service.selected_paper_tool import (
        SelectedPaperToolError,
        inspect_selected_papers,
    )

    try:
        overview = inspect_selected_papers(
            user_id=user_id,
            selected_paper_ids=selected_paper_ids,
        )
    except SelectedPaperToolError as error:
        return tool_result(str(error), success=False)
    return tool_result(json.dumps(overview, ensure_ascii=False), success=True)


def _get_selected_paper_evidence(
    payload: dict[str, Any],
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
) -> dict[str, object]:
    from service.selected_paper_tool import (
        EvidenceRequest,
        SelectedPaperToolError,
        get_selected_paper_evidence,
    )

    requested_ids = payload.get("paper_ids")
    paper_ids = (
        [str(item).strip() for item in requested_ids if str(item).strip()]
        if isinstance(requested_ids, list)
        else None
    )
    try:
        evidence = get_selected_paper_evidence(
            user_id=user_id,
            selected_paper_ids=selected_paper_ids,
            request=EvidenceRequest(
                detail_level=str(payload.get("detail_level") or "auto"),
                limit=payload.get("limit"),
                sampling=str(payload.get("sampling") or "representative"),
                paper_ids=paper_ids,
            ),
        )
    except SelectedPaperToolError as error:
        return tool_result(str(error), success=False)
    return tool_result(json.dumps(evidence, ensure_ascii=False), success=True)


def _arguments(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _started_action_data(
    tool: str, payload: dict[str, Any], selected_paper_ids: list[str]
) -> dict[str, object]:
    """Expose only concise, user-safe tool context to the activity UI."""
    if tool == "search_papers":
        return {
            "query": str(payload.get("query") or "").strip()[:200],
            "mode": str(payload.get("mode") or "bm25"),
        }
    if tool == "present_papers":
        requested = payload.get("paper_ids")
        return {"paperCount": len(requested) if isinstance(requested, list) else 0}
    if tool == "inspect_selected_papers":
        return {"paperCount": len(selected_paper_ids)}
    if tool == "get_selected_paper_evidence":
        requested = payload.get("paper_ids")
        paper_ids = requested if isinstance(requested, list) else selected_paper_ids
        return {
            "paperCount": len(paper_ids),
            "detailLevel": str(payload.get("detail_level") or "auto"),
        }
    return {}


def _completed_action_data(
    tool: str, result: dict[str, object]
) -> dict[str, object]:
    """Summarize a successful tool response without exposing its full payload."""
    items = result.get("contentItems")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return {}
    text = items[0].get("text")
    if not isinstance(text, str):
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    if tool == "search_papers":
        papers = payload.get("papers")
        return {"resultCount": len(papers)} if isinstance(papers, list) else {}
    if tool == "present_papers":
        paper_ids = payload.get("presented_paper_ids")
        return {"paperCount": len(paper_ids)} if isinstance(paper_ids, list) else {}
    if tool == "inspect_selected_papers":
        count = payload.get("selected_count")
        return {"paperCount": count} if isinstance(count, int) else {}
    if tool == "get_selected_paper_evidence":
        data: dict[str, object] = {}
        if isinstance(payload.get("returned_count"), int):
            data["paperCount"] = payload["returned_count"]
        if isinstance(payload.get("detailed_count"), int):
            data["detailedCount"] = payload["detailed_count"]
        return data
    return {}


def _public_action(tool: str) -> str:
    if tool == "search_papers":
        return "search"
    if tool == "present_papers":
        return "present.papers"
    if tool == "inspect_selected_papers":
        return "inspect.selected_papers"
    if tool == "get_selected_paper_evidence":
        return "get.selected_paper_evidence"
    return "tool"


def _public_error(error: Exception) -> str:
    message = str(error).strip()
    if not message or len(message) > 300:
        return "The tool could not be completed."
    return message
