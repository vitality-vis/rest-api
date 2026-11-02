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
        "name": "find_similar_papers",
        "description": (
            "Find corpus papers conceptually similar to trusted papers in the current "
            "selection. Supply only selected seed IDs; the server validates scope, "
            "samples at most 20 spatially representative seeds, and excludes all "
            "currently selected papers from results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_selected_paper_citations",
        "description": (
            "Retrieve references or citing works for up to 5 trusted selected papers. "
            "DOIs are resolved by the server. Corpus matches become presentable "
            "candidates; external works are evidence only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "direction": {
                    "type": "string",
                    "enum": ["references", "cited_by"],
                },
                "limit_per_paper": {
                    "type": "integer", "minimum": 1, "maximum": 20
                },
            },
            "required": ["paper_ids"],
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
            "analysis. Use group=library or group=map to inspect those groups "
            "separately, or group=all for their union. You may request a selected "
            "subset by ID. The server enforces selection scope and evidence budgets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "detail_level": {
                    "type": "string",
                    "enum": ["auto", "titles", "abstracts"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 100,
                },
                "group": {
                    "type": "string",
                    "description": "Named current-selection group such as library or map; use all for the union.",
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
    selected_paper_groups: dict[str, list[str]] | None = None,
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
            result = _search_papers(
                payload, candidates=candidates, excluded_ids=selected_paper_ids
            )
        elif tool == "find_similar_papers":
            result = _find_similar_papers(
                payload,
                selected_paper_ids=selected_paper_ids,
                candidates=candidates,
            )
        elif tool == "get_selected_paper_citations":
            result = _get_selected_paper_citations(
                payload,
                user_id=user_id,
                selected_paper_ids=selected_paper_ids,
                candidates=candidates,
            )
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
                selected_paper_groups=selected_paper_groups,
            )
        elif tool == "get_selected_paper_evidence":
            result = _get_selected_paper_evidence(
                payload,
                user_id=user_id,
                selected_paper_ids=selected_paper_ids,
                selected_paper_groups=selected_paper_groups,
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
    excluded_ids: list[str] | None = None,
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
    fetch_limit = min(limit * 3, 60)
    try:
        found = search(
            SearchRequest(search_query=query, search_mode=mode, limit=fetch_limit)
        )
    except SearchUnavailableError as error:
        return tool_result(str(error), success=False)

    papers: list[dict[str, object]] = []
    excluded = set(excluded_ids or [])
    for paper in found.papers:
        paper_id = paper.get("ID") or paper.get("id")
        if not paper_id:
            continue
        paper_id = str(paper_id)
        if paper_id in excluded:
            continue
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
        if len(papers) >= limit:
            break
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


def _selected_subset(
    payload: dict[str, Any], selected_paper_ids: list[str], *, maximum: int
) -> tuple[list[str], str | None]:
    requested = payload.get("paper_ids")
    ids = (
        list(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))
        if isinstance(requested, list)
        else list(selected_paper_ids)
    )
    if not ids:
        return [], "Select at least one seed paper first."
    selected = set(selected_paper_ids)
    if any(paper_id not in selected for paper_id in ids):
        return [], "Seed papers must belong to the current selection."
    if len(ids) > maximum:
        return [], f"At most {maximum} paper IDs may be supplied."
    return ids, None


def _candidate_record(paper: dict[str, object], *, source: str) -> tuple[str, dict[str, object]] | None:
    raw_id = paper.get("ID") or paper.get("id") or paper.get("paper_id")
    if not raw_id:
        return None
    paper_id = str(raw_id)
    return paper_id, {
        "paper_id": paper_id,
        "title": paper.get("Title") or paper.get("title") or "",
        "abstract": str(paper.get("Abstract") or paper.get("abstract") or "")[:2500],
        "authors": list(paper.get("Authors") or paper.get("authors") or [])[:12],
        "keywords": list(paper.get("Keywords") or paper.get("keywords") or [])[:20],
        "year": paper.get("Year") or paper.get("year"),
        "venue": paper.get("Source") or paper.get("source") or paper.get("venue") or "",
        "citation_count": paper.get("CitationCounts") or paper.get("citation_count"),
        "doi": paper.get("doi"),
        "discovery_source": source,
    }


def _is_search_unavailable(error: Exception) -> bool:
    from service.search import SearchUnavailableError

    return isinstance(error, SearchUnavailableError)


def _run_similar_search(seed_ids: list[str], limit: int):
    from model.paper import SimilarPapersRequest
    from service.search import find_similar_by_papers

    return find_similar_by_papers(
        SimilarPapersRequest(seed_ids=seed_ids, limit=limit)
    )


def _resolve_selected_for_tool(*, user_id: str | None, paper_ids: list[str]):
    from service.paper_registry import resolve_selected_papers

    return resolve_selected_papers(user_id=user_id, paper_ids=paper_ids)


def _fetch_citations(doi: str, limit: int, direction: str | None):
    from service.citations import get_paper_citations

    return get_paper_citations(doi, limit, direction=direction)


def _find_similar_papers(
    payload: dict[str, Any],
    *,
    selected_paper_ids: list[str],
    candidates: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    from service.selected_paper_tool import select_representative_paper_ids

    # The trusted clicked+pinned selection is injected by the server. Keeping
    # seed IDs out of the model-facing schema prevents stale/history IDs from
    # causing a false tool failure or widening the authorized scope.
    raw = list(dict.fromkeys(selected_paper_ids))
    if not raw:
        return tool_result("Select or pin at least one seed paper first.", success=False)
    seeds, policy = select_representative_paper_ids(
        raw, limit=20, prefer_map_coverage=True
    )
    try:
        limit = min(max(int(payload.get("limit") or 20), 1), 20)
    except (TypeError, ValueError):
        limit = 20
    fetch_limit = min(limit * 3, 60)
    try:
        found = _run_similar_search(seeds, fetch_limit)
    except Exception as error:
        if _is_search_unavailable(error):
            return tool_result(str(error), success=False)
        raise
    excluded = set(selected_paper_ids)
    papers: list[dict[str, object]] = []
    for paper in found.papers:
        record = _candidate_record(paper, source="similar")
        if record is None or record[0] in excluded:
            continue
        paper_id, item = record
        papers.append(item)
        if candidates is not None:
            candidates[paper_id] = item
        if len(papers) >= limit:
            break
    return tool_result(json.dumps({
        "requested_seed_count": len(raw),
        "used_seed_count": len(seeds),
        "seed_sampling_policy": policy,
        "papers": papers,
        "instruction": "Combine these candidates with text search when it improves coverage, then curate with present_papers.",
    }, ensure_ascii=False), success=True)


def _get_selected_paper_citations(
    payload: dict[str, Any],
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
    candidates: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    from service.citations import (
        PaperCitationsNotFoundError, PaperCitationsProviderError,
        PaperCitationsUnavailableError,
    )
    from service.paper_registry import LibraryPaperResolutionError

    ids, error = _selected_subset(payload, selected_paper_ids, maximum=5)
    if error:
        return tool_result(error, success=False)
    direction = payload.get("direction")
    if direction not in {"references", "cited_by"}:
        direction = None
    try:
        per_paper = min(max(int(payload.get("limit_per_paper") or 10), 1), 20)
    except (TypeError, ValueError):
        per_paper = 10
    try:
        resolved = _resolve_selected_for_tool(user_id=user_id, paper_ids=ids)
    except LibraryPaperResolutionError as error_value:
        return tool_result(str(error_value), success=False)
    excluded = set(selected_paper_ids)
    sources: list[dict[str, object]] = []
    for source in resolved:
        doi = source.metadata.get("doi")
        if not isinstance(doi, str) or not doi.strip():
            sources.append({"paper_id": source.paper_id, "skipped": "doi_unavailable"})
            continue
        try:
            result = _fetch_citations(doi, per_paper, direction)
        except PaperCitationsNotFoundError:
            sources.append({"paper_id": source.paper_id, "skipped": "not_found"})
            continue
        except (PaperCitationsUnavailableError, PaperCitationsProviderError) as error_value:
            return tool_result(str(error_value), success=False)
        groups = [direction] if direction else ["references", "cited_by"]
        evidence: dict[str, object] = {"paper_id": source.paper_id, "doi": doi}
        for group in groups:
            raw_group = result.get(group) if isinstance(result, dict) else None
            raw_papers = raw_group.get("papers") if isinstance(raw_group, dict) else []
            records: list[dict[str, object]] = []
            for paper in raw_papers if isinstance(raw_papers, list) else []:
                if not isinstance(paper, dict):
                    continue
                records.append(paper)
                paper_id = paper.get("paper_id")
                if paper.get("in_corpus") and isinstance(paper_id, str) and paper_id not in excluded:
                    record = _candidate_record(paper, source=group)
                    if record and candidates is not None:
                        candidates[record[0]] = record[1]
            evidence[group] = records
        sources.append(evidence)
    return tool_result(json.dumps({
        "sources": sources,
        "instruction": "Corpus matches may be presented; external works are citation evidence only and must not be passed to present_papers.",
    }, ensure_ascii=False), success=True)


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
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None,
) -> dict[str, object]:
    from service.selected_paper_tool import (
        SelectedPaperToolError,
        inspect_selected_papers,
    )

    try:
        overview = inspect_selected_papers(
            user_id=user_id,
            selected_paper_ids=selected_paper_ids,
            selected_paper_groups=selected_paper_groups,
        )
    except SelectedPaperToolError as error:
        return tool_result(str(error), success=False)
    return tool_result(json.dumps(overview, ensure_ascii=False), success=True)


def _get_selected_paper_evidence(
    payload: dict[str, Any],
    *,
    user_id: str | None,
    selected_paper_ids: list[str],
    selected_paper_groups: dict[str, list[str]] | None,
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
            selected_paper_groups=selected_paper_groups,
            request=EvidenceRequest(
                detail_level=str(payload.get("detail_level") or "auto"),
                limit=payload.get("limit"),
                paper_ids=paper_ids,
                group=str(payload.get("group") or "all"),
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
    if tool == "find_similar_papers":
        return {"paperCount": len(selected_paper_ids)}
    if tool == "get_selected_paper_citations":
        requested = payload.get("paper_ids")
        return {"paperCount": len(requested) if isinstance(requested, list) else 0}
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
    if tool in {"find_similar_papers", "get_selected_paper_citations"}:
        papers = payload.get("papers")
        if isinstance(papers, list):
            return {"resultCount": len(papers)}
        sources = payload.get("sources")
        return {"sourceCount": len(sources)} if isinstance(sources, list) else {}
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
    if tool == "find_similar_papers":
        return "find.similar_papers"
    if tool == "get_selected_paper_citations":
        return "get.selected_paper_citations"
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
