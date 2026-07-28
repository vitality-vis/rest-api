"""Experimental chat-v2 runner: paper finding uses v2, everything else falls back."""
from __future__ import annotations

import json
from typing import AsyncIterator

from agents.search_v1_legacy import AgentRequest, run as run_search_v1_legacy
from agents.search_v1_legacy.rag_core import PAPERS_PAYLOAD_END, PAPERS_PAYLOAD_START

from .models import SearchV2Paper, SearchV2Request
from .parser import parse_chat_route
from .runner import FUSED_CANDIDATE_LIMIT, SearchCriteriaRequiredError, run_search


CHAT_RESULT_LIMIT = FUSED_CANDIDATE_LIMIT
CHAT_PREVIEW_LIMIT = 5


def _paper_id(paper: dict) -> str | None:
    value = paper.get("ID") or paper.get("id")
    return str(value) if value else None


def _format_authors(authors: object) -> str:
    if isinstance(authors, list):
        return ", ".join(str(author) for author in authors if author) or "(Unknown)"
    if isinstance(authors, str) and authors.strip():
        return authors.strip()
    return "(Unknown)"


def _format_preview(papers: list[SearchV2Paper], *, total: int) -> str:
    lines = ["Here are the top results:"]
    for index, item in enumerate(papers[:CHAT_PREVIEW_LIMIT], start=1):
        paper = item.paper
        title = str(paper.get("Title") or paper.get("title") or "(No title)").strip()
        paper_id = _paper_id(paper)
        authors = _format_authors(paper.get("Authors") or paper.get("authors"))
        year = paper.get("Year") if paper.get("Year") is not None else paper.get("year")
        source = paper.get("Source") or paper.get("source") or "(N/A)"
        title_line = f"{index}. **{title}**"
        if paper_id:
            title_line += f" [[ID:{paper_id}]]"
        lines.append(title_line)
        lines.append(f"   Authors: {authors}")
        lines.append(f"   Year: {year if year is not None else '(N/A)'} · Source: {source}")

    remaining = total - min(total, CHAT_PREVIEW_LIMIT)
    if remaining > 0:
        lines.append(f"\nShowing {CHAT_PREVIEW_LIMIT} papers.")
    return "\n".join(lines)


async def run(request: AgentRequest) -> AsyncIterator[str]:
    route = parse_chat_route(request.text)
    if route.route != "search" or route.search_intent is None:
        async for chunk in run_search_v1_legacy(request):
            yield chunk
        return

    effort = request.effort if request.effort in {"low", "medium", "high"} else "low"
    # Chat sends the full bounded ranked list to the paper panel. The standalone
    # /search/v2 endpoint keeps its smaller default result_limit for JSON clients.
    try:
        result = run_search(
            SearchV2Request(query=request.text, effort=effort, result_limit=CHAT_RESULT_LIMIT),
            intent=route.search_intent,
        )
    except SearchCriteriaRequiredError as error:
        yield str(error)
        return
    ranked: list[SearchV2Paper] = []
    ids: list[str] = []
    for item in result.papers:
        paper_id = _paper_id(item.paper)
        if paper_id:
            ranked.append(item)
            ids.append(paper_id)
    if ids:
        yield _format_preview(ranked, total=len(ids))
        yield "\n\n"
        yield f"{PAPERS_PAYLOAD_START}{json.dumps({'ids': ids, 'ranked_ids': ids, 'count_known': False, 'search_version': 'v2', 'policy': result.policy, 'effort': effort}, separators=(',', ':'))}{PAPERS_PAYLOAD_END}"
    else:
        yield "I couldn't find papers matching that request. Try broadening the topic or filters."
