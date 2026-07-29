"""Experimental chat-v2 runner: paper finding uses v2, everything else falls back."""
from __future__ import annotations

import json
from typing import AsyncIterator

from agents.agent_v1_legacy import AgentRequest, run as run_search_v1_legacy
from agents.agent_v1_legacy.rag_core import PAPERS_PAYLOAD_END, PAPERS_PAYLOAD_START

from .models import SearchV2Request
from .parser import parse_chat_route
from .runner import FUSED_CANDIDATE_LIMIT, SearchCriteriaRequiredError, run_search
from .logging import SearchV2Trace


CHAT_RESULT_LIMIT = FUSED_CANDIDATE_LIMIT


def _paper_id(paper: dict) -> str | None:
    value = paper.get("ID") or paper.get("id")
    return str(value) if value else None


def _format_search_intro() -> str:
    """Keep chat prose separate from the embedded, interactive paper list."""
    return "Below are the papers I found."


async def run(request: AgentRequest) -> AsyncIterator[str]:
    trace = SearchV2Trace.create(
        trace_id=request.trace_id,
        chat_id=request.chat_id,
        user_message_id=request.user_message_id,
        assistant_message_id=request.assistant_message_id,
    )
    effort = request.effort if request.effort in {"low", "medium", "high"} else "low"
    route = parse_chat_route(
        request.text,
        history=request.history,
        effort=effort,
        trace=trace,
    )
    if route.route != "search" or route.search_intent is None:
        async for chunk in run_search_v1_legacy(request):
            yield chunk
        return

    # Chat sends the full bounded ranked list to the paper panel. The standalone
    # /search/v2 endpoint keeps its smaller default result_limit for JSON clients.
    # TODO: Replace this history-based refinement with an explicit active-search
    # state and deterministic intent-patch merge.
    topic = route.search_intent.topic.strip() if route.search_intent.topic else ""
    use_resolved_topic = bool(topic and topic.casefold() not in request.text.casefold())
    retrieval_query = topic if use_resolved_topic else request.text
    try:
        result = run_search(
            SearchV2Request(query=retrieval_query, effort=effort, result_limit=CHAT_RESULT_LIMIT),
            intent=route.search_intent,
            trace=trace,
        )
    except SearchCriteriaRequiredError as error:
        yield str(error)
        return
    ids: list[str] = []
    for item in result.papers:
        paper_id = _paper_id(item.paper)
        if paper_id:
            ids.append(paper_id)
    if ids:
        yield _format_search_intro()
        yield "\n\n"
        yield f"{PAPERS_PAYLOAD_START}{json.dumps({'ids': ids, 'ranked_ids': ids, 'count_known': False, 'search_version': 'v2', 'policy': result.policy, 'effort': effort}, separators=(',', ':'))}{PAPERS_PAYLOAD_END}"
    else:
        yield "I couldn't find papers matching that request. Try broadening the topic or filters."
