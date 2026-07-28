"""One-call structured extraction for the low-effort search pipeline."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from .models import ChatRoute, SearchIntent


_PROMPT = """Extract a paper-search intent from the user request. Return JSON only.
Do not rewrite the user's query: retrieval will use the original text verbatim.
Use retrieval_target=metadata_browse only when the request has no research topic,
only title/paper ID/author/venue/year/citation filters. Otherwise use topic.
Extract criteria only as explicit content requirements. Do not invent filters.

Schema:
{
  "retrieval_target": "topic" | "metadata_browse",
  "topic": string | null,
  "title": string | null,
  "paper_ids": [string],
  "authors": [string],
  "venues": [string],
  "min_year": integer | null,
  "max_year": integer | null,
  "min_citations": integer | null,
  "criteria": [string]
}

USER REQUEST:
"""


def _model_validate(data: dict[str, Any]) -> SearchIntent:
    if hasattr(SearchIntent, "model_validate"):
        return SearchIntent.model_validate(data)
    return SearchIntent.parse_obj(data)


def _fallback(query: str) -> SearchIntent:
    return SearchIntent(retrieval_target="topic", topic=query)


def parse_search_intent(query: str) -> SearchIntent:
    """Extract filters without changing the raw query used for retrieval."""
    try:
        # Keep model configuration in one place until search v2 owns its own LLM factory.
        from agents.search_v1_legacy.runner import get_azure_llm

        raw = get_azure_llm().invoke([HumanMessage(content=f"{_PROMPT}\nUSER REQUEST: {query}")]).content
        clean = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
        return _model_validate(json.loads(clean))
    except Exception:
        return _fallback(query)


_CHAT_ROUTE_PROMPT = """Classify the user message for an experimental academic chat route.
Choose search only when the user explicitly asks to find/list/recommend academic papers or literature.
For questions to answer, general chat, writing, selected-paper discussion, or ambiguity, choose other.
When route is search, also extract search_intent using the exact SearchIntent schema below.
Do not rewrite the user message. Return JSON only.
{"route":"search"|"other", "search_intent": SearchIntent | null}
SearchIntent schema: {"retrieval_target":"topic"|"metadata_browse","topic":string|null,"title":string|null,"paper_ids":[string],"authors":[string],"venues":[string],"min_year":integer|null,"max_year":integer|null,"min_citations":integer|null,"criteria":[string]}
USER MESSAGE:
"""


def parse_chat_route(query: str) -> ChatRoute:
    """One LLM call for the v2 chat experiment; failure safely falls back to legacy."""
    try:
        from agents.search_v1_legacy.runner import get_azure_llm

        raw = get_azure_llm().invoke([HumanMessage(content=f"{_CHAT_ROUTE_PROMPT}\nUSER MESSAGE: {query}")]).content
        clean = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
        data = json.loads(clean)
        if hasattr(ChatRoute, "model_validate"):
            return ChatRoute.model_validate(data)
        return ChatRoute.parse_obj(data)
    except Exception:
        return ChatRoute(route="other")
