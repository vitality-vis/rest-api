"""
Shared search-tool selection — single source of truth for "given an intent,
which existing retrieval tool + args should run".

Extracted verbatim from MainOrchestratorAgent._select_tool so that BOTH the
orchestrator (deterministic fallback path) and CollectPapersAgent select tools
identically. This guarantees the new ResearchTask path preserves the exact
search behavior/quality of the existing pipeline.

No tool is modified here; this only chooses among the already-registered tools.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def build_filters(slots) -> Dict[str, Any]:
    """Build metadata filters from intent slots (mirrors the orchestrator)."""
    if slots is None:
        return {}
    filters: Dict[str, Any] = {}
    if getattr(slots, "authors",  None): filters["authors"]  = slots.authors
    if getattr(slots, "year_min", None): filters["year_min"] = slots.year_min
    if getattr(slots, "year_max", None): filters["year_max"] = slots.year_max
    if getattr(slots, "venues",   None): filters["venues"]   = slots.venues
    if getattr(slots, "title",    None): filters["title"]    = slots.title
    if getattr(slots, "ids",      None):
        filters["ids"] = [str(i) for i in slots.ids]
    return filters


def select_search_tool(
    intent: Any,
    query: str,
    clean_input: str = "",
) -> Tuple[str, dict]:
    """
    Deterministically map an IntentResult + query to (tool_name, tool_args).

    Identical logic to the orchestrator's original _select_tool, parameterized
    so it can run inside CollectPapersAgent as well.
    """
    # Lazy import to avoid any import cycle through the pipeline package.
    from service.application.pipeline import Intent

    if intent is None:
        return "semantic_search", {"query": query}

    hint  = getattr(intent, "tool_hint", None)
    slots = getattr(intent, "slots", None)

    if intent.intent == Intent.LOAD_MORE:
        return "load_more_papers", {}

    if intent.intent == Intent.RAG_QA or hint == "rag_semantic_qa":
        key_query = (getattr(slots, "key_query", None) if slots else None) or query
        question  = (getattr(slots, "question", None)  if slots else None) or clean_input or query
        return "rag_semantic_qa", {"query": key_query, "question": question}

    if hint == "metadata_search":
        return "metadata_search", {"filters": build_filters(slots)}

    if hint == "mixed_search":
        topic = (getattr(slots, "topic", None) if slots else None) or query
        return "mixed_search", {
            "query_text": topic,
            "filters": build_filters(slots),
        }

    topic = (getattr(slots, "topic", None) if slots else None) or query
    return "semantic_search", {"query": topic}
