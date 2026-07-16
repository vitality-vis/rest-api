"""
Shared agent helper utilities.

Lives at the top level of service.agents so chat orchestration code can import
it without creating cycles with the application service entrypoint.

Import chain (no cycles):
  application/agent_service  →  agents/chat/orchestrator  →  agent_utils
  application/agent_service  →  agent_utils
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def _fallback_direct_answer(llm, user_input: str, mem=None):
    """
    Stream a direct LLM reply when the agent/tool pipeline fails.
    Optionally records the answer in mem (MemoryManager).
    """
    try:
        from langchain.schema import HumanMessage

        messages = [HumanMessage(content=user_input)]
        full = []

        if hasattr(llm, "astream") and callable(getattr(llm, "astream")):
            async for chunk in llm.astream(messages):
                if getattr(chunk, "content", None):
                    full.append(chunk.content)
                    yield chunk.content
        else:
            resp = llm.invoke(messages)
            text = getattr(resp, "content", str(resp))
            full.append(text)
            yield text

        if mem and full:
            mem.add_turn("assistant", "".join(full))

    except Exception as exc:
        logger.warning("[fallback_direct_answer] LLM fallback failed: %s", exc)
        yield "I'm sorry, I couldn't complete your request. Please try again or rephrase your question."


async def async_update_memory(session: dict, final_text: str) -> None:
    """
    Update sliding-window memory after a turn completes.
    Called via asyncio.create_task() so it does not block the response stream.
    """
    try:
        mem = session["mem"]
        mem.add_turn("assistant", final_text)
        if session.get("_turn_docs"):
            mem.set_docs(session["_turn_docs"])
        session["_turn_docs"] = []
    except Exception as exc:
        logger.warning("[async_update_memory] FAILED: %s", exc)


def _resolve_selected_papers_from_cache(
    chat_id: str, requested: list, question: str
) -> tuple[str, bool]:
    """
    Look up requested papers (list of (title, id) tuples) in the session doc cache.
    Returns (resolved_agent_input, from_cache: bool).

    If found in cache: formats docs with abstracts and returns a rich prompt.
    If NOT found:      returns a call-hint prompt instructing the agent to
                       metadata_search the IDs first.
    """
    from service.application.retrieval_service import get_session_docs, format_docs

    requested_ids    = {str(pid).strip() for (_, pid) in requested if pid}
    requested_titles = {str(t).strip().lower() for (t, _) in requested if t}

    docs = get_session_docs(chat_id)
    matched = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        if isinstance(d, dict):
            md = d.get("metadata", d)
        did   = str(md.get("id") or md.get("ID") or "").strip()
        title = str(md.get("title") or "").strip().lower()
        if did and did in requested_ids:
            matched.append(d)
        elif title and title in requested_titles:
            matched.append(d)

    if matched:
        formatted = format_docs(matched, include_abstract=True, include_score=False)
        resolved = (
            "The user has selected the following paper(s). Their full details "
            "(title, abstract, authors, etc.) are provided below. Use these details "
            "to understand what the user is asking and decide which tool(s) to call "
            "— or answer directly if no retrieval is needed.\n\n"
            "Selected paper(s):\n"
            f"{formatted}\n\n"
            f"User request: {question}"
        )
        logger.info(
            "[agent_utils] Selected papers resolved from cache (%d docs).", len(matched)
        )
        return resolved, True

    ids = [i for (_, i) in requested if i]
    resolved = (
        "The user is asking about specific paper(s) that are NOT yet in the session cache. "
        "First call metadata_search with the paper IDs to fetch their details, "
        "then use those details (title, abstract) to decide what to do next "
        "based on the user's request.\n\n"
        f'Call hint: metadata_search(filters={{"ids": <id_list>}})\n\n'
        f"Requested paper IDs: {json.dumps(ids)}\n\n"
        f"User request: {question}"
    )
    logger.info("[agent_utils] Selected papers NOT in cache; agent must fetch then plan.")
    return resolved, False
