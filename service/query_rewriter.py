"""
Query rewriter: turns context-dependent user messages into self-contained queries
using a single LLM call with structured JSON output.
"""
import re
import json
import logging
from typing import Optional, Any

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    rewritten_query: str
    was_rewritten: bool
    rewrite_type: str  


_PROMPT = """\
You are a query rewriter for an academic research assistant.
Given the conversation history and the current message, produce a fully
self-contained query that can be understood with zero context.

RULES:
1. Resolve pronouns: "it", "that paper", "their method" → actual title/name from history
2. Expand ellipsis: "what about transformers?" after BERT discussion → full topic
3. Carry topic: "find more recent ones" → "find recent papers on [topic from history]"
4. If message is already self-contained OR is small talk → return it unchanged, was_rewritten=false
5. For load-more requests ("show more", "next", "more papers") → return unchanged, was_rewritten=false
6. NEVER invent titles, authors, or IDs not present in history

SESSION STATE:
- active_paper: {active_paper_title}
- last_search_topic: {last_search_topic}

CONVERSATION HISTORY (last 3 turns):
{history}

CURRENT MESSAGE: {message}

Respond ONLY with valid JSON:
{{
  "rewritten_query": "<string>",
  "was_rewritten": <bool>,
  "rewrite_type": "<pronoun|ellipsis|topic_carry|none>"
}}"""


def rewrite_query(
    message: str,
    llm: Any,
    *,
    session: Optional[dict] = None,
    chat_id: str = "default",
) -> RewriteResult:
    """
    Rewrite the user message into a self-contained query using conversation context.
    Uses the session's memory for history; pass session from agent_runner or get via chat_id.
    """
    if session is None:
        try:
            from service.session_state import get_session
            session = get_session(chat_id) or {}
        except ImportError:
            session = {}

    mem = session.get("mem") if session else None
    history_text = mem.get_history_text() if mem and hasattr(mem, "get_history_text") else ""

    if not history_text.strip():
        return RewriteResult(
            rewritten_query=message,
            was_rewritten=False,
            rewrite_type="none",
        )

    prompt = _PROMPT.format(
        active_paper_title=session.get("active_paper_title", "None"),
        last_search_topic=session.get("last_search_topic", "None"),
        history=history_text,
        message=message,
    )

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        return RewriteResult(
            rewritten_query=data.get("rewritten_query", message),
            was_rewritten=bool(data.get("was_rewritten", False)),
            rewrite_type=data.get("rewrite_type", "none"),
        )
    except Exception as e:
        logger.warning("query_rewriter failed: %s — passing original", e)
        return RewriteResult(
            rewritten_query=message,
            was_rewritten=False,
            rewrite_type="none",
        )


__all__ = ["RewriteResult", "rewrite_query"]
