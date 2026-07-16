"""
Intent classifier — single responsibility: routing + slot extraction.

ONE job: decide whether this query needs retrieval (and what kind) or can be
answered directly.  Everything else (query expansion, confirmation items,
vagueness expansion) is handled by downstream modules so this classifier
stays fast, simple, and reliable.

What it outputs
---------------
  intent           SEARCH_PAPER | RAG_QA | SMALL_TALK | LOAD_MORE | CLARIFY
  confidence       0.0 – 1.0
  tool_hint        which tool to call
  slots            typed structured extraction (year, author, venue, …)
  is_vague         True when the query is too broad for good results
  expansions       5 specific alternatives when is_vague=True (empty otherwise)
  needs_clarification / clarification_question

What it does NOT do (moved to other modules)
---------------------------------------------
  × expanded_queries / expansion_confidence  → query_gateway._build_agent_input
  × confirmation_items                       → query_gateway._gate_confirmation (heuristic)
  × pipeline_hints                           → session["last_intent"]

Pipeline order (application/agent_service.py)
---------------------------------------------
  L0   FastRouter keyword pool           (zero LLM)
  L2   query_rewriter                    (LLM)
  L3   THIS classifier                   (LLM — one focused call)
  L4   SMALL_TALK direct reply
  L4.x QueryGateway (clarify / vague / confirm)
  L5   agent input hint from slots
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    SEARCH_PAPER = "SEARCH_PAPER"
    LOAD_MORE    = "LOAD_MORE"
    RAG_QA       = "RAG_QA"
    SMALL_TALK   = "SMALL_TALK"
    CLARIFY      = "CLARIFY"


# ─── Slot model ───────────────────────────────────────────────────────────────

class ExtractedSlots(BaseModel):
    """Typed slot extraction.  All fields are optional; only populate what the user stated."""
    year_min:  Optional[int]  = None
    year_max:  Optional[int]  = None
    authors:   List[str]      = Field(default_factory=list)
    venues:    List[str]      = Field(default_factory=list)
    topic:     Optional[str]  = None
    key_query: Optional[str]  = None   # core term for RAG_QA ("What is X?" → "X")
    question:  Optional[str]  = None   # full question text for RAG_QA
    paper_ids: List[str]      = Field(default_factory=list)


# ─── Result ───────────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    # Core routing
    intent:     Intent
    confidence: float = Field(ge=0.0, le=1.0)
    tool_hint:  str

    # Slot extraction
    slots: ExtractedSlots = Field(default_factory=ExtractedSlots)

    # Vague-query detection (query too broad → show expansion card)
    is_vague:   bool       = False
    expansions: List[str]  = Field(default_factory=list)

    # Clarification (intent genuinely ambiguous)
    needs_clarification:   bool           = False
    clarification_question: Optional[str] = None


# ─── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = """\
You are an intent classifier for an academic research assistant.

TOOLS:
- metadata_search  : filter by author / year / venue / title / paper IDs
- semantic_search  : topic or concept search (no metadata filters)
- mixed_search     : topic + at least one metadata filter
- load_more_papers : pagination ("more", "next")
- rag_semantic_qa  : retrieve by topic then answer a question
- direct_reply     : small talk, greetings, off-topic

SESSION STATE:
- has_active_paper: {has_active_paper}
- active_paper_title: {active_paper_title}
- has_prior_search: {has_prior_search}
  (if has_prior_search=false → LOAD_MORE is impossible, use SEARCH_PAPER)

═══ STEP 1 — CLASSIFY INTENT ════════════════════════════════════════════════

SMALL_TALK  : greetings, thanks, chitchat, off-topic → direct_reply
LOAD_MORE   : "more", "next page", "show more" (only if has_prior_search=true)
RAG_QA      : question that needs retrieved papers to answer (academic concepts)
SEARCH_PAPER: user wants to find / list papers
CLARIFY     : intent is genuinely ambiguous

"What is X?" → SMALL_TALK if everyday fact; RAG_QA if academic/research concept.

═══ STEP 2 — EXTRACT SLOTS (for SEARCH_PAPER and RAG_QA only) ══════════════

Fill only what the user explicitly stated or clearly implied:

TEMPORAL:
  "recent" / "latest" / "new"      → year_min=2023, year_max=2026
  "last year"                       → year_min=2025, year_max=2025
  "in 2023"                         → year_min=2023, year_max=2023
  "after 2020"                      → year_min=2021, year_max=null
  "2020s"                           → year_min=2020, year_max=2029

RAG_QA:
  key_query = the core concept ("What is RAG?" → key_query="RAG")
  question  = the full question as stated

═══ STEP 3 — VAGUE QUERY DETECTION (for SEARCH_PAPER / RAG_QA) ══════════════

is_vague=true ONLY when the query is a bare concept with no modifiers,
filters, or specificity ("neural networks", "machine learning", "NLP").
If is_vague=true: provide exactly 5 specific alternative queries in expansions[].
If is_vague=false: expansions=[].

══════════════════════════════════════════════════════════════════════════════

Respond ONLY with valid JSON (no markdown):
{{
  "intent": "SEARCH_PAPER|RAG_QA|SMALL_TALK|LOAD_MORE|CLARIFY",
  "confidence": 0.95,
  "tool_hint": "metadata_search|semantic_search|mixed_search|load_more_papers|rag_semantic_qa|direct_reply",
  "slots": {{
    "year_min": null, "year_max": null,
    "authors": [], "venues": [],
    "topic": null, "key_query": null, "question": null,
    "paper_ids": []
  }},
  "is_vague": false,
  "expansions": [],
  "needs_clarification": false,
  "clarification_question": null
}}

MESSAGE: {message}"""


# ─── Few-shot examples ────────────────────────────────────────────────────────

def _j(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"))

_S = {"year_min": None, "year_max": None, "authors": [], "venues": [],
      "topic": None, "key_query": None, "question": None, "paper_ids": []}

_FEW_SHOTS = [
    # 1. RAG_QA — clear abbreviation, not vague
    (
        "What is RAG?",
        _j({"intent": "RAG_QA", "confidence": 0.97, "tool_hint": "rag_semantic_qa",
            "slots": {**_S, "topic": "Retrieval-Augmented Generation",
                      "key_query": "RAG", "question": "What is RAG?"},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 2. SEARCH_PAPER — clean metadata (fully specified, not vague)
    (
        "papers by Hinton in NeurIPS",
        _j({"intent": "SEARCH_PAPER", "confidence": 0.97, "tool_hint": "metadata_search",
            "slots": {**_S, "authors": ["Hinton"], "venues": ["NeurIPS"]},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 3. SEARCH_PAPER — temporal inference (recent → year range)
    (
        "find recent papers on world model combining with MCTS",
        _j({"intent": "SEARCH_PAPER", "confidence": 0.93, "tool_hint": "mixed_search",
            "slots": {**_S, "year_min": 2023, "year_max": 2026,
                      "topic": "world model combining with MCTS"},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 4. SEARCH_PAPER with topic + venue (explicit year, not inferred)
    (
        "What do CHI 2020 papers say about usability?",
        _j({"intent": "SEARCH_PAPER", "confidence": 0.96, "tool_hint": "mixed_search",
            "slots": {**_S, "year_min": 2020, "year_max": 2020,
                      "venues": ["CHI"], "topic": "usability"},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 5. SEARCH_PAPER — vague query, provide 5 specific alternatives
    (
        "neural networks",
        _j({"intent": "SEARCH_PAPER", "confidence": 0.88, "tool_hint": "semantic_search",
            "slots": {**_S, "topic": "neural networks"},
            "is_vague": True,
            "expansions": [
                "convolutional neural networks for image classification",
                "recurrent neural networks for sequence modeling",
                "graph neural networks for node classification",
                "neural network pruning and compression techniques",
                "neural architecture search automated ML",
            ],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 6. LOAD_MORE — pagination
    (
        "show me more",
        _j({"intent": "LOAD_MORE", "confidence": 0.98, "tool_hint": "load_more_papers",
            "slots": _S, "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 7. SMALL_TALK
    (
        "thanks that helps!",
        _j({"intent": "SMALL_TALK", "confidence": 0.99, "tool_hint": "direct_reply",
            "slots": _S, "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 8. RAG_QA — medium confidence; still not vague and no clarification needed
    (
        "what is active learning in machine learning",
        _j({"intent": "RAG_QA", "confidence": 0.72, "tool_hint": "rag_semantic_qa",
            "slots": {**_S, "topic": "active learning",
                      "key_query": "active learning",
                      "question": "What is active learning in machine learning?"},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
    # 9. SEARCH_PAPER — specific paper by title → metadata_search
    (
        "What datasets were used in the paper Attention Is All You Need?",
        _j({"intent": "SEARCH_PAPER", "confidence": 0.95, "tool_hint": "metadata_search",
            "slots": {**_S, "topic": "Attention Is All You Need",
                      "question": "What datasets were used?"},
            "is_vague": False, "expansions": [],
            "needs_clarification": False, "clarification_question": None}),
    ),
]


# ─── classify_intent ──────────────────────────────────────────────────────────

def classify_intent(
    message: str,
    llm:     Any,
    *,
    session:  Optional[dict] = None,
    chat_id:  str = "default",
) -> IntentResult:
    """
    Route the message to the correct intent + extract slots.

    The L0 FastRouter already filtered out greetings, acknowledgments, and
    farewells before this is called, so this classifier only sees real queries.
    """
    if session is None:
        try:
            from service.memory.session_state import get_session
            session = get_session(chat_id) or {}
        except Exception:
            session = {}

    prompt = _PROMPT.format(
        has_active_paper=bool(session.get("active_paper_id")),
        active_paper_title=session.get("active_paper_title", "None"),
        has_prior_search=bool(session.get("search_cache")),
        message=message,
    )

    messages: list = []
    for user_ex, asst_ex in _FEW_SHOTS:
        messages.append(HumanMessage(content=user_ex))
        messages.append(AIMessage(content=asst_ex))
    messages.append(HumanMessage(content=prompt))

    try:
        raw   = llm.invoke(messages).content
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data  = json.loads(clean)

        if isinstance(data.get("intent"), str):
            data["intent"] = Intent(data["intent"])

        data["expansions"] = [str(e).strip() for e in (data.get("expansions") or []) if e][:5]

        raw_slots = data.get("slots") or {}
        data["slots"] = ExtractedSlots(
            year_min  = _safe_int(raw_slots.get("year_min")),
            year_max  = _safe_int(raw_slots.get("year_max")),
            authors   = [str(a) for a in (raw_slots.get("authors") or [])],
            venues    = [str(v) for v in (raw_slots.get("venues")  or [])],
            topic     = raw_slots.get("topic")     or None,
            key_query = raw_slots.get("key_query") or None,
            question  = raw_slots.get("question")  or None,
            paper_ids = [str(p) for p in (raw_slots.get("paper_ids") or [])],
        )

        result = IntentResult(**data)

    except Exception as e:
        logger.warning("[classifier] parse error: %s — defaulting to SEARCH_PAPER", e)
        result = IntentResult(
            intent=Intent.SEARCH_PAPER,
            confidence=0.5,
            tool_hint="semantic_search",
        )

    # Safety-net: only fires on near-zero confidence (temperature=1.0 variance).
    # At 0.30+ the LLM's own needs_clarification field is the authoritative signal.
    if result.confidence < 0.30 and result.intent not in (Intent.SMALL_TALK, Intent.LOAD_MORE):
        logger.warning(
            "[classifier] safety-net: confidence=%.2f — forcing clarification", result.confidence
        )
        result.needs_clarification = True
        result.clarification_question = (
            "Could you clarify — are you looking for papers on a topic, "
            "asking a research question, or something else?"
        )

    return result


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["Intent", "ExtractedSlots", "IntentResult", "classify_intent"]
