"""
Intent classifier for the academic research assistant.
Outputs intent, tool_hint, confidence, and optional slots for downstream routing.
"""
import re
import json
import logging
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    SEARCH_PAPER = "SEARCH_PAPER"
    LOAD_MORE = "LOAD_MORE"
    RAG_QA = "RAG_QA"
    SMALL_TALK = "SMALL_TALK"
    CLARIFY = "CLARIFY"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    tool_hint: str  # metadata_search | semantic_search | mixed_search |
    # load_more_papers | direct_reply | scoped_search
    slots: dict = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


_PROMPT = """\
You are an intent classifier for an academic research assistant with these tools:
- metadata_search   : (list papers) author / year / source / title / paper ID filters
- semantic_search   : (list papers) topic or concept queries
- mixed_search      : (list papers) topic + at least one filter; use for "papers on X from venue Y" or "What do CHI papers say about X?" — agent answers from results
- load_more_papers  : pagination — "more", "next", "show more"
- rag_semantic_qa   : (answer question) retrieve by topic, then answer
- metadata_search   : (list papers) filter-based; also when user asks about a specific paper (by title/author/venue)
- direct_reply      : small talk, greetings, thanks, off-topic

SESSION STATE (use to resolve ambiguity):
- has_active_paper: {has_active_paper}
- active_paper_title: {active_paper_title}
- has_prior_search: {has_prior_search}
- NOTE: if has_prior_search=false, LOAD_MORE is impossible → use SEARCH_PAPER instead

CRITICAL — Commonsense vs academic "What is X?":
- SMALL_TALK (direct_reply): General knowledge, everyday facts, geography, non-academic. Examples: "Where is London?", "What is a dog?" → answer from general knowledge; no paper retrieval.
- RAG_QA (rag_semantic_qa): Research/academic/topic concepts. Examples: "What is RAG?", "What is data preprocessing?" → retrieve by topic and answer.
- SEARCH_PAPER (metadata_search): Question about a specific paper by title/author/venue → metadata_search, then agent answers. (mixed_search): Question that combines topic + filters (e.g. "What do CHI papers from 2020 say about usability?") → mixed_search, then agent answers.

Classify this message. Respond ONLY with valid JSON:
{{
  "intent": "<SEARCH_PAPER|LOAD_MORE|RAG_QA|SMALL_TALK|CLARIFY>",
  "confidence": <0.0-1.0>,
  "tool_hint": "<metadata_search|semantic_search|mixed_search|load_more_papers|rag_semantic_qa|direct_reply>",
  "slots": {{
    "authors": [], "year_min": null, "year_max": null,
    "venues": [], "topic": null, "paper_id": null
  }},
  "needs_clarification": false,
  "clarification_question": null
}}

MESSAGE: {message}"""

_FEW_SHOTS = [
    (
        "papers by Hinton in NeurIPS",
        '{"intent":"SEARCH_PAPER","confidence":0.97,"tool_hint":"metadata_search",'
        '"slots":{"authors":["Hinton"],"venues":["NeurIPS"],"topic":null},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "what is retrieval augmented generation",
        '{"intent":"RAG_QA","confidence":0.93,"tool_hint":"rag_semantic_qa",'
        '"slots":{"topic":"retrieval augmented generation"},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "RAG papers from 2023",
        '{"intent":"SEARCH_PAPER","confidence":0.95,"tool_hint":"mixed_search",'
        '"slots":{"topic":"RAG","year_min":2023},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "show me more",
        '{"intent":"LOAD_MORE","confidence":0.98,"tool_hint":"load_more_papers",'
        '"slots":{},"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "Where is London?",
        '{"intent":"SMALL_TALK","confidence":0.98,"tool_hint":"direct_reply",'
        '"slots":{},"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "What is a dog?",
        '{"intent":"SMALL_TALK","confidence":0.99,"tool_hint":"direct_reply",'
        '"slots":{},"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "What is RAG?",
        '{"intent":"RAG_QA","confidence":0.95,"tool_hint":"rag_semantic_qa",'
        '"slots":{"topic":"RAG"},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "What is data preprocessing?",
        '{"intent":"RAG_QA","confidence":0.94,"tool_hint":"rag_semantic_qa",'
        '"slots":{"topic":"data preprocessing"},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "what dataset did they use in this paper",
        '{"intent":"RAG_QA","confidence":0.94,"tool_hint":"rag_semantic_qa",'
        '"slots":{"topic":"dataset"},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "What do CHI papers from 2020 say about usability?",
        '{"intent":"SEARCH_PAPER","confidence":0.96,"tool_hint":"mixed_search",'
        '"slots":{"topic":"usability","year_min":2020,"venues":["CHI"]},'
        '"needs_clarification":false,"clarification_question":null}',
    ),
    (
        "thanks that helps!",
        '{"intent":"SMALL_TALK","confidence":0.99,"tool_hint":"direct_reply",'
        '"slots":{},"needs_clarification":false,"clarification_question":null}',
    ),
]


def classify_intent(
    message: str,
    llm: Any,
    *,
    session: Optional[dict] = None,
    chat_id: str = "default",
) -> IntentResult:
    """Classify user message into intent, tool hint, and optional slots."""
    if session is None:
        try:
            from service.session_state import get_session
            session = get_session(chat_id) or {}
        except Exception:
            session = {}

    prompt = _PROMPT.format(
        has_active_paper=bool(session.get("active_paper_id")),
        active_paper_title=session.get("active_paper_title", "None"),
        has_prior_search=bool(session.get("search_cache")),
        message=message,
    )

    messages = [HumanMessage(content=prompt)]
    for user_ex, asst_ex in _FEW_SHOTS:
        messages.append(HumanMessage(content=user_ex))
        messages.append(AIMessage(content=asst_ex))
    messages.append(HumanMessage(content=message))

    try:
        raw = llm.invoke(messages).content
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        data["intent"] = Intent(data["intent"]) if isinstance(data.get("intent"), str) else data["intent"]
        result = IntentResult(**data)
    except Exception as e:
        logger.warning("[classifier] parse error: %s — defaulting to SEARCH_PAPER", e)
        result = IntentResult(
            intent=Intent.SEARCH_PAPER,
            confidence=0.5,
            tool_hint="semantic_search",
        )

    if result.confidence < 0.60 and result.intent != Intent.SMALL_TALK:
        result.needs_clarification = True
        result.clarification_question = (
            "Could you clarify — are you looking for papers on a topic, "
            "asking about a specific paper, or something else?"
        )
    elif 0.60 <= result.confidence < 0.85:
        if result.intent == Intent.PAPER_QA and not session.get("active_paper_id"):
            result.needs_clarification = True
            result.clarification_question = "Which paper are you asking about?"

    return result


__all__ = ["Intent", "IntentResult", "classify_intent"]
