"""
Assessor agent — evaluates retrieval quality for self-RAG.

Role in the multi-agent pipeline
---------------------------------
Orchestrator  ──► [/root/assessor mailbox]
                       │
                  AssessorAgent.run(input_data)
                       │  LLM call: score relevance
                       ▼
                  self.send("/root", assessment_result)

The orchestrator reads the result from its own mailbox and decides:
  - score >= threshold  → proceed to answer generation
  - score < threshold   → ask RefinerAgent for a better query, then re-retrieve

Aligns with Codex's guardian/review.rs approval pattern:
  each tool output passes through an evaluation step before the
  orchestrator decides the next action.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage

from ..sub_agent_base import SubAgentBase
from ..status import AgentStatus

if TYPE_CHECKING:
    from ..registry import AgentRegistry

logger = logging.getLogger(__name__)

# ── Assessment result ─────────────────────────────────────────────────────────

@dataclass
class AssessmentResult:
    score:         float           # 0.0 – 1.0
    is_sufficient: bool            # score >= threshold
    refined_query: Optional[str]   # non-None when refinement is suggested
    original_query: str

    THRESHOLD: float = 0.65        # class-level constant


# ── Prompt ────────────────────────────────────────────────────────────────────

_ASSESS_PROMPT = """\
You are a retrieval quality assessor for an academic research assistant.

USER QUERY:
{query}

RETRIEVED DOCUMENTS (first 5):
{docs}

Task: Decide whether the retrieved documents adequately answer the user query.

Rules:
1. Score 0.0–1.0 based on topical relevance and coverage of the query.
2. If score < 0.65, provide a refined_query that would retrieve better documents.
   The refined query should be more specific or use different terminology.
3. If score >= 0.65, set refined_query to null.

Respond ONLY with valid JSON:
{{
  "score": <0.0-1.0>,
  "is_sufficient": <true|false>,
  "refined_query": "<improved query string or null>"
}}"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class AssessorAgent(SubAgentBase):
    """
    Receives (query, docs, reply_to) from the orchestrator's mailbox,
    calls the LLM to score relevance, and routes an assessment_result
    message back to the orchestrator.

    Does NOT yield user-visible text — it is a background worker.
    """

    def __init__(self, chat_id: str, registry: "AgentRegistry", llm: Any) -> None:
        super().__init__(chat_id, registry)
        self.llm = llm

    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """
        Execute one assessment turn.

        Expected input_data keys
        ------------------------
        query    : str  — the user's original search query
        docs     : str  — formatted retrieved documents passed from the chat application service
        reply_to : str  — agent path to route the result back to (typically "/root")
        """
        self.set_status(AgentStatus.RUNNING)

        query    = str(input_data.get("query", ""))
        docs     = str(input_data.get("docs", ""))
        reply_to = str(input_data.get("reply_to", ""))

        result = await self._assess(query, docs)

        if reply_to:
            self.send(
                recipient=reply_to,
                content={
                    "type":          "assessment_result",
                    "score":         result.score,
                    "is_sufficient": result.is_sufficient,
                    "refined_query": result.refined_query,
                    "original_query": result.original_query,
                },
                trigger_turn=True,
            )

        self.set_status(AgentStatus.COMPLETED)
        return  # worker — yields nothing to the user stream
        yield   # satisfy AsyncGenerator type

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _assess(self, query: str, docs: str) -> AssessmentResult:
        try:
            prompt = _ASSESS_PROMPT.format(query=query, docs=docs[:3000])
            raw    = self.llm.invoke([HumanMessage(content=prompt)]).content
            clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
            data   = json.loads(clean)

            score         = float(data.get("score", 0.5))
            is_sufficient = bool(data.get("is_sufficient", score >= AssessmentResult.THRESHOLD))
            refined_query = data.get("refined_query") or None

            logger.info(
                "[AssessorAgent] query=%r score=%.2f sufficient=%s refined=%r",
                query[:60], score, is_sufficient, refined_query,
            )
            return AssessmentResult(
                score=score,
                is_sufficient=is_sufficient,
                refined_query=refined_query,
                original_query=query,
            )

        except Exception as exc:
            logger.warning("[AssessorAgent] LLM assessment failed: %s — treating as sufficient", exc)
            # Fail-safe: don't block the answer if assessor errors
            return AssessmentResult(
                score=0.5,
                is_sufficient=True,
                refined_query=None,
                original_query=query,
            )
