"""
Pre-turn query understanding gate (L4.x).

Aligned with Codex codex-rs/core/src/hook_runtime.rs (inspect_pending_input).

Gate order:
  1. needs_clarification  → SIGNAL: ask user to clarify
  2. is_vague             → SIGNAL: expansion card (user picks specific direction)
  3. temporal inference   → SIGNAL: confirmation card (year range inferred from word)
  4. all pass             → PROCEED: build enriched INTENT_HINT for AgentExecutor
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .classifier import IntentResult


class GatewayAction(str, Enum):
    PROCEED = "proceed"
    SIGNAL  = "signal"


@dataclass
class GatewayResult:
    action:      GatewayAction
    agent_input: Optional[str] = None
    signal:      Optional[str] = None


class QueryGateway:
    """
    Pre-turn gate: runs before the AgentExecutor, may halt the turn with a
    frontend signal.  Each gate method returns GatewayResult or None.
    """

    def process(
        self,
        intent:      Optional["IntentResult"],
        clean_input: str,
        session:     dict,
    ) -> GatewayResult:
        if intent is None:
            return GatewayResult(action=GatewayAction.PROCEED, agent_input=clean_input)

        for gate in (
            self._gate_clarification,
            self._gate_vague_query,
            self._gate_confirmation,
        ):
            result = gate(intent, clean_input, session)
            if result is not None:
                return result

        return GatewayResult(
            action=GatewayAction.PROCEED,
            agent_input=self._build_agent_input(intent, clean_input),
        )

    # ── Gate 1 ───────────────────────────────────────────────────────────────

    def _gate_clarification(self, intent, clean_input, session):
        if intent.needs_clarification and intent.clarification_question:
            return GatewayResult(action=GatewayAction.SIGNAL,
                                 signal=intent.clarification_question)
        return None

    # ── Gate 2 ───────────────────────────────────────────────────────────────

    def _gate_vague_query(self, intent, clean_input, session):
        from .classifier import Intent as I

        already = session.pop("_expansion_offered", False)
        if (
            not already
            and intent.is_vague
            and intent.expansions
            and intent.intent in (I.SEARCH_PAPER, I.RAG_QA)
        ):
            session["_expansion_offered"] = True
            signal = (
                "[SIGNAL:QUERY_EXPANSION:"
                + json.dumps({"original": clean_input,
                               "expansions": intent.expansions},
                              ensure_ascii=False)
                + "]"
            )
            return GatewayResult(action=GatewayAction.SIGNAL, signal=signal)
        return None

    # ── Gate 3 (heuristic — no LLM) ──────────────────────────────────────────
    # Fires when a temporal word ("recent", "latest", …) was inferred as a
    # year range by the classifier.  Items are built from the slots directly.

    _TEMPORAL_WORDS: frozenset = frozenset({
        "recent", "recently", "latest", "newest", "new",
        "current", "currently", "contemporary", "modern",
        "last year", "this year", "past few years",
        "past year", "in recent years",
    })

    def _gate_confirmation(self, intent, clean_input, session):
        from .classifier import Intent as I

        if session.pop("_confirmation_offered", False):
            return None
        if intent.intent not in (I.SEARCH_PAPER, I.RAG_QA):
            return None

        s = intent.slots
        if s.year_min is None and s.year_max is None:
            return None

        query_lower = clean_input.lower()
        matched = next((w for w in self._TEMPORAL_WORDS if w in query_lower), None)
        if matched is None:
            return None  # year was explicit ("in 2023") — no confirmation needed

        year_display = (
            f"{s.year_min}–{s.year_max}"
            if s.year_min and s.year_max
            else (f"from {s.year_min}" if s.year_min else f"until {s.year_max}")
        )
        topic_hint = s.topic or s.key_query or "this topic"
        items = [{
            "field":      "year_range",
            "label":      "Year range",
            "display":    f"'{matched}' → {year_display}",
            "editable":   (f"papers from {s.year_min or 2020} to "
                           f"{s.year_max or 2026} on {topic_hint}"),
            "confidence": 0.75,
        }]

        session["_confirmation_offered"] = True
        session["last_intent"] = intent
        signal = (
            "[SIGNAL:INTENT_CONFIRM:"
            + json.dumps({"original": clean_input, "items": items,
                           "slots": s.model_dump(exclude_none=True)},
                          ensure_ascii=False)
            + "]"
        )
        return GatewayResult(action=GatewayAction.SIGNAL, signal=signal)

    # ── INTENT_HINT builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_agent_input(intent, clean_input: str) -> str:
        s     = intent.slots
        parts = [f"intent={intent.intent.value}", f"tool={intent.tool_hint}"]
        if s.year_min is not None: parts.append(f"year_min={s.year_min}")
        if s.year_max is not None: parts.append(f"year_max={s.year_max}")
        if s.authors:              parts.append(f"authors={s.authors}")
        if s.venues:               parts.append(f"venues={s.venues}")
        if s.topic:                parts.append(f"topic={s.topic!r}")
        if s.key_query:            parts.append(f"key_query={s.key_query!r}")
        if s.question:             parts.append(f"question={s.question!r}")
        if s.paper_ids:            parts.append(f"paper_ids={s.paper_ids}")
        return clean_input + "\n[INTENT_HINT: " + ", ".join(parts) + "]"


__all__ = ["GatewayAction", "GatewayResult", "QueryGateway"]
