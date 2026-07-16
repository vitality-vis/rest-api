"""
Fast-path keyword router (L0).

Detects simple unambiguous queries via a frozen word pool — zero LLM cost for
classification.  The LLM still generates the actual reply; only the routing
decision is bypassed.

Rules
-----
1. Exact match after normalization (lowercase + strip punctuation).
2. Only match queries that are ≤ 6 words.
3. Pagination ("show more", "load more") is intentionally NOT here —
   those need a session-state check that only L3 can do correctly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FastRouteResult:
    """
    Returned when a query matches the pool.
    Carries only the intent label — the LLM generates the actual reply text.
    """
    intent: str   # always "SMALL_TALK"


# ── Greetings ─────────────────────────────────────────────────────────────────

_GREET_1W: frozenset = frozenset({
    "hi", "hello", "hey", "howdy", "hiya", "heya", "yo", "sup",
    "greetings", "salutations", "morning", "afternoon", "evening",
    "bonjour", "hola", "ciao", "hallo", "ola",
})

_GREET_MW: frozenset = frozenset({
    "good morning", "good afternoon", "good evening",
    "good night", "good day",
    "how are you", "how r u", "how do you do",
    "hows it going", "how's it going", "how is it going",
    "whats up", "what's up", "what up", "wassup",
    "nice to meet you", "pleased to meet you",
    "long time no see",
})

# ── Acknowledgments / affirmatives ────────────────────────────────────────────

_ACK_1W: frozenset = frozenset({
    "thanks", "thx", "ty", "cheers",
    "ok", "okay", "k", "kk", "yep", "yeah", "yup", "yes",
    "sure", "indeed", "correct", "right", "exactly", "alright",
    "great", "perfect", "excellent", "awesome", "amazing",
    "wonderful", "fantastic", "brilliant", "superb", "outstanding",
    "nice", "good", "cool", "neat", "sweet",
    "got", "noted", "understood", "clear",
    "interesting", "helpful", "useful",
})

_ACK_MW: frozenset = frozenset({
    "thank you", "many thanks", "much appreciated",
    "thank u", "thanks a lot", "thanks so much",
    "got it", "i got it", "got that",
    "no problem", "no worries", "not a problem",
    "sounds good", "looks good", "thats good", "that's good",
    "good job", "well done", "great job",
    "of course", "sure thing",
    "thats great", "that's great", "thats helpful", "that's helpful",
    "thats perfect", "that's perfect", "thats useful", "that's useful",
    "thats interesting", "that's interesting",
    "very helpful", "very useful", "very good",
    "i see", "i understand", "makes sense",
    "i understand that", "that makes sense",
    "ok thanks", "okay thanks", "ok thank you", "okay thank you",
    "ok great", "okay great", "alright thanks",
})

# ── Farewells ─────────────────────────────────────────────────────────────────

_BYE_1W: frozenset = frozenset({
    "bye", "goodbye", "farewell", "later", "cya",
    "tata", "cheerio",
})

_BYE_MW: frozenset = frozenset({
    "good bye", "see you", "see ya", "see you later",
    "catch you later", "talk to you later", "ttyl",
    "take care", "have a nice day", "have a good day",
    "have a great day", "have a good one",
    "until next time",
})


# ── Router ────────────────────────────────────────────────────────────────────

class FastRouter:
    """
    Zero-LLM keyword router.

    Returns FastRouteResult(intent="SMALL_TALK") on a hit — the caller is
    responsible for generating the actual reply text (via LLM).
    Returns None to fall through to the normal L2 → L3 pipeline.
    """

    def check(self, raw_query: str) -> Optional[FastRouteResult]:
        normalized = _normalize(raw_query)
        words      = normalized.split()

        if not words or len(words) > 6:
            return None

        if len(words) == 1:
            w = words[0]
            if w in _GREET_1W or w in _ACK_1W or w in _BYE_1W:
                return FastRouteResult(intent="SMALL_TALK")
        else:
            if normalized in _GREET_MW or normalized in _ACK_MW or normalized in _BYE_MW:
                return FastRouteResult(intent="SMALL_TALK")

        return None


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


__all__ = ["FastRouter", "FastRouteResult"]
