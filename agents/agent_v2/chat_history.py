"""Shared conversion of persisted chat history into bounded LLM messages."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage


def history_messages(
    history: list[dict[str, str]] | None,
) -> list[HumanMessage | AIMessage]:
    """Return recent user-visible turns without frontend machine payloads."""
    messages: list[HumanMessage | AIMessage] = []
    remaining = 6_000
    for turn in (history or [])[-6:]:
        role, content = turn.get("role"), turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = re.sub(
            r"\[\[VITALITY_PAPERS_JSON\]\][\s\S]*?\[\[/VITALITY_PAPERS_JSON\]\]",
            "",
            content,
        )
        content = re.sub(
            r"\[\[VITALITY_FILE_SEARCH_SCOPE_WARNING\]\][\s\S]*?"
            r"\[\[/VITALITY_FILE_SEARCH_SCOPE_WARNING\]\]",
            "",
            content,
        ).strip()
        if not content or remaining <= 0:
            continue
        content = content[: min(1_000, remaining)]
        messages.append(
            HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        )
        remaining -= len(content)
    return messages
