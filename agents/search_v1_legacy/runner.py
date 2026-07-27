"""Thin entrypoint for the current production search agent.

This package is a seam only: HTTP and callers go through ``run`` / ``AgentRequest``.
The implementation still lives in ``service.agent_runner`` until a later move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

from agents import agent
from service.agent_runner import run_two_stage_rag_stream


@dataclass(frozen=True)
class AgentRequest:
    text: str
    chat_id: str = "default"
    history: Optional[list[dict[str, str]]] = None
    selected_paper_ids: Optional[list] = None
    selected_paper_titles: Optional[list] = None


@agent("search_v1_legacy")
async def run(request: AgentRequest) -> AsyncIterator[str]:
    """Stream plain-text chunks for one turn (same contract as ``/chat`` today)."""
    async for chunk in run_two_stage_rag_stream(
        request.text,
        request.chat_id,
        selected_paper_ids=request.selected_paper_ids,
        selected_paper_titles=request.selected_paper_titles,
        history=request.history,
    ):
        yield chunk


AGENT_ID = run.agent_id
