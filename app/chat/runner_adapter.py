"""Adapt Chat turn requests to Agent runners and wrap string chunks as events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from app.chat.events import AgentAction, PapersResult, RunnerEvent, TextDelta
from app.chat.models import PreparedChatTurn

ChatRunner = Callable[[Any], AsyncIterator[Any]]


def build_agent_request(prepared: PreparedChatTurn) -> Any:
    """Build the pipeline-specific Agent request object."""
    request = prepared.request
    if request.pipeline == "v2":
        from agents.agent_v2.models import V2ChatRequest

        return V2ChatRequest(
            text=request.text,
            chat_id=request.chat_id,
            history=prepared.history,
            selected_paper_ids=request.paper_ids,
            context=request.message_context,
            effort=request.effort,
            model=request.model,
            trace_id=request.trace_id,
            user_message_id=request.user_message_id,
            assistant_message_id=request.assistant_message_id,
            requested_mode=request.requested_mode,
            user_id=prepared.user_id,
            advanced=request.advanced,
        )

    from agents.agent_v1_legacy import AgentRequest

    return AgentRequest(
        text=request.text,
        chat_id=request.chat_id,
        history=prepared.history,
        effort=request.effort,
        model=request.model,
        trace_id=request.trace_id,
        user_message_id=request.user_message_id,
        assistant_message_id=request.assistant_message_id,
    )


async def adapt_runner_output(
    run_agent: ChatRunner,
    prepared: PreparedChatTurn,
) -> AsyncIterator[RunnerEvent]:
    """Pass typed runner output through and wrap legacy strings as text deltas."""
    agent_request = build_agent_request(prepared)
    async for chunk in run_agent(agent_request):
        if isinstance(chunk, (AgentAction, TextDelta, PapersResult)):
            yield chunk
        else:
            yield TextDelta(text=str(chunk))
