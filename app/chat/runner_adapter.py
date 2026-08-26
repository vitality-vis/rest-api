"""Adapt Chat turn requests to Agent runners and wrap string chunks as events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from app.chat.events import AgentAction, PapersResult, RunnerEvent, TextDelta
from app.chat.models import PreparedChatTurn
from app.chat.run_control import RunControl

ChatRunner = Callable[[Any], AsyncIterator[Any]]


def build_agent_request(
    prepared: PreparedChatTurn,
    *,
    control: RunControl | None = None,
) -> Any:
    """Build the v2 Agent request object."""
    from agents.agent_v2.models import V2ChatRequest

    request = prepared.request
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
        control=control,
    )


async def adapt_runner_output(
    run_agent: ChatRunner,
    prepared: PreparedChatTurn,
    *,
    control: RunControl | None = None,
) -> AsyncIterator[RunnerEvent]:
    """Pass typed runner output through and wrap legacy strings as text deltas."""
    agent_request = build_agent_request(prepared, control=control)
    async for chunk in run_agent(agent_request):
        if control is not None:
            control.raise_if_aborted()
        if isinstance(chunk, (AgentAction, TextDelta, PapersResult)):
            yield chunk
        else:
            yield TextDelta(text=str(chunk))
