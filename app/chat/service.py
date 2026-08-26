"""Framework-independent Chat turn orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from time import monotonic

from app.chat.events import (
    AgentAction,
    ChatEvent,
    PapersResult,
    RunCompleted,
    RunFailed,
    TextDelta,
)
from app.chat.models import PreparedChatTurn
from app.chat.persistence import save_assistant_result
from app.chat.runner_adapter import ChatRunner, adapt_runner_output

FALLBACK_TEXT = "I'm sorry, something went wrong on our side. Please try again."


async def iter_chat_turn_events(
    prepared: PreparedChatTurn,
    *,
    run_agent: ChatRunner,
    logger: logging.Logger | None = None,
) -> AsyncIterator[ChatEvent]:
    """Run one prepared Chat turn and yield internal typed events.

    Callable without Flask ``request`` / ``current_app``. Assistant persistence
    runs in ``finally`` so client disconnect (``GeneratorExit`` / ``aclose``)
    still saves a ``failed`` message, matching the previous Flask behavior.
    Persistence failures after streaming starts are logged and do not abort the
    event stream.
    """
    active_logger = logger or logging.getLogger(__name__)
    started_at = monotonic()
    assistant_chunks: list[str] = []
    stream_completed = False
    stream_error: str | None = None
    papers_result: PapersResult | None = None
    degraded = False

    try:
        try:
            async for event in adapt_runner_output(run_agent, prepared):
                if isinstance(event, TextDelta):
                    assistant_chunks.append(event.text)
                elif isinstance(event, PapersResult):
                    papers_result = event
                elif isinstance(event, AgentAction) and event.status == "failed":
                    degraded = True
                yield event
            stream_completed = True
        except Exception as error:  # pylint: disable=broad-except
            # Status 200 and earlier chunks are already conceptually "sent", so
            # no failure may escape: every error degrades into fallback text.
            # GeneratorExit is not caught here; finally still persists.
            active_logger.warning("Chat stream error: %s", error, exc_info=True)
            stream_error = str(error)[:500]
            assistant_chunks.append(FALLBACK_TEXT)
            yield TextDelta(text=FALLBACK_TEXT)

        duration_ms = round((monotonic() - started_at) * 1000)
        if stream_completed:
            yield RunCompleted(duration_ms=duration_ms, degraded=degraded)
        else:
            yield RunFailed(
                # Raw provider/LLM errors belong in logs and persistence only.
                # The public stream must expose a stable, sanitized message.
                message=FALLBACK_TEXT,
                duration_ms=duration_ms,
            )
    finally:
        if prepared.user_id:
            save_assistant_result(
                conversation_id=prepared.request.chat_id,
                text="".join(assistant_chunks),
                status="completed" if stream_completed else "failed",
                duration_ms=round((monotonic() - started_at) * 1000),
                error_message=stream_error,
                message_id=prepared.request.assistant_message_id,
                context=(
                    {
                        "papersResult": {
                            "ids": papers_result.ids,
                            "rankedIds": papers_result.ranked_ids,
                            "policy": papers_result.policy,
                            "effort": papers_result.effort,
                            "countKnown": papers_result.count_known,
                        }
                    }
                    if papers_result is not None
                    else None
                ),
                log=active_logger,
            )
