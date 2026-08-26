"""Framework-independent Chat turn orchestration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from time import monotonic

from app.chat.events import ChatEvent, RunCompleted, RunFailed, TextDelta
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
    still saves a ``failed`` message, matching the pre-Phase-4 Flask behavior.
    Persistence failures after streaming starts are logged and do not abort the
    event stream.
    """
    active_logger = logger or logging.getLogger(__name__)
    started_at = monotonic()
    assistant_chunks: list[str] = []
    stream_completed = False
    stream_error: str | None = None

    try:
        try:
            async for delta in adapt_runner_output(run_agent, prepared):
                assistant_chunks.append(delta.text)
                yield delta
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
            yield RunCompleted(duration_ms=duration_ms)
        else:
            yield RunFailed(
                message=stream_error or FALLBACK_TEXT,
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
                log=active_logger,
            )
