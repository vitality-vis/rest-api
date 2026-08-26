"""Bridge Agent events from a dedicated worker thread to the ASGI event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from app.chat.events import ChatEvent, RunFailed
from app.chat.execution import AgentExecutionRuntime
from app.chat.models import PreparedChatTurn
from app.chat.runner_adapter import ChatRunner
from app.chat.service import iter_chat_turn_events
from app.chat.persistence import save_assistant_result


@dataclass(frozen=True)
class _BridgeDone:
    pass


@dataclass(frozen=True)
class _BridgeFailure:
    message: str


BridgeItem = ChatEvent | _BridgeDone | _BridgeFailure


async def bridge_agent_events(
    prepared: PreparedChatTurn,
    *,
    run_agent: ChatRunner,
    runtime: AgentExecutionRuntime,
    logger: logging.Logger,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[ChatEvent]:
    """Yield worker events without using an Agent or WSGI thread for waiting."""
    main_loop = asyncio.get_running_loop()
    output: asyncio.Queue[BridgeItem] = asyncio.Queue()
    cancel_event = threading.Event()

    def publish(item: BridgeItem) -> None:
        try:
            main_loop.call_soon_threadsafe(output.put_nowait, item)
        except RuntimeError:
            # The ASGI loop has already closed; the worker will observe cancel at
            # its next event boundary and unwind its async generator.
            cancel_event.set()

    def run_job() -> None:
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)

        async def produce() -> None:
            agen = iter_chat_turn_events(
                prepared,
                run_agent=run_agent,
                logger=logger,
            ).__aiter__()
            try:
                async for event in agen:
                    if cancel_event.is_set():
                        break
                    publish(event)
            finally:
                await agen.aclose()

        try:
            worker_loop.run_until_complete(produce())
        except BaseException as error:  # worker boundary; never leak raw details
            logger.warning("Agent bridge worker failed: %s", error, exc_info=True)
            publish(_BridgeFailure(message="The Agent run could not be completed."))
        finally:
            pending = asyncio.all_tasks(loop=worker_loop)
            for task in pending:
                task.cancel()
            if pending:
                worker_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            worker_loop.close()
            publish(_BridgeDone())

    try:
        future = runtime.submit(run_job)
    except RuntimeError as error:
        # Shutdown can race a request that already passed the route-level
        # readiness check. The SSE has started, so return a typed terminal and
        # persist the assistant failure here because no service job can own it.
        logger.warning("Agent executor rejected a Chat run: %s", error)
        if prepared.user_id:
            await asyncio.to_thread(
                save_assistant_result,
                conversation_id=prepared.request.chat_id,
                text="",
                status="failed",
                duration_ms=0,
                error_message="Agent executor unavailable",
                message_id=prepared.request.assistant_message_id,
                log=logger,
            )
        yield RunFailed(
            message="Chat is temporarily unavailable. Please try again.",
            duration_ms=0,
            error_code="agent_unavailable",
        )
        return
    try:
        while True:
            try:
                item = await asyncio.wait_for(output.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    break
                continue

            if isinstance(item, _BridgeDone):
                return
            if isinstance(item, _BridgeFailure):
                yield RunFailed(
                    message=item.message,
                    duration_ms=0,
                    error_code="agent_worker_failed",
                )
                continue
            yield item
    finally:
        cancel_event.set()
        # Cancels a queued job. A running Python thread stops cooperatively at
        # the next emitted Agent event; Phase 7 adds finer node-level cancellation.
        cancelled_before_start = future.cancel()
        if cancelled_before_start and prepared.user_id:
            # The service never started, so it cannot own terminal persistence.
            # Persist here only after Future.cancel() proves no worker can race it.
            await asyncio.to_thread(
                save_assistant_result,
                conversation_id=prepared.request.chat_id,
                text="",
                status="failed",
                duration_ms=0,
                error_message="Client disconnected before the Agent run started",
                message_id=prepared.request.assistant_message_id,
                log=logger,
            )
