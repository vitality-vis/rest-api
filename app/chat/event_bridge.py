"""Bridge Agent events from a dedicated worker thread to the ASGI event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from app.chat.events import ChatEvent, RunCompleted, RunFailed
from app.chat.execution import AgentExecutionRuntime, AgentRunReservation
from app.chat.models import PreparedChatTurn
from app.chat.runner_adapter import ChatRunner
from app.chat.service import iter_chat_turn_events
from app.chat.persistence import save_assistant_result

# How long the worker waits on a full SSE queue before re-checking cancel.
_PUBLISH_WAIT_SLICE_SECONDS = 0.25


@dataclass(frozen=True)
class _BridgeDone:
    pass


@dataclass(frozen=True)
class _BridgeFailure:
    message: str


BridgeItem = ChatEvent | _BridgeDone | _BridgeFailure


def _blocking_publish(
    *,
    main_loop: asyncio.AbstractEventLoop,
    output: asyncio.Queue[BridgeItem],
    item: BridgeItem,
    cancel_event: threading.Event,
) -> bool:
    """Put onto the bounded ASGI queue with real backpressure.

    Returns False if cancelled while waiting for queue space. Typed events are
    never dropped silently while still accepted: cancel aborts the wait instead.
    """
    future = asyncio.run_coroutine_threadsafe(output.put(item), main_loop)
    while True:
        if cancel_event.is_set() and not future.done():
            future.cancel()
            try:
                future.result(timeout=0.5)
            except (FutureCancelledError, asyncio.CancelledError, FutureTimeoutError):
                pass
            except Exception:
                pass
            return False
        try:
            future.result(timeout=_PUBLISH_WAIT_SLICE_SECONDS)
            return True
        except FutureTimeoutError:
            continue
        except FutureCancelledError:
            return False


async def bridge_agent_events(
    prepared: PreparedChatTurn,
    *,
    run_agent: ChatRunner,
    runtime: AgentExecutionRuntime,
    reservation: AgentRunReservation,
    logger: logging.Logger,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[ChatEvent]:
    """Yield worker events without using an Agent or WSGI thread for waiting."""
    main_loop = asyncio.get_running_loop()
    output: asyncio.Queue[BridgeItem] = asyncio.Queue(maxsize=runtime.sse_queue_size)
    cancel_event = threading.Event()
    saw_terminal = False

    def publish(item: BridgeItem) -> bool:
        try:
            return _blocking_publish(
                main_loop=main_loop,
                output=output,
                item=item,
                cancel_event=cancel_event,
            )
        except RuntimeError:
            # The ASGI loop has already closed; stop at the next cancel check.
            cancel_event.set()
            return False

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
                    if not publish(event):
                        break
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
        future = runtime.submit(reservation, run_job)
    except RuntimeError as error:
        # Shutdown can race a request that already reserved a slot.
        logger.warning("Agent executor rejected a Chat run: %s", error)
        reservation.mark_failed()
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
                saw_terminal = True
                reservation.mark_failed()
                yield RunFailed(
                    message=item.message,
                    duration_ms=0,
                    error_code="agent_worker_failed",
                )
                continue
            if isinstance(item, (RunCompleted, RunFailed)):
                saw_terminal = True
                if isinstance(item, RunCompleted):
                    reservation.mark_completed()
                else:
                    reservation.mark_failed()
            yield item
    finally:
        cancel_event.set()
        # Cancels a pending (not yet started) Future. A running thread stops
        # cooperatively at the next event / publish boundary (Phase 7B: nodes).
        cancelled_before_start = False
        if reservation.future is not None and not reservation.future.done():
            cancelled_before_start = bool(reservation.future.cancel())
        if cancelled_before_start and prepared.user_id:
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
        if not saw_terminal:
            reservation.mark_cancelled()
        else:
            reservation.ensure_closed()
