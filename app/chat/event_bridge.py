"""Bridge Agent events from a dedicated worker thread to the ASGI event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import CancelledError as FutureCancelledError
from dataclasses import dataclass

from app.chat.events import ChatEvent, RunCompleted, RunFailed
from app.chat.execution import AgentExecutionRuntime, AgentRunReservation
from app.chat.models import PreparedChatTurn
from app.chat.run_control import AbortKind, RunAborted, RunControl
from app.chat.runner_adapter import ChatRunner
from app.chat.service import iter_chat_turn_events
from app.chat.persistence import save_assistant_result

# How long the worker waits on a full SSE queue before re-checking stream close.
_PUBLISH_WAIT_SLICE_SECONDS = 0.25

_TIMEOUT_ERROR_CODES = {
    AbortKind.QUEUE_WAIT_TIMEOUT.value,
    AbortKind.AGENT_TIMEOUT.value,
}


@dataclass(frozen=True)
class _BridgeDone:
    pass


@dataclass(frozen=True)
class _BridgeFailure:
    message: str
    error_code: str = "agent_worker_failed"


BridgeItem = ChatEvent | _BridgeDone | _BridgeFailure


def _run_failed_for_abort(kind: AbortKind, *, duration_ms: int = 0) -> RunFailed:
    aborted = RunAborted(kind)
    return RunFailed(
        message=aborted.public_message,
        duration_ms=duration_ms,
        error_code=aborted.error_code,
        retryable=aborted.retryable,
    )


async def _cancellable_put(
    output: asyncio.Queue[BridgeItem],
    item: BridgeItem,
    stream_closed_event: threading.Event,
) -> bool:
    """Wait for queue space unless the consumer stream is closed."""
    put_task = asyncio.create_task(output.put(item))
    try:
        while not put_task.done():
            if stream_closed_event.is_set():
                put_task.cancel()
                try:
                    await put_task
                except asyncio.CancelledError:
                    pass
                return False
            await asyncio.wait({put_task}, timeout=_PUBLISH_WAIT_SLICE_SECONDS)
        await put_task
        return True
    except asyncio.CancelledError:
        if not put_task.done():
            put_task.cancel()
            try:
                await put_task
            except asyncio.CancelledError:
                pass
        raise


def _blocking_publish(
    *,
    main_loop: asyncio.AbstractEventLoop,
    output: asyncio.Queue[BridgeItem],
    item: BridgeItem,
    stream_closed_event: threading.Event,
) -> bool:
    """Put onto the bounded ASGI queue with real backpressure.

    Returns False if the consumer stream closed while waiting for queue space.
    Typed events are never dropped silently while the stream is still accepted:
    stream close aborts the wait instead. Agent abort alone must not block put.
    """
    future = asyncio.run_coroutine_threadsafe(
        _cancellable_put(output, item, stream_closed_event),
        main_loop,
    )
    try:
        return bool(future.result())
    except FutureCancelledError:
        return False
    except Exception:
        if stream_closed_event.is_set():
            return False
        raise


def _mark_reservation_for_event(
    reservation: AgentRunReservation,
    event: ChatEvent,
) -> None:
    if isinstance(event, RunCompleted):
        reservation.mark_completed()
        return
    if isinstance(event, RunFailed):
        if event.error_code in _TIMEOUT_ERROR_CODES:
            reservation.mark_timed_out()
        elif event.error_code == AbortKind.CANCELLED.value:
            reservation.mark_cancelled()
        else:
            reservation.mark_failed()


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
    stream_closed_event = threading.Event()
    control = RunControl(
        queue_wait_timeout_s=runtime.queue_wait_timeout_seconds,
        agent_timeout_s=runtime.agent_run_timeout_seconds,
        cancel_event=cancel_event,
        stream_closed_event=stream_closed_event,
    )
    saw_terminal = False

    def publish(item: BridgeItem) -> bool:
        if stream_closed_event.is_set() and not main_loop.is_running():
            return False
        try:
            return _blocking_publish(
                main_loop=main_loop,
                output=output,
                item=item,
                stream_closed_event=stream_closed_event,
            )
        except RuntimeError:
            # The ASGI loop has already closed; stop at the next cancel check.
            control.cancel(AbortKind.CANCELLED)
            control.close_stream()
            return False

    def run_job() -> None:
        # Evaluate queue-wait before marking execution started. Once started,
        # only agent-total timeout and cancel apply.
        try:
            control.raise_if_aborted()
        except RunAborted as aborted:
            publish(_run_failed_for_abort(aborted.kind))
            publish(_BridgeDone())
            return

        control.mark_started()
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)

        async def produce() -> None:
            agen = iter_chat_turn_events(
                prepared,
                run_agent=run_agent,
                logger=logger,
                control=control,
            ).__aiter__()
            try:
                async for event in agen:
                    # Publish first so a terminal RunFailed from the service is
                    # not dropped by a post-abort checkpoint.
                    if not publish(event):
                        break
                    if isinstance(event, (RunCompleted, RunFailed)):
                        break
                    try:
                        control.raise_if_aborted()
                    except RunAborted:
                        break
            finally:
                await agen.aclose()

        try:
            worker_loop.run_until_complete(produce())
        except BaseException as error:  # worker boundary; never leak raw details
            logger.warning("Agent bridge worker failed: %s", error, exc_info=True)
            publish(
                _BridgeFailure(
                    message="The Agent run could not be completed.",
                    error_code="agent_worker_failed",
                )
            )
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
        control.mark_queued()
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
            abort_kind = control.peek_abort()
            if abort_kind in {
                AbortKind.QUEUE_WAIT_TIMEOUT,
                AbortKind.AGENT_TIMEOUT,
            }:
                control.cancel(abort_kind)
                if not saw_terminal:
                    failed = _run_failed_for_abort(abort_kind)
                    saw_terminal = True
                    _mark_reservation_for_event(reservation, failed)
                    yield failed
                break

            try:
                item = await asyncio.wait_for(output.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    control.cancel(AbortKind.CANCELLED)
                    control.close_stream()
                    break
                continue

            if isinstance(item, _BridgeDone):
                if not saw_terminal:
                    abort_kind = control.peek_abort()
                    if abort_kind is not None:
                        failed = _run_failed_for_abort(abort_kind)
                        saw_terminal = True
                        _mark_reservation_for_event(reservation, failed)
                        yield failed
                return
            if isinstance(item, _BridgeFailure):
                saw_terminal = True
                failed = RunFailed(
                    message=item.message,
                    duration_ms=0,
                    error_code=item.error_code,
                )
                _mark_reservation_for_event(reservation, failed)
                yield failed
                continue
            if isinstance(item, (RunCompleted, RunFailed)):
                saw_terminal = True
                _mark_reservation_for_event(reservation, item)
            yield item
    finally:
        if not control.cancel_event.is_set():
            control.cancel(AbortKind.CANCELLED)
        control.close_stream()
        # Cancels a pending (not yet started) Future. A running thread stops
        # cooperatively at the next node / publish boundary.
        cancelled_before_start = False
        worker_future = reservation.future
        if worker_future is not None and not worker_future.done():
            cancelled_before_start = bool(worker_future.cancel())
        if cancelled_before_start and prepared.user_id:
            abort_kind = control.peek_abort()
            if abort_kind is AbortKind.QUEUE_WAIT_TIMEOUT:
                error_message = AbortKind.QUEUE_WAIT_TIMEOUT.value
            elif abort_kind is AbortKind.AGENT_TIMEOUT:
                error_message = AbortKind.AGENT_TIMEOUT.value
            else:
                error_message = "Client disconnected before the Agent run started"
            await asyncio.to_thread(
                save_assistant_result,
                conversation_id=prepared.request.chat_id,
                text="",
                status="failed",
                duration_ms=0,
                error_message=error_message,
                message_id=prepared.request.assistant_message_id,
                log=logger,
            )
        elif worker_future is not None and not cancelled_before_start:
            # Keep the ASGI loop alive until the worker observes cancel and
            # unwinds, so thread-safe publishes are not scheduled onto a closed loop.
            try:
                await asyncio.to_thread(worker_future.result, 5.0)
            except Exception:
                logger.debug(
                    "Agent worker did not finish within join timeout after cancel",
                    exc_info=True,
                )
        if not saw_terminal:
            abort_kind = control.peek_abort()
            if abort_kind in {
                AbortKind.QUEUE_WAIT_TIMEOUT,
                AbortKind.AGENT_TIMEOUT,
            }:
                reservation.mark_timed_out()
            else:
                reservation.mark_cancelled()
        else:
            reservation.ensure_closed()
