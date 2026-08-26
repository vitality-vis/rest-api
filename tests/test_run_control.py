"""Unit tests for RunControl deadlines and cooperative abort."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.chat.event_bridge import bridge_agent_events
from app.chat.events import RunFailed, TextDelta
from app.chat.execution import AgentExecutionRuntime
from app.chat.models import ChatTurnRequest, PreparedChatTurn
from app.chat.run_control import AbortKind, RunAborted, RunControl


def _prepared(
    *,
    chat_id: str,
    assistant_message_id: str,
    user_id: str | None = None,
) -> PreparedChatTurn:
    return PreparedChatTurn(
        request=ChatTurnRequest(
            text="Hello",
            chat_id=chat_id,
            assistant_message_id=assistant_message_id,
            pipeline="v2",
        ),
        user_id=user_id,
        history=[],
    )


def test_run_control_queue_wait_timeout():
    control = RunControl(queue_wait_timeout_s=0.05, agent_timeout_s=10.0)
    control.mark_queued()
    time.sleep(0.06)
    with pytest.raises(RunAborted) as raised:
        control.raise_if_aborted()
    assert raised.value.kind is AbortKind.QUEUE_WAIT_TIMEOUT


def test_run_control_agent_timeout_after_start():
    control = RunControl(queue_wait_timeout_s=10.0, agent_timeout_s=0.05)
    control.mark_queued()
    control.mark_started()
    time.sleep(0.06)
    with pytest.raises(RunAborted) as raised:
        control.raise_if_aborted()
    assert raised.value.kind is AbortKind.AGENT_TIMEOUT


def test_run_control_cancel_wins():
    control = RunControl(queue_wait_timeout_s=10.0, agent_timeout_s=10.0)
    control.mark_queued()
    control.mark_started()
    control.cancel(AbortKind.CANCELLED)
    with pytest.raises(RunAborted) as raised:
        control.raise_if_aborted()
    assert raised.value.kind is AbortKind.CANCELLED


def test_run_control_abort_does_not_close_stream():
    control = RunControl(queue_wait_timeout_s=10.0, agent_timeout_s=10.0)
    control.cancel(AbortKind.AGENT_TIMEOUT)
    assert control.is_aborted()
    assert not control.is_stream_closed()
    control.close_stream()
    assert control.is_stream_closed()


def test_bridge_emits_agent_timeout(monkeypatch):
    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", lambda **_k: None)
    runtime = AgentExecutionRuntime(
        max_workers=1,
        max_pending=1,
        agent_run_timeout_seconds=0.05,
        queue_wait_timeout_seconds=10.0,
    )
    prepared = _prepared(chat_id="timeout-1", assistant_message_id="asst-timeout")

    async def slow_run(_request: Any) -> AsyncIterator[Any]:
        await asyncio.sleep(2)
        yield TextDelta(text="late")

    async def exercise() -> list[str]:
        reservation = runtime.try_reserve()
        assert reservation is not None
        codes: list[str] = []
        async for event in bridge_agent_events(
            prepared,
            run_agent=slow_run,
            runtime=runtime,
            reservation=reservation,
            logger=logging.getLogger("test-timeout"),
        ):
            code = getattr(event, "error_code", None)
            if code:
                codes.append(code)
        return codes

    codes = asyncio.run(exercise())
    snap = runtime.snapshot()
    asyncio.run(runtime.shutdown())
    assert "agent_timeout" in codes
    assert snap.timed_out >= 1


def test_bridge_worker_first_timeout_does_not_stream_ended(monkeypatch):
    """Worker hits abort before ASGI's 250ms poll; terminal must stay agent_timeout."""
    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", lambda **_k: None)
    runtime = AgentExecutionRuntime(
        max_workers=1,
        max_pending=0,
        agent_run_timeout_seconds=0.05,
        queue_wait_timeout_seconds=10.0,
        # Large queue so publish is never blocked on backpressure.
        sse_queue_size=8,
    )
    prepared = _prepared(
        chat_id="worker-first-timeout",
        assistant_message_id="asst-wft",
    )

    async def run_past_deadline(_request: Any) -> AsyncIterator[Any]:
        # Exceed agent timeout, then yield so adapt_runner_output checkpoints.
        await asyncio.sleep(0.08)
        yield TextDelta(text="should-be-aborted")

    async def exercise() -> list[Any]:
        reservation = runtime.try_reserve()
        assert reservation is not None
        events: list[Any] = []
        async for event in bridge_agent_events(
            prepared,
            run_agent=run_past_deadline,
            runtime=runtime,
            reservation=reservation,
            logger=logging.getLogger("test-worker-first"),
        ):
            events.append(event)
        return events

    events = asyncio.run(exercise())
    asyncio.run(runtime.shutdown())
    failed = [event for event in events if isinstance(event, RunFailed)]
    assert len(failed) == 1
    assert failed[0].error_code == "agent_timeout"
    assert all(getattr(event, "error_code", None) != "stream_ended" for event in events)


def test_bridge_queue_wait_timeout(monkeypatch):
    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", lambda **_k: None)
    runtime = AgentExecutionRuntime(
        max_workers=1,
        max_pending=2,
        agent_run_timeout_seconds=30.0,
        queue_wait_timeout_seconds=0.05,
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_job() -> None:
        started.set()
        release.wait(timeout=5)

    blocker_res = runtime.try_reserve()
    assert blocker_res is not None
    blocker = runtime.submit(blocker_res, blocking_job)
    assert started.wait(timeout=2)

    prepared = _prepared(chat_id="queue-wait-1", assistant_message_id="asst-qw")

    async def never_run(_request: Any) -> AsyncIterator[Any]:
        if False:  # pragma: no cover
            yield TextDelta(text="")
        raise AssertionError("should not start after queue wait timeout")

    async def exercise() -> str | None:
        reservation = runtime.try_reserve()
        assert reservation is not None
        error_code = None
        async for event in bridge_agent_events(
            prepared,
            run_agent=never_run,
            runtime=runtime,
            reservation=reservation,
            logger=logging.getLogger("test-queue-wait"),
        ):
            error_code = getattr(event, "error_code", error_code)
        return error_code

    code = asyncio.run(exercise())
    release.set()
    blocker.result(timeout=2)
    blocker_res.mark_completed()
    asyncio.run(runtime.shutdown())
    assert code == "queue_wait_timeout"


def test_bridge_queue_wait_timeout_persists_for_logged_in_user(monkeypatch):
    saved: dict[str, Any] = {}

    def fake_save(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", fake_save)
    runtime = AgentExecutionRuntime(
        max_workers=1,
        max_pending=2,
        agent_run_timeout_seconds=30.0,
        queue_wait_timeout_seconds=0.05,
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_job() -> None:
        started.set()
        release.wait(timeout=5)

    blocker_res = runtime.try_reserve()
    assert blocker_res is not None
    blocker = runtime.submit(blocker_res, blocking_job)
    assert started.wait(timeout=2)

    prepared = _prepared(
        chat_id="queue-wait-auth",
        assistant_message_id="asst-qw-auth",
        user_id="user-1",
    )

    async def never_run(_request: Any) -> AsyncIterator[Any]:
        if False:  # pragma: no cover
            yield TextDelta(text="")
        raise AssertionError("should not start after queue wait timeout")

    async def exercise() -> str | None:
        reservation = runtime.try_reserve()
        assert reservation is not None
        error_code = None
        async for event in bridge_agent_events(
            prepared,
            run_agent=never_run,
            runtime=runtime,
            reservation=reservation,
            logger=logging.getLogger("test-queue-wait-auth"),
        ):
            error_code = getattr(event, "error_code", error_code)
        return error_code

    code = asyncio.run(exercise())
    release.set()
    blocker.result(timeout=2)
    blocker_res.mark_completed()
    asyncio.run(runtime.shutdown())

    assert code == "queue_wait_timeout"
    assert saved["conversation_id"] == "queue-wait-auth"
    assert saved["status"] == "failed"
    assert saved["error_message"] == "queue_wait_timeout"
    assert "disconnected" not in (saved["error_message"] or "").lower()


def test_bridge_timeout_keeps_admission_until_thread_exits(monkeypatch):
    monkeypatch.setattr("app.chat.event_bridge.save_assistant_result", lambda **_k: None)
    runtime = AgentExecutionRuntime(
        max_workers=1,
        max_pending=0,
        agent_run_timeout_seconds=0.05,
        queue_wait_timeout_seconds=10.0,
    )
    started = threading.Event()
    release = threading.Event()
    prepared = _prepared(chat_id="hold-slot", assistant_message_id="asst-hold")

    async def stuck_run(_request: Any) -> AsyncIterator[Any]:
        started.set()
        # Block the worker thread so abort cannot free the slot yet.
        release.wait(timeout=30)
        yield TextDelta(text="late")

    async def exercise() -> None:
        reservation = runtime.try_reserve()
        assert reservation is not None
        async for event in bridge_agent_events(
            prepared,
            run_agent=stuck_run,
            runtime=runtime,
            reservation=reservation,
            logger=logging.getLogger("test-hold-slot"),
        ):
            if getattr(event, "error_code", None) == "agent_timeout":
                assert started.wait(timeout=2)
                snap = runtime.snapshot()
                assert snap.active == 1
                assert snap.timed_out >= 1
                assert snap.accepting is False
                assert runtime.try_reserve() is None
                release.set()

    asyncio.run(exercise())
    # After the stuck thread exits, occupancy is released via Future callback.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.snapshot().active == 0:
            break
        time.sleep(0.01)
    snap = runtime.snapshot()
    assert snap.active == 0
    assert snap.accepting is True
    asyncio.run(runtime.shutdown())
