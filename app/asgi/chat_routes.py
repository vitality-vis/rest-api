"""Native ASGI Chat v2 route with typed SSE transport."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.chat.event_bridge import bridge_agent_events
from app.chat.events import RunCompleted, RunFailed, RunStarted
from app.chat.models import ChatDomainError
from app.chat.request_service import build_chat_turn_request, prepare_chat_turn
from app.chat.sse import SSEEncoder

if TYPE_CHECKING:
    from app.profiles import ApplicationBundle


def _error_response(error: ChatDomainError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.message},
    )


def register_chat_routes(app: FastAPI, bundle: ApplicationBundle) -> None:
    """Register full-only routes before the catch-all Flask mount."""

    @app.post("/chat/v2")
    async def chat_v2(request: Request):
        runtime = bundle.agent_runtime
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "Chat execution is unavailable"},
            )

        try:
            raw = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})
        if not isinstance(raw, dict):
            return JSONResponse(status_code=400, content={"detail": "JSON body must be an object"})

        client_request_id = raw.get("client_request_id")
        if not isinstance(client_request_id, str) or not client_request_id.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "client_request_id is required"},
            )

        agent_run_id = str(uuid4())
        assistant_message_id = str(uuid4())
        payload = {
            **raw,
            "client_request_id": client_request_id.strip(),
            "agent_run_id": agent_run_id,
            "assistant_message_id": assistant_message_id,
        }
        try:
            turn_request = build_chat_turn_request(
                payload,
                pipeline="v2",
                max_text_length=10_000,
                authorization_header=request.headers.get("Authorization"),
                trace_id=agent_run_id,
            )
            # Supabase auth/history/message persistence use synchronous clients.
            prepared = await asyncio.to_thread(prepare_chat_turn, turn_request)
        except ChatDomainError as error:
            return _error_response(error)

        from agents.agent_v2.runner import run as run_agent_v2

        async def event_stream():
            encoder = SSEEncoder()
            terminal_sent = False
            events = bridge_agent_events(
                prepared,
                run_agent=run_agent_v2,
                runtime=runtime,
                logger=bundle.logger,
                is_disconnected=request.is_disconnected,
            )
            # Start/queue the job before exposing run.started so an immediate
            # disconnect always has a Future whose persistence owner is known.
            first_event = asyncio.create_task(events.__anext__())
            try:
                yield encoder.encode(
                    RunStarted(
                        client_request_id=client_request_id.strip(),
                        agent_run_id=agent_run_id,
                        conversation_id=turn_request.chat_id,
                        assistant_message_id=assistant_message_id,
                        effort=turn_request.effort,
                    )
                )
                try:
                    event = await first_event
                except StopAsyncIteration:
                    event = None

                while event is not None:
                    if await request.is_disconnected():
                        return
                    yield encoder.encode(event)
                    if isinstance(event, (RunCompleted, RunFailed)):
                        terminal_sent = True
                        break
                    try:
                        event = await events.__anext__()
                    except StopAsyncIteration:
                        event = None

                if not terminal_sent and not await request.is_disconnected():
                    yield encoder.encode(
                        RunFailed(
                            message="The Agent stream ended unexpectedly.",
                            duration_ms=0,
                            error_code="stream_ended",
                        )
                    )
            finally:
                if not first_event.done():
                    first_event.cancel()
                await asyncio.gather(first_event, return_exceptions=True)
                await events.aclose()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Agent-Run-Id": agent_run_id,
            },
        )
