"""Serialize typed Chat events into SSE frames."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from app.chat.events import (
    AgentAction,
    ChatEvent,
    PapersResult,
    RunCompleted,
    RunFailed,
    RunStarted,
    TextDelta,
)

SCHEMA_VERSION = 1


def encode_keepalive_comment() -> str:
    """SSE comment frame; does not advance seq or carry typed Chat events."""
    return ": keepalive\n\n"


def _event_data(event: ChatEvent) -> dict[str, object]:
    if isinstance(event, RunStarted):
        return {
            "clientRequestId": event.client_request_id,
            "agentRunId": event.agent_run_id,
            "conversationId": event.conversation_id,
            "assistantMessageId": event.assistant_message_id,
            "pipeline": event.pipeline,
            "effort": event.effort,
        }
    if isinstance(event, AgentAction):
        return {
            "actionId": event.action_id,
            "action": event.action,
            "status": event.status,
            **({"data": event.data} if event.data else {}),
        }
    if isinstance(event, TextDelta):
        return {"text": event.text}
    if isinstance(event, PapersResult):
        return {
            "ids": event.ids,
            "rankedIds": event.ranked_ids,
            "policy": event.policy,
            "effort": event.effort,
            "countKnown": event.count_known,
        }
    if isinstance(event, RunCompleted):
        return {
            "durationMs": event.duration_ms,
            **({"degraded": True} if event.degraded else {}),
        }
    if isinstance(event, RunFailed):
        return {
            "durationMs": event.duration_ms,
            "error": {
                "code": event.error_code,
                "message": event.message,
                "retryable": event.retryable,
            },
        }
    raise TypeError(f"Unsupported Chat event: {type(event)!r}")


@dataclass
class SSEEncoder:
    seq: int = 0

    def encode(self, event: ChatEvent) -> str:
        self.seq += 1
        envelope = {
            "schemaVersion": SCHEMA_VERSION,
            "type": event.type,
            "seq": self.seq,
            "timestamp": int(time.time() * 1000),
            "data": _event_data(event),
        }
        data = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        return f"id: {self.seq}\nevent: {event.type}\ndata: {data}\n\n"
