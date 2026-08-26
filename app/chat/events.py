"""Transport-independent typed Chat stream events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True)
class RunStarted:
    client_request_id: str
    agent_run_id: str
    conversation_id: str
    assistant_message_id: str
    effort: str
    pipeline: Literal["v2"] = "v2"
    type: Literal["run.started"] = "run.started"


@dataclass(frozen=True)
class AgentAction:
    action_id: str
    action: str
    status: Literal["started", "completed", "failed"]
    data: dict[str, object] = field(default_factory=dict)
    type: Literal["agent.action"] = "agent.action"


@dataclass(frozen=True)
class TextDelta:

    text: str
    type: Literal["text.delta"] = "text.delta"


@dataclass(frozen=True)
class RunCompleted:
    duration_ms: int
    degraded: bool = False
    type: Literal["run.completed"] = "run.completed"


@dataclass(frozen=True)
class PapersResult:
    ids: list[str]
    ranked_ids: list[str]
    policy: str
    effort: str
    count_known: bool = False
    type: Literal["papers.result"] = "papers.result"


@dataclass(frozen=True)
class RunFailed:

    message: str
    duration_ms: int
    error_code: str = "agent_execution_failed"
    retryable: bool = True
    type: Literal["run.failed"] = "run.failed"


RunnerEvent = Union[AgentAction, TextDelta, PapersResult]
ChatEvent = Union[
    RunStarted,
    AgentAction,
    TextDelta,
    PapersResult,
    RunCompleted,
    RunFailed,
]
