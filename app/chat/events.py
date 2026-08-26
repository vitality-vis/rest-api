"""Internal typed Chat stream events (not yet on the wire as SSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class TextDelta:
    """Append assistant answer text (future wire type ``text.delta``)."""

    text: str
    type: Literal["text.delta"] = "text.delta"


@dataclass(frozen=True)
class RunCompleted:
    """Successful terminal state (future wire type ``run.completed``)."""

    duration_ms: int
    type: Literal["run.completed"] = "run.completed"


@dataclass(frozen=True)
class RunFailed:
    """Failed terminal state (future wire type ``run.failed``)."""

    message: str
    duration_ms: int
    type: Literal["run.failed"] = "run.failed"


ChatEvent = Union[TextDelta, RunCompleted, RunFailed]
