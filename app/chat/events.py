"""Internal typed Chat stream events (Phase 4; not yet on the wire as SSE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class TextDelta:
    """Append assistant answer text (wire type ``text.delta`` in Phase 6)."""

    text: str
    type: Literal["text.delta"] = "text.delta"


@dataclass(frozen=True)
class RunCompleted:
    """Successful terminal state (wire type ``run.completed`` in Phase 6)."""

    duration_ms: int
    type: Literal["run.completed"] = "run.completed"


@dataclass(frozen=True)
class RunFailed:
    """Failed terminal state (wire type ``run.failed`` in Phase 6)."""

    message: str
    duration_ms: int
    type: Literal["run.failed"] = "run.failed"


ChatEvent = Union[TextDelta, RunCompleted, RunFailed]
