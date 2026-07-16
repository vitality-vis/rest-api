"""
Async mailbox for inter-agent communication.

Aligned with Codex codex-rs/core/src/agent/mailbox.rs:
  - Unbounded message queue (asyncio.Queue ↔ mpsc::unbounded_channel)
  - Monotonically-increasing sequence numbers (itertools.count ↔ AtomicU64)
  - trigger_turn flag: recipient should start a new reasoning turn
  - drain() returns messages in delivery order (FIFO)
"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentMessage:
    """
    One inter-agent communication unit.

    Aligns with Codex's InterAgentCommunication protocol type.

    Fields
    ------
    author       : sender's agent path  (e.g. "/root")
    recipient    : target agent path    (e.g. "/root/retriever")
    content      : arbitrary payload dict  (type-tagged with "type" key)
    trigger_turn : if True the recipient should immediately start a new turn
    seq          : assigned by the Mailbox — do not set manually
    """
    author:       str
    recipient:    str
    content:      Dict[str, Any]
    trigger_turn: bool = True
    seq:          int  = field(default=0, compare=False)


class Mailbox:
    """
    Per-agent async-safe unbounded message queue.

    Aligns with Codex's Mailbox + MailboxReceiver pair.
    The separation into send / receive halves is kept logical (single class)
    because Python async tasks don't need the split-ownership that Rust requires.

    Usage
    -----
    mailbox = Mailbox()
    seq = mailbox.send(AgentMessage(...))   # enqueue; returns seq number
    msgs = mailbox.drain()                  # dequeue all pending
    """

    def __init__(self) -> None:
        self._pending: deque[AgentMessage] = deque()
        self._counter = itertools.count(1)

    # ── Sender side ────────────────────────────────────────────────────────────

    def send(self, message: AgentMessage) -> int:
        """Enqueue a message. Returns the monotonically-increasing sequence number."""
        seq = next(self._counter)
        message.seq = seq
        self._pending.append(message)
        return seq

    # ── Receiver side ──────────────────────────────────────────────────────────

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def has_pending_trigger_turn(self) -> bool:
        """True if any queued message has trigger_turn=True."""
        return any(m.trigger_turn for m in self._pending)

    def drain(self) -> List[AgentMessage]:
        """Return all pending messages in delivery order and clear the queue."""
        messages = list(self._pending)
        self._pending.clear()
        return messages

    def peek(self) -> Optional[AgentMessage]:
        """Return the next message without removing it."""
        return self._pending[0] if self._pending else None
