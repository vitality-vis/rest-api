"""
Abstract base class for all agents.

Aligns with Codex's session/turn.rs run_turn() contract:
  - Each agent has a path in the registry
  - Each agent owns a mailbox it reads from
  - Each agent exposes a run() async generator that yields text chunks
  - Status transitions mirror Codex's AgentStatus watch channel updates

Sub-classes implement run() for their specific reasoning/execution logic.
The orchestrator coordinates sub-agents by:
  1. registry.send(author, recipient, content)   — put work in sub-agent mailbox
  2. await sub_agent.run_from_mailbox()          — execute and drain results
  3. registry.drain_inbox("/root")               — read replies
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from .mailbox import Mailbox
from .status import AgentStatus

if TYPE_CHECKING:
    from .registry import AgentRegistry


class BaseAgent(ABC):
    """
    Abstract base for every agent in the multi-agent system.

    Attributes
    ----------
    path     : hierarchical string path set by the registry on registration
    chat_id  : conversation this agent belongs to
    registry : shared registry for routing messages to other agents
    mailbox  : receive-side mailbox; set by registry.register()
    status   : current lifecycle state
    """

    def __init__(self, chat_id: str, registry: "AgentRegistry") -> None:
        self.path:     str             = ""          # filled by registry.register()
        self.chat_id:  str             = chat_id
        self.registry: "AgentRegistry" = registry
        self.mailbox:  Optional[Mailbox] = None      # filled by registry.register()
        self.status:   AgentStatus     = AgentStatus.PENDING_INIT

    # ── Lifecycle helpers ─────────────────────────────────────────────────────

    def set_status(self, status: AgentStatus) -> None:
        """Update own status and mirror it to the registry."""
        self.status = status
        self.registry.set_status(self.path, status)

    # ── Messaging helpers ─────────────────────────────────────────────────────

    def send(self, recipient: str, content: dict, trigger_turn: bool = True) -> bool:
        """Send a message to another agent via the registry router."""
        return self.registry.send(
            author=self.path,
            recipient=recipient,
            content=content,
            trigger_turn=trigger_turn,
        )

    def drain_inbox(self) -> list:
        """Return and clear all pending messages in this agent's mailbox."""
        return self.mailbox.drain() if self.mailbox else []

    def has_pending(self) -> bool:
        return bool(self.mailbox and self.mailbox.has_pending())

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, input_data: Any) -> AsyncGenerator[str, None]:
        """
        Main execution entry point.

        Aligns with Codex's run_turn() async fn.

        Yields text chunks for streaming to the user.
        Sub-classes that don't produce user-visible output should yield nothing
        and instead route results back to the orchestrator via self.send().
        """
        ...
        yield  # make the type checker happy — sub-classes must yield

    async def run_from_mailbox(self) -> None:
        """
        Convenience: drain the mailbox and run on each pending message.

        Used by the orchestrator after routing work to a worker agent:
            registry.send("/root", "/root/assessor", {...})
            await assessor.run_from_mailbox()
            results = registry.drain_inbox("/root")
        """
        for msg in self.drain_inbox():
            async for _ in self.run(msg.content):
                pass  # worker agents don't yield user-visible text
