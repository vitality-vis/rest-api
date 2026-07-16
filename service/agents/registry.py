"""
Per-conversation agent registry.

Aligned with Codex codex-rs/core/src/agent/registry.rs:
  - Tracks every agent spawned for a conversation by its path string
  - Routes messages between agents via their mailboxes
  - Exposes status for all registered agents

Agent paths follow a hierarchical convention (same as Codex AgentPath):
  /root                  — orchestrator (the main reasoning agent)
  /root/retriever        — document retrieval worker
  /root/assessor         — RAG quality assessor (self-RAG)
  /root/refiner          — query refinement worker (self-RAG phase 2)

Two-layer design
----------------
AgentRegistryBase   : ABC that defines the interface.
                      Swap implementations (e.g. distributed registry)
                      by giving the orchestrator a different concrete class
                      without changing any orchestrator code.

AgentRegistry       : Concrete single-process synchronous implementation.
                      One instance per chat session.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .mailbox import AgentMessage, Mailbox
from .status import AgentStatus

if TYPE_CHECKING:
    from .base_agent import BaseAgent


# ── Abstract interface ────────────────────────────────────────────────────────

class AgentRegistryBase(ABC):
    """
    Minimal interface every registry implementation must satisfy.

    The orchestrator types its _registry field as AgentRegistryBase so that
    the concrete implementation can be swapped without touching the orchestrator.

    Well-known path constants are defined here so they remain consistent
    across implementations.
    """

    # Path constants — mirrors Codex's AgentPath::root() convention
    ROOT      = "/root"
    RETRIEVER = "/root/retriever"
    ASSESSOR  = "/root/assessor"
    REFINER   = "/root/refiner"
    PLANNER   = "/root/planner"
    # Research task agents (Planner → ResearchTask → Agent layer)
    TASK_COLLECT_PAPERS = "/root/task/collect_papers"

    @abstractmethod
    def register(self, path: str, agent: "BaseAgent") -> Mailbox:
        """Register an agent and return its assigned mailbox."""
        ...

    @abstractmethod
    def register_mailbox(self, path: str) -> Mailbox:
        """Register a bare mailbox (no agent) at path."""
        ...

    @abstractmethod
    def send(
        self,
        author: str,
        recipient: str,
        content: dict,
        trigger_turn: bool = True,
    ) -> bool:
        """Build and route an AgentMessage. Returns True if recipient exists."""
        ...

    @abstractmethod
    def drain_inbox(self, path: str) -> list:
        """Return and clear all pending messages for the agent at path."""
        ...

    @abstractmethod
    def get_agent(self, path: str) -> Optional["BaseAgent"]:
        """Return the agent registered at path, or None."""
        ...

    @abstractmethod
    def set_status(self, path: str, status: AgentStatus) -> None:
        """Record a status update for the agent at path."""
        ...


# ── Concrete implementation ───────────────────────────────────────────────────

class AgentRegistry(AgentRegistryBase):
    """
    Lightweight in-process registry for one chat session.

    Aligns with Codex's AgentRegistry + ThreadManagerState pair; simplified to
    a single class because our sessions are single-threaded asyncio tasks.
    """

    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self._agents:    Dict[str, "BaseAgent"] = {}
        self._statuses:  Dict[str, AgentStatus] = {}
        self._mailboxes: Dict[str, Mailbox]     = {}

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, path: str, agent: "BaseAgent") -> Mailbox:
        """
        Register an agent at *path* and return the mailbox the agent should
        read from.  Idempotent: re-registering the same path replaces the
        previous entry.

        Aligns with AgentControl::spawn_agent() which records the new thread
        in the registry before starting its event loop.
        """
        mailbox = Mailbox()
        agent.path    = path
        agent.mailbox = mailbox
        self._agents[path]    = agent
        self._statuses[path]  = AgentStatus.PENDING_INIT
        self._mailboxes[path] = mailbox
        return mailbox

    def register_mailbox(self, path: str) -> Mailbox:
        """
        Register a bare mailbox at *path* without an associated agent.

        Use this for the root orchestrator ("/root") which is not an agent
        object but needs a mailbox so sub-agents can route replies to it.
        Idempotent: returns the existing mailbox if already registered.
        """
        if path not in self._mailboxes:
            self._mailboxes[path] = Mailbox()
        return self._mailboxes[path]

    def unregister(self, path: str) -> None:
        self._agents.pop(path, None)
        self._statuses.pop(path, None)
        self._mailboxes.pop(path, None)

    # ── Routing ────────────────────────────────────────────────────────────────

    def route(self, message: AgentMessage) -> bool:
        """
        Deliver *message* to the recipient's mailbox.
        Returns True if the recipient exists, False otherwise.
        """
        mailbox = self._mailboxes.get(message.recipient)
        if mailbox is None:
            return False
        mailbox.send(message)
        return True

    def send(
        self,
        author: str,
        recipient: str,
        content: dict,
        trigger_turn: bool = True,
    ) -> bool:
        """Convenience wrapper: build an AgentMessage and route it."""
        return self.route(AgentMessage(
            author=author,
            recipient=recipient,
            content=content,
            trigger_turn=trigger_turn,
        ))

    # ── Status ─────────────────────────────────────────────────────────────────

    def set_status(self, path: str, status: AgentStatus) -> None:
        self._statuses[path] = status

    def get_status(self, path: str) -> Optional[AgentStatus]:
        return self._statuses.get(path)

    def all_statuses(self) -> List[Tuple[str, AgentStatus]]:
        return list(self._statuses.items())

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_agent(self, path: str) -> Optional["BaseAgent"]:
        return self._agents.get(path)

    def get_mailbox(self, path: str) -> Optional[Mailbox]:
        return self._mailboxes.get(path)

    def drain_inbox(self, path: str) -> list:
        """Drain all pending messages for the agent at *path*."""
        mailbox = self._mailboxes.get(path)
        return mailbox.drain() if mailbox else []
