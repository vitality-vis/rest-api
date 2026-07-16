"""
Tool execution approval gate.

Aligned with Codex codex-rs/core/src/tools/orchestrator.rs + guardian/.

Every tool call passes through this gate before executing.  The gate decides:

  ALLOW   — execute immediately (all current read-only search tools)
  DENY    — block execution and return an error string
  PROMPT  — ask the user for approval  (Phase 2; currently falls back to ALLOW)

How it fits into application/agent_service.py
---------------------------------------------
gate = ToolApprovalGate()

# In each tool factory:
def _make_semantic_fn(orig, cid, gate):
    def fn(query: str):
        ar = gate.evaluate("semantic_search", {"query": query})
        if ar.decision == ApprovalDecision.DENY:
            return f"_(Tool blocked: {ar.reason})_"
        return orig(query=query, chat_id=cid)
    return fn

# The gate instance is stored in the session so future tools can read its policy:
session["approval_gate"] = gate

Extending (Phase 2 — writable/export tools)
--------------------------------------------
1. Add the new tool name to _NEEDS_APPROVAL.
2. In the tool factory, check ar.decision == ApprovalDecision.PROMPT and route a
   [SIGNAL:TOOL_APPROVAL:{...}] to the frontend (same pattern as QUERY_EXPANSION).
3. Resume the turn once the user approves — store the pending call in the session
   (session["pending_tool_call"]) and pop it after approval.

This mirrors Codex's GuardianReviewSession which holds the pending approval
request and resumes the tool call after the user's decision arrives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── Decision types ────────────────────────────────────────────────────────────

class ApprovalDecision(str, Enum):
    ALLOW  = "allow"    # execute immediately
    DENY   = "deny"     # block; return error to agent
    PROMPT = "prompt"   # ask user (Phase 2 — currently treated as ALLOW)


@dataclass
class ApprovalResult:
    decision: ApprovalDecision
    reason:   Optional[str] = None


# ── Gate ─────────────────────────────────────────────────────────────────────

class ToolApprovalGate:
    """
    Synchronous approval gate called inside each tool wrapper.

    Aligns with Codex's ToolOrchestrator three-phase model:
      Phase 1 — Approval check        (this class)
      Phase 2 — Sandbox selection     (N/A for search tools; reserved)
      Phase 3 — Execute with retry    (handled by LangChain AgentExecutor)

    All current tools (read-only academic search) are in _ALWAYS_ALLOW.
    The structure is the hook point for when riskier tools are added.
    """

    # ── Policy sets ──────────────────────────────────────────────────────────

    # Read-only retrieval — always safe to execute without asking
    _ALWAYS_ALLOW: frozenset = frozenset({
        "semantic_search",
        "metadata_search",
        "mixed_search",
        "load_more_papers",
        "rag_semantic_qa",
    })

    # Future write/export tools that should ask the user first (Phase 2)
    # Example: "export_to_csv", "save_to_library", "send_summary_email"
    _NEEDS_APPROVAL: frozenset = frozenset()

    # Tools that are always blocked regardless of context
    _DENY_LIST: frozenset = frozenset()

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, tool_name: str, tool_args: dict) -> ApprovalResult:
        """
        Decide whether tool_name with tool_args may execute.
        Called synchronously inside each tool wrapper closure.
        """
        if tool_name in self._DENY_LIST:
            logger.warning("[ToolApprovalGate] DENY tool=%s (deny-list)", tool_name)
            return ApprovalResult(
                decision=ApprovalDecision.DENY,
                reason=f"Tool '{tool_name}' is not permitted in this context.",
            )

        if tool_name in self._ALWAYS_ALLOW:
            return ApprovalResult(decision=ApprovalDecision.ALLOW)

        if tool_name in self._NEEDS_APPROVAL:
            # Phase 2: route to frontend approval signal and pause the turn.
            # For now, log and allow so the system keeps working.
            logger.info(
                "[ToolApprovalGate] PROMPT for tool=%s args=%s (Phase 2 pending — allowing)",
                tool_name, list(tool_args.keys()),
            )
            return ApprovalResult(decision=ApprovalDecision.ALLOW, reason="approval_pending")

        # Unknown tool — deny by default (fail-safe)
        logger.warning("[ToolApprovalGate] DENY unknown tool=%s", tool_name)
        return ApprovalResult(
            decision=ApprovalDecision.DENY,
            reason=f"Tool '{tool_name}' is not registered in the approval gate.",
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    def wrap(self, tool_name: str, fn: Callable) -> Callable:
        """
        Return a gated version of fn: checks approval before delegating.

        Usage in a factory:
            gated_fn = gate.wrap("semantic_search", raw_fn)
        """
        gate = self

        def gated(*args, **kwargs):
            result = gate.evaluate(tool_name, kwargs)
            if result.decision == ApprovalDecision.DENY:
                return f"_(Tool blocked: {result.reason})_"
            return fn(*args, **kwargs)

        return gated


__all__ = ["ApprovalDecision", "ApprovalResult", "ToolApprovalGate"]
