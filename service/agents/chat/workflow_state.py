"""
WorkflowState: per-turn mutable container for orchestrator execution state.

Created fresh for every call to run_two_stage_rag_stream().
Passed by reference through pipeline stages; mutated in place.

All fields are plain Python types, making WorkflowState straightforwardly
serializable to JSON for async resumption, distributed continuation, or
turn-level replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkflowState:
    """Mutable state for one orchestrator turn."""

    # ── Request ───────────────────────────────────────────────────────────────
    chat_id: str
    user_input: str
    selected_paper_ids: List[str] = field(default_factory=list)
    selected_paper_titles: List[str] = field(default_factory=list)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    clean_input: str = ""    # after L2 rewrite (or trimmed original if rewrite skipped)
    agent_input: str = ""    # enriched prompt after gateway (may include INTENT_HINT)

    # ── Intent / classification ───────────────────────────────────────────────
    intent: Optional[Any] = None        # IntentResult from classify_intent()

    # ── Retrieval results ─────────────────────────────────────────────────────
    retrieved_docs: List[Any] = field(default_factory=list)   # raw docs from last tool call
    tool_outputs: List[str] = field(default_factory=list)     # formatted string per tool call

    # ── Self-RAG control ──────────────────────────────────────────────────────
    use_self_rag: bool = False
    max_rag_iter: int = 1
    rag_iter: int = 0                         # incremented on each refinement
    assessment: Optional[Dict] = None         # assessment_result dict from AssessorAgent
    refined_query: Optional[str] = None       # set when assessment is insufficient

    # ── Research task layer (Planner → ResearchTask → Agent) ──────────────────
    # Additive, optional. Populated only when research planning is enabled;
    # otherwise the deterministic intent→tool path runs unchanged.
    research_plan: Optional[Any] = None            # ResearchPlan from the planner
    research_plan_executed: bool = False           # TaskExecutor ran the plan

    # ── Loop control ──────────────────────────────────────────────────────────
    current_action: Optional[Any] = None      # OrchestratorAction being executed
    iteration: int = 0                        # outer loop counter
    done: bool = False                        # loop termination flag
    errors: List[str] = field(default_factory=list)

    # ── Output accumulation ───────────────────────────────────────────────────
    final_answer: str = ""

    # ── Internal pipeline stage flags (set by execute_action) ────────────────
    _rewrite_done: bool = False       # L2 rewrite was attempted
    _gateway_done: bool = False       # L4.x gateway was processed
    _gateway_signaled: bool = False   # gateway issued a SIGNAL (pause turn)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def has_selected_papers(self) -> bool:
        """True when the user pinned specific papers for this turn."""
        return bool(self.selected_paper_ids or self.selected_paper_titles)
