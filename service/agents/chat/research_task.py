"""
ResearchTask layer — the central abstraction for the planner→agent pipeline.

A ResearchTask names *what* to accomplish (e.g. "collect_papers"), never *how*
(it must never name a tool like "semantic_search"). The "how" lives inside the
specialized agent that handles the task. This mirrors the existing
PlanStep/ExecutionPlan shape but at the task level.

Pure, JSON-serializable data — consistent with WorkflowState / OrchestratorAction
so a plan can be logged, replayed, or resumed.

NOTE: typed for Python 3.9 (Optional/List/Dict, not 3.10 union syntax).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ResearchTaskType(str, Enum):
    """Known research task types.

    The full set is enumerated for documentation/extensibility, but only the
    tasks present in the TaskCatalog are actually plannable/runnable. Today only
    COLLECT_PAPERS has a registered agent — the rest are placeholders for the
    incremental migration and are intentionally NOT wired yet.
    """

    COLLECT_PAPERS         = "collect_papers"
    COMPARE_METHODS        = "compare_methods"
    TIMELINE_ANALYSIS      = "timeline_analysis"
    TAXONOMY_ANALYSIS      = "taxonomy_analysis"
    IDENTIFY_LIMITATIONS   = "identify_limitations"
    GENERATE_RESEARCH_GAPS = "generate_research_gaps"
    GENERATE_SYNTHESIS     = "generate_synthesis"

    @classmethod
    def from_value(cls, value: Any) -> "ResearchTaskType":
        """Coerce a raw string into a ResearchTaskType, or raise ValueError."""
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass
class ResearchTask:
    """A single planned research task.

    Fields
    ------
    type       : the kind of task (never a tool name)
    params     : task inputs, e.g. {"topic": "GraphRAG"} — NOT tool args
    depends_on : indices (1-based) of prior tasks whose output this consumes
    """

    type: ResearchTaskType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """An ordered list of research tasks (plus any planning errors).

    Mirrors ExecutionPlan's contract: an empty/errored plan signals the
    orchestrator to fall back to the deterministic intent→tool path.
    """

    tasks: List[ResearchTask] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def is_runnable(self) -> bool:
        return not self.errors and not self.is_empty()
