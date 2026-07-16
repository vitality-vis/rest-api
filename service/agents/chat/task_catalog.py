"""
TaskCatalog — the task-level analogue of ToolRegistry.

The ResearchPlannerAgent reads this catalog at runtime to build its prompt and
to validate the plan it produces. Only tasks present here are plannable, which
is precisely what enforces "the planner outputs tasks, never tools" and keeps
the planner from emitting task types that have no agent yet.

Today the catalog contains ONLY `collect_papers`. Adding a new task later =
register one TaskDescriptor here + register its agent + add one TaskExecutor
route. Kept deliberately lightweight (no dynamic plugin machinery).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .research_task import ResearchTaskType


@dataclass
class TaskDescriptor:
    """Self-description of a research task for the planner prompt."""

    type: ResearchTaskType
    description: str
    params_hint: str = ""          # human description of expected params
    produces: str = ""             # what downstream tasks can consume


class TaskCatalog:
    """Owns the registered TaskDescriptors and validates research plans."""

    def __init__(self) -> None:
        self._tasks: Dict[ResearchTaskType, TaskDescriptor] = {}

    def register(self, descriptor: TaskDescriptor) -> None:
        self._tasks[descriptor.type] = descriptor

    def get(self, t: ResearchTaskType) -> Optional[TaskDescriptor]:
        return self._tasks.get(t)

    def types(self) -> List[ResearchTaskType]:
        return list(self._tasks.keys())

    def get_catalog_text(self) -> str:
        """Render the catalog injected into the planner prompt."""
        lines: List[str] = []
        for td in self._tasks.values():
            lines.append(f"- {td.type.value}: {td.description}")
            if td.params_hint:
                lines.append(f"    params: {td.params_hint}")
            if td.produces:
                lines.append(f"    produces: {td.produces}")
        return "\n".join(lines).rstrip()

    def validate_plan(self, raw_tasks: List[dict]) -> List[str]:
        """Validate a raw [{task, params}] list; return error strings (empty=ok)."""
        errors: List[str] = []
        known = {t.value for t in self._tasks}
        for i, step in enumerate(raw_tasks, start=1):
            name = str(step.get("task", "")).strip().lower()
            if not name:
                errors.append(f"task {i}: missing 'task' field")
            elif name not in known:
                errors.append(
                    f"task {i}: unknown task '{name}'; known: [{', '.join(sorted(known))}]"
                )
        return errors


# ── Singleton ──────────────────────────────────────────────────────────────
# Mirrors ToolRegistry's get_registry() pattern. Always call get_task_catalog()
# at use time rather than caching the module-level object.

_catalog: Optional[TaskCatalog] = None


def _build_default_catalog() -> TaskCatalog:
    cat = TaskCatalog()
    # Only collect_papers is wired today. Other ResearchTaskTypes are
    # intentionally NOT registered until their agents exist.
    cat.register(
        TaskDescriptor(
            type=ResearchTaskType.COLLECT_PAPERS,
            description=(
                "Find and retrieve papers relevant to the user's request. Use this "
                "to gather candidate papers for any search or question-answering need."
            ),
            params_hint='{"topic": "<search topic or question>"}',
            produces="A formatted list of retrieved papers (used as evidence).",
        )
    )
    return cat


def get_task_catalog() -> TaskCatalog:
    global _catalog
    if _catalog is None:
        _catalog = _build_default_catalog()
    return _catalog


def set_task_catalog(cat: TaskCatalog) -> None:
    """Replace the process-level catalog (tests / hot-reload only)."""
    global _catalog
    _catalog = cat
