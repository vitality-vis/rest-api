"""
Dynamic tool registry for the planner agent.

Each tool registers a ToolDescriptor here once (at module load time).
The PlannerAgent reads from this registry at runtime to build its tool catalog —
no hardcoded tool knowledge lives in the planner prompt.

Adding a new tool:
  1. Call get_registry().register(ToolDescriptor(...)) in agent_tools.py.
  2. Done.  Zero changes to the planner.

Test isolation:
  Use set_registry(ToolRegistry()) at the start of each test so state does
  not bleed across tests.  Never import get_registry() at module level and
  cache the result — always call get_registry() at use time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


# ── Runtime type map ──────────────────────────────────────────────────────────

_TYPE_MAP: Dict[str, Type] = {
    "str":   str,
    "dict":  dict,
    "list":  list,
    "int":   int,
    "float": float,
    "bool":  bool,
}

# Matches inter-step reference tokens like {{step_2.abstract}}
_REF_PATTERN = re.compile(r"\{\{step_\d+\.\w+\}\}")


# ── Per-argument spec ─────────────────────────────────────────────────────────

@dataclass
class ArgSpec:
    type:        str            # must be a key in _TYPE_MAP
    description: str
    required:    bool = True
    default:     Any  = None


# ── Tool self-description ─────────────────────────────────────────────────────

@dataclass
class ToolDescriptor:
    name:        str
    description: str                         # when and why to use this tool
    args:        Dict[str, ArgSpec]          # arg_name → ArgSpec
    output:      str                         # what the tool returns + referenceable fields
    constraints: List[str] = field(default_factory=list)  # "do NOT use when …"


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Owns all registered ToolDescriptors.

    The PlannerAgent calls get_catalog_text() to build its prompt at runtime,
    and validate_plan() to check each step before handing off to the executor.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, descriptor: ToolDescriptor) -> None:
        """
        Register a tool descriptor.
        Raises ValueError on duplicate names to prevent silent overwrites
        from accidental double-imports.
        """
        if descriptor.name in self._tools:
            raise ValueError(
                f"Tool '{descriptor.name}' is already registered. "
                "Use force_register() if you intentionally want to replace it "
                "(e.g. in tests or hot-reload scenarios)."
            )
        self._tools[descriptor.name] = descriptor

    def force_register(self, descriptor: ToolDescriptor) -> None:
        """Unconditionally register or replace — use only in tests / hot-reload."""
        self._tools[descriptor.name] = descriptor

    def get(self, name: str) -> Optional[ToolDescriptor]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    # ── Dynamic catalog ───────────────────────────────────────────────────────

    def get_catalog_text(self, tools: Optional[List[str]] = None) -> str:
        """
        Build the tool catalog string injected into the planner prompt.
        Fully derived from registered descriptors — no hardcoding needed.

        Args:
            tools: optional allowlist of tool names to include. When None,
                   all registered tools are rendered. Use this to give the
                   planner a focused subset based on intent (e.g. search-only
                   tools for SEARCH_PAPER intent).
        """
        lines: List[str] = []
        descriptors = (
            {n: td for n, td in self._tools.items() if n in tools}
            if tools is not None
            else self._tools
        )
        for td in descriptors.values():
            sig = ", ".join(
                f"{n}: {s.type}{'?' if not s.required else ''}"
                for n, s in td.args.items()
            )
            lines.append(f"{td.name}({sig})")
            lines.append(f"  Description: {td.description}")

            if td.args:
                lines.append("  Args:")
                for n, s in td.args.items():
                    req_str = "required" if s.required else f"optional (default={s.default!r})"
                    lines.append(f"    {n} ({s.type}, {req_str}): {s.description}")

            lines.append(f"  Returns: {td.output}")

            if td.constraints:
                lines.append("  Constraints (you MUST respect these when selecting this tool):")
                for c in td.constraints:
                    lines.append(f"    - {c}")

            lines.append("")
        return "\n".join(lines).rstrip()

    # ── Plan validation ───────────────────────────────────────────────────────

    def validate_step(self, step: dict) -> Optional[str]:
        """
        Validate one plan step.
        Returns an error string if invalid, None if acceptable.

        Checks:
        - 'tool' field exists and is a registered tool name
        - All required args are present
        - Arg values match declared types
          (inter-step references {{step_N.field}} bypass type-checking because
           they are resolved at execution time)
        """
        tool_name = step.get("tool", "")
        if not tool_name:
            return "step is missing the 'tool' field"

        td = self._tools.get(tool_name)
        if td is None:
            known = ", ".join(self._tools) or "(none registered)"
            return f"unknown tool '{tool_name}'; known tools: [{known}]"

        args = step.get("args") or {}

        for arg_name, spec in td.args.items():
            value = args.get(arg_name)

            if value is None:
                if spec.required:
                    return (
                        f"tool '{tool_name}': required arg '{arg_name}' "
                        f"({spec.type}) is missing"
                    )
                continue

            # Inter-step references are resolved at runtime — skip type check
            if isinstance(value, str) and _REF_PATTERN.search(value):
                continue

            expected_type = _TYPE_MAP.get(spec.type)
            if expected_type and not isinstance(value, expected_type):
                return (
                    f"tool '{tool_name}': arg '{arg_name}' expected {spec.type} "
                    f"but got {type(value).__name__}"
                )

        return None

    def validate_plan(self, steps: List[dict]) -> List[str]:
        """Validate all steps; return list of error strings (empty = all valid)."""
        errors: List[str] = []
        for i, step in enumerate(steps, start=1):
            err = self.validate_step(step)
            if err:
                errors.append(f"step {i}: {err}")
        return errors


# ── Singleton management ──────────────────────────────────────────────────────
# Never import _registry at module level and cache the result.
# Always call get_registry() at use time so set_registry() takes effect.

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Return the process-level registry, creating it on first call."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def set_registry(r: ToolRegistry) -> None:
    """
    Replace the process-level registry.
    For tests only — lets each test start with an isolated, empty registry:

        def test_something():
            set_registry(ToolRegistry())
            get_registry().register(ToolDescriptor(...))
            ...
    """
    global _registry
    _registry = r
