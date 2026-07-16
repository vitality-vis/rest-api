"""
ToolExecutor: centralized tool invocation with approval gate enforcement.

All explicit tool calls route through this class for consistent gate checking
and logging. The LangChain path has been removed; every tool call now goes
through execute() which returns a ToolExecutionResult.

One ToolExecutor instance is created per chat session.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STEP_REF = re.compile(r"\{\{step_(\d+)\.(\w+)\}\}")


@dataclass
class ToolExecutionResult:
    """Structured result from a single tool invocation."""
    tool_name: str
    output: str                   # formatted string returned by the tool
    docs: List[Any]               # raw Document objects extracted from session
    status: str                   # "ok" | "blocked" | "error" | "unknown_tool"
    error: Optional[str] = None

    def is_ok(self) -> bool:
        return self.status == "ok"


class ToolExecutor:
    """
    Executes individual tool calls and sequential plan steps,
    routing every invocation through the ToolApprovalGate.
    """

    def __init__(self, gate, chat_id: str) -> None:
        self._gate    = gate
        self._chat_id = chat_id
        self._tool_fns: Dict[str, callable] = self._build_tool_map()

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: dict) -> ToolExecutionResult:
        """
        Execute one tool call with gate check.

        Resets the session's _turn_docs buffer before calling so the result
        contains only the docs produced by this specific invocation.
        Returns a ToolExecutionResult with tool output and extracted raw docs.
        """
        from service.application.pipeline import ApprovalDecision
        from service.memory.session_state import get_session

        fn = self._tool_fns.get(tool_name)
        if not fn:
            logger.warning("[ToolExecutor] Unknown tool: %s", tool_name)
            return ToolExecutionResult(
                tool_name=tool_name, output="", docs=[],
                status="unknown_tool", error=f"Unknown tool: {tool_name}",
            )

        if self._gate:
            ar = self._gate.evaluate(tool_name, args)
            if ar.decision == ApprovalDecision.DENY:
                logger.warning("[ToolExecutor] %s blocked: %s", tool_name, ar.reason)
                return ToolExecutionResult(
                    tool_name=tool_name, output=f"_(Tool blocked: {ar.reason})_",
                    docs=[], status="blocked", error=ar.reason,
                )

        # Reset turn buffer so we capture only this call's docs
        sess = get_session(self._chat_id) or {}
        sess["_turn_docs"] = []

        try:
            output = fn(args)
            if output is None:
                output = ""
            docs = self._read_tool_docs(sess)
            return ToolExecutionResult(
                tool_name=tool_name, output=output, docs=docs, status="ok",
            )
        except Exception as exc:
            logger.warning("[ToolExecutor] %s failed: %s", tool_name, exc)
            return ToolExecutionResult(
                tool_name=tool_name, output="", docs=[],
                status="error", error=str(exc),
            )

    def execute_plan(self, steps: list) -> str:
        """
        Execute a list of plan steps sequentially.

        Resolves {{step_N.field}} inter-step references using the output of
        previous steps plus the first cached paper's title/abstract fields.
        Returns the last non-empty step output.
        """
        step_outputs: Dict[int, dict] = {}
        last_output = ""

        for i, step in enumerate(steps, start=1):
            tool_name     = step.get("tool", "")
            resolved_args = self._resolve_args(step.get("args", {}), step_outputs)

            logger.info(
                "[ToolExecutor] step %d: %s(%s) — %s",
                i, tool_name, list(resolved_args), step.get("purpose", ""),
            )

            result = self.execute(tool_name, resolved_args)
            if not result.is_ok() or not result.output:
                continue

            paper_fields = self._peek_paper_fields()
            step_outputs[i] = {"text": result.output, **paper_fields}
            last_output = result.output

        return last_output

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tool_map(self) -> Dict[str, callable]:
        """Build the name → callable map once at construction time."""
        from service.tools.agent_tools import (
            metadata_search, semantic_search, mixed_search,
            rag_semantic_qa, load_more_papers,
        )
        cid = self._chat_id
        return {
            "metadata_search": lambda args: metadata_search.func(
                filters=args.get("filters", {}), user_request="", chat_id=cid,
            ),
            "semantic_search": lambda args: semantic_search.func(
                query=str(args.get("query", "")), chat_id=cid,
            ),
            "mixed_search": lambda args: mixed_search.func(
                query_text=str(args.get("query_text", "")),
                filters=args.get("filters", {}),
                chat_id=cid,
            ),
            "rag_semantic_qa": lambda args: rag_semantic_qa.func(
                query=str(args.get("query", "")),
                question=str(args.get("question", "")),
                chat_id=cid,
            ),
            "load_more_papers": lambda _args: load_more_papers.func(chat_id=cid),
        }

    def _read_tool_docs(self, sess: dict) -> List[Any]:
        """
        Extract raw docs produced by the last tool call.

        Priority:
          1. session["_turn_docs"] — set by save_session_docs (metadata/mixed search)
          2. session["search_cache"] — set by semantic_search / multi_source search
        """
        turn_docs = list(sess.get("_turn_docs", []))
        if turn_docs:
            return turn_docs
        return list(sess.get("search_cache", []))

    def _resolve_args(self, args: dict, step_outputs: Dict[int, dict]) -> dict:
        return {k: self._resolve_value(v, step_outputs) for k, v in args.items()}

    def _resolve_value(self, value, step_outputs: Dict[int, dict]):
        if not isinstance(value, str):
            return value
        return _STEP_REF.sub(
            lambda m: str(step_outputs.get(int(m.group(1)), {}).get(m.group(2), "")),
            value,
        )

    def _peek_paper_fields(self) -> dict:
        """Extract title/abstract from the first cached paper for step reference resolution."""
        from service.memory.session_state import get_session
        sess = get_session(self._chat_id) or {}
        cached = sess.get("search_cache", [])
        if not cached:
            return {}
        first = cached[0]
        if hasattr(first, "metadata"):
            return {
                "title":    first.metadata.get("title", ""),
                "abstract": first.metadata.get("abstract", ""),
            }
        if isinstance(first, dict):
            return {
                "title":    first.get("Title", "") or first.get("title", ""),
                "abstract": first.get("Abstract", "") or first.get("abstract", ""),
            }
        return {}
