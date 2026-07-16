"""Checkpoint-driven session manager for the LR human-in-the-loop graph.

This is the LR *checkpoint protocol*. It is intentionally SEPARATE from the
existing chat tool-approval backend: it does not share approval IDs, approval
gate logic, or any chat/RAG state. It only drives the LR LangGraph thread.

Flow
----
The LR graph pauses via ``interrupt()`` at the plan and retrieval nodes. Each
pause is surfaced to the frontend as a structured *checkpoint* that the user
must explicitly act on. Only an explicit action resumes the graph:

    POST /lrChat {chat_id, message}                      -> start, returns checkpoint
    POST /lrChat {chat_id, checkpoint_id, action, ...}   -> resume, returns next

A plain message sent while a checkpoint is pending does NOT auto-continue the
graph; the user must press an action on the card.

Response shapes
---------------
    {"type": "lr_checkpoint", "checkpoint_id", "checkpoint_type", "title",
     "content", "payload", "actions": [{"id","label"}, ...]}
    {"type": "lr_message", "content", ...}
    {"type": "lr_final", "checkpoint_type": "draft_review", "title",
     "content", "payload", "actions": []}

State is keyed by ``chat_id`` and held in-process (per the graph's
``InMemorySaver``); it clears when a run finishes, is rejected, or on restart.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import Command

from service.agents.lr.graph import build_lr_graph

logger = logging.getLogger("service.agents.lr")

# One compiled graph (with its InMemorySaver); per-chat ``thread_id`` isolates
# each conversation's state.
_graph = None
_graph_lock = threading.Lock()

# chat_id -> {"thread_id", "phase", "checkpoint_id", "checkpoint_type"}
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

_PHASE_RUNNING = "running"
_PHASE_AWAITING = "awaiting_checkpoint"

# Action buttons offered for each checkpoint type. These are LR-specific and do
# not reuse any existing chat approval action set.
_ACTIONS_BY_TYPE: Dict[str, List[Dict[str, str]]] = {
    "plan_review": [
        {"id": "approve", "label": "Approve"},
        {"id": "edit", "label": "Edit"},
        {"id": "reject", "label": "Reject"},
    ],
    "retrieval_review": [
        {"id": "accept", "label": "Accept"},
        {"id": "refine_search", "label": "Refine Search"},
        {"id": "reject", "label": "Reject"},
    ],
}
_DEFAULT_ACTIONS: List[Dict[str, str]] = [
    {"id": "approve", "label": "Approve"},
    {"id": "reject", "label": "Reject"},
]


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_lr_graph()
    return _graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_interrupt(result: Any) -> Optional[dict]:
    """Pull the first interrupt payload out of an invoke() result, if any."""
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def reset_session(chat_id: str) -> None:
    """Forget any in-progress LR run for this chat."""
    with _sessions_lock:
        _sessions.pop(chat_id, None)


def _message(content: str, **extra: Any) -> dict:
    return {"type": "lr_message", "content": content, **extra}


def _render_content(payload: dict) -> str:
    """Build a human-readable markdown summary for a checkpoint payload."""
    ctype = payload.get("checkpoint_type")
    if ctype == "plan_review":
        lines = [
            f"{s.get('stage')}. **{s.get('title')}** — {s.get('description')}"
            for s in payload.get("plan", [])
        ]
        goal = payload.get("user_goal")
        head = f"**Goal:** {goal}\n\n" if goal else ""
        return f"{head}**Proposed plan**\n" + "\n".join(lines)
    if ctype == "retrieval_review":
        blocks = []
        for p in payload.get("retrieved_papers", []):
            authors = p.get("Authors")
            authors_str = ", ".join(authors) if isinstance(authors, list) else (authors or "")
            blocks.append(
                f"- **{p.get('Title')}** ({p.get('Year')})\n  {authors_str}\n  {p.get('Abstract', '')}"
            )
        return "**Retrieved papers**\n" + "\n\n".join(blocks)
    return payload.get("message", "")


def _build_checkpoint(chat_id: str, payload: dict) -> dict:
    checkpoint_id = uuid.uuid4().hex
    checkpoint_type = payload.get("checkpoint_type", "stage_review")
    with _sessions_lock:
        sess = _sessions.get(chat_id)
        if sess is not None:
            sess["phase"] = _PHASE_AWAITING
            sess["checkpoint_id"] = checkpoint_id
            sess["checkpoint_type"] = checkpoint_type
    logger.info(
        "LR checkpoint created | chat_id=%s checkpoint_id=%s type=%s",
        chat_id, checkpoint_id, checkpoint_type,
    )
    return {
        "type": "lr_checkpoint",
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": checkpoint_type,
        "title": payload.get("title", "Checkpoint"),
        "content": _render_content(payload),
        "payload": payload,
        "actions": _ACTIONS_BY_TYPE.get(checkpoint_type, _DEFAULT_ACTIONS),
    }


def _build_final(chat_id: str, state: dict) -> dict:
    reset_session(chat_id)
    draft = state.get("draft", "") if isinstance(state, dict) else ""
    logger.info(
        "LR final draft generated | chat_id=%s draft_chars=%d",
        chat_id, len(draft or ""),
    )
    return {
        "type": "lr_final",
        "checkpoint_id": None,
        "checkpoint_type": "draft_review",
        "title": "Literature review draft",
        "content": draft or "(no draft produced)",
        "payload": {
            "draft": draft,
            "analysis_result": state.get("analysis_result", {}) if isinstance(state, dict) else {},
            "user_goal": state.get("user_goal", "") if isinstance(state, dict) else "",
        },
        "actions": [],
    }


def _interpret(chat_id: str, result: Any) -> dict:
    payload = _extract_interrupt(result)
    if payload is not None:
        return _build_checkpoint(chat_id, payload)
    return _build_final(chat_id, result if isinstance(result, dict) else {})


def _snapshot(chat_id: str) -> Tuple[Optional[dict], bool, Optional[str], Optional[str]]:
    with _sessions_lock:
        sess = _sessions.get(chat_id)
        awaiting = bool(sess and sess.get("phase") == _PHASE_AWAITING)
        active_cp = sess.get("checkpoint_id") if sess else None
        thread_id = sess.get("thread_id") if sess else None
    return sess, awaiting, active_cp, thread_id


def lr_handle(
    chat_id: str,
    message: str = "",
    checkpoint_id: Optional[str] = None,
    action: Optional[str] = None,
    feedback: Optional[str] = None,
) -> dict:
    """Advance the LR workflow for ``chat_id``.

    Either starts a new run (no pending checkpoint) using ``message`` as the
    goal, or resumes a pending checkpoint when an explicit ``action`` is given.
    Raises on graph errors so the route can log the real traceback.
    """
    chat_id = (chat_id or "lr-default").strip() or "lr-default"
    graph = _get_graph()

    _sess, awaiting, active_cp, thread_id = _snapshot(chat_id)
    action_norm = (action or "").strip().lower() or None

    if awaiting:
        # Reject drops the thread entirely; we never resume the graph.
        if action_norm == "reject":
            reset_session(chat_id)
            logger.info(
                "LR checkpoint resumed | chat_id=%s checkpoint_id=%s action=reject (cancelled)",
                chat_id, active_cp,
            )
            return _message(
                "Literature-review workflow cancelled. Send a new goal to start over."
            )

        # Do NOT auto-continue on a plain message while a checkpoint is pending.
        if not action_norm:
            logger.info(
                "LR plain message ignored (checkpoint pending) | chat_id=%s checkpoint_id=%s",
                chat_id, active_cp,
            )
            return _message(
                "There's a pending checkpoint. Use an action on the card above "
                "(e.g. Approve / Edit / Reject) to continue."
            )

        if checkpoint_id and active_cp and checkpoint_id != active_cp:
            logger.info(
                "LR stale checkpoint action | chat_id=%s sent=%s active=%s",
                chat_id, checkpoint_id, active_cp,
            )
            return _message("This checkpoint is no longer active. Please act on the latest one.")

        resume_value = {"action": action_norm, "feedback": (feedback or message or "").strip()}
        logger.info(
            "LR checkpoint resumed | chat_id=%s checkpoint_id=%s action=%s",
            chat_id, active_cp, action_norm,
        )
        result = graph.invoke(Command(resume=resume_value), config=_config(thread_id))
        return _interpret(chat_id, result)

    # No pending checkpoint -> start a fresh run; the message is the goal.
    goal = (message or "").strip()
    if not goal:
        return _message("Please describe your literature-review goal to begin.")

    thread_id = f"lr-{chat_id}-{uuid.uuid4().hex[:8]}"
    with _sessions_lock:
        _sessions[chat_id] = {"thread_id": thread_id, "phase": _PHASE_RUNNING}
    logger.info(
        "LR graph start | chat_id=%s thread_id=%s goal=%r",
        chat_id, thread_id, goal[:160],
    )
    try:
        result = graph.invoke({"user_goal": goal}, config=_config(thread_id))
    except Exception:
        reset_session(chat_id)
        raise
    return _interpret(chat_id, result)
