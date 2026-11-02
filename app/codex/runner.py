"""Chat runner backed by a local Codex app-server process."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from app.chat.events import AgentAction, PapersResult, TextDelta
from app.chat.run_control import RunControl
from app.codex.client import CodexClient, get_shared_client
from app.codex.threads import drop_thread_id, get_thread_id, put_thread_id
from app.codex.tools import DYNAMIC_TOOLS, execute_tool

logger = logging.getLogger(__name__)


@dataclass
class _ConversationLock:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


_conversation_locks_guard = threading.Lock()
_conversation_locks: dict[tuple[str, str], _ConversationLock] = {}


@asynccontextmanager
async def _serialize_conversation(
    user_id: object, chat_id: object, control: RunControl | None
):
    """Serialize one conversation across worker loops with abortable waiting."""
    key = (str(user_id or ""), str(chat_id or ""))
    with _conversation_locks_guard:
        entry = _conversation_locks.setdefault(key, _ConversationLock())
        entry.users += 1

    acquired = False
    try:
        while not acquired:
            if control is not None:
                control.raise_if_aborted()
            # A bounded acquire lets cancellation/deadlines release the executor
            # worker promptly and can never acquire later as an orphaned waiter.
            attempt = asyncio.get_running_loop().run_in_executor(
                None, entry.lock.acquire, True, 0.05
            )
            try:
                acquired = await asyncio.shield(attempt)
            except BaseException:
                # Shield keeps the bounded attempt observable. If it won the
                # race, release the lock before propagating task cancellation.
                if await attempt:
                    entry.lock.release()
                raise
        if control is not None:
            control.raise_if_aborted()
        yield
    finally:
        if acquired:
            entry.lock.release()
        _release_conversation_entry(key, entry)


def _release_conversation_entry(
    key: tuple[str, str], entry: _ConversationLock
) -> None:
    with _conversation_locks_guard:
        entry.users -= 1
        if entry.users == 0 and _conversation_locks.get(key) is entry:
            del _conversation_locks[key]

from app.codex.resources import load_developer_instructions

DEVELOPER_INSTRUCTIONS = load_developer_instructions()


async def run(request: Any, *, client: CodexClient | None = None) -> AsyncIterator[object]:
    """Run one serialized Codex turn on the conversation's thread."""
    control = request.control if isinstance(request.control, RunControl) else None
    async with _serialize_conversation(
        getattr(request, "user_id", None),
        getattr(request, "chat_id", None),
        control,
    ):
        async for event in _run_locked(request, client=client):
            yield event


async def _run_locked(
    request: Any, *, client: CodexClient | None = None
) -> AsyncIterator[object]:
    active = client or await get_shared_client()
    control = request.control if isinstance(request.control, RunControl) else None
    output: asyncio.Queue[object] = asyncio.Queue()
    candidates: dict[str, dict[str, object]] = {}
    presented_paper_ids: list[str] = []
    thread_id, resumed = await _open_thread(active, request)
    history = list(request.history or [])
    if not resumed and not history and getattr(request, "has_codex_thread", False):
        # A stale mapping caused request preparation to omit history. Recover it
        # before starting (and committing) the replacement thread.
        history = await asyncio.to_thread(_load_history_for_recovery, request)

    loop = asyncio.get_running_loop()

    def emit(event: object) -> None:
        loop.call_soon_threadsafe(output.put_nowait, event)

    async def handle_tool(params: dict[str, Any]) -> dict[str, object]:
        return await asyncio.to_thread(
            execute_tool,
            user_id=request.user_id,
            selected_paper_ids=list(request.selected_paper_ids or []),
            selected_paper_groups=dict(
                getattr(request, "selected_paper_groups", None) or {}
            ),
            effort=str(request.effort or "low"),
            call_id=str(params.get("callId") or params.get("itemId") or "tool"),
            tool=str(params.get("tool") or ""),
            arguments=params.get("arguments"),
            emit=emit,
            candidates=candidates,
            presented_paper_ids=presented_paper_ids,
        )

    active.listen(thread_id, output.put_nowait)
    active.handle_tool(thread_id, handle_tool)
    completed = False
    try:
        if control is not None:
            control.raise_if_aborted()
        turn = await active.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [
                    {
                        "type": "text",
                        "text": _turn_text(
                            request, history=history if not resumed else []
                        ),
                    }
                ],
                "approvalPolicy": "never",
                # thread/start uses kebab-case; turn/start sandboxPolicy uses camelCase.
                # networkAccess stays on so Codex can offer hosted web search.
                # The filesystem remains read-only.
                "sandboxPolicy": {"type": "readOnly", "networkAccess": True},
            },
            timeout=30,
        )
        turn_id = str(turn["turn"]["id"])
        async for event in _iter_turn_events(
            active, output, thread_id=thread_id, turn_id=turn_id, control=control
        ):
            yield event
        completed = True
        # Publish a fresh mapping only after the turn stream completed normally.
        # Failures, cancellation, or generator close leave no empty mapping.
        if not resumed:
            chat_id = str(getattr(request, "chat_id", "") or "")
            if chat_id:
                put_thread_id(getattr(request, "user_id", None), chat_id, thread_id)
                logger.info(
                    "Codex thread committed conversation=%s thread=%s",
                    chat_id,
                    thread_id,
                )
    finally:
        if resumed and not completed:
            chat_id = str(getattr(request, "chat_id", "") or "")
            if chat_id:
                drop_thread_id(
                    getattr(request, "user_id", None), chat_id, thread_id
                )
                logger.info(
                    "Codex thread dropped after incomplete turn conversation=%s thread=%s",
                    chat_id,
                    thread_id,
                )
        active.clear_thread(thread_id)


async def _open_thread(client: CodexClient, request: Any) -> tuple[str, bool]:
    """Resume this conversation's thread, or start one and remember it.

    A resumed thread already contains earlier turns, so the caller must not
    replay history. A brand-new thread gets history once.
    """
    chat_id = str(getattr(request, "chat_id", "") or "")
    user_id = getattr(request, "user_id", None)
    if chat_id:
        existing = get_thread_id(user_id, chat_id)
        if existing:
            try:
                await client.request(
                    "thread/resume", {"threadId": existing}, timeout=30
                )
                logger.info(
                    "Codex thread resumed conversation=%s thread=%s",
                    chat_id,
                    existing,
                )
                return existing, True
            except Exception:
                logger.warning(
                    "Codex thread resume failed conversation=%s thread=%s",
                    chat_id,
                    existing,
                    exc_info=True,
                )
                drop_thread_id(user_id, chat_id, existing)
    started = await client.request(
        "thread/start",
        _thread_params(client.workspace_path, persist=bool(chat_id)),
        timeout=30,
    )
    thread_id = str(started["thread"]["id"])
    logger.info("Codex thread started conversation=%s thread=%s", chat_id, thread_id)
    return thread_id, False


def _thread_params(workspace: str, *, persist: bool) -> dict[str, Any]:
    from app.codex.azure_config import azure_codex_ready, resolve_codex_model

    params: dict[str, Any] = {
        "ephemeral": not persist,
        "cwd": workspace,
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "developerInstructions": DEVELOPER_INSTRUCTIONS,
        "dynamicTools": DYNAMIC_TOOLS,
    }
    model = resolve_codex_model()
    if model:
        params["model"] = model
    if azure_codex_ready():
        params["modelProvider"] = "azure"
    return params


def _load_history_for_recovery(request: Any) -> list[dict[str, str]]:
    from app.chat.request_service import normalise_history
    from repositories.supabase.chat_repository import load_completed_history

    return normalise_history(
        load_completed_history(
            conversation_id=str(request.chat_id),
            user_id=str(request.user_id),
            exclude_message_id=getattr(request, "user_message_id", None),
            ownership_verified=True,
        )
    )


def _turn_text(request: Any, *, history: list[dict[str, str]]) -> str:
    sections: list[str] = []
    selected_count = len({str(item) for item in request.selected_paper_ids or [] if item})
    if selected_count:
        groups = dict(getattr(request, "selected_paper_groups", None) or {})
        group_summary = ", ".join(
            f"{name}={len(ids)}" for name, ids in groups.items()
        )
        sections.append(
            f"Current paper selection: {selected_count} unique papers"
            f"{f' across named groups ({group_summary})' if group_summary else ''}. "
            "Use the selected-paper tools to inspect IDs and retrieve evidence."
        )
    if history:
        rendered = [
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in history
            if isinstance(item, dict)
        ]
        if rendered:
            sections.append("Recent conversation:\n" + "\n".join(rendered))
    sections.append(str(request.text or ""))
    return "\n\n".join(sections)


async def _iter_turn_events(
    client: CodexClient,
    output: asyncio.Queue[object],
    *,
    thread_id: str,
    turn_id: str,
    control: RunControl | None,
) -> AsyncIterator[object]:
    saw_text = False
    while True:
        if control is not None:
            try:
                control.raise_if_aborted()
            except Exception as aborted:
                try:
                    await client.request(
                        "turn/interrupt",
                        {"threadId": thread_id, "turnId": turn_id},
                        timeout=10,
                    )
                except Exception:
                    pass
                raise aborted
        try:
            item = await asyncio.wait_for(output.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        if isinstance(item, (TextDelta, AgentAction, PapersResult)):
            if isinstance(item, TextDelta):
                saw_text = True
            yield item
            continue
        event = _event_from_notification(item)
        if event is None:
            continue
        if isinstance(event, TextDelta):
            saw_text = True
            yield event
            continue
        if event == "completed":
            if not saw_text:
                yield TextDelta(text="I couldn't produce a response. Please try again.")
            return
        if event == "failed":
            raise RuntimeError("Codex could not complete this response.")
        yield event


def _web_search_query(item: dict[str, Any]) -> str:
    """Return a short user-visible query, never search result payloads."""
    query = item.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()[:200]
    action = item.get("action")
    if not isinstance(action, dict):
        return ""
    action_query = action.get("query")
    if isinstance(action_query, str) and action_query.strip():
        return action_query.strip()[:200]
    queries = action.get("queries")
    if not isinstance(queries, list):
        return ""
    parts = [str(value).strip() for value in queries if str(value).strip()]
    return "; ".join(parts)[:200]


def _web_search_action(method: str, item: dict[str, Any]) -> AgentAction | None:
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        return None
    status: Literal["started", "completed", "failed"] = (
        "started" if method == "item/started" else "completed"
    )
    if item.get("status") == "failed" or item.get("error"):
        status = "failed"
    query = _web_search_query(item)
    return AgentAction(
        action_id=f"web-search-{item_id}",
        action="web.search",
        status=status,
        data={"query": query} if query else {},
    )


def _event_from_notification(message: object) -> object | None:
    if not isinstance(message, dict):
        return None
    method = message.get("method")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if method in {"item/started", "item/completed"}:
        item = params.get("item")
        if isinstance(item, dict) and item.get("type") == "reasoning":
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                # Expose lifecycle only, never reasoning text or summaries.
                return AgentAction(
                    action_id=f"thinking-{item_id}",
                    action="thinking",
                    status="started" if method == "item/started" else "completed",
                )
        if isinstance(item, dict) and item.get("type") == "webSearch":
            return _web_search_action(method, item)
        return None
    if method == "item/agentMessage/delta":
        delta = params.get("delta")
        return TextDelta(text=str(delta)) if isinstance(delta, str) and delta else None
    if method != "turn/completed":
        return None
    turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
    status = turn.get("status")
    if status == "failed":
        return "failed"
    return "completed"
