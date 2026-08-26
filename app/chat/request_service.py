"""Validate Chat turn payloads; auth, history, and user-message persistence."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.chat.models import (
    ChatForbiddenError,
    ChatPipeline,
    ChatTurnRequest,
    ChatUnauthorizedError,
    ChatUnavailableError,
    ChatValidationError,
    PreparedChatTurn,
)
from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.chat_repository import (
    ChatPersistenceError,
    ConversationOwnershipError,
    ensure_conversation,
    load_completed_history,
    save_message,
)

# TODO: Replace these fixed client-history limits with recent turns plus a
# compact summary and on-demand retrieval of relevant older history.
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_MESSAGE_CHARS = 8_000
MAX_HISTORY_TOTAL_CHARS = 24_000
MAX_MESSAGE_CONTEXT_CHARS = 6_000

# Keep in sync with agents.agent_v1_legacy.rag_core markers. Defined locally so
# the Chat application layer does not import the Agent/search stack.
PAPERS_PAYLOAD_START = "[[VITALITY_PAPERS_JSON]]"
PAPERS_PAYLOAD_END = "[[/VITALITY_PAPERS_JSON]]"
CONTEXT_START = "<CONTEXT>"
CONTEXT_END = "</CONTEXT>"

logger = logging.getLogger(__name__)


def strip_machine_markers(content: str) -> str:
    """Remove frontend-only markers before feeding history to the agent."""
    cleaned = content
    while True:
        start = cleaned.find(PAPERS_PAYLOAD_START)
        if start < 0:
            break
        end = cleaned.find(PAPERS_PAYLOAD_END, start)
        if end < 0:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[end + len(PAPERS_PAYLOAD_END) :]
    return cleaned.replace("[SIGNAL:SHOW_LOAD_MORE]", "").strip()


def normalise_context(value: object) -> dict[str, object]:
    """Return bounded JSON-object context attached to one chat message."""
    if not isinstance(value, dict):
        return {}
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > MAX_MESSAGE_CONTEXT_CHARS:
            # Keep a useful, valid paper reference rather than persisting a
            # partial JSON payload when many/full abstracts were selected.
            papers = value.get("selectedPapers")
            if not isinstance(papers, list):
                return {}
            compact_papers: list[dict[str, object]] = []
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                paper_id = paper.get("id")
                title = paper.get("title")
                if not isinstance(paper_id, str) or not isinstance(title, str):
                    continue
                compact: dict[str, object] = {"id": paper_id, "title": title}
                if isinstance(paper.get("abstract"), str):
                    compact["abstract"] = paper["abstract"][:1_000]
                candidate = {"selectedPapers": [*compact_papers, compact]}
                if (
                    len(
                        json.dumps(
                            candidate, ensure_ascii=False, separators=(",", ":")
                        )
                    )
                    > MAX_MESSAGE_CONTEXT_CHARS
                ):
                    break
                compact_papers.append(compact)
            value = {"selectedPapers": compact_papers}
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def context_marker(context: dict[str, object]) -> str:
    if not context:
        return ""
    return (
        f"\n{CONTEXT_START}\n"
        f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
        f"{CONTEXT_END}"
    )


def normalise_history(value: object) -> list[dict[str, str]]:
    """Return bounded, user/assistant text turns from an untrusted request body."""
    if not isinstance(value, list):
        return []

    turns: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = strip_machine_markers(content)
        if content:
            # Assistant attachments belong to rendering, not future prompts.
            context = normalise_context(item.get("context")) if role == "user" else {}
            turns.append(
                {
                    "role": role,
                    "content": f"{content}{context_marker(context)}"[
                        :MAX_HISTORY_MESSAGE_CHARS
                    ],
                }
            )

    bounded: list[dict[str, str]] = []
    remaining = MAX_HISTORY_TOTAL_CHARS
    for turn in reversed(turns[-MAX_HISTORY_MESSAGES:]):
        if remaining <= 0:
            break
        content = turn["content"][:remaining]
        bounded.append({"role": turn["role"], "content": content})
        remaining -= len(content)
    return list(reversed(bounded))


def parse_access_token(authorization_header: str | None) -> str | None:
    """Return a bearer token, None for guest, or raise on malformed headers."""
    if not authorization_header:
        return None
    scheme, _, access_token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise ChatUnauthorizedError("Unauthorized")
    return access_token.strip()


def build_chat_turn_request(
    data: dict[str, Any],
    *,
    pipeline: ChatPipeline = "v1",
    max_text_length: int | None = None,
    authorization_header: str | None = None,
    trace_id: str | None = None,
) -> ChatTurnRequest:
    """Parse and validate an untrusted Chat turn body into a domain request."""
    text = data.get("text", "")
    text = text.strip() if isinstance(text, str) else ""
    chat_id = str(data.get("chat_id", "default"))
    raw_title = data.get("title", "New chat")
    title = (
        raw_title.strip()[:200]
        if isinstance(raw_title, str) and raw_title.strip()
        else "New chat"
    )
    user_message_id = data.get("user_message_id")
    assistant_message_id = data.get("assistant_message_id")
    client_request_id = data.get("client_request_id")
    agent_run_id = data.get("agent_run_id")
    message_created_at = data.get("message_created_at")
    effort = data.get("effort", "low")
    effort = effort if effort in {"low", "medium", "high"} else "low"
    raw_model = data.get("model")
    if raw_model is None or raw_model == "":
        model = None
    elif isinstance(raw_model, str) and raw_model.strip():
        model = raw_model.strip()
    else:
        raise ChatValidationError("model must be a non-empty string")
    if model is not None:
        import config as app_config

        try:
            model = app_config.resolve_chat_model(model)
        except ValueError as error:
            raise ChatValidationError(str(error)) from error

    user_message_id = str(user_message_id) if user_message_id is not None else None
    assistant_message_id = (
        str(assistant_message_id) if assistant_message_id is not None else None
    )
    client_request_id = (
        str(client_request_id) if client_request_id is not None else None
    )
    agent_run_id = str(agent_run_id) if agent_run_id is not None else None
    message_created_at = (
        str(message_created_at) if message_created_at is not None else None
    )
    message_context = normalise_context(data.get("context"))
    guest_history = normalise_history(data.get("history"))

    requested_mode: Any = "auto"
    paper_ids: list[str] = []
    advanced = None
    if pipeline == "v2":
        from agents.agent_v2.models import AdvancedSearchConfig

        try:
            advanced = AdvancedSearchConfig.model_validate(data.get("advanced") or {})
        except Exception as error:
            raise ChatValidationError(
                f"advanced configuration is invalid: {error}"
            ) from error
        requested_mode = data.get("mode", "auto")
        if requested_mode not in {"auto", "chat", "search", "synthesis"}:
            raise ChatValidationError(
                "mode must be one of: auto, chat, search, synthesis"
            )
        paper_ids = data.get("paper_ids", [])
        if not isinstance(paper_ids, list) or not all(
            isinstance(item, str) and item for item in paper_ids
        ):
            raise ChatValidationError("paper_ids must be an array of strings")

    if not text:
        # Preserve the historical bare 400 body (no mimetype override in original).
        raise ChatValidationError("Please Input Your Text")
    if max_text_length is not None and len(text) > max_text_length:
        raise ChatValidationError(
            f"Your message is too long. Please keep it within {max_text_length:,} characters."
        )

    return ChatTurnRequest(
        text=text,
        chat_id=chat_id,
        title=title,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        message_created_at=message_created_at,
        effort=effort,
        model=model,
        message_context=message_context,
        guest_history=guest_history,
        authorization_header=authorization_header,
        pipeline=pipeline,
        max_text_length=max_text_length,
        trace_id=trace_id,
        client_request_id=client_request_id,
        agent_run_id=agent_run_id,
        requested_mode=requested_mode,
        paper_ids=paper_ids,
        advanced=advanced,
    )


def prepare_chat_turn(request: ChatTurnRequest) -> PreparedChatTurn:
    """Authenticate, ensure conversation, load history, then save the user message."""
    try:
        access_token = parse_access_token(request.authorization_header)
        user_id = verify_access_token(access_token) if access_token else None
    except SupabaseConfigurationError as error:
        logger.error("Supabase is not configured for authenticated chat")
        raise ChatUnavailableError("Authenticated chat is unavailable") from error
    except SupabaseAuthenticationError as error:
        raise ChatUnauthorizedError("Unauthorized") from error
    except ChatUnauthorizedError:
        raise

    if user_id:
        try:
            ensure_conversation(
                conversation_id=request.chat_id,
                user_id=user_id,
                title=request.title,
            )
            # Read history before inserting this turn so the Agent does not see
            # the current question both in history and as its explicit input.
            history = normalise_history(
                load_completed_history(
                    conversation_id=request.chat_id, user_id=user_id
                )
            )
            save_message(
                conversation_id=request.chat_id,
                role="user",
                text=request.text,
                message_id=request.user_message_id,
                created_at=request.message_created_at,
                context=request.message_context,
            )
        except ConversationOwnershipError as error:
            raise ChatForbiddenError("Forbidden") from error
        except ChatPersistenceError as error:
            logger.error("Could not initialise authenticated chat: %s", error)
            raise ChatUnavailableError("Authenticated chat is unavailable") from error
    else:
        history = list(request.guest_history)

    return PreparedChatTurn(request=request, user_id=user_id, history=history)
