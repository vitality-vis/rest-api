"""Flask Chat transport: route parsing, error mapping, and text/plain encoding."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

from agents.agent_v1_legacy import run as run_search_v1_legacy
from app.chat.events import TextDelta
from app.chat.models import ChatDomainError
from app.chat.request_service import (
    build_chat_turn_request,
    normalise_context,
    parse_access_token,
    prepare_chat_turn,
)
from app.chat.runner_adapter import ChatRunner
from app.chat.service import iter_chat_turn_events
from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.chat_repository import (
    ChatPersistenceError,
    ConversationOwnershipError,
    ensure_conversation,
    load_user_conversations,
    save_message,
    set_conversation_closed,
)

chat_bp = Blueprint("chat", __name__)

# TODO: Import oversized guest histories in resumable batches instead of
# rejecting them at these single-request safety limits.
MAX_IMPORT_CONVERSATIONS = 100
MAX_IMPORT_MESSAGES_PER_CONVERSATION = 500
MAX_IMPORT_MESSAGE_CHARS = 50_000


def _get_authenticated_user_id() -> str | None:
    """Return the verified Supabase user ID, or None for a guest request."""
    try:
        access_token = parse_access_token(request.headers.get("Authorization"))
    except ChatDomainError as error:
        raise SupabaseAuthenticationError(str(error)) from error
    if not access_token:
        return None
    return verify_access_token(access_token.strip())


def _require_uuid(value: object, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field_name} must be a UUID") from error


def _require_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO timestamp") from error
    return value


def _importable_message(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("message must be an object")
    role = value.get("role")
    status = value.get("status")
    content = value.get("content")
    if role not in {"user", "assistant"}:
        raise ValueError("message role is invalid")
    if status not in {"streaming", "completed", "failed"}:
        raise ValueError("message status is invalid")
    if not isinstance(content, list):
        raise ValueError("message content is invalid")

    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )[:MAX_IMPORT_MESSAGE_CHARS]
    if not text:
        if status != "failed":
            raise ValueError("message text is required")
        text = "Unable to complete this response. Please try again."

    error_message = value.get("errorMessage")
    return {
        "id": _require_uuid(value.get("id"), "message id"),
        "role": role,
        "status": status,
        "text": text,
        "created_at": _require_timestamp(value.get("createdAt"), "message createdAt"),
        "error_message": error_message[:500] if isinstance(error_message, str) else None,
        "context": normalise_context(value.get("context")),
    }


@chat_bp.route("/chat/import", methods=["POST"])
@cross_origin()
def import_guest_chats():
    """Idempotently import browser-only chats for the verified signed-in user."""
    logger = current_app.logger
    try:
        user_id = _get_authenticated_user_id()
    except SupabaseConfigurationError:
        logger.error("Supabase is not configured for chat import")
        return Response("Chat import is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    if not user_id:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    data = request.get_json(force=True) or {}
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        return Response("Invalid chat import payload", status=400, mimetype="text/plain")

    # Truncate to the most recent conversations/messages; pagination TODO.
    conversations = conversations[:MAX_IMPORT_CONVERSATIONS]

    imported_ids: list[str] = []
    try:
        for value in conversations:
            if not isinstance(value, dict):
                raise ValueError("conversation must be an object")
            messages = value.get("messages")
            if not isinstance(messages, list):
                raise ValueError("conversation messages are invalid")
            messages = messages[:MAX_IMPORT_MESSAGES_PER_CONVERSATION]

            conversation_id = _require_uuid(value.get("id"), "conversation id")
            title = value.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("conversation title is invalid")
            ensure_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                title=title.strip()[:200],
                created_at=_require_timestamp(
                    value.get("createdAt"), "conversation createdAt"
                ),
                updated_at=_require_timestamp(
                    value.get("updatedAt"), "conversation updatedAt"
                ),
                is_closed=bool(value.get("closed", False)),
            )
            for message in messages:
                imported_message = _importable_message(message)
                save_message(
                    conversation_id=conversation_id,
                    role=imported_message["role"],
                    text=imported_message["text"],
                    status=imported_message["status"],
                    error_message=imported_message["error_message"],
                    context=imported_message["context"],
                    message_id=imported_message["id"],
                    created_at=imported_message["created_at"],
                )
            imported_ids.append(conversation_id)
    except ValueError as error:
        return Response(str(error), status=400, mimetype="text/plain")
    except ConversationOwnershipError:
        return Response("Forbidden", status=403, mimetype="text/plain")
    except ChatPersistenceError as error:
        logger.error("Could not import guest chats: %s", error)
        return Response("Chat import is unavailable", status=503, mimetype="text/plain")

    return jsonify(
        {
            "imported_conversation_ids": imported_ids,
            "truncated": len(data.get("conversations", [])) > MAX_IMPORT_CONVERSATIONS,
        }
    )


@chat_bp.route("/chat/conversations", methods=["GET"])
@cross_origin()
def get_chat_conversations():
    """Return the verified user's cloud-backed chat history."""
    logger = current_app.logger
    try:
        user_id = _get_authenticated_user_id()
    except SupabaseConfigurationError:
        logger.error("Supabase is not configured for chat retrieval")
        return Response("Chat history is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    if not user_id:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    try:
        conversations = load_user_conversations(user_id=user_id)
    except ChatPersistenceError as error:
        logger.error("Could not load authenticated chats: %s", error)
        return Response("Chat history is unavailable", status=503, mimetype="text/plain")

    return jsonify({"conversations": conversations})


@chat_bp.route("/chat/conversations/<conversation_id>/closed", methods=["PUT"])
@cross_origin()
def update_chat_conversation_closed(conversation_id: str):
    """Save whether an authenticated user's chat tab is hidden."""
    try:
        user_id = _get_authenticated_user_id()
    except SupabaseConfigurationError:
        return Response("Chat tab state is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    if not user_id:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    data = request.get_json(force=True) or {}
    is_closed = data.get("is_closed")
    if not isinstance(is_closed, bool):
        return Response("is_closed must be a boolean", status=400, mimetype="text/plain")

    try:
        set_conversation_closed(
            conversation_id=conversation_id,
            user_id=user_id,
            is_closed=is_closed,
        )
    except ConversationOwnershipError:
        return Response("Forbidden", status=403, mimetype="text/plain")
    except ChatPersistenceError as error:
        current_app.logger.error("Could not update chat tab state: %s", error)
        return Response("Chat tab state is unavailable", status=503, mimetype="text/plain")

    return Response(status=204)


def _domain_error_response(error: ChatDomainError) -> Response:
    """Map domain errors to the historical HTTP bodies."""
    if error.message == "Please Input Your Text":
        # Preserve the historical bare 400 body (no explicit mimetype).
        return Response(error.message, status=error.status_code)
    return Response(error.message, status=error.status_code, mimetype="text/plain")


def _encode_text_plain_stream(
    prepared: Any,
    run_agent: ChatRunner,
    *,
    logger: logging.Logger,
) -> Response:
    """Encode internal typed events as the current text/plain chunk stream."""
    loop = asyncio.new_event_loop()

    def stream_sync():
        agen = None
        try:
            agen = iter_chat_turn_events(
                prepared,
                run_agent=run_agent,
                logger=logger,
            ).__aiter__()
            while True:
                event = loop.run_until_complete(agen.__anext__())
                if isinstance(event, TextDelta):
                    yield event.text
        except StopAsyncIteration:
            pass
        finally:
            # Client disconnect injects GeneratorExit at yield. aclose() so the
            # Chat service ``finally`` still persists the assistant message.
            if agen is not None and not loop.is_closed():
                try:
                    loop.run_until_complete(agen.aclose())
                except Exception:
                    pass
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()
                try:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass
                loop.close()

    return Response(stream_sync(), status=200, mimetype="text/plain")


def _chat_response(
    run_agent: ChatRunner,
    *,
    pipeline: str = "v1",
    max_text_length: int | None = None,
    trace_id: str | None = None,
) -> Response:
    """Flask transport for one Chat turn (parse → service → encode)."""
    data = request.get_json(force=True) or {}
    try:
        turn_request = build_chat_turn_request(
            data,
            pipeline=pipeline,  # type: ignore[arg-type]
            max_text_length=max_text_length,
            authorization_header=request.headers.get("Authorization"),
            trace_id=trace_id,
        )
        prepared = prepare_chat_turn(turn_request)
    except ChatDomainError as error:
        return _domain_error_response(error)

    return _encode_text_plain_stream(
        prepared,
        run_agent,
        logger=current_app.logger,
    )


@chat_bp.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    """Stream a research-assistant response using the production legacy runner."""
    return _chat_response(run_search_v1_legacy, pipeline="v1")


def _get_chat_v2_runner() -> ChatRunner:
    """Lazy import keeps the experimental v2 stack out of the production route startup."""
    from agents.agent_v2.runner import run as run_chat_v2

    return run_chat_v2


@chat_bp.route("/chat/v2", methods=["POST"])
@cross_origin()
def chat_v2():
    """Experimental chat route that uses v2 for explicit paper-finding turns."""
    from agents.agent_v2.logging import SearchV2Trace

    trace = SearchV2Trace.create()
    response = _chat_response(
        _get_chat_v2_runner(),
        pipeline="v2",
        max_text_length=10_000,
        trace_id=trace.trace_id,
    )
    response.headers["X-Trace-Id"] = trace.trace_id
    return response
