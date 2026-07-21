"""Streaming chat endpoint."""

from __future__ import annotations

import asyncio
from datetime import datetime
from time import monotonic
from uuid import UUID

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin

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
    load_user_conversations,
    save_message,
)
from service.agent_runner import run_two_stage_rag_stream


chat_bp = Blueprint("chat", __name__)

# TODO: Replace these fixed client-history limits with recent turns plus a
# compact summary and on-demand retrieval of relevant older history.
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_MESSAGE_CHARS = 4_000
MAX_HISTORY_TOTAL_CHARS = 24_000
# TODO: Import oversized guest histories in resumable batches instead of
# rejecting them at these single-request safety limits.
MAX_IMPORT_CONVERSATIONS = 100
MAX_IMPORT_MESSAGES_PER_CONVERSATION = 500
MAX_IMPORT_MESSAGE_CHARS = 50_000


def _normalise_history(value: object) -> list[dict[str, str]]:
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
        content = content.strip()
        if content:
            turns.append({"role": role, "content": content[:MAX_HISTORY_MESSAGE_CHARS]})

    bounded: list[dict[str, str]] = []
    remaining = MAX_HISTORY_TOTAL_CHARS
    for turn in reversed(turns[-MAX_HISTORY_MESSAGES:]):
        if remaining <= 0:
            break
        content = turn["content"][:remaining]
        bounded.append({"role": turn["role"], "content": content})
        remaining -= len(content)
    return list(reversed(bounded))


def _get_authenticated_user_id() -> str | None:
    """Return the verified Supabase user ID, or None for a guest request."""
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise SupabaseAuthenticationError("Malformed authorization header")
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


def _importable_message(value: object) -> dict[str, str | None]:
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
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
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
                created_at=_require_timestamp(value.get("createdAt"), "conversation createdAt"),
                updated_at=_require_timestamp(value.get("updatedAt"), "conversation updatedAt"),
            )
            for message in messages:
                imported_message = _importable_message(message)
                save_message(
                    conversation_id=conversation_id,
                    role=imported_message["role"],
                    text=imported_message["text"],
                    status=imported_message["status"],
                    error_message=imported_message["error_message"],
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

    return jsonify({
        "imported_conversation_ids": imported_ids,
        "truncated": len(data.get("conversations", [])) > MAX_IMPORT_CONVERSATIONS,
    })


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


@chat_bp.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    """Stream a research-assistant response for one chat session."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    text = text.strip() if isinstance(text, str) else ""
    chat_id = data.get("chat_id", "default")
    chat_id = str(chat_id)
    title = data.get("title", "New chat")
    title = title.strip()[:200] if isinstance(title, str) and title.strip() else "New chat"
    user_message_id = data.get("user_message_id")
    assistant_message_id = data.get("assistant_message_id")
    message_created_at = data.get("message_created_at")
    user_message_id = str(user_message_id) if user_message_id is not None else None
    assistant_message_id = str(assistant_message_id) if assistant_message_id is not None else None
    message_created_at = str(message_created_at) if message_created_at is not None else None
    if not text:
        return Response("Please Input Your Text", status=400)

    logger = current_app.logger
    try:
        user_id = _get_authenticated_user_id()
    except SupabaseConfigurationError:
        logger.error("Supabase is not configured for authenticated chat")
        return Response("Authenticated chat is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return Response("Unauthorized", status=401, mimetype="text/plain")

    if user_id:
        try:
            ensure_conversation(conversation_id=chat_id, user_id=user_id, title=title)
            # Read history before inserting this turn so the Agent does not see
            # the current question both in history and as its explicit input.
            history = _normalise_history(
                load_completed_history(conversation_id=chat_id, user_id=user_id)
            )
            save_message(
                conversation_id=chat_id,
                role="user",
                text=text,
                message_id=user_message_id,
                created_at=message_created_at,
            )
        except ConversationOwnershipError:
            return Response("Forbidden", status=403, mimetype="text/plain")
        except ChatPersistenceError as error:
            logger.error("Could not initialise authenticated chat: %s", error)
            return Response("Authenticated chat is unavailable", status=503, mimetype="text/plain")
    else:
        history = _normalise_history(data.get("history"))

    loop = asyncio.new_event_loop()
    started_at = monotonic()

    async def agen():
        async for chunk in run_two_stage_rag_stream(text, chat_id, history=history):
            yield chunk

    def stream_sync():
        assistant_chunks: list[str] = []
        stream_completed = False
        stream_error: str | None = None
        try:
            agen_obj = agen().__aiter__()
            while True:
                chunk = loop.run_until_complete(agen_obj.__anext__())
                text_chunk = str(chunk)
                assistant_chunks.append(text_chunk)
                yield text_chunk
        except StopAsyncIteration:
            stream_completed = True
        except Exception as error:
            logger.warning("Chat stream error: %s", error)
            stream_error = str(error)[:500]
            fallback_text = "I'm sorry, something went wrong on our side. Please try again."
            assistant_chunks.append(fallback_text)
            yield fallback_text
        finally:
            if user_id:
                response_text = "".join(assistant_chunks)
                try:
                    save_message(
                        conversation_id=chat_id,
                        role="assistant",
                        text=response_text,
                        status="completed" if stream_completed else "failed",
                        duration_ms=round((monotonic() - started_at) * 1000),
                        error_message=stream_error,
                        message_id=assistant_message_id,
                        created_at=message_created_at,
                    )
                except ChatPersistenceError as error:
                    logger.error("Could not save assistant chat message: %s", error)

            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    return Response(stream_sync(), status=200, mimetype="text/plain")
