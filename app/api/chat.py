"""Streaming chat endpoint."""

from __future__ import annotations

import asyncio

from flask import Blueprint, Response, current_app, request
from flask_cors import cross_origin

from service.agent_runner import run_two_stage_rag_stream


chat_bp = Blueprint("chat", __name__)

# TODO: Replace these fixed client-history limits with recent turns plus a
# compact summary and on-demand retrieval of relevant older history.
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_MESSAGE_CHARS = 4_000
MAX_HISTORY_TOTAL_CHARS = 24_000


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


@chat_bp.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    """Stream a research-assistant response for one chat session."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    chat_id = data.get("chat_id", "default")
    history = _normalise_history(data.get("history"))

    if not text:
        return Response("Please Input Your Text", status=400)

    loop = asyncio.new_event_loop()
    logger = current_app.logger

    async def agen():
        async for chunk in run_two_stage_rag_stream(text, chat_id, history=history):
            yield chunk

    def stream_sync():
        try:
            agen_obj = agen().__aiter__()
            while True:
                chunk = loop.run_until_complete(agen_obj.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass
        except Exception as error:
            logger.warning("Chat stream error: %s", error)
            yield "I'm sorry, something went wrong on our side. Please try again."
        finally:
            pending = asyncio.all_tasks(loop=loop)
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    return Response(stream_sync(), status=200, mimetype="text/plain")
