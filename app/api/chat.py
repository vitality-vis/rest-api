"""Streaming chat endpoint."""

from __future__ import annotations

import asyncio

from flask import Blueprint, Response, current_app, request
from flask_cors import cross_origin

from service.agent_runner import run_two_stage_rag_stream


chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    """Stream a research-assistant response for one chat session."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    chat_id = data.get("chat_id", "default")

    if not text:
        return Response("Please Input Your Text", status=400)

    loop = asyncio.new_event_loop()
    logger = current_app.logger

    async def agen():
        async for chunk in run_two_stage_rag_stream(text, chat_id):
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
