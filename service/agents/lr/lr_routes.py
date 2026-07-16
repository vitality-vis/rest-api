"""Flask blueprint exposing the LR human-in-the-loop checkpoint chat.

Registered from ``main.py`` via ``app.register_blueprint(lr_bp)``. This route
(``POST /lrChat``) is additive and entirely separate from the existing
``/chat`` endpoint and its tool-approval backend.

Request body
------------
    {
      "chat_id": "...",            # stable per chat (falls back to sessionId)
      "message": "...",            # goal text (start) or free text
      "checkpoint_id": "...",      # the checkpoint being acted on (resume)
      "action": "approve" | "edit" | "reject" | "accept" | "refine_search" | "continue",
      "feedback": "..."            # text for edit / refine_search
    }

Responses follow the LR checkpoint protocol (see ``lr_session``):
``lr_checkpoint`` | ``lr_message`` | ``lr_final``.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

from service.agents.lr.lr_session import lr_handle, reset_session

logger = logging.getLogger("service.agents.lr.routes")

lr_bp = Blueprint("lr", __name__)


@lr_bp.route("/lrChat", methods=["POST"])
@cross_origin()
def lr_chat():
    data = request.get_json(force=True) or {}
    chat_id = (data.get("chat_id") or data.get("sessionId") or "lr-default")
    message = data.get("message") or ""
    checkpoint_id = data.get("checkpoint_id")
    action = data.get("action")
    feedback = data.get("feedback")

    try:
        result = lr_handle(
            chat_id=chat_id,
            message=message,
            checkpoint_id=checkpoint_id,
            action=action,
            feedback=feedback,
        )
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        # Log the real traceback rather than hiding it behind a generic message.
        logger.exception("LR /lrChat failed | chat_id=%s action=%s", chat_id, action)
        return (
            jsonify({
                "type": "lr_message",
                "error": str(exc),
                "content": f"LR backend error: {exc}",
            }),
            500,
        )


@lr_bp.route("/lrChat/reset", methods=["POST"])
@cross_origin()
def lr_chat_reset():
    data = request.get_json(force=True) or {}
    chat_id = (data.get("chat_id") or data.get("sessionId") or "lr-default")
    reset_session(chat_id)
    logger.info("LR session reset | chat_id=%s", chat_id)
    return jsonify({"ok": True})
