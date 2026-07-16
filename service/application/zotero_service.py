"""Application service for backend-only Zotero import workflows."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
import time
from typing import Any, Dict, List, Optional

import config
from logger_config import get_logger
from service.infrastructure import embeddings as embed_service
from service.infrastructure import zilliz
from service.integrations.zotero import ZoteroAPIError, ZoteroClient

logger = get_logger()

_CONNECTION_TTL_SECONDS = int(getattr(config, "ZOTERO_CONNECT_TTL_SECONDS", 43200))
_ZOTERO_CONNECTIONS: Dict[str, Dict[str, Any]] = {}


class ZoteroServiceError(Exception):
    """Structured service-layer error for Zotero endpoints."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def connect_zotero_library(
    *,
    chat_id: str,
    zotero_user_id: str,
    zotero_api_key: str,
    app_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    chat_id = _validate_chat_id(chat_id)
    zotero_user_id = str(zotero_user_id or "").strip()
    zotero_api_key = str(zotero_api_key or "").strip()
    app_user_id = str(app_user_id or "").strip()

    if not zotero_user_id or not zotero_api_key:
        raise ZoteroServiceError(
            400,
            "zotero_missing_credentials",
            "Both Zotero User ID and API Key are required.",
        )

    client = ZoteroClient(zotero_user_id, zotero_api_key)
    try:
        key_info = client.validate_credentials()
        collections = client.list_collections(limit=100)
    except ZoteroAPIError as exc:
        raise ZoteroServiceError(exc.status_code, exc.code, exc.message) from exc

    connection_id = secrets.token_urlsafe(24)
    _purge_expired_connections()
    _ZOTERO_CONNECTIONS[connection_id] = {
        "chat_id": chat_id,
        "app_user_id": app_user_id,
        "zotero_user_id": zotero_user_id,
        "zotero_api_key": zotero_api_key,
        "created_at": time.time(),
    }

    return {
        "connection_id": connection_id,
        "zotero_user_id": zotero_user_id,
        "collections": collections,
        "key_access": key_info.get("access", {}),
    }


def list_zotero_library_items(
    *,
    connection_id: str,
    chat_id: str,
    app_user_id: Optional[str] = None,
    collection_key: Optional[str] = None,
    limit: int = 25,
    start: int = 0,
) -> Dict[str, Any]:
    connection = _get_connection(
        connection_id=connection_id,
        chat_id=chat_id,
        app_user_id=app_user_id,
    )
    client = ZoteroClient(
        connection["zotero_user_id"],
        connection["zotero_api_key"],
    )

    try:
        collections = client.list_collections(limit=100)
        items = client.list_items(
            collection_key=collection_key,
            limit=limit,
            start=start,
        )
        for item in items:
            if (
                item.get("item_type") == "attachment"
                and item.get("content_type") == "application/pdf"
            ):
                # Standalone PDF: the item IS its own downloadable attachment.
                item["pdf_attachments"] = [
                    {
                        "key": item.get("key"),
                        "title": item.get("title"),
                        "filename": item.get("filename"),
                        "content_type": item.get("content_type"),
                        "link_mode": item.get("link_mode"),
                    }
                ]
                continue
            attachments = client.list_pdf_attachments(item["key"])
            item["pdf_attachments"] = [
                {
                    "key": attachment.get("key"),
                    "title": attachment.get("title"),
                    "filename": attachment.get("filename"),
                    "content_type": attachment.get("content_type"),
                    "link_mode": attachment.get("link_mode"),
                }
                for attachment in attachments
            ]
    except ZoteroAPIError as exc:
        raise ZoteroServiceError(exc.status_code, exc.code, exc.message) from exc

    return {
        "collections": collections,
        "items": items,
        "collection_key": collection_key,
        "limit": max(1, min(int(limit or 25), 100)),
        "start": max(0, int(start or 0)),
    }


def import_zotero_pdf_attachment(
    *,
    connection_id: str,
    chat_id: str,
    item_key: str,
    attachment_key: Optional[str] = None,
    app_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    chat_id = _validate_chat_id(chat_id)
    item_key = str(item_key or "").strip()
    attachment_key = str(attachment_key or "").strip()

    if not item_key:
        raise ZoteroServiceError(400, "zotero_missing_item_key", "A Zotero item key is required.")

    connection = _get_connection(
        connection_id=connection_id,
        chat_id=chat_id,
        app_user_id=app_user_id,
    )
    client = ZoteroClient(
        connection["zotero_user_id"],
        connection["zotero_api_key"],
    )
    scope_id = _build_scope_id(chat_id, connection.get("app_user_id"))

    try:
        item = client.get_item(item_key)
        if (
            item.get("item_type") == "attachment"
            and item.get("content_type") == "application/pdf"
        ):
            # Standalone PDF attachment: it is its own downloadable file, so there
            # are no child attachments to enumerate — treat the item as the attachment.
            attachments = [item]
        else:
            attachments = client.list_pdf_attachments(item_key)
    except ZoteroAPIError as exc:
        raise ZoteroServiceError(exc.status_code, exc.code, exc.message) from exc

    if not attachments:
        raise ZoteroServiceError(
            404,
            "zotero_no_pdf_attachment",
            "No PDF attachment was found for the selected Zotero item.",
        )

    selected_attachment = None
    if attachment_key:
        selected_attachment = next(
            (attachment for attachment in attachments if attachment.get("key") == attachment_key),
            None,
        )
        if not selected_attachment:
            raise ZoteroServiceError(
                404,
                "zotero_attachment_not_found",
                "The requested PDF attachment was not found under this Zotero item.",
            )
    else:
        selected_attachment = attachments[0]

    existing_chunks = zilliz.zotero_chunk_count(scope_id, selected_attachment.get("key"))
    if existing_chunks > 0:
        return {
            "status": "already_imported",
            "scope_id": scope_id,
            "item_key": item_key,
            "attachment_key": selected_attachment.get("key"),
            "chunk_count": existing_chunks,
            "title": item.get("title", ""),
            "filename": selected_attachment.get("filename", ""),
        }

    temp_path = None
    try:
        temp_path = _download_attachment_pdf(client, selected_attachment)
        extracted_text, page_count = _extract_pdf_text(temp_path)
        chunks = _chunk_text(
            extracted_text,
            chunk_size=int(getattr(config, "ZOTERO_CHUNK_SIZE", 1800)),
            overlap=int(getattr(config, "ZOTERO_CHUNK_OVERLAP", 250)),
        )
        if not chunks:
            raise ZoteroServiceError(
                422,
                "zotero_empty_pdf_text",
                "The PDF was downloaded, but no readable text could be extracted.",
            )

        embeddings = _embed_chunks(chunks)
        if len(embeddings) != len(chunks):
            raise ZoteroServiceError(
                500,
                "zotero_embedding_failed",
                "Could not generate embeddings for the extracted PDF text.",
            )

        records = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.sha1(
                f"{scope_id}|{selected_attachment.get('key')}|{idx}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "id": chunk_id,
                    "embedding": embedding,
                    "scope_id": scope_id,
                    "chat_id": chat_id,
                    "app_user_id": connection.get("app_user_id", ""),
                    "zotero_user_id": connection["zotero_user_id"],
                    "parent_item_key": item_key,
                    "attachment_key": selected_attachment.get("key"),
                    "title": item.get("title", ""),
                    "filename": selected_attachment.get("filename", ""),
                    "content_type": selected_attachment.get("content_type", "application/pdf"),
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                }
            )

        inserted = zilliz.insert_zotero_chunks(records)
        return {
            "status": "imported",
            "scope_id": scope_id,
            "item_key": item_key,
            "attachment_key": selected_attachment.get("key"),
            "title": item.get("title", ""),
            "filename": selected_attachment.get("filename", ""),
            "page_count": page_count,
            "chunk_count": inserted,
            "extracted_chars": len(extracted_text),
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _validate_chat_id(chat_id: str) -> str:
    chat_id = str(chat_id or "").strip()
    if not chat_id or chat_id == "default":
        raise ZoteroServiceError(
            400,
            "zotero_invalid_chat_id",
            "A non-default chat_id is required for Zotero import isolation.",
        )
    return chat_id


def _build_scope_id(chat_id: str, app_user_id: Optional[str]) -> str:
    app_user_id = str(app_user_id or "").strip()
    if app_user_id:
        return f"user:{app_user_id}|chat:{chat_id}"
    return f"chat:{chat_id}"


def _purge_expired_connections() -> None:
    now = time.time()
    expired = [
        key
        for key, value in _ZOTERO_CONNECTIONS.items()
        if now - float(value.get("created_at", 0)) > _CONNECTION_TTL_SECONDS
    ]
    for key in expired:
        _ZOTERO_CONNECTIONS.pop(key, None)


def _get_connection(
    *,
    connection_id: str,
    chat_id: str,
    app_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    _purge_expired_connections()
    chat_id = _validate_chat_id(chat_id)
    connection_id = str(connection_id or "").strip()
    if not connection_id:
        raise ZoteroServiceError(
            400,
            "zotero_missing_connection_id",
            "A Zotero connection_id is required.",
        )

    connection = _ZOTERO_CONNECTIONS.get(connection_id)
    if not connection:
        raise ZoteroServiceError(
            401,
            "zotero_connection_not_found",
            "The Zotero connection has expired or is invalid.",
        )

    if connection.get("chat_id") != chat_id:
        raise ZoteroServiceError(
            403,
            "zotero_scope_mismatch",
            "This Zotero connection does not belong to the provided chat_id.",
        )

    expected_user = str(connection.get("app_user_id") or "").strip()
    provided_user = str(app_user_id or "").strip()
    if expected_user and provided_user and expected_user != provided_user:
        raise ZoteroServiceError(
            403,
            "zotero_scope_mismatch",
            "This Zotero connection does not belong to the provided user_id.",
        )

    return connection


def _download_attachment_pdf(client: ZoteroClient, attachment: Dict[str, Any]) -> str:
    try:
        response = client.download_attachment_pdf(attachment["key"])
    except ZoteroAPIError as exc:
        if exc.code == "zotero_not_found":
            raise ZoteroServiceError(
                404,
                "zotero_pdf_download_unavailable",
                "The PDF attachment exists in Zotero metadata, but the file could not be downloaded.",
            ) from exc
        raise ZoteroServiceError(exc.status_code, exc.code, exc.message) from exc

    suffix = ".pdf"
    filename = str(attachment.get("filename") or "")
    if filename.lower().endswith(".pdf"):
        suffix = ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                tmp_file.write(chunk)
        return tmp_file.name


def _extract_pdf_text(pdf_path: str) -> tuple[str, int]:
    try:
        import fitz
    except ImportError as exc:
        raise ZoteroServiceError(
            500,
            "pdf_parser_missing",
            "PyMuPDF is not installed on the backend.",
        ) from exc

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ZoteroServiceError(
            422,
            "pdf_parse_failed",
            f"Could not open the PDF attachment: {exc}",
        ) from exc

    try:
        texts = []
        for page in doc:
            texts.append(page.get_text("text") or "")
        full_text = "\n\n".join(texts)
        clean_text = re.sub(r"\x00", " ", full_text).strip()
        if not clean_text:
            raise ZoteroServiceError(
                422,
                "zotero_empty_pdf_text",
                "The PDF contains no extractable text.",
            )
        return clean_text, int(doc.page_count)
    finally:
        doc.close()


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> List[str]:
    clean_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean_text:
        return []

    chunk_size = max(400, int(chunk_size or 1800))
    overlap = max(0, min(int(overlap or 0), chunk_size // 2))

    chunks: List[str] = []
    start = 0
    text_len = len(clean_text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            split_at = clean_text.rfind(" ", start + (chunk_size // 2), end)
            if split_at > start:
                end = split_at

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []

    model = getattr(embed_service, "glove_model_instance", None)
    if model and hasattr(model, "embed_documents"):
        try:
            vectors = model.embed_documents(chunks)
            return [_normalize_embedding(vec) for vec in vectors if vec]
        except Exception as exc:
            logger.warning("[Zotero] batch embedding failed, falling back to per-chunk: %s", exc)

    vectors = []
    for chunk in chunks:
        vector = embed_service.glove_embedding(chunk)
        if not vector:
            raise ZoteroServiceError(
                500,
                "zotero_embedding_failed",
                "The embedding service returned an empty vector.",
            )
        vectors.append(_normalize_embedding(vector))
    return vectors


def _normalize_embedding(vector: Any) -> List[float]:
    normalized = []
    for value in list(vector):
        normalized.append(float(value))
    return normalized
