"""Authenticated API endpoints for a user's personal paper library."""

from __future__ import annotations

import json
import math
import tempfile
from typing import BinaryIO
from uuid import UUID, uuid4

from flask import Blueprint, Response, current_app, jsonify, request
from flask_cors import cross_origin
from werkzeug.datastructures import FileStorage

import config
from repositories.azure_openai.files import (
    AzureFilesConfigurationError,
    AzureFilesError,
    AzureFilesTransientError,
    delete_file as delete_azure_file,
    upload_pdf_file,
)
from repositories.supabase.auth import (
    SupabaseAuthenticationError,
    SupabaseConfigurationError,
    verify_access_token,
)
from repositories.supabase.user_papers_repository import (
    UserPaperNotFoundError,
    UserPapersPersistenceError,
    clear_user_paper_file,
    get_user_paper,
    import_user_papers,
    list_user_papers,
    save_user_paper,
    save_user_papers,
    unsave_user_paper,
    upsert_user_paper_file,
)
from service.fulltext import LibraryIndexError, detach_user_paper_file, index_user_paper


library_bp = Blueprint("library", __name__)

MAX_PAPER_ID_LENGTH = 1024
MAX_DOI_LENGTH = 512
MAX_TITLE_LENGTH = 2_000
MAX_ABSTRACT_LENGTH = 100_000
MAX_SOURCE_LENGTH = 2_000
MAX_LIST_ITEMS = 500
MAX_LIST_ITEM_LENGTH = 2_000
MAX_BULK_SAVE_PAPERS = 100
MAX_IMPORT_ITEMS = 100
MAX_RAW_METADATA_BYTES = 512 * 1024
MAX_RAW_FORMAT_LENGTH = 64
PDF_MAGIC = b"%PDF-"
READ_CHUNK_BYTES = 64 * 1024
SPOOLED_PDF_MEMORY_BYTES = 8 * 1024 * 1024


def _pdf_max_bytes() -> int:
    return config.LIBRARY_PDF_MAX_BYTES


def _get_authenticated_user_id() -> str:
    authorization = request.headers.get("Authorization")
    if not authorization:
        raise SupabaseAuthenticationError("Missing authorization header")
    scheme, _, access_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access_token.strip():
        raise SupabaseAuthenticationError("Malformed authorization header")
    return verify_access_token(access_token.strip())


def _validate_paper_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PAPER_ID_LENGTH:
        raise ValueError("paper_id is invalid")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("paper_id is invalid")
    return value


def _bounded_text(value: object, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{field_name} is invalid")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise ValueError(f"{field_name} is invalid")
    return [_bounded_text(item, field_name, MAX_LIST_ITEM_LENGTH) for item in value]


def _nullable_integer(value: object, field_name: str) -> int | None:
    """Accept ints or whole-number floats (Zilliz/JSON may surface either)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} is invalid")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)
    raise ValueError(f"{field_name} is invalid")


def _metadata_paper_id(value: object) -> str:
    """Normalise body ID to the string form used in path params and filters."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Paper metadata ID must match paper_id")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return str(int(value))
    raise ValueError("Paper metadata ID must match paper_id")


def _nullable_coordinate(value: object, field_name: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} is invalid")
    coordinates: dict[str, object] = {}
    for axis in ("x", "y"):
        coordinate = value.get(axis)
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{field_name} is invalid")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError(f"{field_name} is invalid")
        coordinates[axis] = coordinate
    embedding_model = value.get("embedding_model")
    if embedding_model is not None:
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError(f"{field_name} is invalid")
        coordinates["embedding_model"] = embedding_model
    return coordinates


def _has_valid_doi(value: dict[str, object]) -> bool:
    """Whether this paper can be rehydrated from the canonical catalog by DOI."""
    doi = value.get("doi")
    if doi is None:
        return False
    if not isinstance(doi, str) or len(doi) > MAX_DOI_LENGTH:
        raise ValueError("doi is invalid")
    return bool(doi.strip())


def _validate_metadata_snapshot(value: object, paper_id: str) -> dict[str, object] | None:
    """Return a fallback snapshot only for papers that do not have a DOI."""
    if not isinstance(value, dict):
        raise ValueError("Paper metadata must be an object")

    # Ignore response-only retrieval scores from paper_to_api_response.
    if _metadata_paper_id(value.get("ID")) != paper_id:
        raise ValueError("Paper metadata ID must match paper_id")

    # DOI-backed papers are resolved from the current catalog when a library is
    # loaded, so avoid persisting duplicate metadata. Papers without a DOI keep
    # this snapshot as their display fallback.
    if _has_valid_doi(value):
        return None

    return {
        "ID": paper_id,
        "Title": _bounded_text(value.get("Title"), "Title", MAX_TITLE_LENGTH),
        "Abstract": _bounded_text(value.get("Abstract"), "Abstract", MAX_ABSTRACT_LENGTH),
        "Authors": _string_list(value.get("Authors"), "Authors"),
        "Keywords": _string_list(value.get("Keywords"), "Keywords"),
        "Source": _bounded_text(value.get("Source"), "Source", MAX_SOURCE_LENGTH),
        "Year": _nullable_integer(value.get("Year"), "Year"),
        "CitationCounts": _nullable_integer(value.get("CitationCounts"), "CitationCounts"),
        "umap": _nullable_coordinate(value.get("umap"), "umap"),
    }


def _require_authenticated_user_id() -> tuple[str | None, Response | None]:
    try:
        return _get_authenticated_user_id(), None
    except SupabaseConfigurationError:
        current_app.logger.error("Supabase is not configured for the library")
        return None, Response("Library is unavailable", status=503, mimetype="text/plain")
    except SupabaseAuthenticationError:
        return None, Response("Unauthorized", status=401, mimetype="text/plain")


def _save_response(paper: dict[str, object]) -> dict[str, object]:
    """Return the compact representation defined by the PUT /saved contract."""
    snapshot = paper.get("metadata_snapshot")
    title = snapshot.get("Title") if isinstance(snapshot, dict) else None
    return {
        "id": paper.get("id"),
        "paper_id": paper.get("paper_id"),
        "title": title,
        "is_saved": bool(paper.get("is_saved")),
        "azure_file_id": paper.get("azure_file_id"),
    }


def _file_response(paper: dict[str, object]) -> dict[str, object]:
    return {
        "paper_id": paper.get("paper_id"),
        "azure_file_id": paper.get("azure_file_id"),
        "filename": paper.get("uploaded_filename"),
        "bytes": paper.get("uploaded_bytes"),
        "uploaded_at": paper.get("uploaded_at"),
        "vs_file_status": paper.get("vs_file_status"),
        "vs_last_error": paper.get("vs_last_error"),
    }


def _validate_bulk_save_papers(value: object) -> list[tuple[str, dict[str, object]]]:
    if not isinstance(value, dict) or not isinstance(value.get("papers"), list):
        raise ValueError("papers must be an array")
    papers = value["papers"]
    if not papers or len(papers) > MAX_BULK_SAVE_PAPERS:
        raise ValueError(f"papers must contain between 1 and {MAX_BULK_SAVE_PAPERS} items")

    validated: list[tuple[str, dict[str, object] | None]] = []
    paper_ids: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            raise ValueError("Paper metadata must be an object")
        paper_id = _validate_paper_id(_metadata_paper_id(paper.get("ID")))
        if paper_id in paper_ids:
            raise ValueError("papers must not contain duplicate IDs")
        paper_ids.add(paper_id)
        validated.append((paper_id, _validate_metadata_snapshot(paper, paper_id)))
    return validated


def _required_import_text(value: object, field_name: str, maximum: int) -> str:
    text = _bounded_text(value, field_name, maximum).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_import_text(value: object, field_name: str, maximum: int) -> str:
    if value is None:
        return ""
    return _bounded_text(value, field_name, maximum).strip()


def _optional_import_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in _string_list(value, field_name) if item.strip()]


def _validate_user_import_id(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("user:"):
        raise ValueError("Imported paper ID must use the user:<uuid> format")
    raw_uuid = value.removeprefix("user:")
    try:
        parsed = UUID(raw_uuid)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Imported paper ID must use the user:<uuid> format") from error
    if parsed.version != 4 or str(parsed) != raw_uuid:
        raise ValueError("Imported paper ID must use the user:<uuid> format")
    return value


def _validate_import_raw(value: object) -> dict[str, object] | None:
    """Validate a bounded source-payload envelope without interpreting it."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("raw must be an object")

    format_name = value.get("format")
    if (
        not isinstance(format_name, str)
        or not format_name.strip()
        or len(format_name) > MAX_RAW_FORMAT_LENGTH
    ):
        raise ValueError("raw.format is invalid")
    if "payload" not in value:
        raise ValueError("raw.payload is required")

    raw = {"format": format_name.strip(), "payload": value["payload"]}
    parser_version = value.get("parser_version")
    if parser_version is not None:
        raw["parser_version"] = _bounded_text(
            parser_version, "raw.parser_version", MAX_RAW_FORMAT_LENGTH
        ).strip()

    try:
        encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("raw is not JSON serializable") from error
    if len(encoded) > MAX_RAW_METADATA_BYTES:
        raise ValueError("raw exceeds the 512 KiB limit")
    return raw


def _validate_user_import_item(value: object) -> tuple[str, dict[str, object], dict[str, object] | None]:
    """Map one normalized import item into its immutable shelf snapshot."""
    if not isinstance(value, dict):
        raise ValueError("Import item must be an object")
    paper = value.get("paper")
    if not isinstance(paper, dict):
        raise ValueError("paper must be an object")

    requested_id = paper.get("ID")
    paper_id = (
        _validate_user_import_id(requested_id)
        if requested_id is not None
        else f"user:{uuid4()}"
    )
    doi = paper.get("doi")
    if doi is not None:
        doi = _bounded_text(doi, "doi", MAX_DOI_LENGTH).strip() or None

    snapshot: dict[str, object] = {
        "ID": paper_id,
        "Title": _required_import_text(paper.get("title"), "title", MAX_TITLE_LENGTH),
        "Abstract": _required_import_text(
            paper.get("abstract"), "abstract", MAX_ABSTRACT_LENGTH
        ),
        "Authors": _optional_import_string_list(paper.get("authors"), "authors"),
        "Keywords": _optional_import_string_list(paper.get("keywords"), "keywords"),
        "Source": _optional_import_text(paper.get("source"), "source", MAX_SOURCE_LENGTH),
        "Year": _nullable_integer(paper.get("year"), "year"),
        "CitationCounts": None,
        "doi": doi,
        "umap": None,
    }
    return paper_id, snapshot, _validate_import_raw(value.get("raw"))


def _parse_saved_only_query() -> bool:
    raw = request.args.get("saved")
    if raw is None:
        return False
    if raw.lower() in {"1", "true", "yes"}:
        return True
    if raw.lower() in {"0", "false", "no"}:
        return False
    raise ValueError("saved must be a boolean")


def _read_and_validate_pdf(upload: FileStorage, maximum_bytes: int) -> tuple[tempfile.SpooledTemporaryFile, int, str]:
    """Stream the upload into a spooled file while checking PDF magic and size."""
    content_type = (upload.mimetype or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
        raise UnsupportedMediaType("Only PDF uploads are supported")

    filename = upload.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"

    spooled: tempfile.SpooledTemporaryFile = tempfile.SpooledTemporaryFile(
        max_size=SPOOLED_PDF_MEMORY_BYTES
    )
    total = 0
    header = b""

    stream: BinaryIO = upload.stream
    while True:
        chunk = stream.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        if len(header) < len(PDF_MAGIC):
            needed = len(PDF_MAGIC) - len(header)
            header += chunk[:needed]
            if len(header) >= len(PDF_MAGIC) and not header.startswith(PDF_MAGIC):
                spooled.close()
                raise UnsupportedMediaType("Only PDF uploads are supported")
        total += len(chunk)
        if total > maximum_bytes:
            spooled.close()
            raise PayloadTooLarge("PDF exceeds the configured size limit")
        spooled.write(chunk)

    if total == 0 or not header.startswith(PDF_MAGIC):
        spooled.close()
        raise UnsupportedMediaType("Only PDF uploads are supported")

    spooled.seek(0)
    return spooled, total, filename


class UnsupportedMediaType(ValueError):
    """Raised when the upload is not a PDF."""


class PayloadTooLarge(ValueError):
    """Raised when the upload exceeds the configured limit."""


@library_bp.route("/library/papers", methods=["GET"])
@cross_origin()
def get_library_papers():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        saved_only = _parse_saved_only_query()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        return jsonify({"papers": list_user_papers(user_id=user_id, saved_only=saved_only)})
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not load library papers: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")


@library_bp.route("/library/papers/saved", methods=["POST"])
@cross_origin()
def post_library_papers_saved():
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        papers = _validate_bulk_save_papers(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        saved_papers = save_user_papers(user_id=user_id, papers=papers)
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not bulk-save library papers: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return jsonify({"papers": [_save_response(paper) for paper in saved_papers]})


@library_bp.route("/library/papers/import", methods=["POST"])
@cross_origin()
def post_library_papers_import():
    """Save normalized user-supplied paper metadata to the personal shelf.

    Each valid item receives a permanent ``user:`` ID and snapshot. This is
    deliberately separate from the corpus ``/saved`` APIs: imported DOI-backed
    papers must retain their own metadata rather than relying on catalog hydrate.
    """
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return jsonify({"error": "items must be an array"}), 400
    items = payload["items"]
    if not items or len(items) > MAX_IMPORT_ITEMS:
        return jsonify(
            {"error": f"items must contain between 1 and {MAX_IMPORT_ITEMS} items"}
        ), 400

    results: list[dict[str, object]] = []
    valid_items: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        try:
            paper_id, snapshot, raw = _validate_user_import_item(item)
            if paper_id in seen_ids:
                raise ValueError("items must not contain duplicate user paper IDs")
            seen_ids.add(paper_id)
            valid_items.append(
                {
                    "index": index,
                    "paper_id": paper_id,
                    "metadata_snapshot": snapshot,
                    "metadata_raw": raw,
                }
            )
        except ValueError as error:
            results.append({"index": index, "status": "invalid", "reason": str(error)})

    if valid_items:
        try:
            persisted = import_user_papers(user_id=user_id, papers=valid_items)
        except UserPapersPersistenceError as error:
            current_app.logger.error("Could not import library papers: %s", error)
            return Response("Library is unavailable", status=503, mimetype="text/plain")
        if len(persisted) != len(valid_items):
            current_app.logger.error("User paper import returned an unexpected row count")
            return Response("Library is unavailable", status=503, mimetype="text/plain")

        results.extend(
            {
                "index": item["index"],
                "status": "imported",
                "paper_id": item["paper_id"],
                "paper": item["metadata_snapshot"],
            }
            for item in valid_items
        )

    return jsonify({"results": sorted(results, key=lambda result: int(result["index"]))})


@library_bp.route("/library/papers/<path:paper_id>/saved", methods=["PUT"])
@cross_origin()
def put_library_paper_saved(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
        metadata_snapshot = _validate_metadata_snapshot(request.get_json(silent=True), paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        paper, was_created = save_user_paper(
            user_id=user_id, paper_id=paper_id, metadata_snapshot=metadata_snapshot
        )
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not save library paper: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return jsonify(_save_response(paper)), 201 if was_created else 200


@library_bp.route("/library/papers/<path:paper_id>/saved", methods=["DELETE"])
@cross_origin()
def delete_library_paper_saved(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        unsave_user_paper(user_id=user_id, paper_id=paper_id)
    except UserPaperNotFoundError:
        return Response("Not found", status=404, mimetype="text/plain")
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not unsave library paper: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    return Response(status=204)


@library_bp.route("/library/papers/<path:paper_id>/file", methods=["PUT"])
@cross_origin()
def put_library_paper_file(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "file is required"}), 400

    metadata_raw = request.form.get("metadata")
    if not metadata_raw:
        return jsonify({"error": "metadata is required"}), 400
    try:
        metadata_payload = json.loads(metadata_raw)
        metadata_snapshot = _validate_metadata_snapshot(metadata_payload, paper_id)
    except (json.JSONDecodeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    maximum_bytes = _pdf_max_bytes()
    try:
        existing = get_user_paper(user_id=user_id, paper_id=paper_id)
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not load library paper before upload: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")

    old_file_id = existing.get("azure_file_id") if existing else None
    if old_file_id is not None and not isinstance(old_file_id, str):
        old_file_id = None

    spooled = None
    uploaded_azure_id: str | None = None
    try:
        spooled, size, filename = _read_and_validate_pdf(upload, maximum_bytes)
        azure_file = upload_pdf_file(filename=filename, file_obj=spooled)
        uploaded_azure_id = azure_file.id
        persisted_bytes = azure_file.bytes or size
        paper = upsert_user_paper_file(
            user_id=user_id,
            paper_id=paper_id,
            metadata_snapshot=metadata_snapshot,
            azure_file_id=azure_file.id,
            uploaded_filename=azure_file.filename or filename,
            uploaded_bytes=persisted_bytes,
            create_if_missing=True,
        )
    except UnsupportedMediaType as error:
        return Response(str(error), status=415, mimetype="text/plain")
    except PayloadTooLarge as error:
        return Response(str(error), status=413, mimetype="text/plain")
    except AzureFilesConfigurationError:
        current_app.logger.error("Azure OpenAI files are not configured for the library")
        return Response("Library file upload is unavailable", status=503, mimetype="text/plain")
    except AzureFilesTransientError as error:
        current_app.logger.warning(
            "Transient Azure Files failure user_id=%s paper_id=%s error=%s",
            user_id,
            paper_id,
            error,
        )
        return Response("File storage is temporarily unavailable", status=503, mimetype="text/plain")
    except AzureFilesError as error:
        current_app.logger.error(
            "Azure Files upload failed user_id=%s paper_id=%s error=%s",
            user_id,
            paper_id,
            error,
        )
        return Response("File upload failed", status=502, mimetype="text/plain")
    except UserPapersPersistenceError as error:
        current_app.logger.error(
            "Could not persist library file metadata user_id=%s paper_id=%s azure_file_id=%s error=%s",
            user_id,
            paper_id,
            uploaded_azure_id,
            error,
        )
        if uploaded_azure_id:
            try:
                delete_azure_file(file_id=uploaded_azure_id)
            except AzureFilesError as cleanup_error:
                current_app.logger.error(
                    "Orphan Azure file after DB failure file_id=%s user_id=%s paper_id=%s error=%s",
                    uploaded_azure_id,
                    user_id,
                    paper_id,
                    cleanup_error,
                )
        return Response("Library is unavailable", status=503, mimetype="text/plain")
    finally:
        if spooled is not None:
            spooled.close()

    if old_file_id and old_file_id != paper.get("azure_file_id"):
        try:
            delete_azure_file(file_id=old_file_id)
        except AzureFilesError as cleanup_error:
            current_app.logger.warning(
                "Could not delete replaced Azure file file_id=%s user_id=%s paper_id=%s error=%s",
                old_file_id,
                user_id,
                paper_id,
                cleanup_error,
            )

    try:
        paper = index_user_paper(user_id=user_id, paper_id=paper_id)
    except LibraryIndexError as error:
        current_app.logger.warning(
            "Full-text indexing failed user_id=%s paper_id=%s error=%s",
            user_id,
            paper_id,
            error,
            exc_info=True,
        )
        paper = get_user_paper(user_id=user_id, paper_id=paper_id) or paper
    return jsonify(_file_response(paper))


@library_bp.route("/library/papers/<path:paper_id>/file", methods=["DELETE"])
@cross_origin()
def delete_library_paper_file(paper_id: str):
    user_id, error_response = _require_authenticated_user_id()
    if error_response is not None:
        return error_response
    try:
        paper_id = _validate_paper_id(paper_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        paper = get_user_paper(user_id=user_id, paper_id=paper_id)
    except UserPapersPersistenceError as error:
        current_app.logger.error("Could not load library paper before file delete: %s", error)
        return Response("Library is unavailable", status=503, mimetype="text/plain")

    if paper is None or not paper.get("azure_file_id"):
        return Response("Not found", status=404, mimetype="text/plain")

    azure_file_id = paper["azure_file_id"]
    if not isinstance(azure_file_id, str):
        return Response("Not found", status=404, mimetype="text/plain")

    try:
        detach_user_paper_file(user_id=user_id, paper=paper)
        delete_azure_file(file_id=azure_file_id)
    except AzureFilesConfigurationError:
        current_app.logger.error("Azure OpenAI files are not configured for the library")
        return Response("Library file delete is unavailable", status=503, mimetype="text/plain")
    except AzureFilesTransientError as error:
        current_app.logger.warning(
            "Transient Azure Files delete failure user_id=%s paper_id=%s file_id=%s error=%s",
            user_id,
            paper_id,
            azure_file_id,
            error,
        )
        return Response("File storage is temporarily unavailable", status=503, mimetype="text/plain")
    except AzureFilesError as error:
        current_app.logger.error(
            "Azure Files delete failed user_id=%s paper_id=%s file_id=%s error=%s",
            user_id,
            paper_id,
            azure_file_id,
            error,
        )
        return Response("File delete failed", status=502, mimetype="text/plain")

    try:
        clear_user_paper_file(user_id=user_id, paper_id=paper_id)
    except UserPaperNotFoundError:
        return Response("Not found", status=404, mimetype="text/plain")
    except UserPapersPersistenceError as error:
        current_app.logger.error(
            "Cleared Azure file but could not update DB file_id=%s user_id=%s paper_id=%s error=%s",
            azure_file_id,
            user_id,
            paper_id,
            error,
        )
        return Response("Library is unavailable", status=503, mimetype="text/plain")

    return Response(status=204)


__all__ = ["library_bp"]
