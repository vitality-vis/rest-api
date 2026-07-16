"""Backend-only Zotero Web API client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from logger_config import get_logger

logger = get_logger()


class ZoteroAPIError(Exception):
    """Structured error for Zotero API failures."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class ZoteroCredentials:
    user_id: str
    api_key: str


class ZoteroClient:
    """Small read-only client for Zotero user libraries."""

    def __init__(
        self,
        user_id: str,
        api_key: str,
        *,
        timeout: Tuple[int, int] = (10, 60),
    ) -> None:
        self.credentials = ZoteroCredentials(
            user_id=str(user_id).strip(),
            api_key=str(api_key).strip(),
        )
        self.timeout = timeout
        self.base_url = config.ZOTERO_API_BASE_URL.rstrip("/")
        self.api_version = str(config.ZOTERO_API_VERSION)

    def validate_credentials(self) -> Dict[str, Any]:
        """Validate the API key and confirm it can read the target library."""
        try:
            key_info = self._request_json("GET", f"/keys/{self.credentials.api_key}")
        except ZoteroAPIError as exc:
            if exc.code in {"zotero_not_found", "zotero_invalid_key"}:
                raise ZoteroAPIError(
                    401,
                    "zotero_invalid_key",
                    "Invalid Zotero API key.",
                ) from exc
            raise
        resolved_user_id = str(
            key_info.get("userID")
            or key_info.get("userId")
            or key_info.get("user")
            or ""
        ).strip()

        if not resolved_user_id:
            raise ZoteroAPIError(
                401,
                "zotero_invalid_key",
                "Zotero returned an invalid API key response.",
            )

        if resolved_user_id != self.credentials.user_id:
            raise ZoteroAPIError(
                403,
                "zotero_user_mismatch",
                "The provided Zotero User ID does not match the API key owner.",
            )

        # Validate actual library read access as well.
        self._request_json(
            "GET",
            f"/users/{self.credentials.user_id}/collections/top",
            params={"limit": 1, "format": "json"},
        )
        return key_info

    def list_collections(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        payload = self._request_json(
            "GET",
            f"/users/{self.credentials.user_id}/collections/top",
            params={"limit": max(1, min(limit, 100)), "format": "json"},
        )
        return [self._normalize_collection(obj) for obj in payload]

    def list_items(
        self,
        *,
        collection_key: Optional[str] = None,
        limit: int = 25,
        start: int = 0,
    ) -> List[Dict[str, Any]]:
        path = f"/users/{self.credentials.user_id}/items/top"
        if collection_key:
            path = (
                f"/users/{self.credentials.user_id}/collections/"
                f"{collection_key}/items/top"
            )

        payload = self._request_json(
            "GET",
            path,
            params={
                "limit": max(1, min(limit, 100)),
                "start": max(0, int(start)),
                "format": "json",
            },
        )
        items = [self._normalize_item(obj) for obj in payload]
        kept: List[Dict[str, Any]] = []
        for item in items:
            item_type = item.get("item_type")
            if item_type == "note":
                continue
            # Standalone top-level PDF attachments (PDFs dragged straight into the
            # library with no parent item) are valid, importable documents, so keep
            # them. Other attachments (snapshots, links, non-PDF files) are skipped.
            if item_type == "attachment":
                is_pdf = item.get("content_type") == "application/pdf"
                if is_pdf and not item.get("parent_item"):
                    kept.append(item)
                continue
            kept.append(item)
        return kept

    def get_item(self, item_key: str) -> Dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"/users/{self.credentials.user_id}/items/{item_key}",
            params={"format": "json"},
        )
        return self._normalize_item(payload)

    def list_item_children(self, item_key: str) -> List[Dict[str, Any]]:
        payload = self._request_json(
            "GET",
            f"/users/{self.credentials.user_id}/items/{item_key}/children",
            params={"limit": 100, "format": "json"},
        )
        return [self._normalize_item(obj) for obj in payload]

    def list_pdf_attachments(self, item_key: str) -> List[Dict[str, Any]]:
        children = self.list_item_children(item_key)
        attachments = []
        for child in children:
            if child.get("item_type") != "attachment":
                continue
            if child.get("content_type") != "application/pdf":
                continue
            attachments.append(child)
        return attachments

    def download_attachment_pdf(self, attachment_key: str) -> requests.Response:
        return self._request(
            "GET",
            f"/users/{self.credentials.user_id}/items/{attachment_key}/file",
            stream=True,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = self._request(method, path, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ZoteroAPIError(
                502,
                "zotero_invalid_response",
                "Zotero returned a non-JSON response.",
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Zotero-API-Key": self.credentials.api_key,
            "Zotero-API-Version": self.api_version,
        }
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=self.timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise ZoteroAPIError(
                502,
                "zotero_network_error",
                f"Could not reach Zotero: {exc}",
            ) from exc

        if response.status_code in {200, 201, 204}:
            return response

        status_code = response.status_code
        detail = response.text.strip()[:500] if response.text else ""

        if status_code == 401:
            raise ZoteroAPIError(
                401,
                "zotero_invalid_key",
                "Invalid Zotero API key.",
            )
        if status_code == 404:
            raise ZoteroAPIError(
                404,
                "zotero_not_found",
                "The requested Zotero resource was not found.",
            )
        if status_code == 403:
            raise ZoteroAPIError(
                403,
                "zotero_forbidden",
                "The Zotero API key does not have permission to read this library.",
            )
        if status_code == 429:
            raise ZoteroAPIError(
                429,
                "zotero_rate_limited",
                "Zotero rate-limited this request. Please try again shortly.",
            )

        raise ZoteroAPIError(
            502,
            "zotero_api_error",
            f"Zotero request failed with HTTP {status_code}. {detail}".strip(),
        )

    @staticmethod
    def _normalize_collection(raw: Dict[str, Any]) -> Dict[str, Any]:
        data = raw.get("data", raw)
        return {
            "key": data.get("key"),
            "name": data.get("name") or data.get("title") or "",
            "parent_key": data.get("parentCollection"),
        }

    @staticmethod
    def _normalize_item(raw: Dict[str, Any]) -> Dict[str, Any]:
        data = raw.get("data", raw)
        creators = data.get("creators") or []
        authors = []
        for creator in creators:
            if not isinstance(creator, dict):
                continue
            name = creator.get("name")
            if name:
                authors.append(str(name))
                continue
            first_name = str(creator.get("firstName") or "").strip()
            last_name = str(creator.get("lastName") or "").strip()
            full_name = " ".join(part for part in [first_name, last_name] if part)
            if full_name:
                authors.append(full_name)

        return {
            "key": data.get("key"),
            "parent_item": data.get("parentItem"),
            "item_type": data.get("itemType"),
            "title": data.get("title") or "",
            "date": data.get("date") or "",
            "abstract_note": data.get("abstractNote") or "",
            "publication_title": data.get("publicationTitle") or "",
            "url": data.get("url") or "",
            "doi": data.get("DOI") or data.get("doi") or "",
            "collections": data.get("collections") or [],
            "tags": [tag.get("tag") for tag in (data.get("tags") or []) if isinstance(tag, dict)],
            "authors": authors,
            "content_type": data.get("contentType") or "",
            "filename": data.get("filename") or "",
            "link_mode": data.get("linkMode") or "",
        }
