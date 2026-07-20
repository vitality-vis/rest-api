"""MilvusClient connection and collection lifecycle management.

This module is deliberately unaware of papers, filters, and API response
formats. It owns only the Zilliz transport state shared by repositories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Set

import config
from logger_config import get_logger


logging = get_logger()


@dataclass
class _ConnectionState:
    client: Optional[Any] = None
    loaded_collections: Set[str] = field(default_factory=set)


_state = _ConnectionState()


def get_client():
    """Return the shared MilvusClient, creating it on first use."""
    if _state.client is not None:
        return _state.client
    if not config.ZILLIZ_URI or not config.ZILLIZ_TOKEN:
        logging.error("ZILLIZ_URI and ZILLIZ_TOKEN must be set (e.g. in .env)")
        return None

    try:
        from pymilvus import MilvusClient

        _state.client = MilvusClient(
            uri=config.ZILLIZ_URI,
            token=config.ZILLIZ_TOKEN,
        )
        logging.info("Zilliz Cloud MilvusClient established")
        return _state.client
    except ImportError as error:
        logging.error("pymilvus not installed: %s", error)
    except Exception as error:
        logging.error("Failed to create Zilliz MilvusClient: %s", error, exc_info=True)
    return None


def ensure_collection_loaded(collection_name: str) -> bool:
    """Verify that a collection exists and load it once for this process."""
    client = get_client()
    if not client:
        return False

    try:
        if not client.has_collection(collection_name):
            logging.error(
                "Collection '%s' does not exist. Run load_to_zilliz.py first.",
                collection_name,
            )
            return False
        if collection_name not in _state.loaded_collections:
            client.load_collection(collection_name)
            _state.loaded_collections.add(collection_name)
        return True
    except Exception as error:
        logging.error(
            "Failed to prepare collection '%s': %s",
            collection_name,
            error,
            exc_info=True,
        )
        return False
