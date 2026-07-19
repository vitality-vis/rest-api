"""Local static cache for /getMetaData and /getUmapPoints.

Owns fingerprint comparison, facet aggregation, disk snapshots, and in-process
CachedData. Zilliz IO primitives stay in service.zilliz.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from logger_config import get_logger
from model.const import EMBED
from service import zilliz
from service.metadata_normalizer import normalize_aggregated_metadata, parse_string_list

logging = get_logger()


class ZillizFingerprintError(RuntimeError):
    """A collection fingerprint could not be read from Zilliz."""


class ZillizNotConfiguredError(ZillizFingerprintError):
    """Zilliz credentials are not configured for this process."""


def read_collection_fingerprint(embedding_type: str = EMBED.SPECTER) -> Dict[str, Any]:
    """Read a collection change detector, raising a clear error on failure.

    Uses read-only collection metadata and statistics APIs. Callers that can
    safely fall back to local data should use get_collection_fingerprint().
    """
    collection_name = zilliz.COLLECTION_MAPPING.get(embedding_type, "paper_specter")
    if not config.ZILLIZ_URI or not config.ZILLIZ_TOKEN:
        raise ZillizNotConfiguredError("ZILLIZ_URI / ZILLIZ_TOKEN not set")
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
        info = client.describe_collection(collection_name=collection_name) or {}
        update_ts = info.get("update_timestamp")
        if update_ts is None:
            update_ts = info.get("updated_timestamp")

        row_count = None
        try:
            if hasattr(client, "get_collection_stats"):
                stats = client.get_collection_stats(collection_name=collection_name) or {}
            else:
                stats = {}
            if isinstance(stats, dict):
                row_count = stats.get("row_count")
                if row_count is None:
                    row_count = stats.get("rowCount")
        except Exception as e:
            logging.warning(
                "Could not read row count for Zilliz collection %s: %s",
                collection_name,
                e,
            )

        if update_ts is None and row_count is None:
            raise ZillizFingerprintError(
                "Zilliz returned neither update_timestamp nor row_count for "
                f"collection '{collection_name}'"
            )
        return {
            "collection": collection_name,
            "update_timestamp": int(update_ts) if update_ts is not None else None,
            "row_count": int(row_count) if row_count is not None else None,
        }
    except Exception as e:
        if isinstance(e, ZillizFingerprintError):
            raise
        raise ZillizFingerprintError(
            f"Failed to read Zilliz collection fingerprint for "
            f"'{collection_name}': {e}"
        ) from e


def get_collection_fingerprint(embedding_type: str = EMBED.SPECTER) -> Optional[Dict[str, Any]]:
    """Return a fingerprint, or None when the caller can safely fall back."""
    try:
        return read_collection_fingerprint(embedding_type)
    except ZillizFingerprintError as e:
        logging.warning("%s", e)
        return None


def fingerprints_match(
    local: Optional[Dict[str, Any]],
    remote: Optional[Dict[str, Any]],
) -> bool:
    if not local or not remote:
        return False
    if local.get("collection") != remote.get("collection"):
        return False
    if local.get("update_timestamp") != remote.get("update_timestamp"):
        return False
    if local.get("row_count") != remote.get("row_count"):
        return False
    return True


def get_aggregated_metadata(
    embedding_type: str = EMBED.SPECTER,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate filter facets from Zilliz rows.

    Default sample_limit=None fetches the full collection (needed for accurate
    filters / static cache export). Pass an int only for cheap approximate samples.
    """
    docs = zilliz.get_all_metadatas(embedding_type, limit=sample_limit)
    return aggregate_metadata(docs)


def aggregate_metadata(docs: List[dict]) -> Dict[str, Any]:
    """Aggregate filter facets from already-fetched Zilliz rows."""
    if not docs:
        return {
            "authors_summary": [],
            "sources_summary": [],
            "keywords_summary": [],
            "years_summary": [],
            "citation_counts": [],
            "sample_size": 0,
        }

    def aggregate_count(field: str) -> List[Dict[str, Any]]:
        counter = {}
        for doc in docs:
            values = doc.get(field)
            if values is None:
                continue
            if field in ("Authors", "Keywords"):
                values = parse_string_list(values)
            elif not isinstance(values, list):
                values = [values]
            for v in values:
                if v:
                    key_str = str(v).strip()
                    if key_str:
                        counter[key_str] = counter.get(key_str, 0) + 1
        return sorted(
            [{"_id": k, "count": v} for k, v in counter.items()],
            key=lambda x: -x["count"],
        )

    citation_counts = sorted(
        set(
            doc.get("CitationCounts")
            for doc in docs
            if doc.get("CitationCounts") is not None
        )
    )

    return {
        "authors_summary": aggregate_count("Authors"),
        "sources_summary": aggregate_count("Source"),
        "keywords_summary": aggregate_count("Keywords"),
        "years_summary": sorted(aggregate_count("Year"), key=lambda x: x["_id"]),
        "citation_counts": citation_counts,
        "sample_size": len(docs),
    }


def write_static_cache_from_zilliz(
    embedding_type: str = EMBED.SPECTER,
    fingerprint: Optional[dict] = None,
) -> Dict[str, Any]:
    """Pull meta + UMAP from Zilliz and write local snapshot files + fingerprint."""
    if fingerprint is None:
        logging.info("Reading Zilliz collection fingerprint...")
        fingerprint = get_collection_fingerprint(embedding_type)
    if not fingerprint:
        raise RuntimeError("Could not read Zilliz collection fingerprint")
    logging.info("Fingerprint: %s", fingerprint)

    logging.info("Fetching metadata and UMAP points from Zilliz in one pass...")
    rows = zilliz.get_all_static_cache_rows(embedding_type)
    expected_row_count = fingerprint.get("row_count")
    if expected_row_count is not None and len(rows) != int(expected_row_count):
        raise RuntimeError(
            "Refusing to replace static cache with incomplete Zilliz data: "
            f"expected {expected_row_count} rows, received {len(rows)}"
        )
    logging.info("Building metadata and UMAP snapshots from %s rows...", len(rows))
    metadata = normalize_aggregated_metadata(aggregate_metadata(rows))
    umap_points = zilliz.format_umap_points(rows)

    meta_path = Path(config.meta_data_file_path)
    umap_path = Path(config.umap_data_file_path)
    fp_path = Path(config.cache_fingerprint_file_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    umap_path.write_text(json.dumps(umap_points, ensure_ascii=False), encoding="utf-8")
    fp_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info(
        "Static cache written: meta=%s umap_points=%s fingerprint=%s",
        meta_path,
        len(umap_points),
        fingerprint,
    )
    return {
        "metadata": metadata,
        "umap_points": umap_points,
        "fingerprint": fingerprint,
    }


def load_json_file(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Could not load JSON file %s: %s", path, e)
        return default


class CachedData:
    umap_points = None
    meta_datas = None
    aggregated_metadata = None
    fingerprint = None

    def init(self, embedding_type: str = EMBED.SPECTER):
        logging.info("Initializing cached data...")
        meta_path = Path(config.meta_data_file_path)
        umap_path = Path(config.umap_data_file_path)
        fp_path = Path(config.cache_fingerprint_file_path)

        local_fp = load_json_file(str(fp_path), None)
        remote_fp = get_collection_fingerprint(embedding_type)
        files_ok = meta_path.is_file() and umap_path.is_file()

        if files_ok and remote_fp and fingerprints_match(local_fp, remote_fp):
            logging.info(
                "Static cache fingerprint matches Zilliz (%s row_count=%s); loading local files",
                remote_fp.get("collection"),
                remote_fp.get("row_count"),
            )
            self._load_from_disk()
            self.fingerprint = remote_fp
            return

        if remote_fp:
            logging.info(
                "Static cache stale or missing (local=%s remote=%s); refreshing from Zilliz",
                local_fp,
                remote_fp,
            )
            try:
                write_static_cache_from_zilliz(embedding_type, fingerprint=remote_fp)
                self._load_from_disk()
                self.fingerprint = remote_fp
                return
            except Exception as e:
                logging.error(
                    "Failed to refresh static cache from Zilliz: %s",
                    e,
                    exc_info=True,
                )

        if files_ok:
            logging.warning(
                "Using existing local static cache "
                "(Zilliz fingerprint unavailable or refresh failed)"
            )
            self._load_from_disk()
            self.fingerprint = local_fp
            return

        logging.warning(
            "No static cache available; /getMetaData and /getUmapPoints may be empty/slow"
        )
        self.umap_points = []
        self.meta_datas = {}
        self.aggregated_metadata = {}
        self.fingerprint = remote_fp or local_fp

    def _load_from_disk(self):
        self.umap_points = load_json_file(config.umap_data_file_path, [])
        logging.info(
            "Loaded %s UMAP points from file",
            len(self.umap_points) if self.umap_points else 0,
        )
        self.meta_datas = load_json_file(config.meta_data_file_path, {})
        self.aggregated_metadata = normalize_aggregated_metadata(self.meta_datas)
        logging.info("Loaded aggregated metadata from file")

    def get_umap_points(self):
        return self.umap_points

    def get_meta_datas(self):
        return self.meta_datas

    def get_aggregated_metadata(self):
        """Return cached aggregated metadata."""
        return self.aggregated_metadata


cached_data = CachedData()
