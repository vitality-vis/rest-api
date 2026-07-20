"""Conversions between Milvus results and the application's paper dictionaries."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from service.metadata_normalizer import parse_string_list


SCALAR_FIELDS = [
    "ID",
    "Title",
    "Abstract",
    "Authors",
    "Keywords",
    "Source",
    "Year",
    "CitationCounts",
    "Lang",
    "ada_umap",
    "specter_umap",
]


def entity_to_metadata(entity: Any) -> Optional[Dict[str, Any]]:
    """Convert a legacy PyMilvus hit entity into a scalar-field dictionary."""
    if entity is None:
        return None

    metadata: Dict[str, Any] = {}
    if isinstance(entity, dict):
        for key, value in entity.items():
            if key in SCALAR_FIELDS or key == "_score":
                metadata[key] = value
            elif key.lower() in (field.lower() for field in SCALAR_FIELDS):
                canonical_key = next(field for field in SCALAR_FIELDS if field.lower() == key.lower())
                metadata[canonical_key] = value
    else:
        for key in SCALAR_FIELDS:
            if hasattr(entity, "get") and callable(entity.get):
                metadata[key] = entity.get(key) or entity.get(key.lower())
            else:
                metadata[key] = getattr(entity, key, None) or getattr(entity, key.lower(), None)
    if metadata.get("id") is not None and metadata.get("ID") is None:
        metadata["ID"] = metadata["id"]
    return metadata if metadata.get("ID") is not None else None


def row_to_metadata(row: dict) -> dict:
    """Return a Milvus row in the internal metadata representation."""
    return row


def search_hit_to_id_and_distance(hit: Any) -> Tuple[Optional[str], Optional[float]]:
    """Extract the ID and distance from either MilvusClient or legacy ORM hits."""
    if isinstance(hit, dict):
        entity = hit.get("entity") or hit
        document_id = entity.get("ID") or entity.get("id") or hit.get("id")
        distance = hit.get("distance")
    else:
        entity = getattr(hit, "entity", hit)
        if hasattr(entity, "get"):
            document_id = entity.get("ID") or entity.get("id")
        else:
            document_id = getattr(entity, "ID", None) or getattr(entity, "id", None)
        distance = getattr(hit, "distance", None)

    if document_id is None:
        return None, distance
    return str(document_id), distance


def parse_coordinates(value: Any):
    """Decode persisted JSON UMAP coordinates when necessary."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def row_to_umap_point(row: dict) -> dict:
    """Convert one Zilliz row into the UMAP snapshot representation."""
    return {
        "ID": str(row.get("ID")) if row.get("ID") else None,
        "Title": row.get("Title", ""),
        "Year": row.get("Year"),
        "Source": row.get("Source", ""),
        "ada_umap": parse_coordinates(row.get("ada_umap")),
        "specter_umap": parse_coordinates(row.get("specter_umap")),
    }


def rows_to_umap_points(rows: List[dict]) -> List[dict]:
    """Convert Zilliz rows into the UMAP snapshot format."""
    return [row_to_umap_point(row) for row in rows]


def paper_to_api_response(doc: dict, score_key: str = "_score") -> dict:
    """Format a paper dict into the legacy frontend API representation.

    TODO: Move this API-specific serializer to ``api/schemas`` or
    ``api/serializers`` when response schemas are introduced. It remains here
    during the repository migration to keep the existing frontend contract
    stable.
    """
    distance = doc.get(score_key)
    similarity = 0.0
    try:
        numeric_distance = float(distance) if distance is not None else float("nan")
        if not math.isnan(numeric_distance):
            similarity = 1.0 / (1.0 + numeric_distance)
    except Exception:
        pass

    def value_for(key: str):
        return doc.get(key) or doc.get(key.lower()) or ""

    return {
        "ID": doc.get("ID") or doc.get("id"),
        "Title": value_for("Title"),
        "Abstract": value_for("Abstract"),
        "Authors": parse_string_list(doc.get("Authors") or doc.get("authors") or ""),
        "Keywords": parse_string_list(doc.get("Keywords") or doc.get("keywords") or ""),
        "Source": value_for("Source"),
        "Year": doc.get("Year") if doc.get("Year") is not None else doc.get("year"),
        "CitationCounts": (
            doc.get("CitationCounts")
            if doc.get("CitationCounts") is not None
            else doc.get("citationcounts")
        ),
        "_Sim": similarity,
        "Sim": similarity,
        "score": similarity,
        "ada_umap": parse_coordinates(doc.get("ada_umap")),
        "specter_umap": parse_coordinates(doc.get("specter_umap")),
    }
