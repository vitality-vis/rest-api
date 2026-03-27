import ast
import json
from typing import Any, Dict, List


def parse_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        flattened: List[str] = []
        for item in value:
            flattened.extend(parse_string_list(item))
        return flattened

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        parsed = _parse_list_like_string(raw)
        if parsed is not None:
            return parse_string_list(parsed)

        return [part.strip() for part in raw.split(",") if part.strip()]

    normalized = str(value).strip()
    return [normalized] if normalized else []


def normalize_summary_entries(entries: Any) -> List[Dict[str, int]]:
    if not isinstance(entries, list):
        return []

    counter: Dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        try:
            count = int(entry.get("count", 0))
        except Exception:
            count = 0

        if count <= 0:
            continue

        for item in parse_string_list(entry.get("_id")):
            counter[item] = counter.get(item, 0) + count

    return sorted(
        [{"_id": key, "count": value} for key, value in counter.items()],
        key=lambda item: (-item["count"], item["_id"]),
    )


def normalize_aggregated_metadata(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    normalized = dict(data)
    for field in ("authors_summary", "keywords_summary"):
        normalized[field] = normalize_summary_entries(normalized.get(field, []))
    return normalized


def _parse_list_like_string(raw: str):
    if not (raw.startswith("[") and raw.endswith("]")):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed

    return None
