import ast
import json
from typing import Any, Dict, List


def parse_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, tuple):
        return parse_string_list(list(value))

    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes, dict, list)):
        try:
            return parse_string_list(tolist())
        except Exception:
            pass

    if isinstance(value, list):
        flattened: List[str] = []
        for item in value:
            flattened.extend(parse_string_list(item))

        while len(flattened) == 1 and isinstance(flattened[0], str):
            raw = _strip_outer_quotes(flattened[0].strip())
            nested = _parse_list_like_string(raw)
            if nested is None:
                break
            new_flat = [str(x).strip() for x in nested if str(x).strip()]
            if not new_flat or new_flat == flattened:
                break
            flattened = new_flat
        return flattened

    if isinstance(value, str):
        raw = _strip_outer_quotes(value.strip())
        if not raw:
            return []

        parsed = _parse_list_like_string(raw)
        if parsed is not None:
            return parse_string_list(parsed)

        return [part.strip() for part in raw.split(",") if part.strip()]

    coerced = str(value).strip()
    if not coerced:
        return []
    if coerced.startswith("[") and coerced.endswith("]"):
        parsed = _parse_list_like_string(_strip_outer_quotes(coerced))
        if parsed is not None:
            return [str(x).strip() for x in parsed if str(x).strip()]
    return [coerced]


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


def _strip_outer_quotes(s: str) -> str:
    """Remove one layer of matching ' or " wrappers (common with escaped exports)."""
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    return s


def _parse_list_like_string(raw: str):
    raw = _strip_outer_quotes(raw.strip())
    if not (raw.startswith("[") and raw.endswith("]")):
        return None
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed

    return None
