#!/usr/bin/env python3
"""Run labelled, retrieval-free evaluations of the v2 top-level router."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REST_API_ROOT = Path(__file__).resolve().parents[1]
if str(REST_API_ROOT) not in sys.path:
    sys.path.insert(0, str(REST_API_ROOT))

from agents.agent_v2.logging import SearchV2Trace
from agents.agent_v2.models import V2ChatRequest
from agents.agent_v2.router import route


EVALS_DIR = REST_API_ROOT / "evals"
DEFAULT_CASES_PATH = EVALS_DIR / "router_cases.jsonl"


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            expected = case["expected"]
            if not isinstance(case["id"], str) or not isinstance(case["text"], str):
                raise ValueError("id and text must be strings")
            if expected["route"] not in {"talk", "search", "synthesis", "clarify"}:
                raise ValueError("expected.route is invalid")
            if expected.get("response_mode") not in {None, "papers", "grounded_answer"}:
                raise ValueError("expected.response_mode is invalid")
            if "search_intent" in expected and expected["search_intent"] is not None:
                if not isinstance(expected["search_intent"], dict):
                    raise ValueError("expected.search_intent must be an object or null")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid case at {path}:{line_number}: {error}") from error
        cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def _dump_model(value: Any) -> Any:
    if value is None:
        return None
    return value.model_dump() if hasattr(value, "model_dump") else value.dict()


def _search_intent_matches(expected: dict[str, Any], actual_intent: dict[str, Any] | None) -> bool:
    """Partial match: only keys present under expected.search_intent are checked."""
    if "search_intent" not in expected:
        return True
    expected_intent = expected["search_intent"]
    if expected_intent is None:
        return actual_intent is None
    if actual_intent is None:
        return False
    for key, value in expected_intent.items():
        if actual_intent.get(key) != value:
            return False
    return True


def _evaluate_case(case: dict[str, Any], attempt: int) -> dict[str, Any]:
    request = V2ChatRequest(
        text=case["text"],
        history=case.get("history"),
        selected_paper_ids=case.get("selected_paper_ids"),
        context=case.get("context"),
        requested_mode=case.get("requested_mode", "auto"),
    )
    decision = route(
        request,
        trace=SearchV2Trace.create(trace_id=f"router-eval-{case['id']}-{attempt}"),
    )
    expected = case["expected"]
    actual_intent = _dump_model(decision.search_intent)
    route_correct = decision.route == expected["route"]
    mode_correct = decision.response_mode == expected.get("response_mode")
    intent_correct = _search_intent_matches(expected, actual_intent)
    return {
        "case_id": case["id"],
        "attempt": attempt,
        "expected": expected,
        "actual": {
            "route": decision.route,
            "response_mode": decision.response_mode,
            "decision_status": decision.decision_status,
            "search_intent": actual_intent,
        },
        "route_correct": route_correct,
        "response_mode_correct": mode_correct,
        "search_intent_correct": intent_correct,
        "joint_correct": route_correct and mode_correct and intent_correct,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_search = [record for record in records if record["expected"]["route"] == "search"]
    expected_non_search = [record for record in records if record["expected"]["route"] != "search"]
    intent_labelled = [record for record in records if "search_intent" in record["expected"]]
    decision_statuses = Counter(record["actual"]["decision_status"] for record in records)
    actual_search_in_expected_search = sum(
        record["actual"]["route"] == "search" for record in expected_search
    )
    false_searches = sum(
        record["actual"]["route"] == "search" for record in expected_non_search
    )
    return {
        "runs": len(records),
        "route_accuracy": _ratio(sum(record["route_correct"] for record in records), len(records)),
        "search_response_mode_accuracy": _ratio(
            sum(record["response_mode_correct"] for record in expected_search),
            len(expected_search),
        ),
        "search_intent_accuracy": _ratio(
            sum(record["search_intent_correct"] for record in intent_labelled),
            len(intent_labelled),
        ),
        "joint_accuracy": _ratio(sum(record["joint_correct"] for record in records), len(records)),
        "expected_search_recall": _ratio(actual_search_in_expected_search, len(expected_search)),
        "false_search_rate": _ratio(false_searches, len(expected_non_search)),
        "decision_status_counts": dict(sorted(decision_statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be at least 1")

    cases = _read_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or EVALS_DIR / "results" / f"router_eval_{timestamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            for attempt in range(1, args.repeat + 1):
                record = _evaluate_case(case, attempt)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summary(records)
    print(
        json.dumps(
            {"cases": len(cases), "repeat": args.repeat, "output": str(output), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
