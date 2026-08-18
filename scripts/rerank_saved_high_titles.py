#!/usr/bin/env python3
"""Title-rerank a saved benchmark run without repeating retrieval."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent_v2.llm_reranker import score_batch  # noqa: E402
from repositories.zilliz import paper_repository  # noqa: E402
from service.llm import get_llm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--parallel-batches", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paper_id(paper: dict) -> str:
    return str(paper.get("paper_uid") or paper.get("ID") or paper.get("id") or "").strip()


def load_metadata(ids: list[str]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for start in range(0, len(ids), 100):
        for paper in paper_repository.get_papers_by_ids(ids[start : start + 100]):
            identifier = paper_id(paper)
            if identifier:
                records[identifier.casefold()] = paper
    return records


def rerank_row(
    row: dict,
    *,
    metadata: dict[str, dict],
    llm,
    model: str,
    batch_size: int,
    parallel_batches: int,
) -> dict:
    results = list(row.get("results") or [])
    papers = [metadata.get(paper_id(result).casefold(), {"paper_uid": paper_id(result), "title": ""}) for result in results]
    batches = [papers[start : start + batch_size] for start in range(0, len(papers), batch_size)]

    def score(items: list[dict]) -> dict[str, float]:
        return score_batch(row.get("query", ""), items, screening_fields="title", llm=llm)

    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=min(parallel_batches, len(batches))) as executor:
        for batch_scores in executor.map(score, batches):
            scores.update(batch_scores)

    original_rank = {paper_id(result): index for index, result in enumerate(results)}
    reranked = sorted(
        results,
        key=lambda result: (
            -scores.get(paper_id(result), -1.0),
            original_rank[paper_id(result)],
        ),
    )
    output = dict(row)
    output["results"] = [
        {
            **result,
            "rank": rank,
            "title_rerank_score": scores.get(paper_id(result), -1.0),
        }
        for rank, result in enumerate(reranked, start=1)
    ]
    output["title_rerank"] = {
        "model": model,
        "fields": "title",
        "scored": len(scores),
        "policy": "pure_rerank_no_filter",
    }
    return output


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.parallel_batches < 1:
        raise SystemExit("batch sizes must be positive")
    rows = read_jsonl(args.input)
    ids = list(
        dict.fromkeys(
            paper_id(result)
            for row in rows
            for result in row.get("results") or []
            if paper_id(result)
        )
    )
    metadata = load_metadata(ids)
    missing = len(ids) - len(metadata)
    print(f"Loaded metadata for {len(metadata)}/{len(ids)} papers; missing={missing}")
    llm = get_llm(model=args.model)
    reranked = []
    for index, row in enumerate(rows, start=1):
        reranked.append(
            rerank_row(
                row,
                metadata=metadata,
                llm=llm,
                model=args.model,
                batch_size=args.batch_size,
                parallel_batches=args.parallel_batches,
            )
        )
        print(f"Reranked {index}/{len(rows)}: {row.get('id', '')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reranked),
        encoding="utf-8",
    )
    temp.replace(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
