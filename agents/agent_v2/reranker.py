"""CrossEncoder reranking isolated from the legacy agent."""
from __future__ import annotations

from typing import Iterable


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def paper_id(paper: dict) -> str:
    return str(paper.get("ID") or paper.get("id") or paper.get("paper_id") or "").strip()


def paper_text(paper: dict) -> str:
    title = str(paper.get("Title") or paper.get("title") or "").strip()
    abstract = str(paper.get("Abstract") or paper.get("abstract") or "").strip()
    return f"{title}\n\n{abstract}".strip()


def rerank(query: str, papers: Iterable[dict]) -> list[tuple[dict, float]]:
    values = list(papers)
    scores = _get_model().predict([(query, paper_text(paper)) for paper in values])
    ranked = list(zip(values, (float(score) for score in scores)))
    return sorted(ranked, key=lambda item: (-item[1], paper_id(item[0])))
