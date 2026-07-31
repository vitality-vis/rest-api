"""Legacy selected-paper summary and literature-review HTTP endpoints."""
from __future__ import annotations

import json
import os
import traceback
from collections.abc import Iterator
from typing import Any

from flask import Blueprint, Response, request
from flask_cors import cross_origin
from langchain_openai import AzureChatOpenAI

from prompt import LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT
from service import zilliz

from .grounded_writer import (
    extract_citations_metadata_from_content,
    format_papers_with_segments,
)


legacy_summary_bp = Blueprint("legacy_summary", __name__)


def _legacy_llm() -> AzureChatOpenAI:
    """Construct the model used exclusively by the retired summary routes."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=1,
        streaming=True,
    )


def _stream_with_citations(
    *, prompt_template: str, user_prompt: str, papers: list[dict[str, Any]]
) -> Iterator[str]:
    formatted_content, segments_map = format_papers_with_segments(papers)
    full_content = ""
    rendered_prompt = prompt_template.format(prompt=user_prompt, content=formatted_content)
    for chunk in _legacy_llm().stream(rendered_prompt):
        text = str(chunk.content or "")
        full_content += text
        yield text

    citations_metadata = extract_citations_metadata_from_content(
        full_content, papers, segments_map
    )
    yield "\n\n[[CITATIONS_START]]\n"
    yield json.dumps(citations_metadata, ensure_ascii=False)
    yield "\n[[CITATIONS_END]]"


def _selected_papers() -> tuple[str, list[dict[str, Any]] | None]:
    data = request.json or {}
    prompt = data.get("prompt", "")
    paper_ids = data.get("ids", [])
    if not paper_ids:
        return prompt, None
    return prompt, zilliz.query_doc_by_ids(paper_ids)


@legacy_summary_bp.route("/summarize", methods=["POST"])
@cross_origin()
def summarize() -> Response:
    try:
        prompt, selected_papers = _selected_papers()
        if selected_papers is None:
            return Response("Error: Saved paper list is empty", status=400)
        if not selected_papers:
            return Response("Error: No papers found for the given IDs", status=404)
        return Response(
            _stream_with_citations(
                prompt_template=SUMMARIZE_PROMPT,
                user_prompt=prompt,
                papers=selected_papers,
            ),
            mimetype="text/plain",
        )
    except Exception as error:  # pylint: disable=broad-except
        traceback.print_exc()
        return Response(f"An internal error occurred: {error}", status=500)


@legacy_summary_bp.route("/literatureReview", methods=["POST"])
@cross_origin()
def literature_review() -> Response:
    try:
        prompt, selected_papers = _selected_papers()
        if selected_papers is None:
            return Response("Error: Saved paper list is empty", status=400)
        if not selected_papers:
            return Response("Error: No papers found for the given IDs", status=404)
        return Response(
            _stream_with_citations(
                prompt_template=LITERATURE_REVIEW_PROMPT,
                user_prompt=prompt,
                papers=selected_papers,
            ),
            mimetype="text/plain",
        )
    except Exception as error:  # pylint: disable=broad-except
        traceback.print_exc()
        return Response(f"An internal error occurred: {error}", status=500)
