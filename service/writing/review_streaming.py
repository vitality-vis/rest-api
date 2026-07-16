"""Streaming helpers for summaries and literature reviews."""

import json
import os
from typing import Dict, Iterable, List

from langchain_openai import AzureChatOpenAI

from prompt import LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT
from service.writing.grounded_writer import (
    extract_citations_metadata_from_content,
)


_LLM = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=1,  # GPT-5 only supports temperature=1
    streaming=True,
)


def format_papers_in_prompt(papers: List[dict]) -> str:
    lines = []
    for paper in papers:
        title = paper.get("Title", "")
        authors_data = paper.get("Authors", [])
        authors = ", ".join(authors_data) if isinstance(authors_data, list) else str(authors_data)
        abstract = paper.get("Abstract", "")
        source = paper.get("Source", "")
        year = paper.get("Year", "")
        keywords_data = paper.get("Keywords", [])
        keywords = ", ".join(keywords_data) if isinstance(keywords_data, list) else str(keywords_data)

        lines.append(
            f" --- \nTitle: {title}\nAuthors: {authors}\nAbstract: {abstract}\n"
            f"Source: {source}\nYear: {year}\nKeywords: {keywords}\n ---"
        )
    return "\n".join(lines)


def summarize_output(prompt_data: Dict[str, str]) -> Iterable[str]:
    return (chunk.content for chunk in _LLM.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


def literature_review_output(prompt_data: Dict[str, str]) -> Iterable[str]:
    return (chunk.content for chunk in _LLM.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))


def summarize_output_streaming_with_citations(
    prompt_data: Dict[str, str],
    papers: List[dict],
    segments_map: Dict[int, List[dict]],
):
    yield from _stream_with_citations(SUMMARIZE_PROMPT, prompt_data, papers, segments_map)


def literature_review_output_streaming_with_citations(
    prompt_data: Dict[str, str],
    papers: List[dict],
    segments_map: Dict[int, List[dict]],
):
    yield from _stream_with_citations(LITERATURE_REVIEW_PROMPT, prompt_data, papers, segments_map)


def _stream_with_citations(
    prompt_template: str,
    prompt_data: Dict[str, str],
    papers: List[dict],
    segments_map: Dict[int, List[dict]],
):
    formatted_prompt = prompt_template.format(
        prompt=prompt_data["prompt"],
        content=prompt_data["content"],
    )

    full_content = ""
    for chunk in _LLM.stream(formatted_prompt):
        text = chunk.content
        full_content += text
        yield text

    citations_metadata = extract_citations_metadata_from_content(
        full_content,
        papers,
        segments_map,
    )

    yield (
        "\n\n[[CITATIONS_START]]\n"
        + json.dumps(citations_metadata, ensure_ascii=False)
        + "\n[[CITATIONS_END]]"
    )
