"""
Grounded Writer Module

Provides citation tracking functionality for LLM-generated summaries and literature reviews.
Supports streaming mode with inline citations.
"""

import re
import json
from typing import List, Dict, Tuple


# ============================================================================
# Core Functions
# ============================================================================

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.

    This is a simple sentence splitter. For production use, consider
    using NLTK's sent_tokenize for better accuracy.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []

    # Simple regex-based sentence splitting
    # Matches periods, exclamation marks, or question marks followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def format_papers_with_segments(papers: List[dict]) -> Tuple[str, Dict[int, List[dict]]]:
    """
    Format papers for LLM prompt with sentence-level segmentation.

    Args:
        papers: List of paper dictionaries with keys:
                - ID, Title, Authors, Abstract, Year, Source, etc.

    Returns:
        Tuple of (formatted_content, segments_map)
        - formatted_content: String with formatted papers
        - segments_map: Dict mapping paper_index -> list of segment dicts
    """
    formatted_lines = []
    segments_map = {}

    for idx, paper in enumerate(papers):
        paper_id = paper.get('ID', '')
        title = paper.get('Title', '')
        authors = paper.get('Authors', [])
        year = paper.get('Year', '')
        source = paper.get('Source', '')
        abstract = paper.get('Abstract', '')

        # Format authors
        if isinstance(authors, list):
            authors_str = ', '.join(authors)
        else:
            authors_str = str(authors)

        # Paper header
        formatted_lines.append(f"[{idx}] (ID: {paper_id})")
        formatted_lines.append(f"Title: {title}")
        formatted_lines.append(f"Authors: {authors_str}")
        formatted_lines.append(f"Year: {year}")
        formatted_lines.append(f"Source: {source}")
        formatted_lines.append("")

        # Split abstract into sentences
        sentences = split_into_sentences(abstract)
        segments_map[idx] = []

        if sentences:
            formatted_lines.append("Abstract:")
            for seg_idx, sentence in enumerate(sentences):
                segment_id = f"{idx}.{seg_idx}"
                formatted_lines.append(f"[{segment_id}] \"{sentence}\"")

                # Store segment info
                segments_map[idx].append({
                    "index": seg_idx,
                    "id": segment_id,
                    "text": sentence,
                    "field": "abstract"
                })

        formatted_lines.append("")
        formatted_lines.append("---")
        formatted_lines.append("")

    formatted_content = "\n".join(formatted_lines)
    return formatted_content, segments_map


def extract_citations_metadata_from_content(
    content: str,
    papers: List[dict],
    segments_map: Dict[int, List[dict]]
) -> List[dict]:
    """
    Extract citation metadata from streaming content.

    This function is designed for streaming mode:
    1. Parse content to find citation markers [X.Y]
    2. Build metadata for each citation
    3. Return as list of dicts (ready for JSON serialization)

    Args:
        content: Full content with citation markers [X.Y]
        papers: List of paper dicts
        segments_map: Dict mapping paper_index -> segment list

    Returns:
        List of citation metadata dicts
    """
    # Extract citation markers
    citation_pattern = r'\[(\d+)\.(\d+)\]'
    matches = re.findall(citation_pattern, content)

    # Build citation metadata
    citations_metadata = []
    seen_citations = set()

    for paper_idx_str, segment_idx_str in matches:
        paper_idx = int(paper_idx_str)
        segment_idx = int(segment_idx_str)
        citation_key = (paper_idx, segment_idx)

        # Skip duplicates
        if citation_key in seen_citations:
            continue
        seen_citations.add(citation_key)

        # Validate indices
        if paper_idx >= len(papers):
            continue
        if paper_idx not in segments_map or segment_idx >= len(segments_map[paper_idx]):
            continue

        # Get paper and segment info
        paper = papers[paper_idx]
        segment = segments_map[paper_idx][segment_idx]

        # Build metadata dict
        citations_metadata.append({
            "segment_id": f"{paper_idx}.{segment_idx}",
            "paper_id": str(paper.get('ID', '')),
            "paper_index": paper_idx,
            "segment_index": segment_idx,
            "quoted_text": segment['text']
        })

    return citations_metadata
