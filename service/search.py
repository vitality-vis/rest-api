"""
BM25 keyword search service over the paper corpus.
Separated from zilliz.py which handles data access only.
"""
import re
import logging
from typing import List, Optional

from model.const import EMBED
from model.query import QuerySchema

# BM25 index cache: collection_name -> (bm25, corpus, papers)
_bm25_cache = {}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'\w+', text.lower())


def _build_bm25_index(papers: list, collection_name: str):
    """Build and cache a BM25 index over Title + Abstract + Keywords + Authors."""
    from rank_bm25 import BM25Okapi
    corpus = []
    for doc in papers:
        title = str(doc.get("Title") or "")
        abstract = str(doc.get("Abstract") or "")
        keywords = " ".join(doc.get("Keywords") or [])
        authors = " ".join(doc.get("Authors") or [])
        # Weight title more by repeating it
        combined = f"{title} {title} {title} {keywords} {authors} {abstract}"
        corpus.append(_tokenize(combined))
    bm25 = BM25Okapi(corpus)
    _bm25_cache[collection_name] = (bm25, corpus, papers)
    logging.info(f"BM25 index built for {collection_name} ({len(papers)} docs)")
    return bm25, corpus, papers


def _score_distribution(scored: list, num_bins: int = 10) -> list:
    """Return a 10-bin histogram of BM25 scores as list of {min, max, count} buckets."""
    if not scored:
        return []
    scores = [s for s, _ in scored]
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [{"min": round(min_score, 4), "max": round(max_score, 4), "count": len(scores)}]
    bin_size = (max_score - min_score) / num_bins
    buckets = [{"min": round(min_score + i * bin_size, 4), "max": round(min_score + (i + 1) * bin_size, 4), "count": 0} for i in range(num_bins)]
    for s in scores:
        idx = min(int((s - min_score) / bin_size), num_bins - 1)
        buckets[idx]["count"] += 1
    return buckets


def _meaningful_matches(scored: list, num_query_tokens: int = 1) -> dict:
    """
    Estimate meaningful matches using a query-length-aware threshold.
    Threshold = max(num_query_tokens * per_token_baseline, peak_count * 0.1_of_scores_floor).
    A per-token baseline of 1.5 means each query token contributes ~1.5 score units.
    Also floors at 10% of the peak bin score to always cut off very low scorers.
    """
    if not scored:
        return {"threshold": 0.0, "meaningful_matches": 0}
    scores = [s for s, _ in scored]
    per_token_baseline = 1.5
    token_threshold = num_query_tokens * per_token_baseline
    # Floor: 10% of the max score so we always exclude near-zero scorers
    score_floor = max(scores) * 0.1
    threshold = max(token_threshold, score_floor)
    meaningful = sum(1 for s in scores if s >= threshold)
    return {"threshold": round(threshold, 4), "meaningful_matches": meaningful}


def search_papers_bm25(query: str, limit: int = 20, embedding_type: str = EMBED.SPECTER, filters: Optional[QuerySchema] = None) -> dict:
    """
    Google-style fuzzy keyword search using BM25 over Title, Abstract, Keywords, Authors.
    Optionally applies column filters (year, source, author, keyword, etc.) as post-filter.
    Fetches limit * 5 BM25 candidates to ensure enough results survive filtering.
    Returns dict with top `limit` papers sorted by BM25 relevance score, estimated_matches count,
    and a score_distribution histogram (10 bins).
    """
    from service.zilliz import get_cached_papers, match_doc, COLLECTION_MAPPING

    collection_name = COLLECTION_MAPPING.get(embedding_type, "paper_specter")
    all_papers = get_cached_papers(embedding_type)
    if not all_papers:
        return {"papers": [], "estimated_matches": 0, "score_distribution": []}

    # Build or reuse BM25 index
    if collection_name not in _bm25_cache:
        bm25, corpus, papers = _build_bm25_index(all_papers, collection_name)
    else:
        bm25, corpus, papers = _bm25_cache[collection_name]

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return {"papers": [], "estimated_matches": 0, "score_distribution": []}

    scores = bm25.get_scores(tokenized_query)

    # Fetch more candidates than needed so post-filtering still yields `limit` results
    candidates_limit = limit * 5
    scored = [(score, doc) for score, doc in zip(scores, papers) if score > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored[:candidates_limit]

    # Count total matches after applying filters (over all scored candidates, not just top candidates)
    if filters:
        total_matches = sum(1 for _, doc in scored if match_doc(doc, filters))
    else:
        total_matches = len(scored)

    results = []
    for score, doc in top_candidates:
        # Apply column filters if provided
        if filters and not match_doc(doc, filters):
            continue
        result = dict(doc)
        result["bm25_score"] = float(score)
        result["score"] = float(score)
        results.append(result)
        if len(results) >= limit:
            break

    dist = _score_distribution(scored)
    relevance = _meaningful_matches(scored, num_query_tokens=len(tokenized_query))
    return {
        "papers": results,
        "estimated_matches": relevance["meaningful_matches"],
        "score_distribution": dist,
        "score_threshold": relevance["threshold"],
    }


def invalidate_bm25_cache(embedding_type: str = None):
    """Invalidate BM25 cache (call after data reload)."""
    global _bm25_cache
    from service.zilliz import COLLECTION_MAPPING
    if embedding_type:
        collection_name = COLLECTION_MAPPING.get(embedding_type)
        _bm25_cache.pop(collection_name, None)
    else:
        _bm25_cache.clear()


def evaluate_boolean_condition(condition, all_papers):
    """
    Evaluate a single boolean condition and return matching paper IDs.

    Searches across title, abstract, and keywords (combined content).

    Condition format:
    {
        "operator": "AND" | "OR",  # optional, default "OR"
        "keywords": ["keyword1", "keyword2", ...]
    }
    """
    operator = condition.get("operator", "OR").upper()
    keywords = condition.get("keywords", [])

    if not keywords:
        return set()

    matching_ids = set()

    for paper in all_papers:
        paper_id = str(paper.get("ID", ""))
        if not paper_id:
            continue

        # Combine title, abstract, and keywords into one searchable content
        title = str(paper.get("Title", "")).lower()
        abstract = str(paper.get("Abstract", "")).lower()
        kws = paper.get("Keywords", [])
        if isinstance(kws, list):
            keywords_str = " ".join(str(k).lower() for k in kws)
        else:
            keywords_str = str(kws).lower()

        # Combined content: title + abstract + keywords
        content = f"{title} {abstract} {keywords_str}"

        # Check if keywords match in the combined content
        keyword_matches = [kw.lower() in content for kw in keywords]

        if operator == "AND":
            if all(keyword_matches):
                matching_ids.add(paper_id)
        else:  # OR
            if any(keyword_matches):
                matching_ids.add(paper_id)

    return matching_ids


def evaluate_boolean_query(query_tree, all_papers):
    """
    Recursively evaluate a boolean query tree and return matching paper IDs.

    Searches across title, abstract, and keywords fields combined.

    Query tree format:
    {
        "operator": "AND" | "OR" | "NOT",
        "conditions": [
            {
                "operator": "OR",
                "keywords": ["transformer", "attention"]
            },
            {
                "operator": "AND",
                "keywords": ["vision", "image"]
            },
            {
                "operator": "NOT",
                "keywords": ["survey", "review"]
            }
        ]
    }
    """
    operator = query_tree.get("operator", "AND").upper()
    conditions = query_tree.get("conditions", [])

    if not conditions:
        return set()

    # Special handling for NOT operator
    if operator == "NOT":
        # NOT operator: start with all papers and exclude matching ones
        all_paper_ids = {str(p.get("ID", "")) for p in all_papers if p.get("ID")}

        # Evaluate all conditions and combine with OR (exclude any paper matching any condition)
        excluded_ids = set()
        for condition in conditions:
            if "conditions" in condition and isinstance(condition.get("conditions"), list):
                result_set = evaluate_boolean_query(condition, all_papers)
            else:
                result_set = evaluate_boolean_condition(condition, all_papers)
            excluded_ids = excluded_ids.union(result_set)

        return all_paper_ids - excluded_ids

    # Evaluate each condition
    result_sets = []
    for condition in conditions:
        # Check if this is a nested query (has "operator" and "conditions")
        if "conditions" in condition and isinstance(condition.get("conditions"), list):
            # Recursive case: nested query
            result_set = evaluate_boolean_query(condition, all_papers)
        else:
            # Base case: single condition
            result_set = evaluate_boolean_condition(condition, all_papers)
        result_sets.append(result_set)

    # Combine results based on operator
    if operator == "AND":
        return set.intersection(*result_sets) if result_sets else set()
    else:  # OR
        return set.union(*result_sets) if result_sets else set()


def search_papers_boolean(query_tree: dict, limit: int = 100, offset: int = 0, embedding_type: str = EMBED.SPECTER, metadata_filters: Optional[dict] = None) -> dict:
    """
    Boolean query search with structured query format.
    Supports complex AND/OR/NOT logic across title, abstract, and keywords.

    Args:
        query_tree: Query structure with operator and conditions
        limit: Maximum number of results to return (-1 for all)
        offset: Number of results to skip
        embedding_type: Embedding type to use for paper retrieval
        metadata_filters: Optional metadata filters (year, source, citations, etc.)

    Query tree format:
    {
        "operator": "AND" | "OR" | "NOT",
        "conditions": [
            {
                "operator": "OR",
                "keywords": ["machine learning", "deep learning"]
            },
            {
                "operator": "NOT",
                "keywords": ["survey", "review"]
            }
        ]
    }

    Metadata filters format:
    {
        "min_year": 2020,
        "max_year": 2024,
        "sources": ["Nature", "Science"],
        "authors": ["Smith"],
        "keywords": ["neural networks"],
        "min_citations": 10,
        "max_citations": 1000
    }

    Returns: {"papers": [...], "total": count}
    """
    from service.zilliz import get_cached_papers, match_doc

    # Get all papers from cache
    all_papers = get_cached_papers(embedding_type)
    if not all_papers:
        return {"papers": [], "total": 0}

    # Evaluate boolean query
    matching_ids = evaluate_boolean_query(query_tree, all_papers)

    # Filter papers by matching IDs
    matched_papers = [p for p in all_papers if str(p.get("ID", "")) in matching_ids]

    # Apply metadata filters if provided
    if metadata_filters:
        # Convert metadata_filters dict to QuerySchema for compatibility with match_doc
        filters = QuerySchema(
            source=metadata_filters.get("sources"),
            author=metadata_filters.get("authors"),
            keyword=metadata_filters.get("keywords"),
            min_year=metadata_filters.get("min_year"),
            max_year=metadata_filters.get("max_year"),
            min_citation_counts=metadata_filters.get("min_citations"),
            max_citation_counts=metadata_filters.get("max_citations"),
        )
        matched_papers = [p for p in matched_papers if match_doc(p, filters)]

    total_count = len(matched_papers)

    # Apply pagination
    if limit == -1:
        paginated_papers = matched_papers[offset:]
    else:
        paginated_papers = matched_papers[offset:offset + limit]

    logging.info(f"[filter_papers] Query returned {total_count} papers, returning {len(paginated_papers)}")

    return {"papers": paginated_papers, "total": total_count}
