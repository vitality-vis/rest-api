"""Repository API for paper data stored in Zilliz.

This is the migration boundary for the existing ``service.zilliz`` module.
Callers should depend on these data-oriented operations, not on pymilvus
collections or Milvus expression syntax.  The implementation delegates to the
legacy module for now so that route migration does not alter API behaviour.

As ``service.zilliz`` is split, its concrete implementations should move into
this package (connection.py, query_expressions.py, mappers.py, and this file)
without changing this public API.
"""
from typing import Dict, List, Optional

from model.const import EMBED
from model.query import QuerySchema
from service import zilliz as _legacy_zilliz


def search_papers(
    query: QuerySchema,
    embedding_type: str = EMBED.SPECTER,
) -> Dict:
    """Return one filtered, paginated page of papers."""
    return _legacy_zilliz.query_docs(query, embedding_type=embedding_type)


def get_paper_by_id(
    paper_id: str,
    embedding_type: str = EMBED.SPECTER,
) -> Optional[dict]:
    """Return one paper, or ``None`` when it is not present."""
    return _legacy_zilliz.query_doc_by_id(paper_id, embedding_type=embedding_type)


def get_papers_by_ids(
    paper_ids: List[str],
    embedding_type: str = EMBED.SPECTER,
) -> List[dict]:
    """Return the available papers whose IDs are requested."""
    return _legacy_zilliz.query_doc_by_ids(paper_ids, embedding_type=embedding_type)


def search_papers_by_vector(
    vector: List[float],
    *,
    embedding_type: str,
    limit: int,
    exclude_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Return vector-nearest papers, excluding the supplied IDs."""
    return _legacy_zilliz.query_doc_by_embedding(
        paper_ids=exclude_ids or [],
        embedding=vector,
        embedding_type=embedding_type,
        limit=limit,
    )
