"""External integration adapters."""

from .arxiv import arxiv_search
from .openreview import openreview_search
from .semantic_scholar import semantic_scholar_search

__all__ = [
    "arxiv_search",
    "openreview_search",
    "semantic_scholar_search",
]
