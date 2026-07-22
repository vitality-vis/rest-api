"""Registered paper-retrieval profiles.

The public model selector chooses a profile, rather than directly selecting a
Zilliz collection. Add a profile here when a new embedding model is deployed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from model.const import EMBED


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    collection: str
    vector_field: str
    dimension: int
    metric: str
    query_embedder: str
    umap_field: str


RETRIEVAL_PROFILES = {
    EMBED.TEXT_EMBEDDING_3_SMALL: RetrievalProfile(
        name=EMBED.TEXT_EMBEDDING_3_SMALL,
        collection="paper_prod",
        vector_field="embedding",
        dimension=1536,
        metric="COSINE",
        query_embedder="azure_text_embedding_3_small",
        umap_field="umap",
    ),
}

DEFAULT_RETRIEVAL_PROFILE = EMBED.DEFAULT


def get_retrieval_profile(name: Optional[str] = None) -> Optional[RetrievalProfile]:
    """Return a registered profile, or None for an unsupported model selector."""
    return RETRIEVAL_PROFILES.get(str(name or DEFAULT_RETRIEVAL_PROFILE).lower())
