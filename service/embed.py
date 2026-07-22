"""Query-embedding helpers for registered retrieval profiles."""
from __future__ import annotations

import os
from typing import Dict, List, Union

import numpy as np
from openai import AzureOpenAI

from logger_config import get_logger
from model.retrieval import RetrievalProfile


logging = get_logger()

AZURE_EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBED_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_OPENAI_EMBED_API_VERSION")


def _azure_embed_client() -> AzureOpenAI:
    """Create the Azure client only when an embedding is actually requested."""
    if not AZURE_EMBED_ENDPOINT or not AZURE_EMBED_API_KEY or not AZURE_EMBED_DEPLOYMENT:
        raise RuntimeError("Azure OpenAI embedding configuration is incomplete")
    return AzureOpenAI(
        api_version=AZURE_EMBED_API_VERSION,
        azure_endpoint=AZURE_EMBED_ENDPOINT,
        api_key=AZURE_EMBED_API_KEY,
    )


def embed_query(text: str, profile: RetrievalProfile) -> List[float]:
    """Embed text with the query embedder registered by ``profile``."""
    if profile.query_embedder != "azure_text_embedding_3_small":
        raise ValueError(f"No query embedder registered for profile '{profile.name}'")
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        response = _azure_embed_client().embeddings.create(
            model=AZURE_EMBED_DEPLOYMENT,
            input=[text],
        )
        embedding = list(response.data[0].embedding)
    except Exception as error:
        logging.error("Azure embedding failed for profile %s: %s", profile.name, error)
        return []

    if len(embedding) != profile.dimension:
        logging.error(
            "Embedding deployment returned %s dimensions for profile %s; expected %s",
            len(embedding),
            profile.name,
            profile.dimension,
        )
        return []
    return embedding


def embed_paper_query(paper: Union[Dict, str], profile: RetrievalProfile) -> List[float]:
    """Embed a title/abstract pair using the production profile's text space."""
    if isinstance(paper, dict):
        title = str(paper.get("Title") or paper.get("title") or "").strip()
        abstract = str(paper.get("Abstract") or paper.get("abstract") or "").strip()
        return embed_query("\n\n".join(part for part in (title, abstract) if part), profile)
    return embed_query(str(paper or ""), profile)


def mean_embedding(embeddings: List[List[float]]) -> List[float]:
    valid_embeddings = [embedding for embedding in embeddings if embedding]
    if not valid_embeddings:
        return []
    mean_vec = np.mean(np.asarray(valid_embeddings), axis=0)
    norm = np.linalg.norm(mean_vec)
    return (mean_vec / norm if norm else mean_vec).tolist()


def min_max_scaler(arr: List[float]) -> List[float]:
    if not arr:
        return []
    min_val, max_val = min(arr), max(arr)
    if min_val == max_val:
        return [0.0] * len(arr)
    return [(value - min_val) / (max_val - min_val) for value in arr]
