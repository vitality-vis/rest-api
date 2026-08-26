"""Query-embedding helpers for the deployed paper embedding model."""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from openai import AzureOpenAI

import config
from logger_config import get_logger


logging = get_logger()


def _azure_embed_client() -> AzureOpenAI:
    """Create the Azure client only when an embedding is actually requested."""
    config.require_azure_embedding_config()
    return AzureOpenAI(
        api_version=config.AZURE_OPENAI_EMBED_API_VERSION or None,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        timeout=config.EMBEDDING_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def embed_query(text: str) -> List[float]:
    """Embed text in the one vector space currently deployed for papers."""
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        response = _azure_embed_client().embeddings.create(
            model=config.AZURE_OPENAI_EMBED_DEPLOYMENT,
            input=[text],
        )
        embedding = list(response.data[0].embedding)
    except Exception as error:
        logging.error(
            "Azure embedding failed for model %s: %s",
            config.PAPER_EMBEDDING_MODEL,
            error,
        )
        return []

    if len(embedding) != config.PAPER_VECTOR_DIMENSION:
        logging.error(
            "Embedding deployment returned %s dimensions for model %s; expected %s",
            len(embedding),
            config.PAPER_EMBEDDING_MODEL,
            config.PAPER_VECTOR_DIMENSION,
        )
        return []
    return embedding


def embed_paper_query(paper: Union[Dict, str]) -> List[float]:
    """Embed a title/abstract pair using the deployed paper vector space."""
    if isinstance(paper, dict):
        title = str(paper.get("Title") or paper.get("title") or "").strip()
        abstract = str(paper.get("Abstract") or paper.get("abstract") or "").strip()
        return embed_query("\n\n".join(part for part in (title, abstract) if part))
    return embed_query(str(paper or ""))


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
