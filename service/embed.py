from typing import List, Dict

import numpy as np
import requests
# from langchain_openai import OpenAIEmbeddings

import config

ABSTRACT_SIMILARITY_REST_API_URL = 'https://model-apis.semanticscholar.org/specter/v1/invoke'
MAX_BATCH_SIZE = 16

# ada_embedding_func = OpenAIEmbeddings(
#     model='text-embedding-ada-002',
#     openai_api_key=config.OPENAI_API_KEY,
# )

from sentence_transformers import SentenceTransformer

# Local HuggingFace embedding model
hf_ada_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    '''Splits a long list into chunks respecting the batch size'''
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def _specter_embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        response = requests.post(ABSTRACT_SIMILARITY_REST_API_URL, json=chunk)
        if response.status_code != 200:
            raise RuntimeError('Sorry, something went wrong, please try later!')
        for paper in response.json()['preds']:
            embeddings_by_paper_id[paper['paper_id']] = paper['embedding']
    return embeddings_by_paper_id


def specter_embedding(papers) -> list:
    if isinstance(papers, dict):  # In case a single dictionary is passed via Flask request
        papers = [papers]
    paper = papers[0]
    payload = [{
        'paper_id': 'sample_id',
        'title': paper['Title'],
        'abstract': paper['Abstract']
    }]
    embeddings = _specter_embed(payload)
    return embeddings['sample_id']


# def specter_embedding(papers) -> list:
#     paper = papers[0]
#     combined_text = f"{paper['Title']}. {paper['Abstract']}"
#     return hf_specter_model.encode(combined_text, convert_to_numpy=False).tolist()

# def ada_embedding(input_data) -> list:
#     return ada_embedding_func.embed_query(input_data['abstract'])

def ada_embedding(input_data) -> list:
    # Encode abstract text using the local model
    return hf_ada_model.encode(input_data['abstract'], convert_to_numpy=False).tolist()


def mean_embedding(embeddings: List[List[float]]) -> List[float]:
    embeddings = [i for i in embeddings if i is not None]
    if len(embeddings) == 0:
        return []
    embeddings_array = np.array(embeddings)
    # Compute the mean across the first axis (i.e., row-wise average)
    return np.mean(embeddings_array, axis=0).tolist()


def min_max_scaler(arr):
    # Normalize a list of numbers to the [0, 1] range
    min_val = min(arr)
    max_val = max(arr)
    data_range = max_val - min_val
    return [(x - min_val) / data_range for x in arr]