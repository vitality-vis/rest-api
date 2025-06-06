from typing import List, Dict

import numpy as np
import requests
from langchain_openai import OpenAIEmbeddings

import config

ABSTRACT_SIMILARITY_REST_API_URL = 'https://model-apis.semanticscholar.org/specter/v1/invoke'
MAX_BATCH_SIZE = 16

ada_embedding_func = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=config.OPENAI_API_KEY,
)


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    '''Splits a longer list to respect batch size'''
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def _specter_embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(ABSTRACT_SIMILARITY_REST_API_URL, json=chunk)
        if response.status_code != 200:
            raise RuntimeError('Sorry, something went wrong, please try later!')
        for paper in response.json()['preds']:
            embeddings_by_paper_id[paper['paper_id']] = paper['embedding']
    return embeddings_by_paper_id


def specter_embedding(papers) -> list:
    paper = papers[0]
    payload = [{
        'paper_id': 'sample_id',
        'title': paper['Title'],
        'abstract': paper['Abstract']
    }]
    embeddings = _specter_embed(payload)
    return embeddings['sample_id']


def ada_embedding(input_data) -> list:
    return ada_embedding_func.embed_query(input_data['abstract'])


def mean_embedding(embeddings: List[List[float]]) -> List[float]:
    embeddings = [i for i in embeddings if i is not None]
    if len(embeddings) == 0:
        return []
    embeddings_array = np.array(embeddings)
    # Compute the mean along the first axis (axis=0 means take the mean across rows)
    return np.mean(embeddings_array, axis=0).tolist()


def min_max_scaler(arr):
    # Find the minimum and maximum values in the data
    min_val = min(arr)
    max_val = max(arr)
    data_range = max_val - min_val
    return [(x - min_val) / data_range for x in arr]
