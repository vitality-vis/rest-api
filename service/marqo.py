import json
from typing import List
import marqo

from model.const import EMBED
from model.marqo import MarqoQuerySchema
import config
import numpy as np

mq = marqo.Client(url=config.marqo_url)

INDEX_MAPPING = {
    EMBED.ADA: "ada-papers",
    EMBED.GLOVE: "glove-papers",
    EMBED.SPECTER: "specter-papers"
}

def get_all_docs():
    with open(config.raw_json_datafile, "r") as f:
        return json.load(f)

def get_ready_docs():
    with open(config.ready_json_datafile, "r") as f:
        return json.load(f)

def query_docs(query: MarqoQuerySchema):
    all_docs = get_all_docs()
    filtered = []
    for doc in all_docs:
        if query.id_list and doc['ID'] not in query.id_list:
            continue
        if query.author and not any(a in doc.get('Authors', []) for a in query.author):
            continue
        if query.keyword and not any(k in doc.get('Keywords', []) for k in query.keyword):
            continue
        if query.source and doc.get('Source') not in query.source:
            continue
        if query.title and query.title.lower() not in doc.get('Title', '').lower():
            continue
        if query.abstract and query.abstract.lower() not in doc.get('Abstract', '').lower():
            continue
        if query.min_year and doc.get('Year') and doc['Year'] < query.min_year:
            continue
        if query.max_year and doc.get('Year') and doc['Year'] > query.max_year:
            continue
        if query.min_citation_counts is not None and doc.get('CitationCounts', -1) < query.min_citation_counts:
            continue
        if query.max_citation_counts is not None and doc.get('CitationCounts', -1) > query.max_citation_counts:
            continue
        filtered.append(doc)
    return filtered[query.offset:query.offset + query.limit if query.limit != -1 else None]

def query_docs_with_embeddings(query: MarqoQuerySchema):
    return query_docs(query)

def query_doc_by_ids(ids: List[str]):
    return query_docs(MarqoQuerySchema(id_list=ids, offset=0, limit=len(ids))) if ids else []

def query_doc_full_fields_by_ids(ids: List[str]):
    return query_doc_by_ids(ids)

def query_doc_by_id(_id: str):
    results = query_doc_by_ids([_id])
    return results[0] if results else None

def query_doc_by_embedding(paper_ids: List[str], embedding: List[float], embedding_type: str, limit: int):
    index_name = INDEX_MAPPING[embedding_type]
    res = mq.index(index_name).search(q=embedding)
    return [doc for doc in res['hits'] if doc['ID'] not in paper_ids][:limit]

import requests

def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25):
    ready_docs = get_ready_docs()
    for d in ready_docs:
        if '_id' in d and 'ID' not in d:
            d['ID'] = str(d['_id'])

    paper_ids = [p['ID'] for p in papers]
    index_name = INDEX_MAPPING[embedding_type]

    if embedding_type == EMBED.ADA:
        embed_field = "ada_embedding"
    elif embedding_type == EMBED.GLOVE:
        embed_field = "glove_embedding"
    elif embedding_type == EMBED.SPECTER:
        embed_field = "specter_embedding"
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    # Extract vectors (structure: {"vector": [...], "content": ...})
    vectors = []
    for p in ready_docs:
        if p.get("ID") in paper_ids:
            emb = p.get(embed_field, {})
            if isinstance(emb, dict) and "vector" in emb:
                vectors.append(emb["vector"])

    print(f"💡 paper_ids: {paper_ids}")
    print(f"📏 number of vectors: {len(vectors)}")
    if vectors:
        print(f"📐 vector dimension: {len(vectors[0])}")

    if not vectors:
        print(f"⚠️ No valid vector field '{embed_field}', skipping search.")
        return []

    # Compute centroid vector
    mean_vector = np.mean(np.array(vectors), axis=0).tolist()

    try:
        res = mq.index(index_name).search(
            context={
                "tensor": [
                    {"vector": mean_vector, "weight": 1.0}
                ]
            },
            limit=limit
        )
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

    print("✅ Raw Marqo response:", res)

    results = []
    for doc in res.get("hits", []):
        if doc.get("ID") not in paper_ids:
            results.append(format_doc_for_frontend(doc))
        if len(results) >= limit:
            break

    return results

def query_similar_doc_by_embedding_2d(papers: List[dict], embedding_type: str, limit: int = 25):
    return query_similar_doc_by_embedding_full(papers, embedding_type, limit)

def query_similar_doc_by_paper(paper: dict, embedding_type: str, limit: int = 25):
    index_name = INDEX_MAPPING[embedding_type]
    title = paper.get("Title", "")
    abstract = paper.get("Abstract", "")
    q = f"{title}. {abstract}"
    res = mq.index(index_name).search(q=q)

    results = []
    for doc in res['hits']:
        doc['score'] = doc.get('_score', 0.0)   # Equivalent to vectorSearchScore in Mongo
        clean_doc = {k: v for k, v in doc.items() if not k.startswith("_") or k == "score"}
        results.append(clean_doc)
        if len(results) >= limit:
            break

    return results

def query_all_umap_points():
    return [
        {
            'ID': d['ID'],
            'Title': d.get('Title'),
            'ada_umap': d.get('ada_umap'),
            'glove_umap': d.get('glove_umap'),
            'specter_umap': d.get('specter_umap'),
            'Year': d.get('Year'),
            'Source': d.get('Source')
        }
        for d in get_all_docs()
    ]

def _aggregate_count(field):
    counter = {}
    for doc in get_all_docs():
        values = doc.get(field, []) if isinstance(doc.get(field), list) else [doc.get(field)]
        for v in values:
            if v:
                counter[v] = counter.get(v, 0) + 1
    return sorted([{"_id": k, "count": v} for k, v in counter.items()], key=lambda x: -x['count'])

def get_distinct_authors():
    return list(set(a for doc in get_all_docs() for a in doc.get("Authors", [])))

def get_distinct_sources():
    return list(set(doc.get("Source") for doc in get_all_docs() if doc.get("Source")))

def get_distinct_keywords():
    return list(set(k for doc in get_all_docs() for k in doc.get("Keywords", [])))

def get_distinct_years():
    return sorted(set(doc.get("Year") for doc in get_all_docs() if doc.get("Year") is not None))

def get_distinct_titles():
    return list(set(doc.get("Title") for doc in get_all_docs() if doc.get("Title")))

def get_distinct_citation_counts():
    return list(set(doc.get("CitationCounts") for doc in get_all_docs() if doc.get("CitationCounts") is not None))

def get_distinct_authors_with_counts():
    return _aggregate_count("Authors")

def get_distinct_sources_with_counts():
    return _aggregate_count("Source")

def get_distinct_keywords_with_counts():
    return _aggregate_count("Keywords")

def get_distinct_years_with_counts():
    return sorted(_aggregate_count("Year"), key=lambda x: x['_id'])

def get_distinct_titles_with_counts():
    return _aggregate_count("Title")

def get_distinct_citation_counts_with_counts():
    return _aggregate_count("CitationCounts")

def format_doc_for_frontend(doc: dict, score_key="_score") -> dict:
    return {
        "ID": doc.get("ID"),
        "Title": doc.get("Title", ""),
        "Abstract": doc.get("Abstract", ""),
        "Authors": doc.get("Authors", []),
        "Keywords": doc.get("Keywords", []),
        "Source": doc.get("Source", ""),
        "Year": doc.get("Year", None),
        "Sim": float(doc.get(score_key, 0.0)),  # Ensure frontend uses 'Sim' as similarity score key
    }