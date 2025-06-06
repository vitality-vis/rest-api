import json
from typing import List

from pymongo import MongoClient

import config
from model.const import EMBED
from model.mongo import MongoQuerySchema
from service import embed

client = MongoClient(config.mongodb_connection_uri)
db = client[config.mongodb_database]
docs_collection = db[config.mongodb_docs_collection]
docs_embedding_collection = db[config.mongodb_docs_embedding_collection]


def query_docs(query: MongoQuerySchema):
    fields = {
        '_id': 0,
        'ID': 1,
        'Authors': 1,
        'Keywords': 1,
        'Source': 1,
        'Title': 1,
        'Abstract': 1,
        'Year': 1,
        'ada_umap': 1,
        'glove_umap': 1,
        'specter_umap': 1,
        'CitationCounts': 1
    }

    db_query = dict()

    if query.id_list:
        db_query['ID'] = {'$in': query.id_list}
    if query.author:
        db_query['Authors'] = {'$in': query.author}
    if query.keyword:
        db_query['Keywords'] = {'$in': query.keyword}
    if query.source:
        db_query['Source'] = {'$in': query.source}
    if query.title:
        db_query['Title'] = {'$regex': query.title, '$options': 'i'}
    if query.abstract:
        db_query['Abstract'] = {'$regex': query.abstract, '$options': 'i'}
    if query.min_year or query.max_year:
        year_query = dict()
        if query.min_year:
            year_query = {'$gte': query.min_year}
        if query.max_year:
            year_query = {'$lte': query.max_year}
        db_query['Year'] = year_query
    if query.min_citation_counts is not None or query.max_citation_counts is not None:
        citation_query = dict()
        if query.min_citation_counts is not None:
            citation_query['$gte'] = query.min_citation_counts
        if query.max_citation_counts is not None:
            citation_query['$lte'] = query.max_citation_counts
        db_query['CitationCounts'] = citation_query

    if query.limit == -1:
        results = docs_collection.find(db_query, fields).skip(query.offset)  # No limit applied
    else:
        results = docs_collection.find(db_query, fields).skip(query.offset).limit(query.limit)
    results=list(results)
    final_results = []
    for doc in results:
        if 'CitationCounts' not in doc:
            doc['CitationCounts'] = -1
        final_results.append(doc)
    return final_results


def query_docs_with_embeddings(query: MongoQuerySchema):
    # 1) Add the embedding fields to 'fields'.
    fields = {
        '_id': 0,
        'ID': 1,
        'Authors': 1,
        'Keywords': 1,
        'Source': 1,
        'Title': 1,
        'Abstract': 1,
        'Year': 1,
        'ada_umap': 1,
        'ada_embedding': 1,  # <-- now included
        'glove_umap': 1,
        'glove_embedding': 1,  # <-- now included
        'specter_umap': 1,
        'specter_embedding': 1,  # <-- now included
        'CitationCounts': 1
    }

    db_query = {}

    # 2) Build the same query conditions as before.
    if query.id_list:
        db_query['ID'] = {'$in': query.id_list}
    if query.author:
        db_query['Authors'] = {'$in': query.author}
    if query.keyword:
        db_query['Keywords'] = {'$in': query.keyword}
    if query.source:
        db_query['Source'] = {'$in': query.source}
    if query.title:
        db_query['Title'] = {'$regex': query.title, '$options': 'i'}
    if query.abstract:
        db_query['Abstract'] = {'$regex': query.abstract, '$options': 'i'}
    if query.min_year or query.max_year:
        year_query = {}
        if query.min_year:
            year_query['$gte'] = query.min_year
        if query.max_year:
            year_query['$lte'] = query.max_year
        db_query['Year'] = year_query
    if query.min_citation_counts is not None or query.max_citation_counts is not None:
        citation_query = {}
        if query.min_citation_counts is not None:
            citation_query['$gte'] = query.min_citation_counts
        if query.max_citation_counts is not None:
            citation_query['$lte'] = query.max_citation_counts
        db_query['CitationCounts'] = citation_query

    # 3) Fetch results with the new fields projection.
    if query.limit == -1:
        results = docs_collection.find(db_query, fields).skip(query.offset)
    else:
        results = docs_collection.find(db_query, fields).skip(query.offset).limit(query.limit)

    results = list(results)

    # 4) Enforce any post-processing you like (e.g., default CitationCounts).
    final_results = []
    for doc in results:
        if 'CitationCounts' not in doc:
            doc['CitationCounts'] = -1
        final_results.append(doc)

    return final_results


def query_doc_by_ids(ids: list):
    if ids:
        return query_docs(MongoQuerySchema(id_list=ids, offset=0, limit=len(ids)))
    else:
        return []


def query_doc_full_fields_by_ids(ids: list):
    if not ids:
        return []
    results = docs_collection.find({'ID': {'$in': ids}}).skip(0).limit(len(ids))
    return list(results)
    # return results



def query_doc_by_id(_id: str):
    result = query_docs(MongoQuerySchema(id_list=[_id], offset=0, limit=1))
    if len(result) == 1:
        return result[0]
    else:
        return None


def query_similar_doc_by_paper(papers: dict, embedding_type: str, limit: int = 25):
    if embedding_type in (EMBED.GLOVE, EMBED.SPECTER):
        embeddings = embed.specter_embedding(papers)
    elif embedding_type == EMBED.ADA:
        embeddings = embed.ada_embedding(papers)
    else:
        raise RuntimeError('embedding_type not supported')
    a=query_doc_by_embedding([], embeddings, embedding_type, limit)
    return query_doc_by_embedding([], embeddings, embedding_type, limit)


def query_similar_doc_by_embedding_full(paper: dict, embedding_type: str, limit: int = 25):
    if embedding_type in (EMBED.GLOVE, EMBED.SPECTER):
        index_path = 'specter_embedding'
    elif embedding_type == EMBED.ADA:
        index_path = 'ada_embedding' #index_path should be ada_umap instead of ada_embedding
    else:
        raise RuntimeError('embedding_type not supported')

    paper_ids = [i['ID'] for i in paper]
    embedding = embed.mean_embedding([i[index_path] for i in paper if index_path in i and i[index_path] is not None])

    if not embedding_type:
        return []
    return query_doc_by_embedding(paper_ids, embedding, embedding_type, limit)


def query_similar_doc_by_embedding_2d(paper: dict, embedding_type: str, limit: int = 25):
    embed_field = f'{embedding_type}_umap'
    paper_ids = [i['ID'] for i in paper]
    coord = embed.mean_embedding([i[embed_field] for i in paper])
    if not coord:
        return []

    return list(
        docs_collection.aggregate([
            {
                '$geoNear': {
                    'key': embed_field,
                    'near': coord,
                    'distanceField': 'distance',
                    'query': {
                        'ID': {'$nin': paper_ids}
                    },
                    'spherical': True
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'ID': 1,
                    'Authors': 1,
                    'Keywords': 1,
                    'Source': 1,
                    'Title': 1,
                    'Abstract': 1,
                    'Year': 1,
                    'ada_umap': 1,
                    'glove_umap': 1,
                    'specter_umap': 1,
                    'distance': 1
                }
            },
            {'$sort': {'distance': 1}},
            {'$limit': limit}
        ])
    )


    # return list(
    #     docs_collection.find(
    #         {
    #             embed_field: {'$near': coord},
    #             'ID': {'$nin': paper_ids}
    #         },
    #         {
    #             '_id': 0,
    #             'ID': 1,
    #             'Authors': 1,
    #             'Keywords': 1,
    #             'Source': 1,
    #             'Title': 1,
    #             'Abstract': 1,
    #             'Year': 1,
    #             'ada_umap': 1,
    #             'glove_umap': 1,
    #             'specter_umap': 1,
    #         }
    #     )
    #     .skip(0)
    #     .limit(limit)
    # )


def query_doc_by_embedding(paper_ids: List[str], embedding: List[float], embedding_type: str, limit: int):
    if embedding_type in (EMBED.GLOVE, EMBED.SPECTER):
        index, query_path = 'specter_embedding_index', 'specter_embedding'
    elif embedding_type == EMBED.ADA:
        index, query_path = 'ada_embedding_index', 'ada_embedding'
    else:
        raise RuntimeError('embedding_type not supported')

    return list(docs_embedding_collection.aggregate([
        {
            "$vectorSearch": {
                "index": index,
                "path": query_path,
                "queryVector": embedding,
                "numCandidates": limit * 15,
                "limit": limit + len(paper_ids)
            }
        },
        {
            '$match': {
                'ID': {'$nin': paper_ids}
            }
        },
        {
            '$project': {
                '_id': 0,
                'ID': 1,
                'Authors': 1,
                'Keywords': 1,
                'Source': 1,
                'Title': 1,
                'Abstract': 1,
                'Year': 1,
                'ada_umap': 1,
                'glove_umap': 1,
                'specter_umap': 1,
                'score': {'$meta': 'vectorSearchScore'}
            }
        }
    ]))[:limit]


def query_all_umap_points():
    results = docs_collection.find(
        {},
        {
            '_id': 0,
            'ID': 1,
            'Title': 1,
            'ada_umap': 1,
            'glove_umap': 1,
            'specter_umap': 1,
            'Year': 1,
            'Source': 1,
        }
    )
    return list(results)


def query_meta_data():
    authors = docs_collection.aggregate([
        {'$unwind': "$Authors"},
        {'$group': {'_id': "$Authors"}},
        {'$group': {'_id': None, 'distinctValues': {'$addToSet': "$_id"}}},
        {'$project': {'_id': 0, 'distinctValues': 1}}
    ])
    keywords = docs_collection.aggregate([
        {'$unwind': "$Keywords"},
        {'$group': {'_id': "$Keywords"}},
        {'$group': {'_id': None, 'distinctValues': {'$addToSet': "$_id"}}},
        {'$project': {'_id': 0, 'distinctValues': 1}}
    ])
    years = docs_collection.distinct('Year')
    sources = docs_collection.distinct('Source')

    return {
        'authors': list(authors)[0].get('distinctValues', []),
        'keywords': list(keywords)[0].get('distinctValues', []),
        'years': years,
        'sources': sources,
    }


def save_meta_data_local():
    step = 50_000
    start = 0

    meta_data = {
        'authors': {},
        'keywords': {},
        'year': {},
        'source': {},
        'counts': 0,
    }

    while True:
        papers = list(
            docs_collection.find(
                {},
                {
                    '_id': 0,
                    'Authors': 1,
                    'Keywords': 1,
                    'Year': 1,
                    'Source': 1,
                }
            )
            .sort({'_id': 1})
            .skip(start)
            .limit(step)
        )

        if not papers:
            break

        for paper in papers:
            if paper['Authors']:
                for author in paper['Authors']:
                    meta_data['authors'][author] = meta_data['authors'].get(author, 0) + 1
            if paper['Keywords']:
                for kw in paper['Keywords']:
                    meta_data['keywords'][kw] = meta_data['keywords'].get(kw, 0) + 1
            if paper['Year']:
                meta_data['year'][paper['Year']] = meta_data['year'].get(paper['Year'], 0) + 1
            if paper['Source']:
                meta_data['source'][paper['Source']] = meta_data['source'].get(paper['Source'], 0) + 1
            meta_data['counts'] += 1
        start += step

    with open(config.meta_data_file_path, 'w') as f:
        json.dump(meta_data, f)

def get_distinct_authors():
    # Query to get all distinct authors
    return list(docs_collection.distinct("Authors"))

def get_distinct_sources():
    # Query to get all distinct sources
    return list(docs_collection.distinct("Source"))

def get_distinct_keywords():
    # Query to get all distinct keywords
    return list(docs_collection.distinct("Keywords"))

def get_distinct_years():
    # Query to get all distinct years
    return list(docs_collection.distinct("Year"))
def get_distinct_titles():
    collection = db['papers']  # Replace 'papers' with the correct collection name
    return list(docs_collection.distinct("Title"))
def get_distinct_citation_counts():
    collection = db['papers']  # Replace 'papers' with the correct collection name
    return list(docs_collection.distinct("CitationCounts"))
def get_distinct_authors_with_counts():
    # Query to get distinct authors and their counts
    pipeline = [
        {"$unwind": "$Authors"},  # Unwind the Authors array if it exists
        {"$group": {"_id": "$Authors", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}  # Optional: sort by count in descending order
    ]
    return list(docs_collection.aggregate(pipeline))

def get_distinct_sources_with_counts():
    # Query to get distinct sources and their counts
    pipeline = [
        {"$group": {"_id": "$Source", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    return list(docs_collection.aggregate(pipeline))

def get_distinct_keywords_with_counts():
    # Query to get distinct keywords and their counts
    pipeline = [
        {"$unwind": "$Keywords"},  # Unwind the Keywords array if it exists
        {"$group": {"_id": "$Keywords", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    return list(docs_collection.aggregate(pipeline))

def get_distinct_years_with_counts():
    # Query to get distinct years and their counts
    pipeline = [
        {"$group": {"_id": "$Year", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}  # Optional: sort by year in ascending order
    ]
    return list(docs_collection.aggregate(pipeline))

def get_distinct_titles_with_counts():
    # Query to get distinct titles and their counts
    pipeline = [
        {"$group": {"_id": "$Title", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    return list(docs_collection.aggregate(pipeline))

def get_distinct_citation_counts_with_counts():
    # Query to get distinct citation counts and their counts
    pipeline = [
        {"$group": {"_id": "$CitationCounts", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    return list(docs_collection.aggregate(pipeline))