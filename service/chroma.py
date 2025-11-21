import json
import numpy as np
from typing import List, Dict, Any
import config
from model.const import EMBED 
from model.chroma import ChromaQuerySchema
import chromadb 
import logging 
import math 

from typing import Optional
from pydantic import BaseModel
class ChromaQuerySchema(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    author: Optional[List[str]] = None
    source: Optional[str] = None
    keyword: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    id_list: Optional[List[str]] = None
    offset: int = 0
    limit: Optional[int] = None  # ✅ make it optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_PATH = "chroma_db"
COLLECTION_MAPPING = {
    EMBED.ADA: "paper_ada_localized",    
    EMBED.GLOVE: "paper_glove_localized", 
    EMBED.SPECTER: "paper_specter"      
}

try:
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    logging.info(f"ChromaDB client initialized at path: {CHROMA_PATH}")
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB client at {CHROMA_PATH}: {e}")
    _chroma_client = None

_ready_docs_cache = None
_all_docs_cache = None
_ready_docs_by_id = None

# def query_docs(query: ChromaQuerySchema):
#     docs = get_ready_docs()
#     id_list_as_strings = [str(i) for i in query.id_list] if query.id_list else []






# def query_docs(query: ChromaQuerySchema, embedding_type: str = EMBED.SPECTER):
#     if _chroma_client is None:
#         return []

#     try:
#         collection_name = COLLECTION_MAPPING.get(embedding_type)
#         if not collection_name:
#             logging.error(f"No collection found for embedding type: {embedding_type}")
#             return []

#         collection = _chroma_client.get_collection(collection_name)
#         results = collection.get(include=["metadatas"])

#         docs = []
#         for metadata in results.get("metadatas", []):
#             if match_doc(metadata, query):
#                 docs.append(format_doc_for_frontend(metadata))
#         return docs[query.offset: query.offset + query.limit]
#     except Exception as e:
#         logging.error(f"Error in query_docs(): {e}")
#         return []



# def query_docs(query: ChromaQuerySchema, embedding_type: str = EMBED.SPECTER):
#     if _chroma_client is None:
#         return []

#     try:
#         collection_name = COLLECTION_MAPPING.get(embedding_type)
#         if not collection_name:
#             logging.error(f"No collection found for embedding type: {embedding_type}")
#             return []

#         collection = _chroma_client.get_collection(collection_name)
#         results = collection.get(include=["metadatas"])

#         docs = []
#         for metadata in results.get("metadatas", []):
#             if metadata and match_doc(metadata, query):
#                 docs.append(format_doc_for_frontend(metadata))

#         offset = int(query.offset or 0)
#         limit = int(query.limit or 100)
#         return docs[offset: offset + limit]

#     except Exception as e:
#         logging.error(f"Error in query_docs(): {e}", exc_info=True)
#         return []




def query_docs(query: ChromaQuerySchema, embedding_type: str = EMBED.SPECTER):
    if _chroma_client is None:
        return {"papers": [], "total": 0}

    try:
        collection_name = COLLECTION_MAPPING.get(embedding_type)
        if not collection_name:
            logging.error(f"No collection found for embedding type: {embedding_type}")
            return {"papers": [], "total": 0}

        collection = _chroma_client.get_collection(collection_name)

        if query.id_list:
            results = collection.get(
                where={"ID": {"$in": [str(i) for i in query.id_list]}},
                include=["metadatas"]
            )
        else:
            results = collection.get(include=["metadatas"])

        docs = []
        for metadata in results.get("metadatas", []):
            if metadata and match_doc(metadata, query):
                docs.append(format_doc_for_frontend(metadata))

        total_count = len(docs)  # Count BEFORE pagination
        offset = int(query.offset or 0)
        limit = int(query.limit or 100)

        # Handle limit=-1 to return all docs (FIX: don't slice off the last one!)
        if limit == -1:
            paginated_docs = docs[offset:]
        else:
            paginated_docs = docs[offset: offset + limit]

        return {"papers": paginated_docs, "total": total_count}

    except Exception as e:
        logging.error(f"Error in query_docs(): {e}", exc_info=True)
        return {"papers": [], "total": 0}





# def match_doc(doc, query):
#     doc_id_str = str(doc.get("ID")) if doc.get("ID") is not None else None
#     if query.title and query.title.lower() not in doc.get("Title", "").lower(): return False
#     if query.abstract and query.abstract.lower() not in doc.get("Abstract", "").lower(): return False
#     authors_str = " ".join(doc.get("Authors", [])).lower() if isinstance(doc.get("Authors"), list) else str(doc.get("Authors", "")).lower()
#     if query.author and not any(a.lower() in authors_str for a in query.author): return False
#     if query.source and not any(s.lower() in doc.get("Source", "").lower() for s in query.source): return False
#     keywords_str = " ".join(doc.get("Keywords", [])).lower() if isinstance(doc.get("Keywords"), list) else str(doc.get("Keywords", "")).lower()
#     if query.keyword and not any(k.lower() in keywords_str for k in query.keyword): return False
#     try:
#         doc_year = int(doc.get("Year", 0))
#         if query.min_year and doc_year < query.min_year: return False
#         if query.max_year and doc_year > query.max_year: return False
#     except ValueError:
#         logging.warning(f"Invalid year value for doc ID {doc_id_str}: {doc.get('Year')}. Skipping year filter.")
#     try:
#         doc_citation_counts = int(doc.get("CitationCounts", 0))
#         if query.min_citation_counts is not None and doc_citation_counts < query.min_citation_counts: return False
#         if query.max_citation_counts is not None and doc_citation_counts > query.max_citation_counts: return False
#     except ValueError:
#         logging.warning(f"Invalid citation count for doc ID {doc_id_str}: {doc.get('CitationCounts')}. Skipping citation filter.")
#     if query.id_list and (doc_id_str is None or doc_id_str not in id_list_as_strings): return False
#     return True





# def match_doc(doc, query):
#     doc_id_str = str(doc.get("ID")) if doc.get("ID") is not None else None

#     if query.title and query.title.lower() not in str(doc.get("Title", "")).lower():
#         return False

#     if query.abstract and query.abstract.lower() not in str(doc.get("Abstract", "")).lower():
#         return False

#     authors = doc.get("Authors", [])
#     if isinstance(authors, str):
#         authors = [authors]
#     authors_str = " ".join(authors).lower()
#     if query.author and not any(a.lower() in authors_str for a in query.author):
#         return False

#     if query.source and not any(s.lower() in str(doc.get("Source", "")).lower() for s in query.source):
#         return False

#     keywords = doc.get("Keywords", [])
#     if isinstance(keywords, str):
#         keywords = [keywords]
#     keywords_str = " ".join(keywords).lower()
#     if query.keyword and not any(k.lower() in keywords_str for k in query.keyword):
#         return False

#     try:
#         doc_year = int(doc.get("Year", 0))
#         if query.min_year and doc_year < query.min_year:
#             return False
#         if query.max_year and doc_year > query.max_year:
#             return False
#     except Exception:
#         pass

#     try:
#         doc_citation_counts = int(doc.get("CitationCounts", 0))
#         if query.min_citation_counts is not None and doc_citation_counts < query.min_citation_counts:
#             return False
#         if query.max_citation_counts is not None and doc_citation_counts > query.max_citation_counts:
#             return False
#     except Exception:
#         pass

#     if query.id_list and (doc_id_str is None or str(doc_id_str) not in [str(i) for i in query.id_list]):
#         return False

#     return True


def normalize_text(text: str) -> str:
    """title normalization for matching"""
    return str(text or "").strip().lower().rstrip(".!?")

def match_doc(doc, query):
    doc_id_str = str(doc.get("ID")) if doc.get("ID") is not None else None

    # title
    if query.title:
        q_title = normalize_text(query.title)
        d_title = normalize_text(doc.get("Title", ""))

        if q_title == d_title:  
            pass
        elif q_title in d_title:  
            pass
        else:
            try:
                from rapidfuzz import fuzz
                score = fuzz.ratio(q_title, d_title)
                if score < 80:   
                    return False
            except ImportError:
                return False

    # abstract
    if query.abstract and query.abstract.lower() not in str(doc.get("Abstract", "")).lower():
        return False

    # authors
    authors = doc.get("Authors", [])
    if isinstance(authors, str):
        authors = [authors]
    authors_str = " ".join(authors).lower()
    if query.author and not any(a.lower() in authors_str for a in query.author):
        return False

    # source
    if query.source and not any(s.lower() in str(doc.get("Source", "")).lower() for s in query.source):
        return False

    # keywords
    keywords = doc.get("Keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords_str = " ".join(keywords).lower()
    if query.keyword and not any(k.lower() in keywords_str for k in query.keyword):
        return False

    # year
    try:
        doc_year = int(doc.get("Year", 0))
        if query.min_year and doc_year < query.min_year:
            return False
        if query.max_year and doc_year > query.max_year:
            return False
    except Exception:
        pass

    # citation counts
    try:
        doc_citation_counts = int(doc.get("CitationCounts", 0))
        if query.min_citation_counts is not None and doc_citation_counts < query.min_citation_counts:
            return False
        if query.max_citation_counts is not None and doc_citation_counts > query.max_citation_counts:
            return False
    except Exception:
        pass

    # ID list
    if query.id_list and (doc_id_str is None or str(doc_id_str) not in [str(i) for i in query.id_list]):
        return False

    return True




    try:
        offset = int(query.offset)
        limit = int(query.limit)
    except Exception as e:
        logging.warning(f"[query_docs] Offset/Limit parse failed: {e}. Defaulting to offset=0, limit=100.")
        offset = 0
        limit = 100

    return [d for d in docs if match(d)][offset:offset + limit]


def query_docs_with_embeddings(query: ChromaQuerySchema, embedding_type: str = EMBED.SPECTER): 
    return query_docs(query, embedding_type=embedding_type)


# def query_doc_by_id(_id: str, embedding_type: str = EMBED.SPECTER):
#     if _chroma_client is None:
#         return None
#     try:
#         results = _chroma_client.get_collection(COLLECTION_MAPPING[embedding_type]).get(
#             ids=[str(_id)],
#             include=["metadatas"]
#         )
#         if results and results.get("metadatas") and results["metadatas"][0]:
#             return format_doc_for_frontend(results["metadatas"][0])
#     except Exception as e:
#         logging.error(f"Error fetching doc ID {_id} from Chroma: {e}")
#     return None





# def query_doc_by_id(_id: str, embedding_type: str = EMBED.SPECTER):
#     if _chroma_client is None:
#         return None
#     try:
#         collection = _chroma_client.get_collection(COLLECTION_MAPPING[embedding_type])
#         results = collection.get(
#             where={"ID": str(_id)},   # ✅ 用 metadata ID
#             include=["metadatas"]
#         )
#         if results and results.get("metadatas"):
#             return format_doc_for_frontend(results["metadatas"][0])
#     except Exception as e:
#         logging.error(f"Error fetching doc ID {_id} from Chroma: {e}")
#     return None


def query_doc_by_id(_id: str, embedding_type: str = EMBED.SPECTER):
    if _chroma_client is None:
        return None
    try:
        collection = _chroma_client.get_collection(COLLECTION_MAPPING[embedding_type])
        logging.info(f"🔎 Querying by ID={_id}")
        results = collection.get(
            where={"ID": {"$eq": str(_id)}},
            include=["metadatas"]
        )
        logging.info(f"🔎 Raw results: {results}")
        if results and results.get("metadatas") and results["metadatas"][0]:
            return format_doc_for_frontend(results["metadatas"][0])
    except Exception as e:
        logging.error(f"Error fetching doc ID {_id} from Chroma: {e}", exc_info=True)
    return None




def query_doc_by_title(title: str, embedding_type: str = EMBED.SPECTER) -> list:
    """
    through title to get doc metadata
    """
    if _chroma_client is None:
        logging.error("Chroma client not initialized.")
        return []

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"Invalid embedding type: {embedding_type}")
        return []

    try:
        collection = _chroma_client.get_collection(collection_name)
        results = collection.get(include=["metadatas"])
        all_docs = results.get("metadatas", [])

        normalized = title.strip().lower().rstrip(".")

        # First try exact or containment match
        matches = []
        for doc in all_docs:
            doc_title = str(doc.get("Title") or "").strip().lower().rstrip(".")
            if normalized == doc_title or normalized in doc_title:
                matches.append(format_doc_for_frontend(doc))

        # if no exact/contain matches, try fuzzy matching with rapidfuzz
        if not matches:
            try:
                from rapidfuzz import process
                all_titles = [str(d.get("Title") or "") for d in all_docs]
                best_match, score, idx = process.extractOne(normalized, all_titles)
                if score > 80:
                    matches.append(format_doc_for_frontend(all_docs[idx]))
            except ImportError:
                logging.warning("⚠️ rapidfuzz not installed, skip fuzzy matching")

        return matches

    except Exception as e:
        logging.error(f"Error in query_doc_by_title(): {e}", exc_info=True)
        return []








# def query_doc_by_ids(ids: List[str], embedding_type: str = EMBED.SPECTER) -> List[dict]:
#     """
#     批量获取多个文档的 metadata 并进行格式化
#     """
#     if _chroma_client is None:
#         logging.error("Chroma client not initialized.")
#         return []

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not collection_name:
#         logging.error(f"Invalid embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#         results = collection.get(ids=[str(_id) for _id in ids], include=["metadatas"])
#         metadatas = results.get("metadatas", [])

#         docs = []
#         for metadata in metadatas:
#             if metadata:
#                 docs.append(format_doc_for_frontend(metadata))
#         return docs

#     except Exception as e:
#         logging.error(f"Error in query_doc_by_ids(): {e}", exc_info=True)
#         return []



def query_doc_by_ids(ids: List[str], embedding_type: str = EMBED.SPECTER) -> List[dict]:
    if _chroma_client is None:
        logging.error("Chroma client not initialized.")
        return []

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"Invalid embedding type: {embedding_type}")
        return []

    try:
        collection = _chroma_client.get_collection(collection_name)
        results = collection.get(
            where={"ID": {"$in": [str(_id) for _id in ids]}},   
            include=["metadatas"]
        )
        metadatas = results.get("metadatas", [])
        return [format_doc_for_frontend(m) for m in metadatas if m]
    except Exception as e:
        logging.error(f"Error in query_doc_by_ids(): {e}", exc_info=True)
        return []










# def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized. Cannot perform embedding query.")
#         return []

#     paper_ids_to_exclude = [str(p.get("ID") or p.get("id")) for p in papers if p.get("ID") or p.get("id")]








# def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized.")
#         return []

#     # 获取 ID
#     paper_ids_to_exclude = [str(p.get("ID") or p.get("id")) for p in papers]
#     logging.info(f"📌 paper_ids_to_exclude: {paper_ids_to_exclude}")

#     # 选择 collection
#     embed_field = {
#         EMBED.ADA: "ada_embedding",
#         EMBED.GLOVE: "glove_embedding",
#         EMBED.SPECTER: "specter_embedding"
#     }.get(embedding_type)

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not embed_field or not collection_name:
#         logging.error(f"Invalid embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#         results = collection.get(ids=paper_ids_to_exclude, include=["embeddings"])
#     except Exception as e:
#         logging.error(f"Failed to fetch embeddings from Chroma: {e}")
#         return []

#     vectors_for_mean = []
#     for i, emb in enumerate(results.get("embeddings", [])):
#         if isinstance(emb, list) and emb and any(v != 0 for v in emb):
#             vectors_for_mean.append(emb)
#         else:
#             logging.warning(f"❗ No valid embedding for doc: {paper_ids_to_exclude[i]}")

#     if not vectors_for_mean:
#         logging.warning("❌ No valid vectors to calculate mean. Returning empty.")
#         return []

#     mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
#     return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)


# def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized.")
#         return []

#     paper_ids_to_exclude = [str(p.get("ID") or p.get("id")) for p in papers]
#     logging.info(f"📌 paper_ids_to_exclude: {paper_ids_to_exclude}")

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not collection_name:
#         logging.error(f"Invalid embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#         # The 'embeddings' here will be a list of numpy arrays
#         results = collection.get(ids=paper_ids_to_exclude, include=["embeddings"]) 
#     except Exception as e:
#         logging.error(f"Failed to fetch embeddings from Chroma: {e}")
#         return []

#     vectors_for_mean = []
#     # results.get("embeddings", []) will be something like [ array([...]), array([...]) ]
#     for i, emb in enumerate(results.get("embeddings", [])):
#         # This is the corrected check. It handles both lists and numpy arrays.
#         if isinstance(emb, (list, np.ndarray)) and emb is not None and np.any(emb):
#             vectors_for_mean.append(emb)
#         else:
#             logging.warning(f"❗ No valid embedding for doc: {paper_ids_to_exclude[i]}")

#     if not vectors_for_mean:
#         logging.warning("❌ No valid vectors to calculate mean. Returning empty.")
#         return []
    
#     # np.mean will work correctly on the list of numpy arrays
#     mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
#     return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)





# def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized.")
#         return []

#     paper_ids_to_exclude = [str(p.get("ID")) for p in papers if p.get("ID")]
#     logging.info(f"📌 paper_ids_to_exclude: {paper_ids_to_exclude}")

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not collection_name:
#         logging.error(f"Invalid embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#         results = collection.get(
#             where={"ID": {"$in": paper_ids_to_exclude}},   
#             include=["embeddings"]
#         )
#     except Exception as e:
#         logging.error(f"Failed to fetch embeddings from Chroma: {e}")
#         return []

#     vectors_for_mean = []
#     for i, emb in enumerate(results.get("embeddings", [])):
#         if isinstance(emb, (list, np.ndarray)) and np.any(emb):
#             vectors_for_mean.append(emb)
#         else:
#             logging.warning(f"❗ No valid embedding for doc: {paper_ids_to_exclude[i]}")

#     if not vectors_for_mean:
#         logging.warning("❌ No valid vectors to calculate mean. Returning empty.")
#         return []

#     mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
#     return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)


def query_similar_doc_by_embedding_full(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    if _chroma_client is None:
        logging.error("ChromaDB client not initialized.")
        return []

    paper_ids_to_exclude = [str(p.get("ID")) for p in papers if p.get("ID")]
    logging.info(f"[DEBUG] Input papers: {papers}")
    logging.info(f"[DEBUG] Extracted paper_ids_to_exclude: {paper_ids_to_exclude}")

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"Invalid embedding type: {embedding_type}")
        return []

    try:
        collection = _chroma_client.get_collection(collection_name)

        # Try row IDs first
        results = collection.get(
            ids=paper_ids_to_exclude,
            include=["embeddings"]
        )
        logging.info(f"[DEBUG] Results from get(ids=...): {results}")

        # Fallback: try metadata ID if row IDs return nothing
        # if not results.get("embeddings"):
        #     results = collection.get(
        #         where={"ID": {"$in": paper_ids_to_exclude}},
        #         include=["embeddings"]
        #     )
        #     logging.info(f"[DEBUG] Results from get(where metadata.ID=...): {results}")

        embeddings = results.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            logging.warning("❌ No embeddings found for given IDs.")
            return []

    except Exception as e:
        logging.error(f"Failed to fetch embeddings from Chroma: {e}", exc_info=True)
        return []

    vectors_for_mean = []
    for i, emb in enumerate(results.get("embeddings", [])):
        if isinstance(emb, (list, np.ndarray)) and np.any(emb):
            vectors_for_mean.append(emb)
        else:
            logging.warning(f"[WARN] No valid embedding for doc: {paper_ids_to_exclude[i]}")

    if not vectors_for_mean:
        logging.warning("[WARN] No valid vectors to calculate mean. Returning empty list.")
        return []

    mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
    logging.info(f"[DEBUG] Successfully computed mean vector, dimension={len(mean_vector)}")

    return query_doc_by_embedding(paper_ids_to_exclude, mean_vector, embedding_type, limit, lang_filter)














    # ready_docs = get_ready_docs()
    # embed_field = {
    #     EMBED.ADA: "ada_embedding",
    #     EMBED.GLOVE: "glove_embedding",
    #     EMBED.SPECTER: "specter_embedding"
    # }.get(embedding_type)

    # if embed_field is None:
    #     raise ValueError(f"Unsupported embedding type for mean vector calculation: {embedding_type}")

    # vectors_for_mean = []
    # for p in ready_docs:
    #     current_id_str = str(p.get("ID"))
    #     if current_id_str in paper_ids_to_exclude:
    #         embedding_data = p.get(embed_field)
    #         if isinstance(embedding_data, list) and embedding_data and any(v != 0 for v in embedding_data):
    #             vectors_for_mean.append(embedding_data)

    embed_field = {
        EMBED.ADA: "ada_embedding",
        EMBED.GLOVE: "glove_embedding",
        EMBED.SPECTER: "specter_embedding"
    }.get(embedding_type)

    if embed_field is None:
        raise ValueError(f"Unsupported embedding type for mean vector calculation: {embedding_type}")

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        raise ValueError(f"No collection found for embedding type: {embedding_type}")

    try:
        collection = _chroma_client.get_collection(collection_name)

        results = collection.get(
            ids=paper_ids_to_exclude,
            include=["embeddings"]
        )

        vectors_for_mean = []
        for emb in results.get("embeddings", []):
            if isinstance(emb, list) and emb and any(v != 0 for v in emb):
                vectors_for_mean.append(emb)

        logging.debug(f"✅ Retrieved {len(vectors_for_mean)} embeddings for mean vector calculation.")

    except Exception as e:
        logging.error(f"Error retrieving embeddings from ChromaDB: {e}")
        return []

    logging.debug(f"🧪 Number of valid vectors for mean calculation: {len(vectors_for_mean)}")
    if vectors_for_mean:
        logging.debug(f"🧪 Sample vector dimension for mean: {len(vectors_for_mean[0])}")

    if not vectors_for_mean:
        logging.warning("No valid vectors found to calculate mean. Returning empty results.")
        return []

    mean_vector = np.mean(np.array(vectors_for_mean), axis=0).tolist()
    logging.debug(f"🧪 Mean vector Head: {mean_vector[:5]}")

    return query_doc_by_embedding(
        paper_ids_to_exclude, 
        mean_vector, 
        embedding_type, 
        limit, 
        lang_filter
    )


# def query_doc_by_embedding(paper_ids: List[str], embedding: List[float], embedding_type: str, limit: int, lang_filter: Dict = None) -> List[Dict]:
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized. Cannot perform query_doc_by_embedding.")
#         return []

#     if not isinstance(embedding, (list, np.ndarray)):
#         logging.error(f"Invalid embedding type: expected list or np.ndarray, got {type(embedding)}")
#         return []

#     if isinstance(embedding, np.ndarray):
#         embedding = embedding.tolist()

#     if not all(isinstance(x, (float, int)) for x in embedding):
#         logging.error(f"Invalid embedding values: expected flat list of float, got: {embedding[:5]}")
#         return []

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not collection_name:
#         logging.error(f"No ChromaDB collection mapped for embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#     except Exception as e:
#         logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
#         return []

#     query_kwargs = {
#         "query_embeddings": [embedding],
#         "n_results": limit + len(paper_ids) + 5,
#         "include": ["metadatas", "distances"]
#     }

#     if lang_filter:
#         query_kwargs["where"] = lang_filter
#         logging.info(f"ChromaDB query: Applying language filter: {lang_filter}")

#     try:
#         results = collection.query(**query_kwargs)

#         docs = []
#         if results and results.get('ids') and results['ids'][0]:
#             for i in range(len(results['ids'][0])):
#                 doc_chroma_id = results['ids'][0][i]
#                 metadata = results['metadatas'][0][i]
#                 distance = results['distances'][0][i]

#                 doc_id_from_chroma = str(doc_chroma_id)
#                 if doc_id_from_chroma not in paper_ids:
#                     if metadata:
#                         metadata['_score'] = float(distance)
#                         docs.append(format_doc_for_frontend(metadata))
#                     else:
#                         logging.warning(f"Document with ID '{doc_id_from_chroma}' found in ChromaDB but not in _ready_docs_by_id cache. Skipping.")

#                 if len(docs) >= limit:
#                     break

#         return docs
#     except Exception as e:
#         logging.error(f"Error querying ChromaDB collection '{collection_name}': {e}", exc_info=True)
#         return []






# def query_doc_by_embedding(paper_ids: List[str], embedding: List[float], embedding_type: str, limit: int, lang_filter: Dict = None) -> List[Dict]:
#     if _chroma_client is None:
#         logging.error("ChromaDB client not initialized.")
#         return []

#     if isinstance(embedding, np.ndarray):
#         embedding = embedding.tolist()

#     collection_name = COLLECTION_MAPPING.get(embedding_type)
#     if not collection_name:
#         logging.error(f"No ChromaDB collection mapped for embedding type: {embedding_type}")
#         return []

#     try:
#         collection = _chroma_client.get_collection(collection_name)
#     except Exception as e:
#         logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
#         return []

#     query_kwargs = {
#         "query_embeddings": [embedding],
#         "n_results": limit + len(paper_ids) + 5,
#         "include": ["metadatas", "distances"]
#     }

#     # only add where if paper_ids is not empty
#     if paper_ids:
#         query_kwargs["where"] = {"ID": {"$nin": paper_ids}}

#     if lang_filter:
#         query_kwargs["where"].update(lang_filter)

#     try:
#         results = collection.query(**query_kwargs)
#         docs = []
#         if results and results.get('metadatas'):
#             for i, metadata in enumerate(results['metadatas'][0]):
#                 distance = results['distances'][0][i]
#                 if metadata:
#                     metadata["_score"] = float(distance)
#                     docs.append(format_doc_for_frontend(metadata))
#                 if len(docs) >= limit:
#                     break
#         return docs
#     except Exception as e:
#         logging.error(f"Error querying ChromaDB collection '{collection_name}': {e}", exc_info=True)
#         return []


def query_doc_by_embedding(
    paper_ids: List[str],
    embedding: List[float],
    embedding_type: str,
    limit: int,
    lang_filter: Dict = None
) -> List[Dict]:
    if _chroma_client is None:
        logging.error("ChromaDB client not initialized.")
        return []

    # Ensure embedding is always a list
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"No ChromaDB collection mapped for embedding type: {embedding_type}")
        return []

    try:
        collection = _chroma_client.get_collection(collection_name)
    except Exception as e:
        logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
        return []

    query_kwargs = {
        "query_embeddings": [embedding],
        "n_results": limit + len(paper_ids) + 5,
        "include": ["metadatas", "distances"]
    }

    # logging.info(
    #     f"[DEBUG] query_doc_by_embedding | embed_type={embedding_type}, "
    #     f"dim={len(embedding)}, paper_ids={len(paper_ids)}, "
    #     f"limit_param={limit}, n_results={query_kwargs['n_results']}, "
    #     f"where={query_kwargs.get('where', None)}"
    # )





    # Build where clause dynamically
    where_clause = {}
    if paper_ids:
        where_clause["ID"] = {"$nin": paper_ids}
    if lang_filter:
        where_clause.update(lang_filter)
    if where_clause:  # ✅ only set when not empty
        query_kwargs["where"] = where_clause

    logging.info(
        f"[DEBUG] query_doc_by_embedding | embed_type={embedding_type}, "
        f"dim={len(embedding)}, paper_ids={len(paper_ids)}, "
        f"where={query_kwargs.get('where', None)}"
    )

    try:
        results = collection.query(**query_kwargs)
        docs = []

        if results and results.get("metadatas"):
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i]
                if metadata:
                    metadata["_score"] = float(distance)
                    docs.append(format_doc_for_frontend(metadata))
                if len(docs) >= limit:
                    break

        logging.info(f"[DEBUG] Retrieved {len(docs)} docs from collection '{collection_name}'")
        return docs

    except Exception as e:
        logging.error(
            f"Error querying ChromaDB collection '{collection_name}': {e}",
            exc_info=True
        )
        return []





# def query_similar_doc_by_embedding_2d(papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None):
#     logging.info(f"🧪 [2D] Running similarity query on {len(papers)} papers with embedding type: {embedding_type}")
#     for p in papers:
#         logging.info(f"📄 ID={p.get('ID')} Title={p.get('Title', '')}")
#     return query_similar_doc_by_embedding_full(papers, embedding_type, limit, lang_filter)

def query_similar_doc_by_embedding_2d(
    papers: List[dict], embedding_type: str, limit: int = 25, lang_filter: Dict = None
):
    logging.info(f"🧪 [2D] Running similarity query on {len(papers)} papers with embedding type: {embedding_type}")

    umap_field = {
        EMBED.ADA: "ada_umap",
        EMBED.GLOVE: "glove_umap",
        EMBED.SPECTER: "specter_umap"
    }.get(embedding_type)
    if not umap_field:
        logging.error(f"No UMAP field found for embedding type: {embedding_type}")
        return []

    query_points = []
    for p in papers:
        coords = p.get(umap_field)
        if isinstance(coords, str):
            try:
                coords = json.loads(coords)
            except Exception:
                coords = None
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                xy = np.asarray(coords, dtype=float)
                if np.all(np.isfinite(xy)):
                    query_points.append(xy)
            except Exception:
                pass

    if not query_points:
        logging.warning("No valid 2D coords found for query papers.")
        return []

    mean_vector = np.mean(np.vstack(query_points), axis=0)

    all_points = get_all_umap_points(embedding_type)
    results = []
    for doc in all_points:
        coords = doc.get(umap_field)
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                xy = np.asarray(coords, dtype=float)
                if not np.all(np.isfinite(xy)):
                    continue
                dist = float(np.linalg.norm(xy - mean_vector))
                if not math.isfinite(dist):
                    continue 

                # ✅ 转成 score（越小的 distance → 越大的 score）
                score = 1.0 / (1.0 + dist)

                results.append({
                    "ID": str(doc.get("ID")) if doc.get("ID") else None,
                    "Title": doc.get("Title", ""),
                    "Abstract": doc.get("Abstract", ""),
                    "Authors": doc.get("Authors", []),
                    "Keywords": doc.get("Keywords", []),
                    "Source": doc.get("Source", ""),
                    "Year": doc.get("Year"),
                    "ada_umap": doc.get("ada_umap"),
                    "glove_umap": doc.get("glove_umap"),
                    "specter_umap": doc.get("specter_umap"),
                    "distance": dist,  
                    "score": score,     
                    "Sim": score         
                })
            except Exception:
                continue

    results.sort(key=lambda x: x["distance"])
    return results[:limit]

def normalize_results(results, mode="nD"):
    normalized = []
    for doc in results:
        sim = doc.get("score", 0.0)
        try:
            sim = float(sim)
        except Exception:
            sim = 0.0

        if not math.isfinite(sim):
            sim = 0.0

        doc["score"] = float(sim)
        doc["Sim"] = float(sim)
        normalized.append(doc)

    return normalized



def query_similar_doc_by_paper(paper: dict, embedding_type: str, limit: int = 25, lang_filter: Dict = None):
    logging.info(f"🧪 Running similarity query for single paper ID={paper.get('ID')}, Title={paper.get('Title', '')}")
    return query_similar_doc_by_embedding_full([paper], embedding_type, limit, lang_filter)

def get_all_umap_points(embedding_type: str = EMBED.SPECTER):
    try:
        collection = _chroma_client.get_collection(COLLECTION_MAPPING[embedding_type])
        results = collection.get(include=["metadatas"])

        points = []
        for doc in results.get("metadatas", []):
      
            def parse_coords(val):
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except:
                        return None
                return val

            points.append({
                "ID": str(doc.get("ID")) if doc.get("ID") else None,   
                "Title": doc.get("Title", ""),                        
                "Year": doc.get("Year"),
                "Source": doc.get("Source", ""),                      
                "ada_umap": parse_coords(doc.get("ada_umap")),
                "glove_umap": parse_coords(doc.get("glove_umap")),
                "specter_umap": parse_coords(doc.get("specter_umap")),
            })

        # Debug 
        for p in points[:5]:
            logging.info(f"UMAP point sample: {p}")

        return points
    except Exception as e:
        logging.error(f"Failed to load UMAP points from Chroma: {e}", exc_info=True)
        return []



def _aggregate_count(field: str) -> List[Dict[str, Any]]:
    counter = {}
    all_docs = get_all_docs()
    if not all_docs:
        logging.warning("No documents found for aggregation.")
        return []

    for doc in all_docs:
        values = doc.get(field)
        if values is None:
            continue
        
        if not isinstance(values, list):
            values = [values]

        for v in values:
            if v:
                key_str = str(v)
                counter[key_str] = counter.get(key_str, 0) + 1
    
    return sorted([{"_id": k, "count": v} for k, v in counter.items()], key=lambda x: -x["count"])


# def get_distinct_authors(): return list(set(a for doc in get_all_docs() for a in doc.get("Authors", []) if a))
# def get_distinct_sources(): return list(set(doc.get("Source") for doc in get_all_docs() if doc.get("Source")))
# def get_distinct_keywords(): return list(set(k for doc in get_all_docs() for k in doc.get("Keywords", []) if k))
# def get_distinct_years(): return sorted(list(set(doc.get("Year") for doc in get_all_docs() if doc.get("Year") is not None)))
# def get_distinct_titles(): return list(set(doc.get("Title") for doc in get_all_docs() if doc.get("Title")))
# def get_distinct_citation_counts(): return sorted(list(set(doc.get("CitationCounts") for doc in get_all_docs() if doc.get("CitationCounts") is not None)))

# def get_distinct_authors_with_counts(): return _aggregate_count("Authors")
# def get_distinct_sources_with_counts(): return _aggregate_count("Source")
# def get_distinct_keywords_with_counts(): return _aggregate_count("Keywords")
# def get_distinct_years_with_counts(): return sorted(_aggregate_count("Year"), key=lambda x: x['_id'])
# def get_distinct_titles_with_counts(): return _aggregate_count("Title")
# def get_distinct_citation_counts_with_counts(): return _aggregate_count("CitationCounts")

def get_all_metadatas_from_chroma(embedding_type: str = EMBED.SPECTER) -> List[dict]:
    if _chroma_client is None:
        logging.error("Chroma client is not initialized.")
        return []

    collection_name = COLLECTION_MAPPING.get(embedding_type)
    if not collection_name:
        logging.error(f"Invalid embedding type: {embedding_type}")
        return []

    try:
        collection = _chroma_client.get_collection(collection_name)
        results = collection.get(include=["metadatas"])
        return results.get("metadatas", [])
    except Exception as e:
        logging.error(f"Failed to fetch metadatas from Chroma collection '{collection_name}': {e}")
        return []

def get_distinct_authors(embedding_type: str = EMBED.SPECTER) -> List[str]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    return list(set(a for doc in docs for a in doc.get("Authors", []) if a))

# def get_distinct_sources(embedding_type: str = EMBED.SPECTER) -> List[str]:
#     docs = get_all_metadatas_from_chroma(embedding_type)
#     return list(set(doc.get("Source") for doc in docs if doc.get("Source")))
def get_distinct_sources(embedding_type: str = EMBED.SPECTER):
    docs = get_all_metadatas_from_chroma(embedding_type)
    formatted_docs = [format_doc_for_frontend(d) for d in docs]

    # Debug 
    for fd in formatted_docs[:5]:
        print("DEBUG formatted doc:", fd)

    return list(set(doc.get("Source") for doc in formatted_docs if doc.get("Source")))


def get_distinct_keywords(embedding_type: str = EMBED.SPECTER) -> List[str]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    return list(set(k for doc in docs for k in doc.get("Keywords", []) if k))

def get_distinct_years(embedding_type: str = EMBED.SPECTER) -> List[int]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    return sorted(set(doc.get("Year") for doc in docs if doc.get("Year") is not None))

def get_distinct_titles(embedding_type: str = EMBED.SPECTER) -> List[str]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    return list(set(doc.get("Title") for doc in docs if doc.get("Title")))

def get_distinct_citation_counts(embedding_type: str = EMBED.SPECTER) -> List[int]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    return sorted(set(doc.get("CitationCounts") for doc in docs if doc.get("CitationCounts") is not None))

def _aggregate_count_from_chroma(field: str, embedding_type: str = EMBED.SPECTER) -> List[Dict[str, Any]]:
    docs = get_all_metadatas_from_chroma(embedding_type)
    counter = {}

    for doc in docs:
        values = doc.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for v in values:
            if v:
                key_str = str(v)
                counter[key_str] = counter.get(key_str, 0) + 1

    return sorted([{"_id": k, "count": v} for k, v in counter.items()], key=lambda x: -x["count"])

def get_distinct_authors_with_counts(embedding_type: str = EMBED.SPECTER): return _aggregate_count_from_chroma("Authors", embedding_type)
def get_distinct_sources_with_counts(embedding_type: str = EMBED.SPECTER): return _aggregate_count_from_chroma("Source", embedding_type)
def get_distinct_keywords_with_counts(embedding_type: str = EMBED.SPECTER): return _aggregate_count_from_chroma("Keywords", embedding_type)
def get_distinct_years_with_counts(embedding_type: str = EMBED.SPECTER): return sorted(_aggregate_count_from_chroma("Year", embedding_type), key=lambda x: x['_id'])
def get_distinct_titles_with_counts(embedding_type: str = EMBED.SPECTER): return _aggregate_count_from_chroma("Title", embedding_type)
def get_distinct_citation_counts_with_counts(embedding_type: str = EMBED.SPECTER): return _aggregate_count_from_chroma("CitationCounts", embedding_type)


SIMILARITY_DECAY_ALPHA = 3.0

def format_doc_for_frontend(doc: dict, score_key="_score") -> dict:
    distance = doc.get(score_key)
    final_sim_value = 0.0

    for key in ["ada_umap", "glove_umap", "specter_umap"]:
        val = doc.get(key)
        if isinstance(val, str):
            try:
                doc[key] = json.loads(val)
            except json.JSONDecodeError:
                doc[key] = None

    # compute similarity from distance
    try:
        float_distance = float(distance) if distance is not None else float("nan")
        if not math.isnan(float_distance):
            final_sim_value = max(0.0, min(1.0, math.exp(-SIMILARITY_DECAY_ALPHA * float_distance)))
    except Exception as e:
        logging.warning(f"Distance parse error for doc {doc.get('ID')}: {e}")

    # handle Authors
    authors = doc.get("Authors", "")
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    # handle Keywords
    keywords = doc.get("Keywords", "")
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    # return formatted dict
    return {
        "ID": doc.get("ID"),
        "Title": doc.get("Title", ""),
        "Abstract": doc.get("Abstract", ""),
        "Authors": authors,
        "Keywords": keywords,
        "Source": doc.get("Source", ""),
        "Year": doc.get("Year"),
        "_Sim": final_sim_value,
        "Sim": final_sim_value,
        "score": final_sim_value,
        "ada_umap": doc.get("ada_umap"),
        "glove_umap": doc.get("glove_umap"),
        "specter_umap": doc.get("specter_umap"),
    }
