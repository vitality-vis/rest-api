"""
Load paper embeddings and metadata from JSON into Zilliz Cloud collections.

HOW TO UPLOAD YOUR DATA TO ZILLIZ:
----------------------------------
1. Get your Zilliz Cloud URL and API key:
   - Go to https://cloud.zilliz.com and sign in (or create an account).
   - Create or select a cluster.
   - In the cluster dashboard, find "Connect" / "Endpoint" → copy the URI
     (e.g. https://xxx.api.gcp-us-west1.zillizcloud.com).
   - In "API Key" or "Security" → create/copy your API key (token).

2. Create a file named .env in this project root with:
   ZILLIZ_URI=https://your-cluster-endpoint.api.region.zillizcloud.com
   ZILLIZ_TOKEN=your_api_key_here

3. Put your JSON data file in the path set below (JSON_PATH), or set
   raw_json_datafile in config.py. The JSON must contain embedding fields
   (e.g. specter_embedding, ada_embedding, glove_embedding).

4. Run:
   python load_to_zilliz.py

Credentials are read from the .env file (via config.ZILLIZ_URI and config.ZILLIZ_TOKEN).
"""
import os
import json
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from tqdm import tqdm
from logger_config import get_logger

logging = get_logger()

import config

# Collection name mapping
embed_collection_map = {
    "ada_embedding": "paper_ada_localized",
    "glove_embedding": "paper_glove_localized",
    "specter_embedding": "paper_specter",
}

JSON_PATH = getattr(config, "raw_json_datafile", None) or os.path.join(config.PROJ_ROOT_DIR, "data", "VitaLITy-2.0.0.json")
BATCH_SIZE = 500


def _create_schema(dim: int):
    from pymilvus import CollectionSchema, FieldSchema, DataType
    fields = [
        FieldSchema(name="ID", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="Title", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="Abstract", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="Authors", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="Keywords", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="Source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="Year", dtype=DataType.INT64),
        FieldSchema(name="CitationCounts", dtype=DataType.DOUBLE),
        FieldSchema(name="Lang", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="ada_umap", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="glove_umap", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="specter_umap", dtype=DataType.VARCHAR, max_length=256),
    ]
    return CollectionSchema(fields=fields, description="paper collection")


def _iter_json_array(filepath: str, chunk_size: int = 1024 * 1024):
    """
    Stream a top-level JSON array from disk so large datasets do not need to be
    fully materialized in memory.
    """
    decoder = json.JSONDecoder()
    with open(filepath, "r", encoding="utf-8") as f:
        buffer = ""
        in_array = False
        eof = False

        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            buffer = buffer.lstrip()
            if not in_array:
                if not buffer and eof:
                    raise ValueError("JSON file is empty.")
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise ValueError("Expected a top-level JSON array.")
                buffer = buffer[1:]
                in_array = True
                continue

            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if not buffer:
                if eof:
                    raise ValueError("Unexpected end of JSON while parsing array.")
                continue

            try:
                item, idx = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                continue

            yield item
            buffer = buffer[idx:]


def _extract_metadata(doc: dict) -> dict:
    authors = doc.get("Authors")
    if isinstance(authors, list):
        authors_value = json.dumps([str(a).strip() for a in authors if str(a).strip()], ensure_ascii=False)
    else:
        authors_value = str(authors or "")
    keywords = doc.get("Keywords")
    if isinstance(keywords, list):
        keywords_value = json.dumps([str(k).strip() for k in keywords if str(k).strip()], ensure_ascii=False)
    else:
        keywords_value = str(keywords or "")

    year = doc.get("Year")
    try:
        year = int(year) if year is not None else 0
    except Exception:
        year = 0

    citation = doc.get("CitationCounts")
    try:
        citation = float(citation) if citation is not None else 0.0
    except Exception:
        citation = 0.0

    return {
        "Title": (doc.get("Title") or "")[:2047],
        "Abstract": (doc.get("Abstract") or "")[:65534],
        "Authors": authors_value[:8191],
        "Keywords": keywords_value[:8191],
        "Source": (doc.get("Source") or "")[:1023],
        "Year": year if year else 0,
        "CitationCounts": citation if citation is not None else 0.0,
        "Lang": (doc.get("lang", "unknown") or "unknown").lower()[:63],
        "ada_umap": json.dumps(doc.get("ada_umap"))[:255] if doc.get("ada_umap") else "",
        "glove_umap": json.dumps(doc.get("glove_umap"))[:255] if doc.get("glove_umap") else "",
        "specter_umap": json.dumps(doc.get("specter_umap"))[:255] if doc.get("specter_umap") else "",
    }


def _flush_batch(collection, batch: dict) -> int:
    if not batch["ids"]:
        return 0

    collection.insert([
        batch["ids"],
        batch["embeddings"],
        [m["Title"] for m in batch["meta"]],
        [m["Abstract"] for m in batch["meta"]],
        [m["Authors"] for m in batch["meta"]],
        [m["Keywords"] for m in batch["meta"]],
        [m["Source"] for m in batch["meta"]],
        [m["Year"] for m in batch["meta"]],
        [m["CitationCounts"] for m in batch["meta"]],
        [m["Lang"] for m in batch["meta"]],
        [m["ada_umap"] for m in batch["meta"]],
        [m["glove_umap"] for m in batch["meta"]],
        [m["specter_umap"] for m in batch["meta"]],
    ])
    inserted = len(batch["ids"])
    batch["ids"].clear()
    batch["embeddings"].clear()
    batch["meta"].clear()
    return inserted


def main():
    # Credentials come from .env: ZILLIZ_URI and ZILLIZ_TOKEN (see config.py)
    if not config.ZILLIZ_URI or not config.ZILLIZ_TOKEN:
        logging.error(
            "Zilliz URL and API key are missing. Add them to a .env file in the project root:\n"
            "  ZILLIZ_URI=https://your-cluster.api.region.zillizcloud.com\n"
            "  ZILLIZ_TOKEN=your_api_key\n"
            "Get the URI and token from https://cloud.zilliz.com (cluster → Connect / API Key)."
        )
        return

    from pymilvus import connections, Collection, utility

    connections.connect(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
    logging.info("Connected to Zilliz Cloud (using ZILLIZ_URI from .env)")

    if not os.path.exists(JSON_PATH):
        logging.error(f"JSON not found: {JSON_PATH}")
        return

    try:
        stats = {"total_docs": 0, "dims": {field: None for field in embed_collection_map}}
        for doc in _iter_json_array(JSON_PATH):
            stats["total_docs"] += 1
            for embed_field in embed_collection_map:
                if stats["dims"][embed_field] is not None:
                    continue
                emb = doc.get(embed_field)
                if emb:
                    stats["dims"][embed_field] = len(emb)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"JSON parse error while scanning {JSON_PATH}: {e}")
        return

    logging.info(f"Scanned {stats['total_docs']} documents from {JSON_PATH}")

    collections = {}
    for embed_field, collection_name in embed_collection_map.items():
        dim = stats["dims"][embed_field]
        if dim is None:
            dim = config.ZILLIZ_EMBED_DIM.get(collection_name, 768)
            logging.warning(f"No valid '{embed_field}' in data; using config dim={dim} for {collection_name}")
        else:
            logging.info(f"Using embedding dim={dim} for {collection_name} (from data)")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logging.info(f"Dropped existing collection: {collection_name}")

        schema = _create_schema(dim)
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"Created collection: {collection_name} (dim={dim})")
        collections[embed_field] = collection

    batches = {
        embed_field: {"ids": [], "embeddings": [], "meta": []}
        for embed_field in embed_collection_map
    }
    inserted_counts = {embed_field: 0 for embed_field in embed_collection_map}

    try:
        iterator = _iter_json_array(JSON_PATH)
        progress = tqdm(iterator, total=stats["total_docs"], desc="Loading documents")
        for doc in progress:
            doc_id = str(doc.get("ID"))
            meta = None

            for embed_field, collection in collections.items():
                emb = doc.get(embed_field)
                if emb is None or not emb:
                    continue

                emb = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm <= 0:
                    continue

                if meta is None:
                    meta = _extract_metadata(doc)

                batch = batches[embed_field]
                batch["ids"].append(doc_id)
                batch["embeddings"].append((emb / norm).tolist())
                batch["meta"].append(meta)

                if len(batch["ids"]) >= BATCH_SIZE:
                    inserted_counts[embed_field] += _flush_batch(collection, batch)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"JSON parse error while loading {JSON_PATH}: {e}")
        return

    for embed_field, collection in collections.items():
        inserted_counts[embed_field] += _flush_batch(collection, batches[embed_field])

    for embed_field, collection_name in embed_collection_map.items():
        collection = collections[embed_field]
        index_type = getattr(config, "ZILLIZ_INDEX_TYPE", "IVF_FLAT")
        nlist = getattr(config, "ZILLIZ_INDEX_NLIST", 2048)
        if index_type.upper() == "HNSW":
            index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 256}}
        else:
            index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": nlist}}
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logging.info(
            f"Inserted {inserted_counts[embed_field]} rows and loaded collection: {collection_name}"
        )

    logging.info("All data loaded to Zilliz Cloud.")


if __name__ == "__main__":
    main()
