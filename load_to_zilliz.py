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

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON not found: {JSON_PATH}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return

    logging.info(f"Loaded {len(data)} documents from {JSON_PATH}")

    for embed_field, collection_name in embed_collection_map.items():
        # Infer dimension from first valid embedding in data (so schema matches your JSON)
        dim = None
        for d in data:
            if d.get(embed_field) and len(d.get(embed_field)) > 0:
                dim = len(d[embed_field])
                break
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

        ids_batch, embeddings_batch, meta_batch = [], [], []
        BATCH_SIZE = 500

        for d in tqdm(data, desc=f"Preparing {collection_name}"):
            doc_id = str(d.get("ID"))
            if embed_field not in d or d[embed_field] is None or not d[embed_field]:
                continue
            emb = np.array(d[embed_field], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm <= 0:
                continue
            emb = (emb / norm).tolist()

            authors = d.get("Authors")
            authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors or "")
            keywords = d.get("Keywords")
            keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords or "")
            year = d.get("Year")
            try:
                year = int(year) if year is not None else 0
            except Exception:
                year = 0
            citation = d.get("CitationCounts")
            try:
                citation = float(citation) if citation is not None else 0.0
            except Exception:
                citation = 0.0

            ids_batch.append(doc_id)
            embeddings_batch.append(emb)
            meta_batch.append({
                "Title": (d.get("Title") or "")[:2047],
                "Abstract": (d.get("Abstract") or "")[:65534],
                "Authors": authors_str[:8191],
                "Keywords": keywords_str[:8191],
                "Source": (d.get("Source") or "")[:1023],
                "Year": year if year else 0,
                "CitationCounts": citation if citation is not None else 0.0,
                "Lang": (d.get("lang", "unknown") or "unknown").lower()[:63],
                "ada_umap": json.dumps(d.get("ada_umap"))[:255] if d.get("ada_umap") else "",
                "glove_umap": json.dumps(d.get("glove_umap"))[:255] if d.get("glove_umap") else "",
                "specter_umap": json.dumps(d.get("specter_umap"))[:255] if d.get("specter_umap") else "",
            })

            if len(ids_batch) >= BATCH_SIZE:
                collection.insert([
                    ids_batch,
                    embeddings_batch,
                    [m["Title"] for m in meta_batch],
                    [m["Abstract"] for m in meta_batch],
                    [m["Authors"] for m in meta_batch],
                    [m["Keywords"] for m in meta_batch],
                    [m["Source"] for m in meta_batch],
                    [m["Year"] for m in meta_batch],
                    [m["CitationCounts"] for m in meta_batch],
                    [m["Lang"] for m in meta_batch],
                    [m["ada_umap"] for m in meta_batch],
                    [m["glove_umap"] for m in meta_batch],
                    [m["specter_umap"] for m in meta_batch],
                ])
                ids_batch, embeddings_batch, meta_batch = [], [], []

        if ids_batch:
            collection.insert([
                ids_batch,
                embeddings_batch,
                [m["Title"] for m in meta_batch],
                [m["Abstract"] for m in meta_batch],
                [m["Authors"] for m in meta_batch],
                [m["Keywords"] for m in meta_batch],
                [m["Source"] for m in meta_batch],
                [m["Year"] for m in meta_batch],
                [m["CitationCounts"] for m in meta_batch],
                [m["Lang"] for m in meta_batch],
                [m["ada_umap"] for m in meta_batch],
                [m["glove_umap"] for m in meta_batch],
                [m["specter_umap"] for m in meta_batch],
            ])

        index_type = getattr(config, "ZILLIZ_INDEX_TYPE", "IVF_FLAT")
        nlist = getattr(config, "ZILLIZ_INDEX_NLIST", 2048)
        if index_type.upper() == "HNSW":
            index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 256}}
        else:
            index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": nlist}}
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logging.info(f"Inserted and loaded collection: {collection_name}")

    logging.info("All data loaded to Zilliz Cloud.")


if __name__ == "__main__":
    main()
