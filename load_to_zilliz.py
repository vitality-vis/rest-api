import os
import json
import argparse
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from tqdm import tqdm
from logger_config import get_logger

logging = get_logger()

import config
from service.metadata_normalizer import parse_string_list

# Collection name mapping
embed_collection_map = {
    "ada_embedding": "paper_ada_localized",
    "glove_embedding": "paper_glove_localized",
    "specter_embedding": "paper_specter",
}

JSON_PATH = getattr(config, "raw_json_datafile", None) or os.path.join(config.PROJ_ROOT_DIR, "data", "VitaLITy-2.0.0.json")
BATCH_SIZE = 500


def _embedding_seq_len(emb):
    """Length of embedding sequence; avoids truthiness on NumPy arrays."""
    if emb is None:
        return 0
    try:
        return len(emb)
    except TypeError:
        return 0


def _pick(doc: dict, *keys, default=None):
    for key in keys:
        value = doc.get(key)
        if value is not None:
            return value
    return default


def _create_schema(dim: int):
    from pymilvus import CollectionSchema, FieldSchema, DataType
    fields = [
        FieldSchema(name="ID", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="Title", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="Abstract", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="Authors", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=256, max_length=512),
        FieldSchema(name="Keywords", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=256, max_length=512),
        FieldSchema(name="Source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="Year", dtype=DataType.INT64),
        FieldSchema(name="CitationCounts", dtype=DataType.DOUBLE),
        FieldSchema(name="Lang", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="Doi", dtype=DataType.VARCHAR, max_length=256),
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
    def _cap_list(items, max_items=256, max_length=512):
        out = []
        for item in items[:max_items]:
            s = str(item).strip()
            if s:
                out.append(s[:max_length])
        return out

    authors_value = _cap_list(parse_string_list(_pick(doc, "Authors", "authors")))
    keywords_value = _cap_list(parse_string_list(_pick(doc, "Keywords", "keywords")))

    year = _pick(doc, "Year", "year")
    try:
        year = int(year) if year is not None else 0
    except Exception:
        year = 0

    citation = _pick(doc, "CitationCounts", "citationCounts", "citationcounts")
    try:
        citation = float(citation) if citation is not None else 0.0
    except Exception:
        citation = 0.0

    return {
        "Title": (_pick(doc, "Title", "title", default="") or "")[:2047],
        "Abstract": (_pick(doc, "Abstract", "abstract", default="") or "")[:65534],
        "Authors": authors_value,
        "Keywords": keywords_value,
        "Source": (_pick(doc, "Source", "source", default="") or "")[:1023],
        "Year": year if year else 0,
        "CitationCounts": citation if citation is not None else 0.0,
        "Lang": (_pick(doc, "Lang", "lang", default="unknown") or "unknown").lower()[:63],
        "Doi": (_pick(doc, "Doi", "doi", default="unknown") or "unknown").lower()[:255],
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
        [m["Doi"] for m in batch["meta"]],
        [m["ada_umap"] for m in batch["meta"]],
        [m["glove_umap"] for m in batch["meta"]],
        [m["specter_umap"] for m in batch["meta"]],
    ])
    inserted = len(batch["ids"])
    batch["ids"].clear()
    batch["embeddings"].clear()
    batch["meta"].clear()
    return inserted


def _is_array_field(field) -> bool:
    dtype_name = str(getattr(field, "dtype", "")).upper()
    return "ARRAY" in dtype_name


def _validate_existing_schema(collection, collection_name: str) -> bool:
    fields = {field.name: field for field in collection.schema.fields}
    required_array_fields = ("Authors", "Keywords")

    for field_name in required_array_fields:
        field = fields.get(field_name)
        if field is None:
            logging.error(
                f"Collection {collection_name} is missing required field '{field_name}'. "
                "Run loader without --keep-existing to recreate collections."
            )
            return False
        if not _is_array_field(field):
            logging.error(
                f"Collection {collection_name} has incompatible schema: '{field_name}' is not ARRAY. "
                "Run loader without --keep-existing to recreate collections."
            )
            return False
    return True


def _is_collection_empty(collection) -> bool:
    # num_entities can lag until flush on some Milvus/Zilliz versions; flush before counting.
    try:
        collection.flush()
        return int(collection.num_entities) == 0
    except Exception as exc:
        logging.error(f"Failed to inspect existing collection size: {exc}")
        return False


def main(keep_existing: bool = False, check_duplicate_ids: bool = False):
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
                if emb is not None and _embedding_seq_len(emb) > 0:
                    stats["dims"][embed_field] = len(emb)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"JSON parse error while scanning {JSON_PATH}: {e}")
        return

    logging.info(f"Scanned {stats['total_docs']} documents from {JSON_PATH}")

    collections = {}
    needs_index_build = {}
    for embed_field, collection_name in embed_collection_map.items():
        dim = stats["dims"][embed_field]
        if dim is None:
            dim = config.ZILLIZ_EMBED_DIM.get(collection_name, 768)
            logging.warning(f"No valid '{embed_field}' in data; using config dim={dim} for {collection_name}")
        else:
            logging.info(f"Using embedding dim={dim} for {collection_name} (from data)")
        if utility.has_collection(collection_name):
            if keep_existing:
                collection = Collection(name=collection_name)
                if not _validate_existing_schema(collection, collection_name):
                    return
                if not _is_collection_empty(collection):
                    logging.error(
                        f"--keep-existing only supports empty collections, but {collection_name} already has data. "
                        "Use default mode (drop/recreate) to avoid duplicate IDs or insertion conflicts."
                    )
                    return
                logging.info(f"Using existing EMPTY collection without recreating: {collection_name}")
                needs_index_build[embed_field] = False
            else:
                utility.drop_collection(collection_name)
                logging.info(f"Dropped existing collection: {collection_name}")
                schema = _create_schema(dim)
                collection = Collection(name=collection_name, schema=schema)
                logging.info(f"Created collection: {collection_name} (dim={dim})")
                needs_index_build[embed_field] = True
        else:
            schema = _create_schema(dim)
            collection = Collection(name=collection_name, schema=schema)
            logging.info(f"Created collection: {collection_name} (dim={dim})")
            needs_index_build[embed_field] = True
        collections[embed_field] = collection

    batches = {
        embed_field: {"ids": [], "embeddings": [], "meta": []}
        for embed_field in embed_collection_map
    }
    inserted_counts = {embed_field: 0 for embed_field in embed_collection_map}
    seen_ids = set() if check_duplicate_ids else None

    try:
        iterator = _iter_json_array(JSON_PATH)
        progress = tqdm(iterator, total=stats["total_docs"], desc="Loading documents")
        for doc in progress:
            raw_id = _pick(doc, "ID", "id")
            if raw_id is None or str(raw_id).strip() == "":
                continue
            doc_id = str(raw_id).strip()
            if seen_ids is not None:
                if doc_id in seen_ids:
                    logging.warning("Skipping duplicate document ID in JSON: %s", doc_id)
                    continue
                seen_ids.add(doc_id)
            meta = None

            for embed_field, collection in collections.items():
                emb = doc.get(embed_field)
                if emb is None or _embedding_seq_len(emb) == 0:
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
        if needs_index_build.get(embed_field, False):
            collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logging.info(
            f"Inserted {inserted_counts[embed_field]} rows and loaded collection: {collection_name}"
        )

    logging.info("All data loaded to Zilliz Cloud.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load paper embeddings and metadata into Zilliz.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not drop/recreate collections; only allowed for empty existing collections.",
    )
    parser.add_argument(
        "--check-duplicate-ids",
        action="store_true",
        help="Track IDs in memory: warn and skip rows when the same ID appears more than once in the JSON.",
    )
    args = parser.parse_args()
    main(keep_existing=args.keep_existing, check_duplicate_ids=args.check_duplicate_ids)
