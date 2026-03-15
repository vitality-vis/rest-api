# Zilliz search speed and precision tuning

These settings control how fast and how accurate vector search is. Set them in `.env` (or in `config.py`).

## Search parameters (used at query time)

| Variable | Default | Effect |
|----------|---------|--------|
| **ZILLIZ_SEARCH_NPROBE** | 128 | (IVF index only) Number of clusters to search. **Higher** → better recall, slower. Typical: 32–256. |
| **ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER** | 1.5 | Request `limit × multiplier` candidates from search, then trim. Slightly > 1 improves precision when excluding some IDs. |
| **ZILLIZ_SEARCH_EF** | 128 | (HNSW index only) Search range. **Higher** → better recall, slower. Use **128–256** for high precision. |

## Index parameters (used when loading data with `load_to_zilliz.py`)

| Variable | Default | Effect |
|----------|---------|--------|
| **ZILLIZ_INDEX_NLIST** | 2048 | (IVF only) Number of clusters. ~√n to 4√n. 75k rows → 256–2000. |
| **ZILLIZ_INDEX_TYPE** | HNSW | **HNSW**: default for speed + recall; use **ZILLIZ_SEARCH_EF** at query time. **IVF_FLAT**: alternative. |

## Quick tips

- **Faster, less precise**: lower `ZILLIZ_SEARCH_NPROBE` (e.g. 32), or use HNSW with lower `ZILLIZ_SEARCH_EF`.
- **High precision (no loss)**: HNSW with **ZILLIZ_SEARCH_EF=128** (default) or 256.
- After changing index settings, re-run `python load_to_zilliz.py` to rebuild the index.

## Region, CPU, and async (response speed)

- **Region**: Deploy the app in the **same region** (and VPC if possible) as your Zilliz Cloud cluster to reduce network latency. Set the cluster region in Zilliz Cloud and run your API in the same cloud/region.
- **CPU**: Cross-encoder reranking is CPU-bound. Allocate **more CPU** (or a dedicated worker) to shorten rerank time without changing precision.
- **Async**: The chat flow uses async streaming (`run_two_stage_rag_stream`). For high concurrency, run the app with **multiple workers** (e.g. Gunicorn/Uvicorn workers) so one blocking tool run doesn’t stall other requests.
