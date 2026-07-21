
import os
from dotenv import load_dotenv

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_project_environment():
    """Load this project's optional .env without overriding real environment values."""
    return load_dotenv(os.path.join(PROJ_ROOT_DIR, ".env"), override=False)


# Configuration values below are intentionally captured only after the project
# environment has been loaded. This makes API, scripts, and pytest consistent.
load_project_environment()

# === Supabase (authenticated chat persistence) ===
# SUPABASE_SERVICE_ROLE_KEY is server-only and must never be exposed to clients.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# === File path settings ===
meta_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/meta_data.json')
umap_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/umap_data.json')
cache_fingerprint_file_path = os.path.join(PROJ_ROOT_DIR, 'data/cache_fingerprint.json')

# Raw JSON data file (with embeddings). Use the file you want to load into Zilliz.
raw_json_datafile = os.path.join(PROJ_ROOT_DIR, 'data/VitaLITy-2.0.0.json')

# Ready-to-use JSON file (optional: used consistently with the ready_docs function)
ready_json_datafile = raw_json_datafile

# === Data source settings ===
data_source = "json"  # Keep as json, indicating local JSON file is used for loading

# === Zilliz Cloud (vector database) ===
# Set in .env: ZILLIZ_URI (e.g. https://xxx.api.gcp-us-west1.zillizcloud.com), ZILLIZ_TOKEN (API key)
ZILLIZ_URI = os.environ.get("ZILLIZ_URI", "")
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN", "")
# Embedding dimensions per collection (used as fallback; loader can infer from data)
# paper_ada_localized: 1536 = OpenAI text-embedding-3-large/ada-002, 384 = MiniLM
ZILLIZ_EMBED_DIM = {
    "paper_specter": 768,
    "paper_ada_localized": 1536,
}

# === Zilliz search & index tuning (speed vs precision) ===
# nprobe: number of IVF clusters to search. Higher = better recall, slower. Typical 32–256.
ZILLIZ_SEARCH_NPROBE = int(os.environ.get("ZILLIZ_SEARCH_NPROBE", "128"))
# Number of candidates to request from vector search (before applying limit). Slightly more helps precision when excluding IDs.
ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER = float(os.environ.get("ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER", "1.5"))
# Index build: nlist (IVF) = number of clusters. ~sqrt(n) to 4*sqrt(n). 75k → 256–2000.
ZILLIZ_INDEX_NLIST = int(os.environ.get("ZILLIZ_INDEX_NLIST", "2048"))
# Index type: "IVF_FLAT" (good balance) or "HNSW" (often faster + better recall on Zilliz Cloud)
ZILLIZ_INDEX_TYPE = os.environ.get("ZILLIZ_INDEX_TYPE", "HNSW")
# For HNSW index only: search param "ef" (higher = better recall, slower). Use 128–256 for high precision.
ZILLIZ_SEARCH_EF = int(os.environ.get("ZILLIZ_SEARCH_EF", "128"))
