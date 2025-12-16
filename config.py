import os

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# === File path settings ===
meta_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/meta_data.json')
umap_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/umap_data.json')

# Raw JSON data file (with embeddings)
raw_json_datafile = os.path.join(PROJ_ROOT_DIR, 'data/vitality_10000_with_embeddings.json')

# Ready-to-use JSON file (optional: used consistently with the ready_docs function)
ready_json_datafile = raw_json_datafile  # Consistently use the with_embeddings file

# === Data source settings ===
data_source = "json"  # Keep as json, indicating local JSON file is used for loading

# === Chroma settings (local vector database) ===
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "paper_specter"  # The collection name you successfully inserted earlier