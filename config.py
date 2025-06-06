import os

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# File path settings
meta_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/meta_data.json')
umap_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/umap_data.json')
raw_json_datafile = os.path.join(PROJ_ROOT_DIR, 'data/vitality_10000_with_embeddings.json')
ready_json_datafile = os.path.join(PROJ_ROOT_DIR, 'data/vitality_10000_marqo_ready.json')

# Data source: use only JSON + Marqo
data_source = "json"  # Force to use JSON; the system uniformly loads from local JSON files

# Marqo configuration
marqo_url = "http://localhost:8882"  # Update this if you change the port or deployment method

# Local Chroma configuration for LangChain (used only for /chat endpoint)
DB_FOLDER_NAME = 'data.db'
COLLECTION_NAME = 'paper_v3'  # The collection name corresponding to the embedding model (e.g., ada, glove, specter, etc.)