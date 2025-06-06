import os

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

meta_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/meta_data.json')
umap_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/umap_data.json')

# 'json' or 'mongodb'
data_source = "mongodb"

# if `data_source` = 'json'
# Note: 2-D search only works with MongoDB and not a raw JSON file.
# raw_json_datafile = 'data/VitaLITy-1.0.0.json'
raw_json_datafile = 'data/sample_100.json'
# if `data_source` = 'mongodb'
mongodb_connection_uri = 'mongodb://localhost:27017'
mongodb_database = 'vissimilarity'
# mongodb_docs_collection = 'docs'
mongodb_docs_collection = 'docsEmbedding'
mongodb_docs_embedding_collection = 'docsEmbedding'
mongodb_langchain_text_key = 'Abstract'


DB_FOLDER_NAME = 'data.db'
COLLECTION_NAME = 'paper_v3'  # v3 -> Vis and ada embedding