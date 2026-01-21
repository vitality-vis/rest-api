import json
import os

import chromadb

import config

embed_collection_map = {
    'ada_embedding': 'paper_v4_ada_embedding',
    'specter_embedding': 'paper_v4_specter_embedding',
    'glove_embedding': 'paper_v4_glove_embedding',
}


def chroma_collection(collection_name: str):
    chroma_path = os.path.join(config.PROJ_ROOT_DIR, config.DB_FOLDER_NAME)
    vector_db_client = chromadb.PersistentClient(path=chroma_path)
    return vector_db_client.get_or_create_collection(collection_name)


def load_json_to_chroma(embed_name):
    chroma = chroma_collection(embed_collection_map[embed_name])

    json_file_path = os.path.join(config.PROJ_ROOT_DIR, config.raw_json_datafile)
    with open(json_file_path) as f:
        data = json.load(f)

    start_idx = 0
    step = 100

    while True:
        data_slice = data[start_idx:start_idx + step]
        if not data_slice:
            break

        chroma.add(
            embeddings=[i[embed_name] for i in data_slice],
            ids=[i['ID'] for i in data_slice],
        )
        start_idx = start_idx + step
