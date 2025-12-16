import json
import os
import chromadb
from tqdm import tqdm
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure embedding field to collection name mapping
embed_collection_map = {
    'ada_embedding': 'paper_ada_localized',
    'glove_embedding': 'paper_glove_localized',
    'specter_embedding': 'paper_specter',
}

# Define paths
DB_PATH = os.path.join("chroma_db")
# JSON_PATH = "data/vitality_10000_with_embeddings.json"
JSON_PATH = "data/VitaLITy-2.0.0_final.json"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=DB_PATH)
logging.info(f"ChromaDB client initialized at path: {DB_PATH}")

# Load JSON data
try:
    with open(JSON_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Successfully loaded data from {JSON_PATH}. Total documents: {len(data)}")
except FileNotFoundError:
    logging.error(f"Error: JSON data file not found at {JSON_PATH}. Please ensure the file exists.")
    exit(1)
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON from {JSON_PATH}: {e}")
    exit(1)


# Iterate through each embedding type and import its data into the corresponding ChromaDB collection
for embed_field, collection_name in embed_collection_map.items():
    # Delete old collection before re-importing to ensure a clean slate.
    # In a production environment, a more robust data migration strategy is recommended.
    try:
        client.delete_collection(collection_name)
        logging.info(f"🗑️ Successfully deleted old collection: {collection_name}")
    except Exception as e:
        logging.warning(f"⚠️ Could not delete collection {collection_name} (might not exist or be empty): {e}")

    # Get or create the new collection
    collection = client.get_or_create_collection(collection_name)
    logging.info(f"📥 Importing data into collection: {collection_name}...")

    embeddings_batch = []
    ids_batch = []
    metadatas_batch = []

    BATCH_SIZE = 1000

    for d in tqdm(data, desc=f"Preparing and normalizing data ({collection_name})"):
        doc_id = str(d.get("ID"))
        
        # Check if the embedding field exists and is valid
        if embed_field not in d or d[embed_field] is None or not d[embed_field]:
            logging.debug(f"Document ID '{doc_id}' has no valid '{embed_field}' embedding. Skipping.")
            continue

        embedding_vector = np.array(d[embed_field])
        
        # Core change: Perform L2 normalization on stored vectors
        # Ensure all vectors stored in the database are normalized.
        norm = np.linalg.norm(embedding_vector)
        if norm > 0:
            embedding_vector = embedding_vector / norm
        else:
            logging.warning(f"Embedding vector for '{doc_id}' ({embed_field}) is a zero vector. Cannot normalize. Skipping.")
            continue


        # 🚀 CRITICAL FIX: Convert list-type metadata fields to strings
        # ChromaDB metadata values must be str, int, float, bool, or None.
        
        # Handle Authors field
        authors_data = d.get("Authors")
        if isinstance(authors_data, list):
            # Join list elements into a single string
            metadata_authors = ", ".join(authors_data)
        elif authors_data is not None:
            # Convert non-list, non-None data to string
            metadata_authors = str(authors_data)
        else:
            # Default to empty string if None
            metadata_authors = ""

        # Handle Keywords field (similar logic as Authors)
        keywords_data = d.get("Keywords")
        if isinstance(keywords_data, list):
            # Join list elements into a single string
            metadata_keywords = ", ".join(keywords_data)
        elif keywords_data is not None:
            # Convert non-list, non-None data to string
            metadata_keywords = str(keywords_data)
        else:
            # Default to empty string if None
            metadata_keywords = ""

        # Add 'lang' field to metadata
        # metadata = {
        #     "ID": doc_id,
        #     "Title": d.get("Title", ""),
        #     "Abstract": d.get("Abstract", ""),
        #     "Lang": d.get("lang", "unknown").lower(),
        #     "Year": int(d["Year"]) if isinstance(d.get("Year"), (int, str)) and str(d.get("Year")).isdigit() else None,
        #     "Source": d.get("Source", ""),
        #     # "Authors": [a.strip() for a in d.get("Authors", [])] if isinstance(d.get("Authors"), list) else [str(d.get("Authors"))] if d.get("Authors") else [],
        #     # "Keywords": [k.strip() for k in d.get("Keywords", [])] if isinstance(d.get("Keywords"), list) else [str(d.get("Keywords"))] if d.get("Keywords") else []
            
        #     "Authors": ", ".join([a.strip() for a in d.get("Authors", [])]) if isinstance(d.get("Authors"), list) else str(d.get("Authors") or ""),
        #     "Keywords": ", ".join([k.strip() for k in d.get("Keywords", [])]) if isinstance(d.get("Keywords"), list) else str(d.get("Keywords") or "")

        # }

        metadata = {
            "ID": doc_id,
            "Title": d.get("Title", ""),
            "Abstract": d.get("Abstract", ""),
            "Lang": d.get("lang", "unknown").lower(),
            "Year": int(d["Year"]) if isinstance(d.get("Year"), (int, str)) and str(d.get("Year")).isdigit() else None,
            "Source": d.get("Source", ""),
            "Authors": ", ".join([a.strip() for a in d.get("Authors", [])]) if isinstance(d.get("Authors"), list) else str(d.get("Authors") or ""),
            "Keywords": ", ".join([k.strip() for k in d.get("Keywords", [])]) if isinstance(d.get("Keywords"), list) else str(d.get("Keywords") or ""),
            "CitationCounts": float(d.get("CitationCounts", 0.0)) if d.get("CitationCounts") is not None else None,
            "ada_umap": json.dumps(d.get("ada_umap")) if d.get("ada_umap") else None,
            "glove_umap": json.dumps(d.get("glove_umap")) if d.get("glove_umap") else None,
            "specter_umap": json.dumps(d.get("specter_umap")) if d.get("specter_umap") else None,
        }
        
        embeddings_batch.append(embedding_vector.tolist())
        ids_batch.append(doc_id)
        metadatas_batch.append(metadata)

        # Execute batch insertion when batch size is reached
        if len(ids_batch) >= BATCH_SIZE:
            try:
                collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch
                )
                logging.debug(f"Added batch of {len(ids_batch)} documents to {collection_name}.")
            except Exception as e:
                # Log the error, but continue if it's a batch error
                logging.error(f"Error adding batch to {collection_name}: {e}")
            
            # Clear batch lists
            embeddings_batch = []
            ids_batch = []
            metadatas_batch = []

    # Insert any remaining items
    if ids_batch:
        try:
            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                metadatas=metadatas_batch
            )
            logging.debug(f"Added final batch of {len(ids_batch)} documents to {collection_name}.")
        except Exception as e:
            logging.error(f"Error adding final batch to {collection_name}: {e}")

    # Final count check for the collection
    logging.info(f"✅ Successfully imported {collection.count()} documents into {collection_name}")

logging.info("\n🎉 All embedding data successfully loaded to ChromaDB!")