from typing import List, Dict, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from logger_config import get_logger

# === New Imports for Azure OpenAI ===
import os
from openai import AzureOpenAI

# Use centralized logger
logging = get_logger()

# === Constants ===
MAX_BATCH_SIZE = 16
# ==========================================
# Azure OpenAI Configuration
# ==========================================
AZURE_EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBED_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_OPENAI_EMBED_API_VERSION")

# Initialize Azure Client
# Ensure these environment variables are set before running
embed_client = AzureOpenAI(
    api_version=AZURE_EMBED_API_VERSION,
    azure_endpoint=AZURE_EMBED_ENDPOINT,
    api_key=AZURE_EMBED_API_KEY
)

# ==========================================
# Local Embedding Classes (Unchanged)
# ==========================================
class LocalSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        logging.info(f"Loaded local SentenceTransformer model: {model_name}")

    def embed_query(self, text: str) -> List[float]:
        # We get the first and only result from the list
        embedding = self.model.encode([text], convert_to_numpy=False)[0].tolist()
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=False).tolist()


class LocalSpecterEmbedding(Embeddings):
    def __init__(self, model_name="allenai/specter"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda() 
            logging.info(f"Local Specter embedding model '{model_name}' moved to GPU.")
        else:
            logging.info(f"Loaded local Specter embedding model: {model_name} (on CPU).")

    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()} 
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy() 

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        BATCH_SIZE = 32 
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batch Specter Embedding"):
                batch_texts = texts[i:i+BATCH_SIZE]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()

                norms = np.linalg.norm(batch_embeds, axis=1, keepdims=True)
                batch_embeds = np.where(norms != 0, batch_embeds / norms, batch_embeds)
                
                embeddings.extend(batch_embeds.tolist())
        return embeddings


# ==========================================
# Model Instantiation
# ==========================================

# NOTE: Ada model is now handled via API, so we do NOT instantiate a local model for it.

# Glove Model (Still Local)
glove_model_instance = LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Specter Model (Still Local)
specter_model_instance = LocalSpecterEmbedding(model_name="allenai/specter")


# ==========================================
# Embedding Functions
# ==========================================

def ada_embedding(text_to_embed: Union[str, Dict]) -> List[float]:
    """
    Generates embeddings using Azure OpenAI (Ada).
    Accepts string or dictionary (extracting 'abstract').
    """
    # 1. Handle Input Types
    if isinstance(text_to_embed, dict):
        text_to_embed = text_to_embed.get('abstract', '')
    
    # 2. Validation
    if not text_to_embed or not isinstance(text_to_embed, str):
       logging.warning("No valid text found for ADA embedding. Returning empty list.")
       return []

    # 3. Call Azure OpenAI
    try:
        response = embed_client.embeddings.create(
            model=AZURE_EMBED_DEPLOYMENT,
            input=[text_to_embed] # API expects a list
        )
        # Extract embedding
        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        logging.error(f"Error generating Azure OpenAI Ada embedding: {e}")
        return []


# def glove_embedding(text_to_embed: str) -> List[float]:
#     """
#     Generates Glove-like embeddings using local SentenceTransformer.
#     """
#     if not text_to_embed:
#         logging.warning("No text found for Glove embedding. Returning empty list.")
#         return []
#     return glove_model_instance.embed_query(text_to_embed)

def glove_embedding(text_to_embed: Union[str, Dict]) -> List[float]:
    """
    Generates Glove-like embeddings using local SentenceTransformer.
    Robustly handles dict inputs by checking 'abstract', 'text', and 'title'.
    """
    # 1. Handle Dictionary Input (Extract text safely)
    if isinstance(text_to_embed, dict):
        # Try 'abstract' first, then 'text', then 'title', then combine them if needed
        extracted_text = text_to_embed.get('abstract') or text_to_embed.get('text') or text_to_embed.get('Title')
        
        # If still nothing, try lowercase keys
        if not extracted_text:
             extracted_text = text_to_embed.get('title', '')
        
        text_to_embed = extracted_text

    # 2. Validation
    if not text_to_embed or not isinstance(text_to_embed, str) or not text_to_embed.strip():
        logging.warning("[Glove] No valid text found to embed. Returning empty list.")
        return []

    # 3. Generate Embedding
    try:
        # logging.info(f"[Glove] Embedding text: {text_to_embed[:30]}...") # Uncomment to debug
        return glove_model_instance.embed_query(text_to_embed)
    except Exception as e:
        logging.error(f"[Glove] Error during embedding generation: {e}")
        return []


def specter_embedding(papers: Union[Dict, List[Dict]]) -> List[float]:
    """
    Generates Specter embeddings using the local Specter model.
    """
    if isinstance(papers, dict):
        papers = [papers]
    
    if not papers:
        logging.warning("No papers provided for Specter embedding.")
        return []

    paper = papers[0]
    
    title = paper.get('Title', '')
    abstract = paper.get('Abstract', '')
    
    # Specter [SEP] token
    combined_text = f"{title} {specter_model_instance.tokenizer.sep_token} {abstract}"
    
    if not combined_text.strip():
        logging.warning(f"Paper {paper.get('paper_id', 'unknown')} has no title or abstract for Specter embedding. Skipping.")
        return []

    try:
        embedding = specter_model_instance.embed_query(combined_text)
        logging.debug(f"Generated Specter embedding (length: {len(embedding)})")
        return embedding
    except Exception as e:
        logging.error(f"Error generating Specter embedding for paper: '{combined_text[:50]}...': {e}")
        return []


# ==========================================
# Utilities
# ==========================================
def chunks(lst: list, chunk_size: int = MAX_BATCH_SIZE):
    '''Splits a longer list to respect batch size'''
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def mean_embedding(embeddings: List[List[float]]) -> List[float]:
    """
    Computes the mean of a list of embedding vectors and L2 normalizes the result.
    """
    valid_embeddings = [e for e in embeddings if e is not None and len(e) > 0]
    if not valid_embeddings:
        logging.warning("No valid embeddings provided for mean_embedding calculation.")
        return []
    
    mean_vec = np.mean(np.array(valid_embeddings), axis=0)
    
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    else:
        logging.warning("Mean embedding resulted in a zero vector. Normalization skipped.")
        
    logging.debug(f"Calculated mean embedding (length: {len(mean_vec)})")
    return mean_vec.tolist()

def min_max_scaler(arr: List[float]) -> List[float]:
    """
    Applies Min-Max scaling to a list of numerical values.
    """
    if not arr:
        return []
    min_val = min(arr)
    max_val = max(arr)
    data_range = max_val - min_val
    if data_range == 0:
        logging.warning("Min-Max scaler received an array with no range (all values are identical). Returning zeros.")
        return [0.0] * len(arr)
    return [(x - min_val) / data_range for x in arr]






