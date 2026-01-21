## DO NOT RUN THIS CODE

import json
import pandas as pd
import numpy as np
import umap
import spacy
import ast
from tqdm import tqdm
from typing import List
# from gensim.corpora import Dictionary  # Can be removed if not using old GloVe TF-IDF
# from gensim.models import TfidfModel  # Can be removed if not using old GloVe TF-IDF
# from gensim.matutils import sparse2full  # Can be removed if not using old GloVe TF-IDF
# from sklearn.decomposition import TruncatedSVD  # Can be removed if not using old GloVe TF-IDF
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import gc  # For garbage collection and memory/VRAM release

# Language detection with FastText
try:
    import fasttext
    # Download pretrained FastText language detection model (run once if needed)
    # fasttext.util.download_model('lid.176')  # Uncomment if not already downloaded
    lid_model = fasttext.load_model('lid.176.bin')
    print("✅ FastText language detection model loaded successfully.")
except ImportError:
    print("❌ fasttext is not installed. Run `pip install fasttext`. Language detection will be disabled.")
    lid_model = None
except ValueError as e:
    print(f"❌ Failed to load FastText model: {e}. Make sure 'lid.176.bin' is present.")
    lid_model = None

# === Config ===
INPUT_JSON = 'data/VitaLITy-2.0.0.json'
OUTPUT_JSON = 'data/vitality_10000_with_embeddings.json'

# === Step 1: Load and preprocess data ===
with open(INPUT_JSON, 'r', encoding='utf-8') as f:  # Ensure UTF-8 encoding
    raw = json.load(f)

df = pd.DataFrame(raw)
df = df[df['Abstract'].map(lambda x: isinstance(x, str) and len(x) > 50)].copy()
df = df.head(1000)  # Or however many rows you want
print(f"📊 Number of records used: {len(df)}")

df['Authors'] = df['Authors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Keywords'] = df['Keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Language detection
print("🌐 Performing language detection...")
df['lang'] = 'unknown'  # Default to unknown
if lid_model:
    # Concatenate title + abstract to improve language detection accuracy
    texts_for_lang_detection = [f"{t} {a}" for t, a in zip(df["Title"], df["Abstract"])]
    
    # FastText returns something like (('__label__en',), 0.99)
    # We strip the label prefix to get 'en'
    predictions = [lid_model.predict(text) for text in tqdm(texts_for_lang_detection, desc="Detecting languages")]
    df['lang'] = [p[0][0].replace('__label__', '') if p and p[0] else 'unknown' for p in predictions]
else:
    print("⚠️ FastText not loaded. Skipping language detection. All documents will be labeled as 'unknown'.")

# === Step 2: Ada embedding (SentenceTransformer multi-language model) ===
# Using 'sentence-transformers/all-MiniLM-L6-v2', which already supports multiple languages
print("🧠 Generating ada_embedding (Multi-language MiniLM)")
# Use both title and abstract to improve embedding quality
texts_for_ada = [f"{t}. {a}" for t, a in zip(df["Title"], df["Abstract"])]
ada_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# normalize_embeddings=True by default ensures L2 normalization
ada_embeddings = ada_model.encode(texts_for_ada, show_progress_bar=True, normalize_embeddings=True) 

df["ada_embedding"] = ada_embeddings.tolist()

# UMAP for Ada
print("🎯 Running UMAP for ada_embedding...")
ada_umap_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
df["ada_umap"] = ada_umap_reducer.fit_transform(ada_embeddings).tolist()
print("✅ Ada embedding + UMAP done.")

# Free memory
del ada_model, ada_embeddings, ada_umap_reducer, texts_for_ada
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# === Step 3: GloVe-like embedding (using multilingual SentenceTransformer) ===
# 🚀 Key change: Use multilingual model to replace original SpaCy + TF-IDF GloVe
# This gives GloVe-like embeddings multilingual capability
print("🧠 Generating Glove-like embedding (Multi-language SentenceTransformer)")
glove_multilingual_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
texts_for_glove = [f"{t}. {a}" for t, a in zip(df["Title"], df["Abstract"])]

glove_embeddings = glove_multilingual_model.encode(texts_for_glove, show_progress_bar=True, normalize_embeddings=True)
df["glove_embedding"] = glove_embeddings.tolist()

# UMAP for GloVe-like embedding
print("🎯 Running UMAP for glove_embedding...")
umap_glove_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
df["glove_umap"] = umap_glove_reducer.fit_transform(glove_embeddings).tolist()
print("✅ GloVe-like embedding + UMAP done.")

# Free memory
del glove_multilingual_model, glove_embeddings, umap_glove_reducer, texts_for_glove
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# === Step 4: Specter embedding ===
# Specter is English-only, no multilingual support
# Still apply L2 normalization to work with ChromaDB and cosine similarity
print("🧠 Generating Specter embedding (English-specific)")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")
model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("Specter model moved to GPU.")

title_abs = [f"{t} {tokenizer.sep_token} {a}" for t, a in zip(df['Title'], df['Abstract'])]
specter_embeds = []

BATCH_SIZE_SPECTER = 32  # Use small batches for memory efficiency

with torch.no_grad():
    for i in tqdm(range(0, len(title_abs), BATCH_SIZE_SPECTER), desc="Specter embedding"):
        batch = title_abs[i:i+BATCH_SIZE_SPECTER]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model(**inputs)
        batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
        
        # 🚀 Apply L2 normalization to Specter embeddings
        specter_norms = np.linalg.norm(batch_embeds, axis=1, keepdims=True)
        batch_embeds = np.where(specter_norms != 0, batch_embeds / specter_norms, batch_embeds)
        
        specter_embeds.extend(batch_embeds.tolist())

df["specter_embedding"] = specter_embeds

# UMAP for Specter
print("🎯 Running UMAP for specter_embedding...")
umap_specter_reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
df["specter_umap"] = umap_specter_reducer.fit_transform(np.array(specter_embeds)).tolist()
print("✅ Specter embedding + UMAP done.")

# Free memory
del tokenizer, model, specter_embeds, umap_specter_reducer, title_abs
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# === Step 5: Save output ===
# Save with 'lang' field included, and preserve non-English characters
df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
print(f"✅ Processed data saved to: {OUTPUT_JSON}")