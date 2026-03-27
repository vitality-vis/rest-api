# VitaLITy 2.0 – REST API

Backend API for **VitaLITy**, built with **Flask**, **Zilliz Cloud** (vector DB), and **LangChain**. It provides paper retrieval (by ID, similarity, abstract), 2D UMAP endpoints, and LLM-powered chat, summarization, and literature review.

---

## Requirements

- **Python 3.9+**
- **Azure OpenAI** (LLM and optional Ada embeddings)
- **Zilliz Cloud** (vector database)

---

## Setup

### 1. Environment

**Option A – venv + pip:**

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B – Conda (from `environment.yml`):**

```bash
conda env create -f environment.yml
conda activate vitality-rest-api
```

### 2. Data

Place the paper dataset in the `data/` folder. The loader expects **`data/VitaLITy-2.0.0.json`** by default (see `config.py` → `raw_json_datafile`). If your dataset has a different name (e.g. `VitaLITy-2.0.0_final.json`), set the path in `config.py` or use the same filename.

### 3. Environment variables

Create a **`.env`** file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your own values:

```bash
# Azure OpenAI (LLM)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_API_KEY=your-api-key

# Azure OpenAI Embeddings (optional – used when embedding type "ada" is selected)
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBED_API_VERSION=2024-02-01

# Zilliz Cloud (required)
ZILLIZ_URI=https://your-cluster.api.region.zillizcloud.com
ZILLIZ_TOKEN=your-zilliz-api-key

# Optional
SEMANTIC_SCHOLAR_API_KEY=your-key
PORT=3000
```

Get Zilliz credentials from [Zilliz Cloud](https://cloud.zilliz.com).

### 4. Load data into Zilliz

```bash
python load_to_zilliz.py
```

This creates/updates Zilliz collections from your JSON file. Ensure `ZILLIZ_URI` and `ZILLIZ_TOKEN` are set.

### 5. Export cached metadata and UMAP data

```bash
python script/export_zilliz_static_data.py
```

---

## Run

**Development:**

```bash
python main.py
# With auto-reload:
python main.py --debug
```

Server runs at **http://localhost:3000** (or the port in `PORT`).

**Production (Gunicorn):**

```bash
pip install gunicorn eventlet
gunicorn --worker-class eventlet -w 1 --bind 127.0.0.1:8000 --timeout 600 main:app
```

---

## API overview

### Paper retrieval

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/getPapers` | POST | Get papers by IDs or full payload (with filters) |
| `/getSimilarPapers` | POST | Similar papers from a list of papers (by embedding) |
| `/getSimilarPapersByAbstract` | POST | Similar papers from abstract (and optional title) text |
| `/getUmapPoints` | GET | 2D UMAP coordinates for visualization |
| `/getMetaData` | GET | Metadata for UI filters |

### LLM

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat about selected papers (streaming) |
| `/summarize` | POST | Summarize selected papers |
| `/literatureReview` | POST | Generate a literature review |

### Embedding types

Supported for similarity search: **`specter`** (default), **`glove`**, **`ada`**. If Ada is requested but the Azure embed deployment is missing, the API falls back to Specter.

### Example requests

**Similar papers by abstract:**

```json
POST /getSimilarPapersByAbstract
{
  "input_data": "This paper explores neural retrieval and RAG.",
  "title": "Optional paper title",
  "embedding": "specter",
  "limit": 25,
  "lang": "all"
}
```

**Similar papers by paper list:**

```json
POST /getSimilarPapers
{
  "input_data": ["Paper Title 1", "Paper Title 2"],
  "embedding": "specter",
  "limit": 25,
  "dimensions": "nD"
}
```

**Chat:**

```json
POST /chat
{
  "papers": [...],
  "message": "What are the main themes in these papers?"
}
```


---

## Project structure

```
├── main.py              # Flask app and routes
├── config.py            # Paths, Zilliz and search settings
├── logger_config.py     # Logging (including optional Google Cloud)
├── prompt.py            # LLM prompts
├── load_to_zilliz.py    # Load JSON into Zilliz collections
├── requirements.txt
├── environment.yml      # Optional Conda env
├── data/
│   └── VitaLITy-2.0.0.json   # Paper dataset (path configurable in config.py)
├── service/              # Core logic
│   ├── zilliz.py        # Zilliz queries, similarity, UMAP
│   ├── embed.py         # Specter, Glove, Azure Ada embeddings
│   ├── rag_core.py      # RAG retrieval, rerank, formatting
│   ├── agent_runner.py  # LangChain agent and tools
│   ├── agent_tools.py   # RAG/semantic search tools
│   ├── intent_classifier.py
│   ├── query_rewriter.py
│   ├── memory_manager.py
│   ├── session_state.py
│   └── grounded_writer.py
├── model/
│   ├── const.py         # e.g. EMBED (specter, glove, ada)
│   └── query.py         # Query schemas
└── extension/
    └── ext_zilliz.py    # Caching / Zilliz helpers
```

---

## Tuning Zilliz

See **`docs/ZILLIZ_TUNING.md`** for index type (IVF_FLAT vs HNSW), `nprobe`, `ef`, and related options. Key env vars: `ZILLIZ_INDEX_TYPE`, `ZILLIZ_SEARCH_NPROBE`, `ZILLIZ_SEARCH_EF`, `ZILLIZ_SEARCH_CANDIDATES_MULTIPLIER`.

---

## Credits

VitaLITy was created by [Arpit Narechania](https://arpitnarechania.github.io), [Alireza Karduni](https://www.karduni.com/), [Ryan Wesslen](https://wesslen.netlify.app/), and [Emily Wall](https://emilywall.github.io/).

---

## Citation

```bibtex
@article{narechania2021vitality,
  title={vitaLITy: Promoting Serendipitous Discovery of Academic Literature with Transformers \& Visual Analytics},
  author={Narechania, Arpit and Karduni, Alireza and Wesslen, Ryan and Wall, Emily},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2022},
  doi={10.1109/TVCG.2021.3114820},
  publisher={IEEE}
}
```

---

## License

[MIT License](LICENSE).

---

## Contact

For questions or issues, open a [GitHub issue](https://github.com/vitality-vis/rest-api/issues) or contact [Arpit Narechania](https://narechania.com).
