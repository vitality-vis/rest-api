# VitaLITy 2.0 – REST API

Backend API for **VitaLITy**, built with **Flask**, **Zilliz Cloud** (vector DB), and **LangChain**. It provides paper retrieval (by ID, similarity, abstract), 2D UMAP endpoints, and LLM-powered chat, summarization, and literature review.

---

## Requirements

- **Python 3.9+**
- **Azure OpenAI** (LLM and optional Ada embeddings)
- **Zilliz Cloud** (vector database)
- **Supabase** (authenticated chat persistence)

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
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_AVAILABLE_MODELS={"gpt-5-mini":"gpt-5-mini","gpt-5.6-luna":"gpt-5.6-luna","gpt-5.6-terra":"gpt-5.6-terra"}
AZURE_OPENAI_DEFAULT_MODEL=gpt-5.6-luna

# Azure OpenAI Embeddings (optional – used when embedding type "ada" is selected)
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBED_API_VERSION=2024-02-01

# Zilliz Cloud (required)
ZILLIZ_URI=https://your-cluster.api.region.zillizcloud.com
ZILLIZ_TOKEN=your-zilliz-api-key

# Supabase (required for authenticated chat persistence)
# Use the project root URL, without /rest/v1. This is a server-only secret.
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-or-secret-key

# Optional
PORT=3000
```

Get Zilliz credentials from [Zilliz Cloud](https://cloud.zilliz.com).

`SUPABASE_SERVICE_ROLE_KEY` has administrator access and must remain in the backend `.env` or your deployment platform's secret store. Never add it to a Vite `VITE_*` variable, send it to the browser, or commit it. The frontend uses the separate public Supabase URL and anon/publishable key.

### 4. Supabase database migrations

Authenticated chat tables and RLS policies are versioned in `supabase/migrations/`. After authenticating the Supabase CLI, link this backend to the intended Supabase project and apply outstanding migrations:

```bash
npx supabase link --project-ref <your-project-ref>
npx supabase db push
```

The migration creates `chat_conversations` and `chat_messages`, enables Row Level Security (RLS), and limits table access to the owning authenticated user. Do not use `db push` against production until the migration has been reviewed.

### 5. (Optional) Pre-warm the local cache

```bash
python script/export_zilliz_static_data.py
```

This is optional: if the cache is missing or outdated, the API downloads fresh data from Zilliz on startup. Run this command only to pre-warm the local metadata, UMAP, and fingerprint files.

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
| `/getSimilarPapers` | POST | Similar papers for one or more seed paper IDs (bulk vector search + RRF) |
| `/getPaperCitations` | POST | References and cited-by papers from OpenAlex for one DOI |
| `/getUmapPoints` | GET | 2D UMAP coordinates for visualization |
| `/getMetaData` | GET | Metadata for UI filters |

### LLM

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat about selected papers (streaming) |
| `/summarize` | POST | Summarize selected papers |
| `/literatureReview` | POST | Generate a literature review |

`/getSimilarPapers` uses the collection's configured paper embedding and combines
the per-seed result lists with reciprocal rank fusion (RRF).

### Example requests

**Similar papers by paper list:**

```json
POST /getSimilarPapers
{
  "seed_ids": ["paper-id-1", "paper-id-2"],
  "limit": 25,
  "min_year": 2020,
  "source": ["CHI"]
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
├── supabase/
│   └── migrations/       # Versioned Supabase database schema and RLS policies
├── data/
│   └── VitaLITy-2.0.0.json   # Paper dataset (path configurable in config.py)
├── service/              # Core logic
│   ├── bootstrap.py
│   ├── citations.py
│   ├── embed.py
│   ├── fulltext.py
│   ├── lib.py
│   ├── memory_manager.py
│   ├── metadata_normalizer.py
│   ├── paper_qa.py
│   ├── search.py
│   ├── static_cache.py
│   └── zilliz.py
├── agents/
│   ├── agent_v1_legacy/
│   │   ├── agent_tools.py
│   │   ├── grounded_writer.py
│   │   ├── intent_classifier.py
│   │   ├── query_rewriter.py
│   │   ├── rag_core.py
│   │   ├── runner.py
│   │   ├── session_state.py
│   │   └── summary_routes.py
│   └── agent_v2/
│       ├── logging.py
│       ├── models.py
│       ├── reranker.py
│       ├── router.py
│       ├── runner.py
│       └── search_executor.py
├── model/
│   ├── const.py         # e.g. EMBED (specter, ada)
│   └── paper.py         # Paper request and response schemas
```

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
