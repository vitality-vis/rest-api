# VitaLITy 2.0 - REST API

Backend API server for VitaLITy, built with Flask, ChromaDB, and LangChain.

---

## Requirements

- Python 3.9+
- Azure OpenAI API access (for LLM features)

---

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Azure OpenAI (required for LLM features)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key

# Azure OpenAI Embeddings
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBED_API_VERSION=2024-02-01

# Optional: Semantic Scholar API
SEMANTIC_SCHOLAR_API_KEY=your-key
```

### 4. Load Data into ChromaDB

```bash
python load_to_chroma.py
```

This creates the vector database in `chroma_db/` from `data/VitaLITy-2.0.0_final.json`.

---

## Run

### Development

```bash
# Run in production mode (default)
python main_chroma.py

# Run in debug mode with auto-reload
python main_chroma.py --debug
```

The API will be available at `http://localhost:3000` (or the port specified in the `PORT` environment variable).

**Note:** Debug mode enables Flask's debug mode and auto-reloader, which automatically restarts the server when code changes are detected. Use `--debug` only during development.

### Production (with Gunicorn)

```bash
pip install gunicorn eventlet
gunicorn --worker-class eventlet -w 1 --bind 127.0.0.1:8000 --timeout 600 main_chroma:app
```

---

## API Endpoints

### Paper Retrieval

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/getPapers` | POST | Get papers by IDs or titles |
| `/getSimilarPapers` | POST | Find similar papers by embedding |
| `/getSimilarPapersByAbstract` | POST | Find similar papers by abstract text |
| `/getUmapPoints` | GET | Get 2D UMAP coordinates for visualization |
| `/getMetaData` | GET | Get metadata for UI filters |

### LLM Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Chat with LLM about selected papers |
| `/summarize` | POST | Summarize selected papers |
| `/literatureReview` | POST | Generate a literature review |

### Example Requests

**Get Similar Papers:**
```json
POST /getSimilarPapers
{
    "input_data": ["Paper Title 1", "Paper Title 2"],
    "input_type": "Title",
    "limit": 10,
    "embedding": "ada",
    "dimensions": "2D"
}
```

**Find Papers by Abstract:**
```json
POST /getSimilarPapersByAbstract
{
    "input_data": {
        "title": "My Paper Title",
        "abstract": "This paper explores..."
    },
    "limit": 10
}
```

**Chat with Papers:**
```json
POST /chat
{
    "papers": [...],
    "message": "What are the main themes in these papers?"
}
```

---

## Project Structure

```
rest-api-main/
├── main_chroma.py      # Main Flask application
├── config.py           # Configuration settings
├── load_to_chroma.py   # Script to load data into ChromaDB
├── prompt.py           # LLM prompts
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
├── data/
│   └── VitaLITy-2.0.0_final.json  # Paper data with embeddings
├── chroma_db/          # ChromaDB vector database
├── service/            # Query and RAG functions
├── model/              # Constants and schemas
└── extension/          # Data caching
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for VM deployment instructions.

---

## Credits

VitaLITy was created by 
<a target="_blank" href="https://arpitnarechania.github.io">Arpit Narechania</a>, <a target="_blank" href="https://www.karduni.com/">Alireza Karduni</a>, <a target="_blank" href="https://wesslen.netlify.app/">Ryan Wesslen</a>, and <a target="_blank" href="https://emilywall.github.io/">Emily Wall</a>.

---

## Citation

```bibTeX
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

The software is available under the [MIT License](https://github.com/vitality-vis/rest-api/blob/master/LICENSE).

---

## Contact

If you have any questions, feel free to [open an issue](https://github.com/vitality-vis/rest-api/issues/new/choose) or contact [Arpit Narechania](https://narechania.com).
