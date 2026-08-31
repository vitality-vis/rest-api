import json
import os

from dotenv import load_dotenv
from model.const import EMBED

PROJ_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_project_environment():
    """Load this project's optional .env without overriding real environment values.

    Under systemd, prefer EnvironmentFile= in the unit. That is read by PID 1
    before the service drops privileges, so the process may not be able to open
    .env itself — in that case skip quietly and use the already-injected env.
    """
    env_path = os.path.join(PROJ_ROOT_DIR, ".env")
    try:
        return load_dotenv(env_path, override=False)
    except OSError:
        return False


# Configuration values below are intentionally captured only after the project
# environment has been loaded. This makes API, scripts, and pytest consistent.
load_project_environment()


def _positive_int_environment_value(name: str, default: int) -> int:
    """Read a positive integer setting, failing fast for invalid configuration."""
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

# === Supabase (authenticated chat persistence) ===
# SUPABASE_SERVICE_ROLE_KEY is server-only and must never be exposed to clients.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def is_supabase_configured() -> bool:
    """Whether authenticated user persistence is configured."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


# === Public deployment host (CORS, Socket.IO, MCP) ===
VITALITY_PUBLIC_HOST = "vitality.mathcs.emory.edu"
VITALITY_PUBLIC_ORIGIN = f"https://{VITALITY_PUBLIC_HOST}"

LOCAL_DEV_CORS_ORIGINS = [
    "http://localhost:8080",  # User study dev server
    "http://localhost:8081",  # standalone
    "http://localhost:5173",  # rebuild Vite dev server
]
PUBLIC_CORS_ORIGINS = [*LOCAL_DEV_CORS_ORIGINS, VITALITY_PUBLIC_ORIGIN]

_MCP_LOCAL_ALLOWED_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*", "testserver"]
_MCP_LOCAL_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]


def mcp_allowed_hosts() -> list[str]:
    """Host header allowlist for the public /mcp endpoint."""
    return [
        *_MCP_LOCAL_ALLOWED_HOSTS,
        VITALITY_PUBLIC_HOST,
        f"{VITALITY_PUBLIC_HOST}:*",
    ]


def mcp_allowed_origins() -> list[str]:
    """Origin allowlist for the public /mcp endpoint."""
    return [*_MCP_LOCAL_ALLOWED_ORIGINS, *PUBLIC_CORS_ORIGINS]

# === Library PDF uploads ===
# This is safe to expose through the public bootstrap config: it is a UI limit,
# not an Azure credential or endpoint. 100 MiB is the default when unset.
LIBRARY_PDF_MAX_BYTES = _positive_int_environment_value(
    "LIBRARY_PDF_MAX_BYTES", 100 * 1024 * 1024
)

# === File path settings ===
meta_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/meta_data.json')
umap_data_file_path = os.path.join(PROJ_ROOT_DIR, 'data/umap_data.json')
cache_fingerprint_file_path = os.path.join(PROJ_ROOT_DIR, 'data/cache_fingerprint.json')

# Built Vite frontend served by Flask. Deploy with:
#   rsync -a --delete frontend-vitality2-study/dist/ rest-api/static/dist/
# Override with FRONTEND_DIST_DIR if needed.
_DEFAULT_FRONTEND_DIST = os.path.join(PROJ_ROOT_DIR, "static", "dist")
FRONTEND_DIST_DIR = os.path.abspath(
    os.environ.get("FRONTEND_DIST_DIR", _DEFAULT_FRONTEND_DIST)
)
# Raw JSON data file (with embeddings). Use the file you want to load into Zilliz.
raw_json_datafile = os.path.join(PROJ_ROOT_DIR, 'data/VitaLITy-2.0.0.json')

# Ready-to-use JSON file (optional: used consistently with the ready_docs function)
ready_json_datafile = raw_json_datafile

# === Data source settings ===
data_source = "json"  # Keep as json, indicating local JSON file is used for loading

# === Zilliz Cloud (vector database) ===
# Set in .env: ZILLIZ_URI (e.g. https://xxx.api.gcp-us-west1.zillizcloud.com), ZILLIZ_TOKEN (API key)
ZILLIZ_URI = os.environ.get("ZILLIZ_URI", "")
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN", "")

# Hard-coded request budgets for Chat Agent isolation (not env-tunable yet).
ZILLIZ_REQUEST_TIMEOUT_SECONDS = 30.0
EMBEDDING_REQUEST_TIMEOUT_SECONDS = 30.0
LLM_REQUEST_TIMEOUT_SECONDS = 120.0
AGENT_QUEUE_WAIT_TIMEOUT_SECONDS = 120.0
AGENT_RUN_TIMEOUT_SECONDS = 600.0
AGENT_SSE_KEEPALIVE_SECONDS = 15.0
AGENT_SSE_QUEUE_SIZE = 64

# === OpenAlex (citation neighbors; free API key required) ===
# Key: https://openalex.org/settings/api — free daily credit, not a paid plan.
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()


# === Azure OpenAI chat models ===
# Logical keys (API `model` param) map to Azure deployment names.
def _parse_available_chat_models() -> dict[str, str]:
    raw = (os.environ.get("AZURE_OPENAI_AVAILABLE_MODELS") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "AZURE_OPENAI_AVAILABLE_MODELS must be a JSON object of "
                '{ "model_key": "azure_deployment_name", ... }'
            ) from error
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(
                "AZURE_OPENAI_AVAILABLE_MODELS must be a non-empty JSON object"
            )
        models: dict[str, str] = {}
        for key, deployment in parsed.items():
            model_key = str(key).strip()
            deployment_name = str(deployment).strip()
            if not model_key or not deployment_name:
                raise ValueError(
                    "AZURE_OPENAI_AVAILABLE_MODELS entries must be non-empty strings"
                )
            models[model_key] = deployment_name
        return models

    # Backward compatible: a single AZURE_OPENAI_DEPLOYMENT becomes the only model.
    deployment = (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "").strip()
    if deployment:
        return {deployment: deployment}
    return {}


AZURE_OPENAI_AVAILABLE_MODELS = _parse_available_chat_models()
_raw_default_model = (os.environ.get("AZURE_OPENAI_DEFAULT_MODEL") or "").strip()
if _raw_default_model:
    if (
        AZURE_OPENAI_AVAILABLE_MODELS
        and _raw_default_model not in AZURE_OPENAI_AVAILABLE_MODELS
    ):
        raise ValueError(
            f"AZURE_OPENAI_DEFAULT_MODEL={_raw_default_model!r} is not a key in "
            "AZURE_OPENAI_AVAILABLE_MODELS"
        )
    AZURE_OPENAI_DEFAULT_MODEL = _raw_default_model
else:
    AZURE_OPENAI_DEFAULT_MODEL = next(iter(AZURE_OPENAI_AVAILABLE_MODELS), "")


def resolve_chat_model(model: str | None = None) -> str:
    """Return a validated logical chat-model key (request override or default)."""
    key = (model or "").strip() or AZURE_OPENAI_DEFAULT_MODEL
    if not key:
        raise ValueError("No chat model is configured")
    if key not in AZURE_OPENAI_AVAILABLE_MODELS:
        available = ", ".join(AZURE_OPENAI_AVAILABLE_MODELS) or "(none)"
        raise ValueError(f"Unknown model {key!r}. Available: {available}")
    return key


def resolve_chat_deployment(model: str | None = None) -> str:
    """Map a logical chat-model key to its Azure deployment name."""
    return AZURE_OPENAI_AVAILABLE_MODELS[resolve_chat_model(model)]


# === Zilliz paper collection schema ===
PAPER_COLLECTION = "paper_prod"

# === Paper embedding (Azure OpenAI Embeddings API; checked on vector request) ===
PAPER_EMBEDDING_MODEL = EMBED.TEXT_EMBEDDING_3_SMALL
PAPER_VECTOR_FIELD = "embedding"
PAPER_VECTOR_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = PAPER_EMBEDDING_MODEL
PAPER_UMAP_FIELD = "umap"
PAPER_VECTOR_METRIC = "COSINE"

AZURE_OPENAI_ENDPOINT = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
AZURE_OPENAI_API_KEY = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
AZURE_OPENAI_API_VERSION = (os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip()
AZURE_OPENAI_EMBED_DEPLOYMENT = (os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT") or "").strip()
AZURE_OPENAI_EMBED_API_VERSION = (os.environ.get("AZURE_OPENAI_EMBED_API_VERSION") or "").strip()


def is_azure_chat_configured() -> bool:
    """Whether the full-app Azure Chat capability has complete configuration."""
    return bool(
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_API_VERSION
        and AZURE_OPENAI_AVAILABLE_MODELS
        and AZURE_OPENAI_DEFAULT_MODEL
    )


def is_azure_embedding_configured() -> bool:
    """Whether remote vector-query embedding credentials are present.

    Missing values must not block exact/BM25 startup; vector requests check this
    (and raise a capability error) at call time.
    """
    return bool(
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_EMBED_DEPLOYMENT
        and AZURE_OPENAI_EMBED_API_VERSION
    )


def require_azure_embedding_config() -> None:
    """Raise when vector search is requested without embedding configuration."""
    if not is_azure_embedding_configured():
        raise RuntimeError("Azure OpenAI embedding configuration is incomplete")


def is_supported_embedding_model(name=None) -> bool:
    """Whether a request selects the one embedding model currently deployed."""
    return str(name or PAPER_EMBEDDING_MODEL).lower() == PAPER_EMBEDDING_MODEL


# Embedding dimensions retained for ingestion scripts using the collection map.
ZILLIZ_EMBED_DIM = {
    PAPER_COLLECTION: PAPER_VECTOR_DIMENSION,
}
