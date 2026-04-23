# config.py
#
# CHANGES — Supabase Storage integration:
#
#   Added:
#     supabase_url        : Supabase project URL (e.g. https://xxx.supabase.co)
#     supabase_service_key: service_role key (secret — never expose publicly)
#     supabase_bucket     : bucket name (must be public, default "pdfs")
#
#   Backward compatible: if all three are empty, Supabase upload is skipped
#   and the system behaves exactly as before (local-only mode).
#
# CHANGES — Admin panel auth:
#
#   Added:
#     admin_token: Bearer token required to access /admin/* routes.
#                  Leave empty to disable auth (dev mode).
#
# Everything else is UNCHANGED.

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model  : str = "llama-3.1-8b-instant"
    ollama_model: str = "llama3.2"
    max_turns   : int = 20

    # LLM provider selector
    llm_provider: str = "groq"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim  : int = 384
    hf_token       : str = ""

    # Embedder provider selector
    embedder: str = "huggingface"

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size          : int = 500
    chunk_overlap       : int = 50
    child_chunk_size    : int = 512
    child_chunk_overlap : int = 80
    parent_chunk_size   : int = 1500
    parent_chunk_overlap: int = 100

    # Chunker strategy selector
    chunker: str = "hierarchical"

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k           : int   = 20
    rrf_k           : int   = 60
    min_rerank_score: float = 0.1

    # Offline retrieval — chunks returned when LLM is unavailable
    offline_top_k: int = 5

    # ── Reranker (Person B settings) ──────────────────────────────────────
    enable_offline_reranker: bool = True
    reranker_model         : str  = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    reranker_top_k         : int  = 5

    # ── Network / sync ────────────────────────────────────────────────────
    network_check_url: str = "https://8.8.8.8"
    sync_manifest_url: str = ""   # set to your central server manifest endpoint

    network_poll_interval : int = 15   # seconds between connectivity checks
    network_check_timeout : int = 3    # HTTP timeout for each check

    # ── Vector store vendor (Person A) ────────────────────────────────────
    vector_store_vendor: str = "qdrant"

    # ── Qdrant (local mode — unchanged) ───────────────────────────────────
    qdrant_path      : str = str(BASE_DIR / "data" / "qdrant")
    qdrant_collection: str = "rag_docs"

    # Qdrant Cloud (cloud mode — leave empty for local-only)
    qdrant_cloud_url    : str = ""
    qdrant_cloud_api_key: str = ""

    # ── LanceDB ───────────────────────────────────────────────────────────
    lancedb_uri          : str = str(BASE_DIR / "data" / "lancedb")
    lancedb_cloud_uri    : str = ""
    lancedb_cloud_api_key: str = ""
    lancedb_cloud_region : str = ""

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_path: str = str(BASE_DIR / "data" / "chroma")
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_api_key: str = ""

    # ── Supabase Storage ──────────────────────────────────────────────────
    # Used to upload PDFs to a permanent public bucket during ingestion so
    # that the sync engine can later download them on other devices.
    #
    # Leave all three empty to run in local-only mode (no upload, no change
    # to existing behaviour).
    #
    # Setup:
    #   1. Create a Supabase project at https://supabase.com
    #   2. Go to Storage → create a bucket named "pdfs" and make it PUBLIC
    #   3. Go to Project Settings → API → copy the service_role key (secret)
    #   4. Set the three variables below in your .env file
    #
    # Public URL format produced after upload:
    #   https://<project-ref>.supabase.co/storage/v1/object/public/pdfs/<filename>
    supabase_url        : str = ""   # e.g. https://abcxyz.supabase.co
    supabase_service_key: str = ""   # service_role secret key
    supabase_bucket     : str = "pdfs"

    # ── Admin panel auth ──────────────────────────────────────────────────
    # Bearer token required to access /admin/* routes.
    # Leave empty to disable auth (dev mode — all admin routes are open).
    # Set a strong random string in production.
    #
    # Example (generate a token):
    #   python -c "import secrets; print(secrets.token_hex(32))"
    #
    # Usage — include in every admin request:
    #   Authorization: Bearer <your-token>
    admin_token: str = ""

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ── Legacy constants (kept for backward compatibility) ────────────────────────
QDRANT_PATH          = settings.qdrant_path
QDRANT_COLLECTION    = settings.qdrant_collection
EMBEDDING_DIM        = settings.embedding_dim
TOP_K                = settings.top_k
RRF_K                = settings.rrf_k
MIN_RERANK_SCORE     = settings.min_rerank_score
CHUNK_SIZE           = settings.chunk_size
CHUNK_OVERLAP        = settings.chunk_overlap
CHILD_CHUNK_SIZE     = settings.child_chunk_size
CHILD_CHUNK_OVERLAP  = settings.child_chunk_overlap
PARENT_CHUNK_SIZE    = settings.parent_chunk_size
PARENT_CHUNK_OVERLAP = settings.parent_chunk_overlap
GROQ_MODEL           = settings.groq_model
GROQ_API_KEY         = settings.groq_api_key
MAX_TURNS            = settings.max_turns
OLLAMA_MODEL         = settings.ollama_model
EMBEDDING_MODEL      = settings.embedding_model
OLLAMA_EMBED_MODEL   = settings.ollama_model
HF_TOKEN             = settings.hf_token
BM25_PATH            = str(Path(settings.qdrant_path).parent / "bm25.pkl")
IMAGES_DIR           = str(BASE_DIR / "data" / "images")
PDFS_DIR             = str(BASE_DIR / "data" / "pdfs")

# ── Directory creation ────────────────────────────────────────────────────────
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(PDFS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
Path(settings.lancedb_uri).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)