# config.py
#
# CHANGES — Day 1 joint commit (Person A + Person B):
#
#   Person A adds:
#     vector_store_vendor   : which backend to use (qdrant | lancedb | chroma)
#     qdrant_cloud_url      : Qdrant Cloud REST endpoint
#     qdrant_cloud_api_key  : Qdrant Cloud API key
#     lancedb_uri           : local LanceDB path
#     lancedb_cloud_uri     : LanceDB cloud URI (s3:// or lancedb://)
#     lancedb_cloud_api_key : LanceDB cloud key
#     lancedb_cloud_region  : LanceDB cloud region
#     chroma_path           : local ChromaDB persist path
#     chroma_host           : remote Chroma server host
#     chroma_port           : remote Chroma server port
#
#   Person B adds:
#     enable_offline_reranker : toggle offline reranker in chat router
#     reranker_model          : cross-encoder model name
#     reranker_top_k          : how many chunks to keep after offline rerank
#     network_poll_interval   : how often NetworkMonitor checks connectivity
#     network_check_timeout   : HTTP timeout for each connectivity check

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
    child_chunk_size    : int = 300
    child_chunk_overlap : int = 30
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
    # When enable_offline_reranker=True, the chat router applies the
    # cross-encoder in offline mode too (not just online mode).
    # Set ENABLE_OFFLINE_RERANKER=true in .env to activate.
    enable_offline_reranker: bool = True
    reranker_model         : str  = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    reranker_top_k         : int  = 5

    # ── Network / sync ────────────────────────────────────────────────────
    network_check_url: str = "https://8.8.8.8"
    sync_manifest_url: str = ""   # set to your central server manifest endpoint

    # Person B: surface poll interval and timeout to config
    # (were hardcoded 30s / 5s in rag_service.py and network_monitor.py)
    network_poll_interval : int = 15   # seconds between connectivity checks
    network_check_timeout : int = 3    # HTTP timeout for each check

    # ── Vector store vendor (Person A) ────────────────────────────────────
    # Controls which backend all vector operations use.
    # Change VECTOR_STORE_VENDOR in .env to switch the entire backend.
    # Values: "qdrant" (default) | "lancedb" | "chroma"
    vector_store_vendor: str = "qdrant"

    # ── Qdrant (local mode — unchanged) ───────────────────────────────────
    qdrant_path      : str = str(BASE_DIR / "data" / "qdrant")
    qdrant_collection: str = "rag_docs"

    # Qdrant Cloud (cloud mode — leave empty for local-only)
    qdrant_cloud_url    : str = ""
    qdrant_cloud_api_key: str = ""

    # ── LanceDB ───────────────────────────────────────────────────────────
    lancedb_uri          : str = str(BASE_DIR / "data" / "lancedb")
    lancedb_cloud_uri    : str = ""   # s3:// or lancedb:// URI
    lancedb_cloud_api_key: str = ""
    lancedb_cloud_region : str = ""

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_path: str = str(BASE_DIR / "data" / "chroma")
    chroma_host: str = ""        # remote Chroma server (leave empty for local)
    chroma_port: int = 8000

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ── Legacy constants (kept for backward compatibility) ────────────────────────
# Existing routers and services reference these module-level names.
# New code should use settings.* directly.
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