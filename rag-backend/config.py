# config.py
#
# CHANGES vs previous version (Day 2 — A5):
#   - PDFS_DIR added: data/pdfs/ directory where ingested PDF files are
#     copied and served as static files via /pdfs/{filename}.
#   - Directory is created on startup alongside existing dirs.

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_model: str = "llama3.2"
    max_turns: int = 20

    # LLM provider selector
    llm_provider: str = "groq"

    # Qdrant (local only)
    qdrant_path: str = str(BASE_DIR / "data" / "qdrant")
    qdrant_collection: str = "rag_docs"

    # Embeddings — switched to bge-small-en-v1.5 (384-dim, ~130MB, half of base)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    hf_token: str = ""

    # Embedder provider selector
    embedder: str = "huggingface"

    # Chunking — tuned for manual-style content
    chunk_size: int = 500
    chunk_overlap: int = 50
    child_chunk_size: int = 300
    child_chunk_overlap: int = 30
    parent_chunk_size: int = 1500   # increased for long manual sections
    parent_chunk_overlap: int = 100

    # Chunker strategy selector
    chunker: str = "hierarchical"

    # Retrieval
    top_k: int = 20
    rrf_k: int = 60
    min_rerank_score: float = 0.1

    # Offline retrieval — how many chunks to return when LLM is unavailable
    offline_top_k: int = 5

    # Network / sync
    network_check_url: str = "https://8.8.8.8"
    sync_manifest_url: str = ""   # set to your central server manifest endpoint

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ── Legacy constants ───────────────────────────────────────────
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

# ── NEW (A5): PDFs directory ───────────────────────────────────
# Ingested PDF files are copied here so they can be served as static
# files at /pdfs/{filename}. This enables the PDF viewer modal (B1).
PDFS_DIR = str(BASE_DIR / "data" / "pdfs")

# Ensure directories exist
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(PDFS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)