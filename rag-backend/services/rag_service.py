# services/rag_service.py
#
# CHANGES vs original:
#   - Session manager (TTLCache) REMOVED — no per-user sessions needed.
#     Single shared RAGChain instance. Workers don't need isolated histories.
#   - is_online property added — reads from NetworkMonitor singleton.
#   - _build_vector_store() simplified — Qdrant local only (Pinecone removed).
#   - _build_llm() simplified — Ollama removed from service layer.
#   - delete_file_from_stores() unchanged.
#   - NetworkMonitor imported and started at startup.

import threading
from pathlib import Path

from config import settings
from embeddings.embedder        import EmbedderFactory
from generation.groq_llm        import LLMFactory
from retrieval.bm25_store       import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from vectorstore.qdrant_store   import QdrantVectorStore
from chains.rag_chain           import RAGChain


# ── Singletons ────────────────────────────────────────────────
_embedder      : object = None
_reranker      : object = None
_vector_store  : object = None
_bm25_store    : object = None
_chain         : object = None   # single shared chain (no per-session isolation)
_network_monitor : object = None

_lock = threading.Lock()

# ── Background task registry ──────────────────────────────────
_tasks: dict = {}


# ── Factories ─────────────────────────────────────────────────

def _build_vector_store(embedder):
    """Qdrant local only — Pinecone removed for offline-first use."""
    print(f"  [SERVICE] Vector store : Local Qdrant (path='{settings.qdrant_path}')")
    return QdrantVectorStore(embedder=embedder)


def _build_embedder():
    """HuggingFace only — Ollama removed."""
    print(f"  [SERVICE] Embedder     : huggingface (bge-small)")
    return EmbedderFactory.get("huggingface")


def _build_llm():
    """
    LLM_PROVIDER=groq   → GroqLLM   (default, cloud, needs GROQ_API_KEY)
    LLM_PROVIDER=ollama → OllamaLLM (local, needs ollama serve)
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        import generation.ollama_llm  # noqa: F401 — registers into factory
        print(f"  [SERVICE] LLM          : Ollama local (model='{settings.ollama_model}')")
        return LLMFactory.get("ollama")

    if provider == "groq" and not settings.groq_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set in .env"
        )

    print(f"  [SERVICE] LLM          : Groq cloud (model='{settings.groq_model}')")
    return LLMFactory.get("groq")


def _build_chunker():
    from ingestion.chunker import ChunkerFactory
    strategy = settings.chunker.lower().strip()
    print(f"  [SERVICE] Chunker      : {strategy}")
    return ChunkerFactory.get(strategy)


# ── Startup ───────────────────────────────────────────────────

async def startup() -> None:
    """Called once at FastAPI startup. Initialise all singletons."""
    global _embedder, _reranker, _vector_store, _bm25_store, _chain, _network_monitor

    data_dir = Path(settings.qdrant_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [SERVICE] Initialising singletons...")
    _embedder     = _build_embedder()
    _reranker     = Reranker()
    _vector_store = _build_vector_store(_embedder)
    _bm25_store   = BM25Store(path=str(data_dir / "bm25.pkl"))
    _chain        = _build_chain()

    # Start network monitor — polls connectivity every 30 seconds
    from services.network_monitor import NetworkMonitor
    _network_monitor = NetworkMonitor(
        check_url     = settings.network_check_url,
        poll_interval = 30,
    )
    _network_monitor.start()

    print("  [SERVICE] ✅ All singletons ready\n")


# ── Chain ─────────────────────────────────────────────────────

def _build_chain() -> RAGChain:
    llm = _build_llm()

    retriever = HybridRetriever(
        vector_store = _vector_store,
        embedder     = _embedder,
        top_k        = settings.top_k,
    )
    if _bm25_store and _bm25_store._chunks:
        retriever.index_chunks(_bm25_store._chunks)

    return RAGChain(
        llm            = llm,
        vector_store   = _vector_store,
        retriever      = retriever,
        reranker       = _reranker,
        use_reranker   = True,
        retrieve_top_k = settings.top_k,
        rerank_top_k   = 5,
        cite_sources   = True,
    )


def get_chain() -> RAGChain:
    """Return the shared RAGChain instance."""
    return _chain


def clear_chain_memory() -> None:
    """Clear conversation history on the shared chain."""
    with _lock:
        if _chain:
            _chain.reset_memory()


# ── Network status ────────────────────────────────────────────

def is_online() -> bool:
    """
    Returns True if network connectivity is currently available.
    Reads from the NetworkMonitor singleton started at startup.
    Falls back to True if monitor not yet initialised.
    """
    if _network_monitor is None:
        return True
    return _network_monitor.is_online


# ── Per-file deletion ─────────────────────────────────────────

def delete_file_from_stores(filename: str) -> dict:
    vectors_deleted = _vector_store.delete_by_source(filename)
    bm25_deleted    = _bm25_store.delete_by_source(filename)

    with _lock:
        if _chain and hasattr(_chain.retriever, "bm25"):
            _chain.retriever.bm25 = _bm25_store

    return {
        "vectors_deleted": vectors_deleted,
        "bm25_deleted"   : bm25_deleted,
    }


# ── Chunker accessor (used by ingest.py) ──────────────────────

def get_chunker():
    return _build_chunker()


# ── Singleton accessors ───────────────────────────────────────

def get_vector_store():
    return _vector_store

def get_bm25_store() -> BM25Store:
    return _bm25_store

def get_parent_store():
    return None   # inline parents, no separate store

def get_embedder():
    return _embedder


# ── Task registry ─────────────────────────────────────────────

def set_task(task_id, status, progress=0, message="", result=None):
    _tasks[task_id] = {
        "status"  : status,
        "progress": progress,
        "message" : message,
        "result"  : result or {},
    }

def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)