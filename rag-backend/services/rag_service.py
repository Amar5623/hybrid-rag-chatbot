# services/rag_service.py
#
# Person A — Phase 4 (Day 5-6)
#
# CHANGES vs previous version:
#
#   1. FACTORY WIRING
#      _build_vector_store() now calls vectorstore.factory.get_vector_store()
#      instead of directly instantiating QdrantVectorStore.
#      One env var change (VECTOR_STORE_VENDOR=lancedb) switches the backend.
#
#   2. HYBRID MANAGER — two store singletons
#      _local_store  : BaseVectorStore — always initialised (local mode)
#      _cloud_store  : BaseVectorStore — initialised only if cloud creds set
#
#      get_vector_store() returns the cloud store when online + configured,
#      otherwise falls back to the local store.  The rest of the codebase
#      (routers, retrievers) calls get_vector_store() and gets the right one
#      transparently.
#
#   3. DYNAMIC STORE IN RETRIEVER
#      HybridRetriever.retrieve() now accepts an optional `store` parameter.
#      rag_chain._retrieve() passes rag_service.get_vector_store() each call.
#      This means the retriever always uses whichever store is current
#      (online=cloud, offline=local) without rebuilding the chain.
#
#      NOTE: chains/rag_chain.py also needs a one-line change — see the
#            "REQUIRED CHAIN CHANGE" comment below.  The change is minimal:
#            pass `store=rag_service.get_vector_store()` to retriever.retrieve().
#
#   4. get_reranker() ACCESSOR
#      Added alongside existing get_vector_store(), get_bm25_store() etc.
#      Person B's chat.py offline reranker change calls this.
#
#   5. NETWORK MONITOR CONFIG
#      poll_interval and timeout now read from settings (Person B's Day 1 change).
#
# REQUIRED CHAIN CHANGE (in chains/rag_chain.py — minimal edit):
#   In RAGChain._retrieve(), change:
#       retrieval = self.retriever.retrieve(question, ...)
#   to:
#       import services.rag_service as rag_service
#       retrieval = self.retriever.retrieve(
#           question,
#           ...,
#           store=rag_service.get_vector_store(),   # ← add this
#       )
#   See retrieval/hybrid_retriever.py for the `store` parameter addition.

import threading
from pathlib import Path

from config import settings
from embeddings.embedder        import EmbedderFactory
from generation.groq_llm        import LLMFactory
from retrieval.bm25_store       import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from vectorstore.base           import BaseVectorStore
from vectorstore.factory        import get_vector_store as _factory_get_store
from chains.rag_chain           import RAGChain


# ── Singletons ────────────────────────────────────────────────────────────────
_embedder        : object = None
_reranker        : object = None
_local_store     : object = None   # always available (local disk)
_cloud_store     : object = None   # None unless cloud creds are set
_bm25_store      : object = None
_chain           : object = None
_network_monitor : object = None

_lock = threading.Lock()

# ── Background task registry ──────────────────────────────────────────────────
_tasks: dict = {}


# ── HYBRID STORE ACCESSOR ─────────────────────────────────────────────────────

def get_vector_store() -> BaseVectorStore:
    """
    Return the active vector store.

    Online + cloud configured → cloud store (authoritative)
    Otherwise                 → local store (offline-safe fallback)

    Called on every request so the chain always uses the correct store
    without needing to restart.
    """
    if _cloud_store is not None and is_online():
        return _cloud_store
    return _local_store


# ── FACTORIES ─────────────────────────────────────────────────────────────────

def _build_local_store(embedder) -> BaseVectorStore:
    vendor = settings.vector_store_vendor
    print(f"  [SERVICE] Local vector store : {vendor} (local mode)")
    return _factory_get_store(vendor=vendor, mode="local", embedder=embedder)


def _build_cloud_store(embedder) -> BaseVectorStore | None:
    """
    Build the cloud store for the configured vendor.
    Returns None if no cloud credentials are set — safe fallback.
    """
    vendor = settings.vector_store_vendor

    # Qdrant cloud requires URL
    if vendor == "qdrant" and not settings.qdrant_cloud_url:
        print("  [SERVICE] Cloud store skipped — QDRANT_CLOUD_URL not set")
        return None

    # LanceDB cloud requires cloud URI
    if vendor == "lancedb" and not settings.lancedb_cloud_uri:
        print("  [SERVICE] Cloud store skipped — LANCEDB_CLOUD_URI not set")
        return None

    # Chroma cloud requires host
    if vendor in ("chroma", "chromadb") and not settings.chroma_host:
        print("  [SERVICE] Cloud store skipped — CHROMA_HOST not set")
        return None

    print(f"  [SERVICE] Cloud vector store : {vendor} (cloud mode)")
    try:
        return _factory_get_store(vendor=vendor, mode="cloud", embedder=embedder)
    except Exception as e:
        print(f"  [SERVICE] ⚠  Cloud store init failed: {e} — falling back to local only")
        return None


def _build_embedder():
    print("  [SERVICE] Embedder     : huggingface (bge-small)")
    return EmbedderFactory.get("huggingface")


def _build_llm():
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        import generation.ollama_llm  # noqa: F401 — registers into factory
        print(f"  [SERVICE] LLM          : Ollama local (model='{settings.ollama_model}')")
        return LLMFactory.get("ollama")

    if provider == "groq" and not settings.groq_api_key:
        raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set in .env")

    print(f"  [SERVICE] LLM          : Groq cloud (model='{settings.groq_model}')")
    return LLMFactory.get("groq")


def _build_chunker():
    from ingestion.chunker import ChunkerFactory
    strategy = settings.chunker.lower().strip()
    print(f"  [SERVICE] Chunker      : {strategy}")
    return ChunkerFactory.get(strategy)


# ── STARTUP ───────────────────────────────────────────────────────────────────

async def startup() -> None:
    """Called once at FastAPI startup. Initialises all singletons."""
    global _embedder, _reranker, _local_store, _cloud_store, \
           _bm25_store, _chain, _network_monitor

    data_dir = Path(settings.qdrant_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [SERVICE] Initialising singletons...")

    _embedder    = _build_embedder()
    _reranker    = Reranker(model_name=settings.reranker_model)

    # ── Vector stores (hybrid manager) ────────────────────────────────────
    _local_store = _build_local_store(_embedder)
    _cloud_store = _build_cloud_store(_embedder)   # None if no creds

    _bm25_store  = BM25Store(path=str(data_dir / "bm25.pkl"))
    _chain       = _build_chain()

    # ── Network monitor ───────────────────────────────────────────────────
    from services.network_monitor import NetworkMonitor
    _network_monitor = NetworkMonitor(
        check_url     = settings.network_check_url,
        poll_interval = settings.network_poll_interval,
        timeout       = settings.network_check_timeout,
    )
    _network_monitor.start()

    cloud_status = "✅ connected" if _cloud_store else "⬜ not configured"
    print(f"  [SERVICE] Local store  : ✅ ready")
    print(f"  [SERVICE] Cloud store  : {cloud_status}")
    print("  [SERVICE] ✅ All singletons ready\n")


# ── CHAIN ─────────────────────────────────────────────────────────────────────

def _build_chain() -> RAGChain:
    llm = _build_llm()

    # The retriever is initialised with the local store.
    # At retrieval time, get_vector_store() is passed in dynamically
    # (via the `store` param in HybridRetriever.retrieve()) so online/offline
    # switching is transparent without rebuilding the chain.
    retriever = HybridRetriever(
        vector_store = _local_store,
        embedder     = _embedder,
        top_k        = settings.top_k,
    )
    if _bm25_store and _bm25_store._chunks:
        retriever.index_chunks(_bm25_store._chunks)

    return RAGChain(
        llm            = llm,
        vector_store   = _local_store,
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


def rebuild_chain() -> None:
    """
    Rebuild the chain after a store switch or major config change.
    Thread-safe — holds the lock during rebuild.
    """
    global _chain
    with _lock:
        _chain = _build_chain()


def clear_chain_memory() -> None:
    """Clear conversation history on the shared chain."""
    with _lock:
        if _chain:
            _chain.reset_memory()


# ── NETWORK STATUS ────────────────────────────────────────────────────────────

def is_online() -> bool:
    """
    Returns True if network connectivity is currently available.
    Falls back to True if monitor not yet initialised (startup race).
    """
    if _network_monitor is None:
        return True
    return _network_monitor.is_online


# ── PER-FILE DELETION ─────────────────────────────────────────────────────────

def delete_file_from_stores(filename: str) -> dict:
    vectors_deleted = _local_store.delete_by_source(filename)
    bm25_deleted    = _bm25_store.delete_by_source(filename)

    with _lock:
        if _chain and hasattr(_chain.retriever, "bm25"):
            _chain.retriever.bm25 = _bm25_store

    return {
        "vectors_deleted": vectors_deleted,
        "bm25_deleted"   : bm25_deleted,
    }


# ── BM25 REBUILD (called by sync engine after vector sync) ────────────────────

def rebuild_bm25_async() -> None:
    """
    Rebuild the BM25 index from the current local vector store contents.

    Called asynchronously by the sync engine after a vector pull so the
    BM25 index stays in sync with the vector store without blocking reconnect.

    Scrolls the local store for all chunk contents, rebuilds the BM25 store,
    and hot-swaps it on the running retriever.
    """
    import threading

    def _rebuild():
        global _bm25_store
        try:
            print("  [BM25] Starting async BM25 rebuild from local store...")
            # Scroll local store for content field only
            all_ids   = _local_store.get_all_ids(with_payload_fields=["source"])
            if not all_ids:
                print("  [BM25] Local store empty — skipping BM25 rebuild")
                return

            # We need full content for BM25 — fetch in batches
            all_points = _local_store.get_points_by_ids([p["id"] for p in all_ids])
            chunks = [
                {"content": pt["payload"].get("content", ""), "source": pt["payload"].get("source", "")}
                for pt in all_points
                if pt["payload"].get("content")
            ]

            if not chunks:
                print("  [BM25] No content found — skipping BM25 rebuild")
                return

            from pathlib import Path
            data_dir  = Path(settings.qdrant_path).parent
            new_bm25  = BM25Store(path=str(data_dir / "bm25.pkl"))
            new_bm25.build(chunks)

            # Hot-swap
            with _lock:
                _bm25_store = new_bm25
                if _chain and hasattr(_chain.retriever, "bm25"):
                    _chain.retriever.bm25 = new_bm25

            print(f"  [BM25] ✅ Rebuilt BM25 index with {len(chunks)} chunks")

        except Exception as e:
            print(f"  [BM25] ⚠  BM25 rebuild failed: {e}")

    threading.Thread(target=_rebuild, daemon=True).start()


# ── CHUNKER ACCESSOR ──────────────────────────────────────────────────────────

def get_chunker():
    return _build_chunker()


# ── SINGLETON ACCESSORS ───────────────────────────────────────────────────────

def get_local_store() -> BaseVectorStore:
    return _local_store

def get_cloud_store() -> BaseVectorStore | None:
    return _cloud_store

def get_bm25_store() -> BM25Store:
    return _bm25_store

def get_reranker() -> Reranker:
    """Return the shared Reranker singleton. Used by Person B's offline chat path."""
    return _reranker

def get_parent_store():
    return None   # inline parents, no separate store

def get_embedder():
    return _embedder


# ── TASK REGISTRY ─────────────────────────────────────────────────────────────

def set_task(task_id, status, progress=0, message="", result=None):
    _tasks[task_id] = {
        "status"  : status,
        "progress": progress,
        "message" : message,
        "result"  : result or {},
    }

def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)