# vectorstore/factory.py
#
# Person A — Phase 4 (Day 5-6)
#
# Returns the correct BaseVectorStore subclass based on vendor + mode.
#
# Usage:
#   from vectorstore.factory import get_vector_store
#
#   local_store = get_vector_store(vendor="qdrant",   mode="local", embedder=emb)
#   cloud_store = get_vector_store(vendor="qdrant",   mode="cloud", embedder=emb)
#   local_lance = get_vector_store(vendor="lancedb",  mode="local", embedder=emb)
#   local_chroma= get_vector_store(vendor="chroma",   mode="local", embedder=emb)
#
# Env var shortcut:
#   VECTOR_STORE_VENDOR=lancedb  → switches the entire backend
#   VECTOR_STORE_VENDOR=chroma   → same
#   VECTOR_STORE_VENDOR=qdrant   → default
#
# The factory is the only place that imports concrete store classes.
# All other code imports BaseVectorStore from vectorstore.base.

from __future__ import annotations

from vectorstore.base import BaseVectorStore


def get_vector_store(
    vendor  : str = None,
    mode    : str = "local",
    embedder: object = None,
    **kwargs,
) -> BaseVectorStore:
    """
    Instantiate and return the configured vector store.

    Args:
        vendor:   "qdrant" | "lancedb" | "chroma".
                  Defaults to settings.vector_store_vendor.
        mode:     "local" | "cloud".
        embedder: BaseEmbedder instance. If None, the store creates its own.
        **kwargs: Passed through to the store constructor for overrides
                  (e.g. path=, collection_name=, uri=).

    Returns:
        Configured BaseVectorStore instance.

    Raises:
        ValueError: if vendor is unrecognised.
        ImportError: if the vendor's package is not installed.
    """
    from config import settings

    _vendor = (vendor or settings.vector_store_vendor).lower().strip()

    if _vendor == "qdrant":
        from vectorstore.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(embedder=embedder, mode=mode, **kwargs)

    elif _vendor == "lancedb":
        from vectorstore.lancedb_store import LanceDBVectorStore
        return LanceDBVectorStore(embedder=embedder, mode=mode, **kwargs)

    elif _vendor in ("chroma", "chromadb"):
        from vectorstore.chroma_store import ChromaVectorStore
        return ChromaVectorStore(embedder=embedder, mode=mode, **kwargs)

    else:
        raise ValueError(
            f"Unknown vector store vendor: '{_vendor}'. "
            f"Choose from: qdrant, lancedb, chroma."
        )


__all__ = ["get_vector_store"]