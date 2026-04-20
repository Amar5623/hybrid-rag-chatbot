# vectorstore/__init__.py
#
# Phase 1 — expose BaseVectorStore from the new canonical home (base.py)
# and re-export QdrantVectorStore for backward compat.
#
# Old code: from vectorstore.qdrant_store import BaseVectorStore, QdrantVectorStore
# New code: from vectorstore import BaseVectorStore           (cleaner)
#           from vectorstore.factory import get_vector_store   (for switching)

from vectorstore.base        import BaseVectorStore, CHUNK_KEYS, CHUNK_DEFAULTS, make_chunk_dict
from vectorstore.qdrant_store import QdrantVectorStore
from vectorstore.factory      import get_vector_store

__all__ = [
    "BaseVectorStore",
    "CHUNK_KEYS",
    "CHUNK_DEFAULTS",
    "make_chunk_dict",
    "QdrantVectorStore",
    "get_vector_store",
]