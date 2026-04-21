# vectorstore/base.py
#
# Person A — Phase 1 (Day 1-2)
#
# WHAT CHANGED:
#   BaseVectorStore promoted out of qdrant_store.py into its own file.
#   Abstract method set expanded from 4 to 10:
#     OLD: add_documents, search, delete_collection, get_stats
#     NEW: + search_with_filter, delete_by_source, count,
#            list_sources, reset, get_all_ids, upsert_from_points
#
#   get_all_ids() and upsert_from_points() are new additions required
#   by the vector-pull sync engine (Phase 5 / sync_service.py rewrite).
#   They allow the sync engine to copy vectors between stores without
#   re-embedding — the key insight that makes sync fast.
#
# CANONICAL CHUNK DICT SCHEMA
#   Every search() return value and every item in add_documents() MUST
#   contain these keys.  Missing keys must default to None/"" — never KeyError.
#
#   key             type            description
#   ─────────────── ─────────────── ──────────────────────────────────────────
#   content         str             chunk text (child chunk, ~300 tok)
#   score           float           retrieval score (cosine/BM25/RRF/rerank)
#   source          str             original PDF filename
#   page            int | None      1-based page number
#   type            str             "text" | "table" | "image"
#   heading         str             nearest heading in document
#   section_path    str             breadcrumb e.g. "Ch3 > Sec3.2 > Cooling"
#   image_path      str             relative path for image chunks
#   parent_id       str             UUID of parent chunk (hierarchical only)
#   parent_content  str             full parent passage (~1500 tok)
#   chunk_index     int | None      position within parent
#   total_chunks    int | None      total children of this parent
#   bbox            list | None     [x0, y0, x1, y1] in PDF coord space
#   page_width      float | None    PDF page width
#   page_height     float | None    PDF page height
#
# WHY SEPARATE FILE:
#   qdrant_store.py, lancedb_store.py, chroma_store.py, factory.py,
#   rag_service.py, and hybrid_retriever.py all import BaseVectorStore.
#   Putting it in qdrant_store.py causes a circular-like coupling where
#   every new vendor file imports from the Qdrant file — semantically wrong.

from abc import ABC, abstractmethod
from embeddings.embedder import BaseEmbedder, EmbedderFactory


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL CHUNK DICT KEYS
# Used by _payload_to_dict() in every concrete store implementation.
# Centralising the list here means a new field only needs one edit.
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_KEYS: list[str] = [
    "content",
    "score",
    "source",
    "page",
    "type",
    "heading",
    "section_path",
    "image_path",
    "parent_id",
    "parent_content",
    "chunk_index",
    "total_chunks",
    "bbox",
    "page_width",
    "page_height",
    "source_url",
]

# Default values when a payload field is absent (e.g. older indexed chunks).
CHUNK_DEFAULTS: dict = {
    "content"       : "",
    "score"         : 0.0,
    "source"        : "unknown",
    "page"          : None,
    "type"          : "text",
    "heading"       : "",
    "section_path"  : "",
    "image_path"    : "",
    "parent_id"     : "",
    "parent_content": "",
    "chunk_index"   : None,
    "total_chunks"  : None,
    "bbox"          : None,
    "page_width"    : None,
    "page_height"   : None,
    "source_url": "",
}


def make_chunk_dict(payload: dict, score: float = 0.0) -> dict:
    """
    Build a canonical chunk dict from a raw payload dict.

    Used by concrete store implementations in their _payload_to_dict() helpers
    so each vendor only needs one line instead of 15 get() calls.

    Args:
        payload: raw dict from the vendor (Qdrant payload, LanceDB row, etc.)
        score:   retrieval score already extracted from the vendor result

    Returns:
        dict with exactly the keys listed in CHUNK_KEYS, falling back to
        CHUNK_DEFAULTS for any missing field.
    """
    result = {}
    for key in CHUNK_KEYS:
        if key == "score":
            result[key] = round(float(score), 4)
        else:
            result[key] = payload.get(key, CHUNK_DEFAULTS[key])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BASE VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    """
    Abstract interface for all vector store backends.

    Concrete implementations:  QdrantVectorStore, LanceDBVectorStore,
                                ChromaVectorStore
    Factory:                    vectorstore.factory.get_vector_store()

    Every method that any router, service, or retriever calls on a store
    is declared here as abstract, so swapping vendors is a one-line env-var
    change and the rest of the codebase never notices.
    """

    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder   = embedder or EmbedderFactory.get("huggingface")
        self.collection = None   # set by subclass

    # ── WRITE ─────────────────────────────────────────────────────────────

    @abstractmethod
    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed chunks and upsert into the store.

        Args:
            chunks: list of canonical chunk dicts (see CHUNK_KEYS).
                    The 'content' field is embedded; all other fields
                    become the point payload / metadata.
        """

    @abstractmethod
    def upsert_from_points(self, points: list[dict]) -> None:
        """
        Upsert pre-computed vectors + payloads without re-embedding.

        Used by the sync engine to copy points from the cloud store to the
        local store without incurring embedding costs.

        Each item in `points` must have:
            id      : str         — unique point ID (UUID string)
            vector  : list[float] — pre-computed embedding vector
            payload : dict        — canonical chunk payload (no 'score' key)

        Args:
            points: list of dicts with 'id', 'vector', 'payload' keys.
        """

    @abstractmethod
    def delete_by_source(self, filename: str) -> int:
        """
        Delete all vectors whose 'source' payload field equals filename.

        Returns:
            Number of vectors deleted.
        """

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> int:
        """
        Delete vectors by their point IDs.

        Used by the sync engine to remove locally-held points that have been
        deleted from the cloud store.

        Returns:
            Number of vectors deleted.
        """

    @abstractmethod
    def reset(self) -> None:
        """Wipe the collection and recreate it empty."""

    # ── READ ──────────────────────────────────────────────────────────────

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k       : int = 5,
    ) -> list[dict]:
        """
        Nearest-neighbour vector search.

        Returns:
            List of canonical chunk dicts ordered by descending score.
        """

    @abstractmethod
    def search_with_filter(
        self,
        query_vector: list[float],
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5,
    ) -> list[dict]:
        """
        Vector search filtered to points where payload[filter_by] == filter_val.

        Returns:
            List of canonical chunk dicts ordered by descending score.
        """

    @abstractmethod
    def get_all_ids(
        self,
        with_payload_fields: list[str] = None,
    ) -> list[dict]:
        """
        Return all point IDs (and optionally a subset of payload fields).

        Used by the sync engine to diff local vs cloud state without
        fetching full vectors.

        Args:
            with_payload_fields: list of payload keys to include in each
                returned dict, e.g. ["source", "sha256"]. If None, only
                the 'id' field is returned.

        Returns:
            List of dicts with 'id' key and any requested payload fields.
            Example: [{"id": "abc-123", "source": "engine.pdf"}, ...]
        """

    @abstractmethod
    def get_points_by_ids(
        self,
        ids: list[str],
    ) -> list[dict]:
        """
        Fetch full point data (vector + payload) for a list of IDs.

        Used by the sync engine to pull points from the cloud store for
        upsert into the local store.

        Returns:
            List of dicts with 'id', 'vector', 'payload' keys — same
            shape as the input to upsert_from_points().
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors currently in the collection."""

    @abstractmethod
    def list_sources(self) -> list[str]:
        """Return sorted list of unique 'source' values in the collection."""

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return a summary dict for the /kb/stats endpoint.

        Required keys: collection, total_vectors, dimensions, distance.
        Additional vendor-specific keys are welcome.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the entire collection (used by reset-KB endpoints)."""


__all__ = [
    "BaseVectorStore",
    "CHUNK_KEYS",
    "CHUNK_DEFAULTS",
    "make_chunk_dict",
]