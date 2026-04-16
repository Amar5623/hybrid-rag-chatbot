# vectorstore/qdrant_store.py
#
# CHANGES vs previous:
#   - _payload_to_dict() now also returns parent_content from payload
#     (stored inline by HierarchicalChunker — no SQLite needed)
#   - delete_by_source(filename) added — deletes all vectors for a file
#   - Everything else unchanged

import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue
)
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config              import QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM


# ─────────────────────────────────────────────────────────
# BASE VECTOR STORE
# ─────────────────────────────────────────────────────────

class BaseVectorStore:
    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder   = embedder or EmbedderFactory.get("huggingface")
        self.collection = None

    def add_documents(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def get_stats(self) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────
# QDRANT VECTOR STORE
# ─────────────────────────────────────────────────────────

class QdrantVectorStore(BaseVectorStore):
    """
    Local Qdrant vector store — persists to disk.

    Each chunk's full dict becomes the Qdrant payload, including
    parent_content (stored inline by HierarchicalChunker).
    """

    def __init__(
        self,
        embedder        : BaseEmbedder = None,
        collection_name : str          = QDRANT_COLLECTION,
        embedding_dim   : int          = EMBEDDING_DIM,
        path            : str          = QDRANT_PATH
    ):
        super().__init__(embedder)
        self.collection    = collection_name
        self.embedding_dim = embedding_dim
        self.path          = path

        print(f"\n  [QDRANT] Connecting to local DB at: {path}")
        self.client = QdrantClient(path=path)
        self._ensure_collection()

    # ── SETUP ─────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config  = VectorParams(
                    size     = self.embedding_dim,
                    distance = Distance.COSINE
                )
            )
            print(f"  [QDRANT] Created new collection: '{self.collection}'")
        else:
            print(f"  [QDRANT] Using existing collection: '{self.collection}'")

    def reset(self) -> None:
        """Wipe and recreate the collection."""
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name = self.collection,
            vectors_config  = VectorParams(
                size     = self.embedding_dim,
                distance = Distance.COSINE
            )
        )
        print(f"  [QDRANT] Collection reset: '{self.collection}'")

    # Keep old name as alias
    def reset_collection(self) -> None:
        self.reset()

    # ── WRITE ─────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        if not chunks:
            print("  [QDRANT] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {k: v for k, v in chunk.items()}
            points.append(
                PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = vector,
                    payload = payload
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name = self.collection,
                points          = points[i : i + batch_size]
            )

        print(f"  [QDRANT] ✅ Added {len(points)} vectors to '{self.collection}'")

    # ── DELETE BY SOURCE ──────────────────────────────────

    def delete_by_source(self, filename: str) -> int:
        """
        Delete all vectors whose payload 'source' field matches filename.
        Returns the number of vectors deleted.

        Uses Qdrant's filtered delete — efficient, no scroll needed.
        """
        # Count before deletion so we can report how many were removed
        before = self.count()

        self.client.delete(
            collection_name = self.collection,
            points_selector = Filter(
                must=[FieldCondition(
                    key   = "source",
                    match = MatchValue(value=filename)
                )]
            ),
        )

        after   = self.count()
        deleted = before - after
        print(f"  [QDRANT] Deleted {deleted} vectors for source='{filename}'")
        return deleted

    # ── PAYLOAD HELPER ────────────────────────────────────

    @staticmethod
    def _payload_to_dict(r) -> dict:
        """
        Convert a Qdrant search result point into a clean dict.
        Now includes parent_content for inline parent expansion.
        """
        p = r.payload
        return {
            "content"       : p.get("content",        ""),
            "score"         : round(r.score, 4),
            "source"        : p.get("source",         "unknown"),
            "page"          : p.get("page",           None),
            "type"          : p.get("type",           "text"),
            "heading"       : p.get("heading",        ""),
            "section_path"  : p.get("section_path",   ""),
            "image_path"    : p.get("image_path",     ""),
            "parent_id"     : p.get("parent_id",      ""),
            "parent_content": p.get("parent_content", ""),   # ← NEW
            "chunk_index"   : p.get("chunk_index",    None),
            "total_chunks"  : p.get("total_chunks",   None),
        }

    # ── READ ──────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k       : int = 5,
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            limit           = top_k,
            with_payload    = True
        ).points
        return [self._payload_to_dict(r) for r in results]

    def search_with_filter(
        self,
        query_vector: list[float],
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            query_filter    = Filter(
                must=[FieldCondition(
                    key   = filter_by,
                    match = MatchValue(value=filter_val)
                )]
            ),
            limit        = top_k,
            with_payload = True
        ).points
        return [self._payload_to_dict(r) for r in results]

    # ── STATS ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        info  = self.client.get_collection(self.collection)
        total = info.points_count or info.vectors_count or 0
        return {
            "collection"   : self.collection,
            "total_vectors": total,
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "storage_path" : self.path
        }

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        try:
            result = self.client.scroll(
                collection_name = self.collection,
                limit           = 10_000,
                with_payload    = ["source"],
                with_vectors    = False,
            )
            sources = {pt.payload.get("source", "") for pt in result[0]}
            return sorted(s for s in sources if s)
        except Exception:
            return []

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        print(f"  [QDRANT] Deleted collection: '{self.collection}'")


__all__ = ["BaseVectorStore", "QdrantVectorStore"]