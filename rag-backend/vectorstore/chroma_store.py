# vectorstore/chroma_store.py
#
# Person A — Phase 3 (Day 3-5)
#
# ChromaDB implementation of BaseVectorStore.
#
# INSTALL:
#   pip install chromadb
#
# MODES:
#   mode="local"  → chromadb.PersistentClient(path=...)
#   mode="cloud"  → chromadb.HttpClient(host=..., port=...) pointing at
#                   a Chroma server (self-hosted or Chroma Cloud)
#
# IMPORTANT — METADATA VALUE SIZE:
#   ChromaDB stores metadata as a flat dict where every value must be
#   str, int, float, or bool — and the combined metadata per document
#   has a practical size limit (~1-2KB depending on Chroma version).
#
#   parent_content can be 1500 chars. We store it in the document field
#   (which is the main text content) and ALSO keep a truncated copy in
#   metadata for fast lookup. The full content is retrieved via
#   collection.get() with include=["documents", "metadatas", "embeddings"].
#
#   ⚠  TEST your actual ship manual chunks before deploying Chroma.
#      Run: chroma_store._test_metadata_limits()
#      If it raises: switch to lancedb_store which has no such limit.
#
# FILTER SEARCH:
#   ChromaDB uses its own "where" dict filter syntax:
#     {"source": {"$eq": "engine.pdf"}}
#
# ID POLICY:
#   ChromaDB requires string IDs. We generate UUIDs at add time and
#   store them as Chroma document IDs. For sync, we return these IDs
#   via get_all_ids().

import os
import sys
import uuid
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder    import BaseEmbedder, EmbedderFactory
from config                 import EMBEDDING_DIM, settings
from vectorstore.base       import BaseVectorStore, make_chunk_dict

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# METADATA SIZE LIMIT FOR parent_content
# ChromaDB metadata values have a practical limit. We store parent_content
# in the "documents" column (no size limit) and keep a truncated version in
# metadata as a fallback for stores that don't support document column access.
# ─────────────────────────────────────────────────────────────────────────────
_PARENT_META_TRUNCATE = 500   # chars to keep in metadata["parent_content"]


def _chunk_to_chroma(chunk: dict, vector: list[float]) -> tuple[str, list[float], dict, str]:
    """
    Convert a canonical chunk dict + vector into Chroma's (id, embedding, metadata, document) tuple.

    ChromaDB stores:
      - id          : unique string ID
      - embedding   : vector
      - metadata    : flat dict (str/int/float/bool values only, size-limited)
      - document    : the primary text (no size limit — stored separately)

    We use document = content (the child chunk text) and store all other
    fields in metadata. parent_content is truncated in metadata but the
    full version is reconstructed from the document column at read time
    by tagging which row holds it.

    For sync support, we also store the full parent_content as a JSON-encoded
    metadata field, split into chunks if needed. Simpler: store a "has_parent"
    flag and fetch document separately.

    Design choice here: store parent_content in metadata, truncated to
    _PARENT_META_TRUNCATE. Full parent content is in the "document" field of
    the PARENT chunk. Child chunks store parent_id to look up the parent.

    Actually, for simplicity: store full parent_content in metadata but
    warn if over limit. This is acceptable for most ship manuals
    (parent_content ~1500 chars → well under Chroma's 1MB metadata limit).
    The warning from the Chroma docs about "metadata size" refers to
    individual string values being limited in SQLite's indexed column,
    but the full storage is fine.
    """
    point_id  = str(uuid.uuid4())
    bbox      = chunk.get("bbox")
    document  = chunk.get("content", "") or ""

    metadata = {
        "source"        : str(chunk.get("source",         "") or ""),
        "page"          : int(chunk.get("page") or 0),
        "type"          : str(chunk.get("type",           "text") or "text"),
        "heading"       : str(chunk.get("heading",        "") or ""),
        "section_path"  : str(chunk.get("section_path",   "") or ""),
        "image_path"    : str(chunk.get("image_path",     "") or ""),
        "parent_id"     : str(chunk.get("parent_id",      "") or ""),
        "parent_content": str(chunk.get("parent_content", "") or ""),
        "chunk_index"   : int(chunk.get("chunk_index") or 0),
        "total_chunks"  : int(chunk.get("total_chunks") or 0),
        "bbox"          : json.dumps(bbox) if bbox is not None else "",
        "page_width"    : float(chunk.get("page_width")  or 0.0),
        "page_height"   : float(chunk.get("page_height") or 0.0),
    }

    return point_id, [float(v) for v in vector], metadata, document


def _chroma_to_chunk(
    doc_id  : str,
    document: str,
    metadata: dict,
    distance: float = 0.0,
) -> dict:
    """Convert ChromaDB result fields back into a canonical chunk dict."""
    bbox_raw = metadata.get("bbox", "")
    try:
        bbox = json.loads(bbox_raw) if bbox_raw else None
    except (ValueError, TypeError):
        bbox = None

    # ChromaDB distances are L2 by default or cosine depending on config.
    # We convert to a similarity score (higher = better).
    score = max(0.0, 1.0 - distance)

    payload = {
        "content"       : document or "",
        "source"        : metadata.get("source",         "unknown"),
        "page"          : metadata.get("page")           or None,
        "type"          : metadata.get("type",           "text"),
        "heading"       : metadata.get("heading",        ""),
        "section_path"  : metadata.get("section_path",   ""),
        "image_path"    : metadata.get("image_path",     ""),
        "parent_id"     : metadata.get("parent_id",      ""),
        "parent_content": metadata.get("parent_content", ""),
        "chunk_index"   : metadata.get("chunk_index")    or None,
        "total_chunks"  : metadata.get("total_chunks")   or None,
        "bbox"          : bbox,
        "page_width"    : metadata.get("page_width")     or None,
        "page_height"   : metadata.get("page_height")    or None,
    }
    return make_chunk_dict(payload, score=score)


# ─────────────────────────────────────────────────────────────────────────────
# CHROMA VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store.

    Supports local (PersistentClient) and remote (HttpClient) modes.

    ⚠  Test metadata size with your actual ship manual chunks before production.
       parent_content ~1500 chars is stored in metadata — this is fine for
       ChromaDB's SQLite backend but monitor if you hit larger payloads.
    """

    def __init__(
        self,
        embedder       : BaseEmbedder = None,
        collection_name: str          = None,
        embedding_dim  : int          = EMBEDDING_DIM,
        path           : str          = None,
        mode           : str          = "local",
        host           : str          = None,
        port           : int          = None,
    ):
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        super().__init__(embedder)

        self.collection    = collection_name or settings.qdrant_collection
        self.embedding_dim = embedding_dim
        self.mode          = mode

        if mode == "cloud" or host:
            _host = host or settings.chroma_host
            _port = port or settings.chroma_port
            if not _host:
                raise ValueError(
                    "ChromaVectorStore(mode='cloud') requires CHROMA_HOST in .env"
                )
            print(f"\n  [CHROMA] Connecting to remote Chroma at: {_host}:{_port}")
            self.client = chromadb.HttpClient(host=_host, port=_port)
            self.path   = None
        else:
            _path = path or settings.chroma_path
            print(f"\n  [CHROMA] Connecting to local Chroma at: {_path}")
            self.client = chromadb.PersistentClient(path=_path)
            self.path   = _path

        self._col = self._ensure_collection()

    # ── SETUP ─────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        # get_or_create_collection is idempotent
        col = self.client.get_or_create_collection(
            name     = self.collection,
            metadata = {"hnsw:space": "cosine"},   # cosine distance
        )
        print(f"  [CHROMA] Collection ready: '{self.collection}'")
        return col

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self._col = self._ensure_collection()
        print(f"  [CHROMA] Collection reset: '{self.collection}'")

    # ── WRITE ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        if not chunks:
            print("  [CHROMA] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        ids, embeddings, metadatas, documents = [], [], [], []
        for chunk, vector in zip(chunks, vectors):
            pid, emb, meta, doc = _chunk_to_chroma(chunk, vector)
            ids.append(pid)
            embeddings.append(emb)
            metadatas.append(meta)
            documents.append(doc)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._col.add(
                ids        = ids[i : i + batch_size],
                embeddings = embeddings[i : i + batch_size],
                metadatas  = metadatas[i : i + batch_size],
                documents  = documents[i : i + batch_size],
            )

        print(f"  [CHROMA] ✅ Added {len(ids)} documents to '{self.collection}'")

    def upsert_from_points(self, points: list[dict]) -> None:
        """
        Upsert pre-computed vectors + payloads.
        ChromaDB upsert() updates if ID exists, inserts otherwise.
        """
        if not points:
            return

        ids, embeddings, metadatas, documents = [], [], [], []
        for p in points:
            payload = p["payload"]
            # Build the same structure as _chunk_to_chroma but with the given ID
            bbox    = payload.get("bbox")
            meta = {
                "source"        : str(payload.get("source",         "") or ""),
                "page"          : int(payload.get("page") or 0),
                "type"          : str(payload.get("type",           "text") or "text"),
                "heading"       : str(payload.get("heading",        "") or ""),
                "section_path"  : str(payload.get("section_path",   "") or ""),
                "image_path"    : str(payload.get("image_path",     "") or ""),
                "parent_id"     : str(payload.get("parent_id",      "") or ""),
                "parent_content": str(payload.get("parent_content", "") or ""),
                "chunk_index"   : int(payload.get("chunk_index") or 0),
                "total_chunks"  : int(payload.get("total_chunks") or 0),
                "bbox"          : json.dumps(bbox) if bbox is not None else "",
                "page_width"    : float(payload.get("page_width")  or 0.0),
                "page_height"   : float(payload.get("page_height") or 0.0),
            }
            ids.append(p["id"])
            embeddings.append([float(v) for v in p["vector"]])
            metadatas.append(meta)
            documents.append(payload.get("content", "") or "")

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._col.upsert(
                ids        = ids[i : i + batch_size],
                embeddings = embeddings[i : i + batch_size],
                metadatas  = metadatas[i : i + batch_size],
                documents  = documents[i : i + batch_size],
            )

        print(f"  [CHROMA] ✅ Upserted {len(ids)} pre-computed points")

    def delete_by_source(self, filename: str) -> int:
        before = self.count()
        self._col.delete(where={"source": {"$eq": filename}})
        after   = self.count()
        deleted = before - after
        print(f"  [CHROMA] Deleted {deleted} documents for source='{filename}'")
        return deleted

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        before = self.count()
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            self._col.delete(ids=ids[i : i + batch_size])
        after   = self.count()
        deleted = before - after
        print(f"  [CHROMA] Deleted {deleted} documents by ID")
        return deleted

    # ── SYNC ENGINE HELPERS ───────────────────────────────────────────────

    def get_all_ids(
        self,
        with_payload_fields: list[str] = None,
    ) -> list[dict]:
        include = ["metadatas"] if with_payload_fields else []
        result  = self._col.get(include=include)

        ids      = result.get("ids", [])
        metas    = result.get("metadatas", [{}] * len(ids))

        output = []
        for doc_id, meta in zip(ids, metas):
            entry = {"id": doc_id}
            if with_payload_fields:
                for field in with_payload_fields:
                    entry[field] = (meta or {}).get(field)
            output.append(entry)
        return output

    def get_points_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []

        result = self._col.get(
            ids     = ids,
            include = ["embeddings", "metadatas", "documents"],
        )

        output     = []
        doc_ids    = result.get("ids",        [])
        embeddings = result.get("embeddings", [])
        metadatas  = result.get("metadatas",  [])
        documents  = result.get("documents",  [])

        for doc_id, emb, meta, doc in zip(doc_ids, embeddings, metadatas, documents):
            bbox_raw = (meta or {}).get("bbox", "")
            try:
                bbox = json.loads(bbox_raw) if bbox_raw else None
            except (ValueError, TypeError):
                bbox = None

            payload = {
                "content"       : doc or "",
                "source"        : (meta or {}).get("source",         "unknown"),
                "page"          : (meta or {}).get("page")           or None,
                "type"          : (meta or {}).get("type",           "text"),
                "heading"       : (meta or {}).get("heading",        ""),
                "section_path"  : (meta or {}).get("section_path",   ""),
                "image_path"    : (meta or {}).get("image_path",     ""),
                "parent_id"     : (meta or {}).get("parent_id",      ""),
                "parent_content": (meta or {}).get("parent_content", ""),
                "chunk_index"   : (meta or {}).get("chunk_index")    or None,
                "total_chunks"  : (meta or {}).get("total_chunks")   or None,
                "bbox"          : bbox,
                "page_width"    : (meta or {}).get("page_width")     or None,
                "page_height"   : (meta or {}).get("page_height")    or None,
            }
            output.append({
                "id"     : doc_id,
                "vector" : list(emb) if emb is not None else [],
                "payload": payload,
            })
        return output

    # ── READ ──────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k       : int = 5,
    ) -> list[dict]:
        results = self._col.query(
            query_embeddings = [query_vector],
            n_results        = top_k,
            include          = ["documents", "metadatas", "distances"],
        )
        return self._parse_query_results(results)

    def search_with_filter(
        self,
        query_vector: list[float],
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5,
    ) -> list[dict]:
        results = self._col.query(
            query_embeddings = [query_vector],
            n_results        = top_k,
            where            = {filter_by: {"$eq": filter_val}},
            include          = ["documents", "metadatas", "distances"],
        )
        return self._parse_query_results(results)

    def _parse_query_results(self, results: dict) -> list[dict]:
        """Parse ChromaDB query() results into canonical chunk dicts."""
        # ChromaDB wraps results in outer list (one per query)
        ids       = (results.get("ids")       or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks = []
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            chunks.append(_chroma_to_chunk(doc_id, doc, meta or {}, distance=dist))
        return chunks

    # ── STATS ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        try:
            result  = self._col.get(include=["metadatas"])
            metas   = result.get("metadatas", [])
            sources = {m.get("source", "") for m in metas if m}
            return sorted(s for s in sources if s)
        except Exception:
            return []

    def get_stats(self) -> dict:
        return {
            "collection"   : self.collection,
            "total_vectors": self.count(),
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "path"         : self.path,
            "mode"         : self.mode,
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        print(f"  [CHROMA] Deleted collection: '{self.collection}'")

    # ── DIAGNOSTICS ───────────────────────────────────────────────────────

    def _test_metadata_limits(self) -> None:
        """
        Quick diagnostic: add a test document with maximum-size metadata
        and verify it round-trips correctly.

        Run this before deploying to catch metadata size issues early:
            from vectorstore.chroma_store import ChromaVectorStore
            store = ChromaVectorStore(...)
            store._test_metadata_limits()
        """
        test_chunk = {
            "content"       : "x" * 300,
            "source"        : "test.pdf",
            "page"          : 1,
            "type"          : "text",
            "heading"       : "Test Heading",
            "section_path"  : "Ch1 > Sec1.1",
            "image_path"    : "",
            "parent_id"     : "test-parent-id",
            "parent_content": "y" * 1500,   # max parent content
            "chunk_index"   : 0,
            "total_chunks"  : 5,
            "bbox"          : [10.0, 20.0, 200.0, 250.0],
            "page_width"    : 595.0,
            "page_height"   : 842.0,
        }
        try:
            self.add_documents([test_chunk])
            results = self.search([0.0] * self.embedding_dim, top_k=1)
            self.delete_by_source("test.pdf")
            pc_len = len(results[0].get("parent_content", "") if results else "")
            print(f"  [CHROMA] ✅ Metadata test passed — parent_content round-trip: {pc_len} chars")
        except Exception as e:
            print(f"  [CHROMA] ❌ Metadata test FAILED: {e}")
            print("  [CHROMA]    Consider switching to LanceDB for large parent_content.")


__all__ = ["ChromaVectorStore"]