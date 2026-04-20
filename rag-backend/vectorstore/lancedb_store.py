# vectorstore/lancedb_store.py
#
# Person A — Phase 3 (Day 3-5)
#
# LanceDB implementation of BaseVectorStore.
#
# WHY LANCEDB:
#   - Single Python package (no server process needed)
#   - Works for both local path and cloud URI — same client API
#   - Native vector search without a separate extension
#   - Arrow-based columnar storage = fast bulk scans for BM25 rebuild
#
# INSTALL:
#   pip install lancedb
#
# LOCAL vs CLOUD:
#   mode="local"  → lancedb.connect(uri)  where uri is a local path
#   mode="cloud"  → lancedb.connect(uri)  where uri is an s3:// or lancedb:// URI
#   The connect() call is identical — mode is just a label for logging.
#
# DATA MODEL:
#   One LanceDB table per collection. Each row stores:
#     id         : str    — UUID string (LanceDB rowid is separate; we store ours)
#     vector     : list   — fixed-size float32 array (embedding_dim)
#     + all canonical chunk payload fields (see vectorstore.base.CHUNK_KEYS)
#
# PAYLOAD SIZE NOTE:
#   parent_content can be ~1500 chars. LanceDB stores it as a plain string
#   column — no size limit issues unlike ChromaDB.
#
# FILTER SEARCH:
#   LanceDB supports SQL-style WHERE clauses on metadata columns.
#   search_with_filter() uses .where(f"{filter_by} = '{filter_val}'").

import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa

from embeddings.embedder    import BaseEmbedder, EmbedderFactory
from config                 import EMBEDDING_DIM, settings
from vectorstore.base       import BaseVectorStore, CHUNK_KEYS, make_chunk_dict

try:
    import lancedb
    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_schema(embedding_dim: int) -> pa.Schema:
    """
    Build the PyArrow schema for a LanceDB table.

    vector field must be declared as fixed-size list so LanceDB can
    build an ANN index over it.
    """
    return pa.schema([
        pa.field("id",             pa.utf8()),
        pa.field("vector",         pa.list_(pa.float32(), embedding_dim)),
        # ── Canonical chunk payload fields ─────────────────────────────────
        pa.field("content",        pa.utf8()),
        pa.field("source",         pa.utf8()),
        pa.field("page",           pa.int32()),
        pa.field("type",           pa.utf8()),
        pa.field("heading",        pa.utf8()),
        pa.field("section_path",   pa.utf8()),
        pa.field("image_path",     pa.utf8()),
        pa.field("parent_id",      pa.utf8()),
        pa.field("parent_content", pa.utf8()),
        pa.field("chunk_index",    pa.int32()),
        pa.field("total_chunks",   pa.int32()),
        # bbox stored as a JSON string (list[float] → serialised)
        pa.field("bbox",           pa.utf8()),
        pa.field("page_width",     pa.float32()),
        pa.field("page_height",    pa.float32()),
    ])


def _chunk_to_row(chunk: dict, vector: list[float]) -> dict:
    """Convert a canonical chunk dict + vector into a LanceDB row dict."""
    import json
    bbox = chunk.get("bbox")
    return {
        "id"            : str(uuid.uuid4()),
        "vector"        : [float(v) for v in vector],
        "content"       : chunk.get("content",        "") or "",
        "source"        : chunk.get("source",         "") or "",
        "page"          : int(chunk.get("page") or 0),
        "type"          : chunk.get("type",           "text") or "text",
        "heading"       : chunk.get("heading",        "") or "",
        "section_path"  : chunk.get("section_path",   "") or "",
        "image_path"    : chunk.get("image_path",     "") or "",
        "parent_id"     : chunk.get("parent_id",      "") or "",
        "parent_content": chunk.get("parent_content", "") or "",
        "chunk_index"   : int(chunk.get("chunk_index") or 0),
        "total_chunks"  : int(chunk.get("total_chunks") or 0),
        "bbox"          : json.dumps(bbox) if bbox is not None else "",
        "page_width"    : float(chunk.get("page_width")  or 0.0),
        "page_height"   : float(chunk.get("page_height") or 0.0),
    }


def _row_to_chunk(row: dict, score: float = 0.0) -> dict:
    """Convert a LanceDB result row back to a canonical chunk dict."""
    import json
    bbox_raw = row.get("bbox", "")
    try:
        bbox = json.loads(bbox_raw) if bbox_raw else None
    except (ValueError, TypeError):
        bbox = None

    payload = {
        "content"       : row.get("content",        ""),
        "source"        : row.get("source",         "unknown"),
        "page"          : row.get("page")           or None,
        "type"          : row.get("type",           "text"),
        "heading"       : row.get("heading",        ""),
        "section_path"  : row.get("section_path",   ""),
        "image_path"    : row.get("image_path",     ""),
        "parent_id"     : row.get("parent_id",      ""),
        "parent_content": row.get("parent_content", ""),
        "chunk_index"   : row.get("chunk_index")    or None,
        "total_chunks"  : row.get("total_chunks")   or None,
        "bbox"          : bbox,
        "page_width"    : row.get("page_width")     or None,
        "page_height"   : row.get("page_height")    or None,
    }
    return make_chunk_dict(payload, score=score)


# ─────────────────────────────────────────────────────────────────────────────
# LANCEDB VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class LanceDBVectorStore(BaseVectorStore):
    """
    LanceDB vector store.

    Supports local (file-system) and cloud (s3:// or lancedb://) URIs.
    The connect() API is identical for both; only the URI differs.
    """

    def __init__(
        self,
        embedder       : BaseEmbedder = None,
        collection_name: str          = None,
        embedding_dim  : int          = EMBEDDING_DIM,
        uri            : str          = None,
        mode           : str          = "local",
    ):
        if not _LANCEDB_AVAILABLE:
            raise ImportError(
                "lancedb is not installed. Run: pip install lancedb"
            )

        super().__init__(embedder)

        self.collection    = collection_name or settings.qdrant_collection
        self.embedding_dim = embedding_dim
        self.mode          = mode

        _uri = uri or (
            settings.lancedb_cloud_uri if mode == "cloud"
            else settings.lancedb_uri
        )
        if not _uri:
            raise ValueError(
                "LanceDBVectorStore requires a URI. "
                "Set LANCEDB_URI (local) or LANCEDB_CLOUD_URI (cloud) in .env"
            )

        print(f"\n  [LANCEDB] Connecting ({mode}) at: {_uri}")
        self.db  = lancedb.connect(_uri)
        self.uri = _uri
        self._table = self._ensure_table()

    # ── SETUP ─────────────────────────────────────────────────────────────

    def _ensure_table(self):
        existing = self.db.table_names()
        if self.collection not in existing:
            schema = _build_schema(self.embedding_dim)
            table  = self.db.create_table(self.collection, schema=schema)
            print(f"  [LANCEDB] Created table: '{self.collection}'")
        else:
            table = self.db.open_table(self.collection)
            print(f"  [LANCEDB] Opened existing table: '{self.collection}'")
        return table

    def reset(self) -> None:
        """Drop and recreate the table."""
        if self.collection in self.db.table_names():
            self.db.drop_table(self.collection)
        schema = _build_schema(self.embedding_dim)
        self._table = self.db.create_table(self.collection, schema=schema)
        print(f"  [LANCEDB] Table reset: '{self.collection}'")

    # ── WRITE ─────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        if not chunks:
            print("  [LANCEDB] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        rows = [_chunk_to_row(chunk, vec) for chunk, vec in zip(chunks, vectors)]
        self._table.add(rows)
        print(f"  [LANCEDB] ✅ Added {len(rows)} rows to '{self.collection}'")

    def upsert_from_points(self, points: list[dict]) -> None:
        """
        Upsert pre-computed vectors + payloads.
        LanceDB's add() is append; we delete existing IDs first.
        """
        if not points:
            return

        ids = [p["id"] for p in points]
        self._delete_by_id_list(ids)  # remove stale versions if re-syncing

        import json
        rows = []
        for p in points:
            payload = p["payload"]
            bbox    = payload.get("bbox")
            rows.append({
                "id"            : p["id"],
                "vector"        : [float(v) for v in p["vector"]],
                "content"       : payload.get("content",        "") or "",
                "source"        : payload.get("source",         "") or "",
                "page"          : int(payload.get("page") or 0),
                "type"          : payload.get("type",           "text") or "text",
                "heading"       : payload.get("heading",        "") or "",
                "section_path"  : payload.get("section_path",   "") or "",
                "image_path"    : payload.get("image_path",     "") or "",
                "parent_id"     : payload.get("parent_id",      "") or "",
                "parent_content": payload.get("parent_content", "") or "",
                "chunk_index"   : int(payload.get("chunk_index") or 0),
                "total_chunks"  : int(payload.get("total_chunks") or 0),
                "bbox"          : json.dumps(bbox) if bbox is not None else "",
                "page_width"    : float(payload.get("page_width")  or 0.0),
                "page_height"   : float(payload.get("page_height") or 0.0),
            })

        self._table.add(rows)
        print(f"  [LANCEDB] ✅ Upserted {len(rows)} pre-computed points")

    def delete_by_source(self, filename: str) -> int:
        before = self.count()
        # LanceDB DELETE uses SQL WHERE syntax
        self._table.delete(f"source = '{filename}'")
        after   = self.count()
        deleted = before - after
        print(f"  [LANCEDB] Deleted {deleted} rows for source='{filename}'")
        return deleted

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        n = self._delete_by_id_list(ids)
        print(f"  [LANCEDB] Deleted {n} rows by ID")
        return n

    def _delete_by_id_list(self, ids: list[str]) -> int:
        """Delete rows whose 'id' column is in the given list."""
        if not ids:
            return 0
        before = self.count()
        # LanceDB supports IN list via comma-joined quoted strings
        id_list = ", ".join(f"'{i}'" for i in ids)
        self._table.delete(f"id IN ({id_list})")
        after = self.count()
        return before - after

    # ── SYNC ENGINE HELPERS ───────────────────────────────────────────────

    def get_all_ids(
        self,
        with_payload_fields: list[str] = None,
    ) -> list[dict]:
        cols = ["id"] + (with_payload_fields or [])
        # to_pandas() is efficient for column projection on LanceDB Arrow tables
        df     = self._table.to_pandas(columns=cols)
        result = df.to_dict(orient="records")
        # Ensure id is always str
        for row in result:
            row["id"] = str(row["id"])
        return result

    def get_points_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []

        import json
        id_list = ", ".join(f"'{i}'" for i in ids)
        df      = self._table.to_pandas(filter=f"id IN ({id_list})")

        result = []
        for _, row in df.iterrows():
            payload = {
                "content"       : row.get("content",        ""),
                "source"        : row.get("source",         "unknown"),
                "page"          : row.get("page")           or None,
                "type"          : row.get("type",           "text"),
                "heading"       : row.get("heading",        ""),
                "section_path"  : row.get("section_path",   ""),
                "image_path"    : row.get("image_path",     ""),
                "parent_id"     : row.get("parent_id",      ""),
                "parent_content": row.get("parent_content", ""),
                "chunk_index"   : row.get("chunk_index")    or None,
                "total_chunks"  : row.get("total_chunks")   or None,
                "bbox"          : json.loads(row["bbox"]) if row.get("bbox") else None,
                "page_width"    : row.get("page_width")     or None,
                "page_height"   : row.get("page_height")    or None,
            }
            result.append({
                "id"     : str(row["id"]),
                "vector" : list(row["vector"]),
                "payload": payload,
            })
        return result

    # ── READ ──────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k       : int = 5,
    ) -> list[dict]:
        results = (
            self._table
            .search(query_vector)
            .limit(top_k)
            .to_list()
        )
        return [_row_to_chunk(r, score=1.0 - r.get("_distance", 0.0)) for r in results]

    def search_with_filter(
        self,
        query_vector: list[float],
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5,
    ) -> list[dict]:
        results = (
            self._table
            .search(query_vector)
            .where(f"{filter_by} = '{filter_val}'")
            .limit(top_k)
            .to_list()
        )
        return [_row_to_chunk(r, score=1.0 - r.get("_distance", 0.0)) for r in results]

    # ── STATS ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return len(self._table)
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        try:
            df = self._table.to_pandas(columns=["source"])
            return sorted(df["source"].dropna().unique().tolist())
        except Exception:
            return []

    def get_stats(self) -> dict:
        return {
            "collection"   : self.collection,
            "total_vectors": self.count(),
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "uri"          : self.uri,
            "mode"         : self.mode,
        }

    def delete_collection(self) -> None:
        if self.collection in self.db.table_names():
            self.db.drop_table(self.collection)
        print(f"  [LANCEDB] Dropped table: '{self.collection}'")


__all__ = ["LanceDBVectorStore"]