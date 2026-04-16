# retrieval/bm25_store.py
#
# CHANGES:
#   - delete_by_source(filename) added — filters chunks by source,
#     rebuilds and saves. Used by the delete-file endpoint.
#   - Everything else unchanged.

import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_PATH

from rank_bm25 import BM25Okapi


class BM25Store:
    """
    Persistent BM25 sparse index.
    Persists to disk so hybrid retrieval works correctly after restart.
    """

    def __init__(self, path: str = BM25_PATH):
        self.path            = path
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi    = None
        self._load()

    # ── persistence ───────────────────────────────────────

    def _load(self) -> None:
        if not Path(self.path).exists():
            print("  [BM25] No saved index — will build on first ingest")
            return
        try:
            with open(self.path, "rb") as f:
                data         = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self._bm25   = data.get("bm25")
            print(f"  [BM25] Loaded {len(self._chunks)} docs from disk")
        except Exception as e:
            print(f"  [BM25] Load failed ({e}) — starting fresh")
            self._chunks = []
            self._bm25   = None

    def _save(self) -> None:
        try:
            with open(self.path, "wb") as f:
                pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        except Exception as e:
            print(f"  [BM25] Save failed: {e}")

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        tokenized  = [c["content"].lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    # ── write ─────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._rebuild()
        self._save()
        print(f"  [BM25] Index built with {len(chunks)} documents.")

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._rebuild()
        self._save()
        print(f"  [BM25] Index now has {len(self._chunks)} docs")

    def delete_by_source(self, filename: str) -> int:
        """
        Remove all chunks whose 'source' field matches filename.
        Rebuilds and saves the index after removal.
        Returns number of chunks removed.
        """
        before        = len(self._chunks)
        self._chunks  = [c for c in self._chunks if c.get("source") != filename]
        removed       = before - len(self._chunks)
        self._rebuild()
        self._save()
        print(f"  [BM25] Removed {removed} chunks for source='{filename}'. "
              f"Index now has {len(self._chunks)} docs")
        return removed

    def reset(self) -> None:
        self._chunks = []
        self._bm25   = None
        if Path(self.path).exists():
            Path(self.path).unlink()
        print("  [BM25] Index reset")

    # ── read ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self._bm25 or not self._chunks:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            chunk          = self._chunks[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["BM25Store"]