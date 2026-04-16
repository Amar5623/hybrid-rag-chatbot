# retrieval/naive_retriever.py

import os
import sys
import pathlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.qdrant_store import QdrantVectorStore, BaseVectorStore
from embeddings.embedder      import BaseEmbedder, EmbedderFactory
from config import TOP_K


# ─────────────────────────────────────────────────────────
# RETRIEVAL RESULT
# ─────────────────────────────────────────────────────────

class RetrievalResult:
    """
    Wraps the output of a retrieval call.
    Provides raw chunks, formatted context, and citations
    all from one object.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    # ── FORMATTED CONTEXT ─────────────────────────────────

    def to_context_string(self, max_chars: int = 6000) -> str:
        """
        Format chunks into a single LLM-ready context block.
        Includes section_path in header for richer attribution.

        Example output:
            [Source: report.pdf | Page: 2 | Section: Chapter 3 > Results]
            Sales increased by 30% in Q2...
        """
        parts: list[str] = []
        total = 0

        for chunk in self.chunks:
            source  = chunk.get("source", "unknown")
            page    = chunk.get("page")
            section = chunk.get("section_path") or chunk.get("heading") or ""
            content = chunk.get("content", "").strip()

            header = f"[Source: {source}"
            if page is not None:
                header += f" | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += "]"

            block = f"{header}\n{content}"

            if total + len(block) > max_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    # ── CITATIONS ─────────────────────────────────────────

    def get_citations(self) -> list[dict]:
        """
        Return deduplicated citations with source, page,
        heading, and section_path.
        """
        seen:    set        = set()
        results: list[dict] = []

        for chunk in self.chunks:
            key = (chunk.get("source", "unknown"), chunk.get("page"))
            if key not in seen:
                seen.add(key)
                results.append({
                    "source"      : chunk.get("source", "unknown"),
                    "page"        : chunk.get("page"),
                    "heading"     : chunk.get("heading", ""),
                    "section_path": chunk.get("section_path", ""),
                    "type"        : chunk.get("type", "text"),
                })

        return results

    def format_citations(self) -> str:
        """Human-readable citation string."""
        cites = self.get_citations()
        if not cites:
            return ""
        lines = []
        for i, c in enumerate(cites, 1):
            page_str    = f", p.{c['page']}" if c["page"] is not None else ""
            section_str = (
                f" [{c['section_path']}]" if c.get("section_path") else
                f" [{c['heading']}]"      if c.get("heading")      else ""
            )
            lines.append(f"  [{i}] {c['source']}{page_str}{section_str}")
        return "Sources:\n" + "\n".join(lines)

    # ── RAW ACCESS ────────────────────────────────────────

    def get_chunks(self) -> list[dict]:
        return self.chunks

    def get_top_chunk(self) -> dict | None:
        return self.chunks[0] if self.chunks else None

    def get_images(self) -> list[str]:
        """
        Return absolute paths of retrieved image chunks that exist on disk.
        Falls back to searching under data/images/ if the stored path
        doesn't resolve directly.
        """
        project_root = pathlib.Path(__file__).parent.parent
        images_dir   = project_root / "data" / "images"
        results      = []

        for c in self.chunks:
            if c.get("type") != "image" or not c.get("image_path"):
                continue
            p = c["image_path"]
            if os.path.exists(p):
                results.append(p)
                continue
            # Fallback: look in canonical images directory
            alt = str(images_dir / os.path.basename(p))
            if os.path.exists(alt):
                results.append(alt)

        return results

    def best_score(self) -> float:
        """Highest rerank/retrieval score in this result set."""
        if not self.chunks:
            return 0.0
        return self.chunks[0].get("rerank_score", self.chunks[0].get("score", 0.0))

    def __len__(self) -> int:
        return len(self.chunks)


# ─────────────────────────────────────────────────────────
# NAIVE RETRIEVER
# ─────────────────────────────────────────────────────────

class NaiveRetriever:
    """
    Dense-only vector retriever with optional parent expansion.

    Embeds the query once via the embedder and passes the vector
    directly to the store — avoids the double-embedding that
    occurred when passing a string to store.search().

    When parent_store is provided (a ParentStore instance),
    retrieved child chunks are swapped for their parent
    (1200-char) versions before being returned — same
    small-to-big pattern as HybridRetriever.
    """

    def __init__(
        self,
        vector_store : BaseVectorStore = None,
        embedder     : BaseEmbedder    = None,
        top_k        : int             = TOP_K,
    ):
        self.embedder     = embedder or EmbedderFactory.get("huggingface")
        self.store        = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k        = top_k   # ParentStore instance or None

        print(f"  [NAIVE] NaiveRetriever ready. top_k={top_k}")
        if parent_store:
            print(f"  [NAIVE] Parent store attached ({len(parent_store)} entries)")

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Dense vector search with optional parent expansion.

        Args:
            query : search string
            top_k : override instance default

        Returns:
            RetrievalResult with parent-expanded chunks if store provided
        """
        k       = top_k or self.top_k
        q_vec   = self.embedder.embed_text(query)
        results = self.store.search(query_vector=q_vec, top_k=k)

        if self.parent_store:
            results = self._expand_to_parents(results)

        return RetrievalResult(results)

    # ── PARENT EXPANSION ──────────────────────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Swap child chunks for their parent chunks using a single
        batch fetch from ParentStore — one DB query for all IDs
        instead of N individual lookups.

        Falls back to child if parent not found.
        Deduplicates on parent_id so the same parent is never
        sent to the LLM twice.
        """
        # Collect all parent IDs that need fetching
        parent_ids = [
            c.get("parent_id", "") for c in chunks
            if c.get("parent_id")
        ]

        # Batch fetch — single SQLite query
        parent_map   = self.parent_store.get_batch(parent_ids) if parent_ids else {}
        expanded     : list[dict] = []
        seen_parents : set        = set()

        for child in chunks:
            parent_id = child.get("parent_id", "")

            if parent_id and parent_id in parent_map:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent = parent_map[parent_id]
                merged = {
                    # Parent content — larger, more context for LLM
                    "content"     : parent["content"],
                    # Child metadata — for citations and scoring
                    "score"       : child.get("score", 0.0),
                    "source"      : child.get("source",       parent.get("source", "")),
                    "page"        : child.get("page",         parent.get("page")),
                    "type"        : child.get("type",         parent.get("type", "text")),
                    "heading"     : child.get("heading",      parent.get("heading", "")),
                    "section_path": child.get("section_path", parent.get("section_path", "")),
                    "parent_id"   : parent_id,
                    "image_path"  : child.get("image_path", ""),
                }
                expanded.append(merged)
            else:
                # No parent found — use child as-is (graceful fallback)
                expanded.append(child)

        return expanded

    # ── CONVENIENCE ───────────────────────────────────────

    def get_context(self, query: str, **kwargs) -> str:
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"        : "NaiveRetriever",
            "top_k"       : self.top_k,
            "parent_store": len(self.parent_store) if self.parent_store else 0,
            "vector_store": self.store.get_stats(),
        }


__all__ = ["RetrievalResult", "NaiveRetriever"]