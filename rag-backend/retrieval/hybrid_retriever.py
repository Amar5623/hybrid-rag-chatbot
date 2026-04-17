# retrieval/hybrid_retriever.py
#
# CHANGES vs previous version (Day 2 — A4):
#   - _expand_to_parents() is NO LONGER called inside retrieve().
#     retrieve() now always returns raw child chunks (300 tokens each).
#
#   WHY THIS MATTERS:
#     Previously: children → expand to parents → rerank 1500-tok blobs
#     The reranker was scoring 1500-token parent blobs against a short query.
#     A dense parent block buries the precise sentence that matched — the
#     reranker's cross-encoder loses the tight lexical signal it needs.
#
#     Now: children → rerank children (precise signal) → expand in rag_chain
#     The reranker sees 300-token child chunks — exactly the passage that
#     matched the query. After selecting top-N, rag_chain.py expands those
#     children to their parent context for the LLM to generate a full answer.
#
#   NEW public method: expand_to_parents(retrieval: RetrievalResult) -> RetrievalResult
#     Called by rag_chain._retrieve() after reranking in online mode.
#     Offline mode returns child chunks directly (workers see the precise match).
#
#   Everything else (RRF, BM25, dedup, filter) unchanged.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25_store      import BM25Store
from retrieval.naive_retriever import RetrievalResult
from vectorstore.qdrant_store  import QdrantVectorStore, BaseVectorStore
from embeddings.embedder       import BaseEmbedder, EmbedderFactory
from config                    import TOP_K, RRF_K


# ─────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results  : list[dict],
    sparse_results : list[dict],
    k              : int   = RRF_K,
    dense_weight   : float = 1.0,
    sparse_weight  : float = 1.0,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map : dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
        return chunk.get("content", "").strip()[:200]

    for rank, chunk in enumerate(dense_results, start=1):
        key             = _key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (k + rank)
        chunk_map[key]  = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        key             = _key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused: list[dict] = []
    for key in sorted_keys:
        chunk              = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["score"]     = chunk["rrf_score"]
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever: Dense (cosine) + Sparse (BM25) fused with RRF.

    Online pipeline (is_offline=False):
        embed query
        → BM25 + vector search
        → RRF fusion
        → return child chunks
        [caller: rerank children → expand_to_parents → send to LLM]

    Offline pipeline (is_offline=True):
        embed query
        → BM25 + vector search
        → RRF fusion
        → return child chunks directly
        (no reranker, no parent expansion — workers see precise match excerpts)

    KEY CHANGE (Day 2 A4):
        _expand_to_parents() is no longer called inside retrieve().
        retrieve() always returns child chunks (300 tokens).
        Use expand_to_parents(retrieval) AFTER reranking in online mode.
    """

    def __init__(
        self,
        vector_store   : BaseVectorStore = None,
        embedder       : BaseEmbedder    = None,
        top_k          : int             = TOP_K,
        rrf_k          : int             = RRF_K,
        dense_weight   : float           = 1.0,
        sparse_weight  : float           = 1.0,
        deduplicate    : bool            = True,
        score_threshold: float           = 0.0,
        bm25_path      : str             = None,
        parent_store                     = None,   # ignored — kept for compat
    ):
        self.embedder        = embedder or EmbedderFactory.get("huggingface")
        self.store           = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k           = top_k
        self.rrf_k           = rrf_k
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.deduplicate     = deduplicate
        self.score_threshold = score_threshold

        from pathlib import Path
        from config import settings
        default_bm25_path = str(Path(settings.qdrant_path).parent / "bm25.pkl")
        self.bm25 = BM25Store(path=bm25_path or default_bm25_path)

        print(
            f"  [HYBRID] Ready. "
            f"top_k={top_k} | rrf_k={rrf_k} | "
            f"dense={dense_weight} | sparse={sparse_weight} | "
            f"parent_expansion=post-rerank (A4)"
        )

    # ── INDEX ─────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        self.bm25.build(chunks)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.bm25.add(chunks)

    # ── CORE RETRIEVAL ────────────────────────────────────

    def retrieve(
        self,
        query        : str,
        top_k        : int  = None,
        filter_field : str  = None,
        filter_value : str  = None,
        is_offline   : bool = False,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval and return RAW CHILD CHUNKS.

        A4 CHANGE: _expand_to_parents() is NO LONGER called here.
        The returned chunks always contain child content (300 tokens).

        For ONLINE mode:
            The caller (rag_chain._retrieve) should:
              1. rerank against these child chunks  ← precise signal
              2. call self.retriever.expand_to_parents(retrieval)  ← LLM context

        For OFFLINE mode (is_offline=True):
            Child chunks are returned directly to the user.
            This is actually BETTER UX — workers see the precise matched
            passage, not a 1500-token blob they have to read through.

        Args:
            query        : search string
            top_k        : override instance default
            filter_field : optional metadata field to filter by
            filter_value : value to match for filter_field
            is_offline   : flag passed from rag_chain; affects log message only

        Returns:
            RetrievalResult — child chunks sorted by RRF score
        """
        k       = top_k or self.top_k
        fetch_k = max(k * 3, 20)

        # 1. Embed query
        q_vec = self.embedder.embed_text(query)

        # 2. Dense search
        if filter_field and filter_value:
            dense_results = self.store.search_with_filter(
                query_vector = q_vec,
                filter_by    = filter_field,
                filter_val   = filter_value,
                top_k        = fetch_k,
            )
        else:
            dense_results = self.store.search(
                query_vector = q_vec,
                top_k        = fetch_k,
            )

        # 3. BM25 search
        sparse_results = self.bm25.search(query=query, top_k=fetch_k)

        # 4. RRF fusion
        fused = reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results,
            k              = self.rrf_k,
            dense_weight   = self.dense_weight,
            sparse_weight  = self.sparse_weight,
        )

        if self.score_threshold > 0:
            fused = [r for r in fused if r["score"] >= self.score_threshold]

        if self.deduplicate:
            fused = self._deduplicate(fused)

        fused = fused[:k]

        # ── A4: NO _expand_to_parents() call here ──────────────
        # Previously this line existed:
        #   fused = self._expand_to_parents(fused)
        # It has been removed. Expansion now happens in rag_chain._retrieve()
        # AFTER reranking, via self.retriever.expand_to_parents(retrieval).
        # ────────────────────────────────────────────────────────

        mode = "offline" if is_offline else "online (rerank + expand in chain)"
        print(f"  [HYBRID] Returning {len(fused)} child chunks ({mode})")
        return RetrievalResult(fused)

    # ── PARENT EXPANSION (now a public method) ─────────────

    def expand_to_parents(self, retrieval: RetrievalResult) -> RetrievalResult:
        """
        PUBLIC wrapper — expands child chunks to parent context.

        Called by rag_chain._retrieve() in ONLINE mode, AFTER reranking.
        This is the key sequence:
            1. retrieve()       → 20 children, RRF-sorted
            2. reranker.rerank()→ top-5 children, cross-encoder scored
            3. expand_to_parents() → swap child.content for parent_content
               so the LLM receives the full surrounding passage (~1500 tok)

        Args:
            retrieval: RetrievalResult of reranked child chunks

        Returns:
            New RetrievalResult with content replaced by parent_content
            where available. Deduplicates on parent_id so you never send
            two children from the same parent section as separate passages.
        """
        expanded = self._expand_to_parents(retrieval.get_chunks())
        print(f"  [HYBRID] Parent expansion: {len(retrieval)} → {len(expanded)} passages")
        return RetrievalResult(expanded)

    # ── PARENT EXPANSION (private impl) ───────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Replace each child's content with its parent_content if available.
        parent_content is stored directly on the Qdrant payload by
        HierarchicalChunker — no SQLite lookup needed.

        Deduplicates on parent_id: if two children share the same parent,
        only the first (higher-ranked) one is kept. The parent passage
        already contains both children's text, so there's no info loss.
        """
        expanded     : list[dict] = []
        seen_parents : set        = set()

        for child in chunks:
            parent_id      = child.get("parent_id", "")
            parent_content = child.get("parent_content", "")

            if parent_content and parent_id:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                merged            = {k: v for k, v in child.items()}
                merged["content"] = parent_content
                expanded.append(merged)
            else:
                expanded.append(child)

        return expanded

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        seen  : set        = set()
        unique: list[dict] = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content not in seen:
                seen.add(content)
                unique.append(chunk)
        return unique

    def get_context(self, query: str, **kwargs) -> str:
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"          : "HybridRetriever",
            "top_k"         : self.top_k,
            "rrf_k"         : self.rrf_k,
            "dense_weight"  : self.dense_weight,
            "sparse_weight" : self.sparse_weight,
            "deduplicate"   : self.deduplicate,
            "bm25_docs"     : len(self.bm25),
            "parent_mode"   : "post-rerank-expansion (A4)",
        }


__all__ = ["reciprocal_rank_fusion", "HybridRetriever"]