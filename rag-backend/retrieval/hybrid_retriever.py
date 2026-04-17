# retrieval/hybrid_retriever.py
#
# CHANGES vs Day 2 version (Day 3 — A2):
#   - _deduplicate() REPLACED by _mmr_deduplicate().
#     The new method does exact dedup first (cheap), then applies
#     Maximal Marginal Relevance (MMR) to ensure the final chunk set
#     is semantically diverse — not just lexically non-duplicate.
#
#   - Two new constructor params:
#       use_mmr        : bool  = True   — toggle MMR on/off (useful for testing)
#       mmr_threshold  : float = 0.82   — cosine sim ceiling. Chunks more similar
#                                         than this to an already-accepted chunk
#                                         are treated as redundant and skipped.
#
#   WHY MMR IS NEEDED:
#     Ship manuals use large chunks with overlap (parent_chunk_size=1500,
#     overlap=100). Two chunks from the same section can be 70-80% identical
#     in content but survive exact-match dedup because their first 200 chars differ.
#     Result: all 5 offline chunks say "cooling water must not exceed 60°C"
#     in slightly different words, wasting the worker's limited reading time.
#
#     MMR fixes this by asking: "is this chunk telling us something NEW compared
#     to what we've already accepted?" If not — skip it, pick the next candidate.
#     Workers get 5 chunks from genuinely different sections of the manual.
#
#   MMR ALGORITHM (greedy):
#     1. Sort candidates by RRF score (already done before this step).
#     2. Batch-embed all candidates in one forward pass.
#     3. Always accept the top-ranked chunk first.
#     4. For each subsequent candidate, compute cosine similarity to every
#        already-accepted chunk. If max_sim < mmr_threshold → accept.
#        Otherwise skip (it's redundant with something already accepted).
#     5. If accepted set is still < top_k after scanning all candidates,
#        backfill from the remainder (diversity is moot when options are scarce).
#
#   PERFORMANCE:
#     The extra cost vs plain dedup is one embed_documents() batch call on
#     fetch_k (20–40) candidate chunks. On bge-small-en-v1.5 on CPU this is
#     ~100-200ms — acceptable since reranker (online) and BM25 take similar time.
#     In offline mode there is no reranker, so the total latency budget is fine.
#
#   Everything else (RRF, BM25, expand_to_parents, score_threshold) unchanged.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from retrieval.bm25_store      import BM25Store
from retrieval.naive_retriever import RetrievalResult
from vectorstore.qdrant_store  import QdrantVectorStore, BaseVectorStore
from embeddings.embedder       import BaseEmbedder, EmbedderFactory
from config                    import TOP_K, RRF_K


# ─────────────────────────────────────────────────────────
# COSINE SIMILARITY HELPER
# ─────────────────────────────────────────────────────────

def _cosine_sim(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.

    Uses numpy for efficiency since embeddings are typically 384-dim floats.
    Returns 0.0 if either vector is zero-norm (degenerate case).

    Args:
        a, b: embedding vectors (list or 1-D numpy array)

    Returns:
        float in [-1.0, 1.0], typically [0.0, 1.0] for sentence embeddings
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


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
    """
    Merge dense (vector) and sparse (BM25) results with Reciprocal Rank Fusion.

    RRF score for a chunk: sum of weight / (k + rank) across each list
    where the chunk appears. Chunks appearing in both lists are boosted.

    The deduplication key is full content string (not [:200] truncation)
    to avoid false collisions on chunks that start identically.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map : dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
        # Full content as key — safer than truncation.
        # This is only for in-memory dict lookup, not persisted anywhere.
        return chunk.get("content", "").strip()

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
        → MMR deduplication
        → return child chunks
        [caller: rerank children → expand_to_parents → send to LLM]

    Offline pipeline (is_offline=True):
        embed query
        → BM25 + vector search
        → RRF fusion
        → MMR deduplication  ← KEY: ensures 5 diverse manual sections
        → return child chunks directly to worker
        (no reranker, no parent expansion)

    A2 CHANGE (Day 3):
        _deduplicate() replaced by _mmr_deduplicate().
        Workers now see diverse chunks from different sections of the manual
        instead of 5 overlapping passages from the same paragraph.

    A4 CHANGE (Day 2 — unchanged):
        _expand_to_parents() is not called inside retrieve().
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
        # ── NEW (A2) ────────────────────────────────────────
        use_mmr        : bool            = True,
        mmr_threshold  : float           = 0.70,
    ):
        self.embedder        = embedder or EmbedderFactory.get("huggingface")
        self.store           = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k           = top_k
        self.rrf_k           = rrf_k
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.deduplicate     = deduplicate
        self.score_threshold = score_threshold
        # ── NEW (A2) ────────────────────────────────────────
        self.use_mmr         = use_mmr
        self.mmr_threshold   = mmr_threshold

        from pathlib import Path
        from config import settings
        default_bm25_path = str(Path(settings.qdrant_path).parent / "bm25.pkl")
        self.bm25 = BM25Store(path=bm25_path or default_bm25_path)

        print(
            f"  [HYBRID] Ready. "
            f"top_k={top_k} | rrf_k={rrf_k} | "
            f"dense={dense_weight} | sparse={sparse_weight} | "
            f"mmr={'✅' if use_mmr else '❌'} (threshold={mmr_threshold}) | "
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

        A2 CHANGE: dedup step is now _mmr_deduplicate() which:
            1. Removes exact duplicates (cheap hash set)
            2. Embeds remaining candidates in one batch call
            3. Greedily selects top_k diverse chunks using cosine similarity

        A4 CHANGE (Day 2): _expand_to_parents() not called here.
            Online caller should: retrieve → rerank → expand_to_parents.
            Offline caller gets child chunks directly (workers see precise match).

        Args:
            query        : search string
            top_k        : override instance default
            filter_field : optional metadata field to filter by
            filter_value : value to match for filter_field
            is_offline   : affects log message, passed to MMR for consistent logging

        Returns:
            RetrievalResult — child chunks, MMR-diverse, sorted by initial RRF score
        """
        k       = top_k or self.top_k
        fetch_k = max(k * 3, 20)

        # 1. Embed query — also used in MMR cosine comparison
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

        # 5. Score threshold filter
        if self.score_threshold > 0:
            fused = [r for r in fused if r["score"] >= self.score_threshold]

        # 6. Dedup + diversity
        # CHANGED (A2): was _deduplicate(fused) + fused[:k]
        #               now _mmr_deduplicate(fused, q_vec, k) which does both
        if self.deduplicate:
            fused = self._mmr_deduplicate(
                chunks    = fused,
                query_vec = q_vec,
                top_k     = k,
            )
        else:
            fused = fused[:k]

        # ── A4: NO _expand_to_parents() call here ──────────────
        # Expansion happens in rag_chain._retrieve() AFTER reranking.
        # ────────────────────────────────────────────────────────

        mode = "offline" if is_offline else "online (rerank + expand in chain)"
        print(f"  [HYBRID] Returning {len(fused)} child chunks ({mode})")
        return RetrievalResult(fused)

    # ── MMR DEDUPLICATION (A2) ─────────────────────────────

    def _mmr_deduplicate(
        self,
        chunks    : list[dict],
        query_vec : list | np.ndarray,
        top_k     : int,
    ) -> list[dict]:
        """
        Two-stage deduplication: exact match first, then MMR semantic diversity.

        Stage 1 — Exact dedup (O(n) hash set):
            Removes chunks with identical content strings.
            Fast and always done regardless of use_mmr flag.
            Needed because Qdrant and BM25 can return the same chunk.

        Stage 2 — MMR greedy selection (O(n * accepted) cosine similarity):
            Batch-embeds all remaining candidates in one forward pass.
            Greedily accepts chunks that are semantically diverse from
            everything already in the accepted set.

            Selection rule:
                max_sim = max cosine_sim(candidate, s) for s in accepted
                Accept candidate if max_sim < mmr_threshold (default 0.82)

            The accepted set is ordered by original RRF score, not by MMR score.
            This means: relevance is the primary sort, diversity is the filter.
            We don't sacrifice the most relevant chunk for diversity — we just
            make sure positions 2-5 aren't clones of position 1.

        Backfill:
            If fewer than top_k chunks survive MMR (rare — happens when corpus is
            small or query is very narrow), we backfill from the remaining
            candidates in RRF-score order. A chunk that's redundant is still
            better than nothing.

        Fallback:
            If embedding the candidates fails (OOM, model error), we fall back
            to pure score cutoff (fused[:top_k]). This never crashes — the
            retrieval pipeline always returns something.

        Args:
            chunks    : RRF-fused candidates, sorted by score descending
            query_vec : embedded query vector (already computed in retrieve())
            top_k     : desired output size

        Returns:
            list of at most top_k chunks, MMR-diverse
        """
        if not chunks:
            return []

        # ── Stage 1: exact dedup ──────────────────────────────
        seen_content: set        = set()
        unique      : list[dict] = []
        for c in chunks:
            content = c.get("content", "").strip()
            if content not in seen_content:
                seen_content.add(content)
                unique.append(c)

        # If exact dedup already reduced us to <= top_k, we're done
        if len(unique) <= top_k:
            return unique

        # If MMR is disabled, just use score cutoff on the exact-deduped list
        if not self.use_mmr:
            return unique[:top_k]

        # ── Stage 2: MMR semantic dedup ───────────────────────
        try:
            texts         = [c["content"] for c in unique]
            candidate_vecs = self.embedder.embed_documents(texts)
            # embed_documents returns list of lists or numpy arrays — normalise
            candidate_vecs = [np.asarray(v, dtype=np.float32) for v in candidate_vecs]
        except Exception as e:
            print(f"  [HYBRID/MMR] Embedding candidates failed ({e}) "
                  f"— falling back to score cutoff")
            return unique[:top_k]

        # Greedy MMR selection
        # accepted_indices: indices into `unique` of chosen chunks
        # accepted_vecs   : their embedding vectors (for similarity computation)
        accepted_indices: list[int]         = []
        accepted_vecs   : list[np.ndarray]  = []

        for i, (chunk, vec) in enumerate(zip(unique, candidate_vecs)):
            if len(accepted_indices) >= top_k:
                break

            if not accepted_indices:
                # Always accept the top-ranked chunk unconditionally
                accepted_indices.append(i)
                accepted_vecs.append(vec)
                continue

            # Compute max cosine similarity to every already-accepted chunk
            max_sim = max(_cosine_sim(vec, av) for av in accepted_vecs)

            if max_sim < self.mmr_threshold:
                # Low similarity → new information → accept
                accepted_indices.append(i)
                accepted_vecs.append(vec)
            # else: too similar to something already accepted → skip (redundant)

        # ── Backfill if accepted < top_k ─────────────────────
        # This happens when the corpus is small or the query is very narrow
        # (all remaining chunks are about the same sub-topic).
        # We'd rather return a slightly redundant chunk than nothing.
        if len(accepted_indices) < top_k:
            accepted_set   = set(accepted_indices)
            remaining_count = top_k - len(accepted_indices)
            backfill = [
                i for i in range(len(unique))
                if i not in accepted_set
            ][:remaining_count]
            accepted_indices.extend(backfill)

            if backfill:
                print(
                    f"  [HYBRID/MMR] Backfilled {len(backfill)} chunk(s) — "
                    f"all candidates exceeded similarity threshold"
                )

        result = [unique[i] for i in accepted_indices]
        print(
            f"  [HYBRID/MMR] {len(unique)} unique candidates → "
            f"{len(result)} MMR-diverse chunks (threshold={self.mmr_threshold})"
        )
        return result

    # ── PARENT EXPANSION (now a public method) ─────────────

    def expand_to_parents(self, retrieval: RetrievalResult) -> RetrievalResult:
        """
        PUBLIC wrapper — expands child chunks to parent context.

        Called by rag_chain._retrieve() in ONLINE mode, AFTER reranking.
        Sequence in online mode:
            1. retrieve()          → N children, RRF+MMR sorted
            2. reranker.rerank()   → top-5 children, cross-encoder scored
            3. expand_to_parents() → swap child.content for parent_content
               so the LLM receives the full surrounding passage (~1500 tok)

        Args:
            retrieval: RetrievalResult of reranked child chunks

        Returns:
            New RetrievalResult with content replaced by parent_content
            where available. Deduplicates on parent_id so two children
            from the same parent don't send the same passage twice.
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

    def get_context(self, query: str, **kwargs) -> str:
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"         : "HybridRetriever",
            "top_k"        : self.top_k,
            "rrf_k"        : self.rrf_k,
            "dense_weight" : self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "deduplicate"  : self.deduplicate,
            "mmr_enabled"  : self.use_mmr,
            "mmr_threshold": self.mmr_threshold,
            "bm25_docs"    : len(self.bm25),
            "parent_mode"  : "post-rerank-expansion (A4)",
        }


__all__ = ["reciprocal_rank_fusion", "HybridRetriever"]