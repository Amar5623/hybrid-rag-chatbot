# retrieval/reranker.py
#
# CHANGES vs original:
#   - Default model switched from ms-marco-MiniLM-L-6-v2 (~85MB)
#     to ms-marco-TinyBERT-L-2-v2 (~18MB) — 4x smaller, still very
#     good for technical retrieval. Meets the "least resources" requirement.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder
from retrieval.naive_retriever import RetrievalResult


class Reranker:
    """
    Cross-encoder reranker — rescores retrieved chunks against the query.

    Model: cross-encoder/ms-marco-TinyBERT-L-2-v2
      ✅ ~18MB (was ~85MB — 4x smaller)
      ✅ Free  ✅ Runs offline
      ✅ Strong on technical passage ranking
      Trained on MS MARCO passage ranking dataset.

    Usage:
        result   = retriever.retrieve(query, top_k=20)
        reranked = reranker.rerank(query, result, top_k=5)
        context  = reranked.to_context_string()
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    def __init__(
        self,
        model_name : str = DEFAULT_MODEL,
        batch_size : int = 16,
        max_length : int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"  [RERANKER] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length = max_length,
        )
        print(f"  [RERANKER] ✅ Ready!")

    # ── CORE RERANK ──────────────────────────

    def rerank(
        self,
        query          : str,
        retrieval      : RetrievalResult,
        top_k          : int   = 5,
        score_threshold: float = None,
    ) -> RetrievalResult:
        """
        Rerank a RetrievalResult using the cross-encoder.

        Args:
            query           : original search query
            retrieval       : output from any retriever
            top_k           : how many chunks to keep after reranking
            score_threshold : optional min cross-encoder score to keep

        Returns:
            New RetrievalResult sorted by cross-encoder score,
            with 'rerank_score' and updated 'score' fields.
        """
        chunks = retrieval.get_chunks()

        if not chunks:
            print("  [RERANKER] No chunks to rerank.")
            return RetrievalResult([])

        pairs = [(query, c["content"]) for c in chunks]

        scores = self.model.predict(
            pairs,
            batch_size        = self.batch_size,
            show_progress_bar = len(pairs) > 20,
        )

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            c = chunk.copy()
            c["rerank_score"]    = round(float(score), 4)
            c["retrieval_score"] = c.get("score", 0.0)
            c["score"]           = c["rerank_score"]
            scored_chunks.append(c)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        if score_threshold is not None:
            scored_chunks = [
                c for c in scored_chunks
                if c["rerank_score"] >= score_threshold
            ]

        scored_chunks = scored_chunks[:top_k]

        print(f"  [RERANKER] Reranked {len(chunks)} → kept top {len(scored_chunks)}")
        return RetrievalResult(scored_chunks)

    # ── CONVENIENCE ──────────────────────────

    def rerank_chunks(
        self,
        query  : str,
        chunks : list[dict],
        top_k  : int = 5,
    ) -> list[dict]:
        """Rerank raw chunk dicts directly."""
        return self.rerank(
            query     = query,
            retrieval = RetrievalResult(chunks),
            top_k     = top_k,
        ).get_chunks()

    def get_info(self) -> dict:
        return {
            "type"      : "Reranker",
            "model"     : self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


__all__ = ["Reranker"]