# chains/rag_chain.py
#
# CHANGES vs previous version:
#   - OfflineChunk construction in stream() updated to pass 4 new fields:
#       chunk_type, bbox, page_width, page_height
#     These come directly from the chunk dict payload (stored at ingestion
#     time by the updated pdf_loader.py). No logic change — just forwarding
#     the data that now exists in every chunk dict.
#
#   All other logic identical to previous version.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.naive_retriever  import NaiveRetriever, RetrievalResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from generation.groq_llm        import BaseLLM, ChatHistory, LLMFactory
from vectorstore.qdrant_store   import QdrantVectorStore, BaseVectorStore
from embeddings.embedder        import EmbedderFactory
from schemas                    import OfflineQueryResponse, OfflineChunk
from config                     import TOP_K, MIN_RERANK_SCORE, settings


# ─────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a precise ship manual assistant. Answer questions based on the provided manual sections.

Rules:
1. Answer strictly from the provided context. Do not invent facts not in the context.
2. For follow-up questions referencing previous turns, use the conversation history.
3. Be concise and direct. No padding or filler phrases.
4. Do NOT write a 'Sources:' or 'References:' section — citations are handled separately.
5. Preserve technical terminology exactly as it appears in the source.
6. Always format tabular data as a markdown table using | col | col | syntax."""

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}"""

GENERAL_FALLBACK_PROMPT = """\
You are a helpful ship manual assistant. The provided manual sections do not contain
relevant information to answer this question.

Rules:
1. Start with one short sentence noting the manuals didn't cover this topic.
2. If you have general knowledge on it, answer from that.
3. If you don't, say so honestly.
4. Be concise."""


# ─────────────────────────────────────────────────────────
# CHAIN RESPONSE
# ─────────────────────────────────────────────────────────

class ChainResponse:
    """
    Wraps the full output of an online RAG chain call.
    query_type is always "document" now (no router).
    """

    def __init__(
        self,
        answer    : str,
        retrieval : RetrievalResult,
        question  : str,
        model     : str,
        usage     : dict = None,
        query_type: str  = "document",
    ):
        self.answer     = answer
        self.retrieval  = retrieval
        self.question   = question
        self.model      = model
        self.usage      = usage or {}
        self.query_type = query_type

    def get_answer(self) -> str:
        return self.answer

    def get_citations(self) -> list[dict]:
        citations: list[dict] = []
        for chunk in self.retrieval.get_chunks():
            citations.append({
                "source"      : chunk.get("source", "unknown"),
                "page"        : chunk.get("page", "?"),
                "heading"     : chunk.get("heading", ""),
                "section_path": chunk.get("section_path", ""),
                "type"        : chunk.get("type", "text"),
            })
        return citations

    def get_images(self) -> list[str]:
        return self.retrieval.get_images()

    def has_images(self) -> bool:
        return len(self.get_images()) > 0

    def get_chunks(self) -> list[dict]:
        return self.retrieval.get_chunks()

    def get_context(self) -> str:
        return self.retrieval.to_context_string()

    def __repr__(self) -> str:
        return (
            f"ChainResponse("
            f"model={self.model}, "
            f"query_type={self.query_type}, "
            f"chunks={len(self.retrieval)}, "
            f"tokens={self.usage.get('total_tokens', '?')})"
        )


# ─────────────────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────────────────

class RAGChain:
    """
    Simplified RAG pipeline for offline-capable ship manual lookup.

    Online  (is_online=True):
        Retrieve → Rerank → LLM stream → ChainResponse
    Offline (is_online=False):
        Retrieve (no reranker) → OfflineQueryResponse with chunk cards
        No LLM call, no SSE — just JSON with manual excerpts.

    QueryRouter removed: all queries are document queries.
    QueryExpansion removed: no LLM call before retrieval.
    """

    def __init__(
        self,
        llm           : BaseLLM        = None,
        vector_store  : BaseVectorStore = None,
        retriever                       = None,
        reranker      : Reranker        = None,
        use_reranker  : bool            = True,
        retrieve_top_k: int             = TOP_K,
        rerank_top_k  : int             = 5,
        cite_sources  : bool            = True,
        llm_provider  : str             = "groq",
    ):
        # ── LLM ───────────────────────────────────────────
        self.llm = llm or LLMFactory.get(llm_provider)
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)

        # ── Vector store ──────────────────────────────────
        embedder   = EmbedderFactory.get("huggingface")
        self.store = vector_store or QdrantVectorStore(embedder=embedder)

        # ── Retriever ─────────────────────────────────────
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(
                vector_store = self.store,
                embedder     = embedder,
                top_k        = retrieve_top_k,
            )

        # ── Reranker ──────────────────────────────────────
        self.use_reranker = use_reranker
        self.reranker     = reranker or (Reranker() if use_reranker else None)
        self.rerank_top_k = rerank_top_k

        # ── Settings ──────────────────────────────────────
        self.retrieve_top_k  = retrieve_top_k
        self.cite_sources    = cite_sources
        self._source_filter  : str | None = None

        # ── Memory ────────────────────────────────────────
        self.history = self.llm.history

        print(f"\n  [RAG CHAIN] ✅ Ready!")
        print(f"  [RAG CHAIN] LLM       : {self.llm.model_name}")
        print(f"  [RAG CHAIN] Retriever : {type(self.retriever).__name__}")
        print(f"  [RAG CHAIN] Reranker  : {'✅ (online only)' if use_reranker else '❌'}")
        print(f"  [RAG CHAIN] Mode      : online/offline branching enabled")

    # ── INDEXING ──────────────────────────────────────────

    def index_documents(self, chunks: list[dict]) -> None:
        self.store.add_documents(chunks)
        if hasattr(self.retriever, "index_chunks"):
            self.retriever.index_chunks(chunks)
        print(f"  [RAG CHAIN] Indexed {len(chunks)} chunks.")

    # ── RETRIEVAL ─────────────────────────────────────────

    def _retrieve(self, question: str, is_offline: bool = False) -> RetrievalResult:
        """
        Run retrieval. In offline mode, passes is_offline=True to the retriever
        so the reranker step inside HybridRetriever is skipped.
        After retrieval, if we're online, we apply the cross-encoder reranker here.
        """
        retrieval = self.retriever.retrieve(
            question,
            filter_field = "source" if self._source_filter else None,
            filter_value = self._source_filter,
            is_offline   = is_offline,
        )

        # Online only: apply reranker
        if not is_offline and self.use_reranker and self.reranker and len(retrieval) > 0:
            retrieval = self.reranker.rerank(
                query     = question,
                retrieval = retrieval,
                top_k     = self.rerank_top_k,
            )

        return retrieval

    # ── PROMPT BUILDING ───────────────────────────────────

    def _build_prompt(self, question: str, context: str) -> str:
        return RAG_USER_TEMPLATE.format(context=context, question=question)

    # ── ASK (blocking, online only) ───────────────────────

    def ask(self, question: str, has_kb: bool = True) -> ChainResponse:
        """
        Blocking online RAG pipeline.
        Always assumes online — use stream() for online/offline branching.
        """
        if not has_kb:
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            result = self.llm.generate(
                prompt   = question,
                history  = self.history,
                store_as = question,
            )
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            return ChainResponse(
                answer     = result["content"],
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = result["model"],
                usage      = result["usage"],
                query_type = "general",
            )

        retrieval = self._retrieve(question, is_offline=False)
        context   = retrieval.to_context_string()

        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            result = self.llm.generate(
                prompt   = question,
                history  = self.history,
                store_as = question,
            )
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            return ChainResponse(
                answer     = result["content"],
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = result["model"],
                usage      = result["usage"],
                query_type = "general",
            )

        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
        prompt = self._build_prompt(question, context)
        result = self.llm.generate(
            prompt   = prompt,
            history  = self.history,
            store_as = question,
        )
        return ChainResponse(
            answer     = result["content"],
            retrieval  = retrieval,
            question   = question,
            model      = result["model"],
            usage      = result["usage"],
            query_type = "document",
        )

    # ── STREAM (generator, online OR offline) ─────────────

    def stream(self, question: str, has_kb: bool = True, is_online: bool = True):
        """
        Main entry point — handles both online and offline modes.

        Online  → yields str tokens then final ChainResponse (SSE streaming)
        Offline → yields a single OfflineQueryResponse (normal JSON, no SSE)

        Args:
            question  : user's question
            has_kb    : whether any documents are indexed
            is_online : network status from NetworkMonitor / rag_service

        Yields:
            str              — text tokens (online only)
            ChainResponse    — final metadata object (online only)
            OfflineQueryResponse — all chunks at once (offline only)
        """

        # ── OFFLINE BRANCH ────────────────────────────────
        if not is_online:
            if not has_kb:
                yield OfflineQueryResponse(
                    query      = question,
                    chunks     = [],
                    total      = 0,
                    is_offline = True,
                )
                return

            offline_top_k = settings.offline_top_k
            retrieval = self._retrieve(question, is_offline=True)
            chunks    = retrieval.get_chunks()[:offline_top_k]

            # ── CHANGED: pass bbox, page_width, page_height, chunk_type ──
            # These 4 fields are now stored on every chunk by pdf_loader.py.
            # We simply forward whatever is in the chunk dict.
            # chunk.get() with a default handles old chunks that pre-date
            # this change (e.g. chunks ingested before the upgrade) gracefully.
            offline_chunks = [
                OfflineChunk(
                    source       = c.get("source", "unknown"),
                    page         = c.get("page"),
                    heading      = c.get("heading", ""),
                    section_path = c.get("section_path", ""),
                    content      = c.get("content", ""),
                    score        = round(float(c.get("score", 0.0)), 4),
                    # ── NEW ──────────────────────────────────────────────
                    chunk_type   = c.get("type", "text"),
                    bbox         = c.get("bbox"),         # list[float] or None
                    page_width   = c.get("page_width"),   # float or None
                    page_height  = c.get("page_height"),  # float or None
                )
                for c in chunks
            ]

            yield OfflineQueryResponse(
                query      = question,
                chunks     = offline_chunks,
                total      = len(offline_chunks),
                is_offline = True,
            )
            return

        # ── ONLINE BRANCH ─────────────────────────────────

        # No KB
        if not has_kb:
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            full_reply: list[str] = []
            usage: dict = {}
            for chunk in self.llm.stream(
                prompt   = question,
                history  = self.history,
                store_as = question,
            ):
                if isinstance(chunk, str):
                    full_reply.append(chunk)
                    yield chunk
                else:
                    usage = chunk.get("usage", {})
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            yield ChainResponse(
                answer     = "".join(full_reply),
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = self.llm.model_name,
                usage      = usage,
                query_type = "general",
            )
            return

        # Retrieve + rerank (online)
        retrieval = self._retrieve(question, is_offline=False)
        context   = retrieval.to_context_string()

        # Weak context fallback
        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            full_reply = []
            usage = {}
            for chunk in self.llm.stream(
                prompt   = question,
                history  = self.history,
                store_as = question,
            ):
                if isinstance(chunk, str):
                    full_reply.append(chunk)
                    yield chunk
                else:
                    usage = chunk.get("usage", {})
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            yield ChainResponse(
                answer     = "".join(full_reply),
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = self.llm.model_name,
                usage      = usage,
                query_type = "general",
            )
            return

        # Full RAG stream
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
        prompt     = self._build_prompt(question, context)
        full_reply = []
        usage      = {}
        for chunk in self.llm.stream(
            prompt   = prompt,
            history  = self.history,
            store_as = question,
        ):
            if isinstance(chunk, str):
                full_reply.append(chunk)
                yield chunk
            else:
                usage = chunk.get("usage", {})

        yield ChainResponse(
            answer     = "".join(full_reply),
            retrieval  = retrieval,
            question   = question,
            model      = self.llm.model_name,
            usage      = usage,
            query_type = "document",
        )

    # ── MEMORY ────────────────────────────────────────────

    def reset_memory(self) -> None:
        self.llm.reset_history()
        print("  [RAG CHAIN] Memory cleared.")

    def set_source_filter(self, filename: str) -> None:
        self._source_filter = filename
        print(f"  [RAG CHAIN] Pinned to: '{filename}'")

    def clear_source_filter(self) -> None:
        self._source_filter = None
        print(f"  [RAG CHAIN] Pin cleared")

    def get_source_filter(self) -> str | None:
        return self._source_filter

    def get_history(self) -> list[dict]:
        return self.history.to_messages()

    # ── INFO ──────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "llm"           : self.llm.get_info(),
            "retriever"     : type(self.retriever).__name__,
            "reranker"      : self.reranker.get_info() if self.reranker else None,
            "retrieve_top_k": self.retrieve_top_k,
            "rerank_top_k"  : self.rerank_top_k,
            "cite_sources"  : self.cite_sources,
            "history_turns" : len(self.history),
            "vector_store"  : self.store.get_stats(),
            "last_query_type": "document",
        }


__all__ = ["ChainResponse", "RAGChain"]