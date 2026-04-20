# chains/rag_chain.py
#
# CHANGES vs previous version (Day 2 — A4):
#   - _retrieve() restructured: rerank children FIRST, expand to parents AFTER.
#
#   OLD flow (broken):
#       retrieve()        → children + expand → parent blobs (1500 tok)
#       reranker.rerank() → scores parent blobs  ← low precision
#
#   NEW flow (A4):
#       retrieve()                       → raw child chunks (300 tok)
#       reranker.rerank()                → top-N children  ← precise signal
#       retriever.expand_to_parents()    → parent blobs (1500 tok) for LLM
#
# ── BUG 2 FIX — Online mode citations are inaccurate ─────────────────────
#   PROBLEM:
#     get_citations() returned one citation for every chunk in the retrieval
#     result, even when 5 expanded chunks all came from the same parent
#     passage.  The LLM may only have used 2 of the 5 contexts but all 5
#     showed up as sources.  Additionally, because expand_to_parents() swaps
#     content for parent_content but leaves all other metadata as the child's,
#     the page reported in the citation was the child's page, not necessarily
#     the page that contained the answer.
#
#   FIX (short-term, effective immediately):
#     Deduplicate citations on parent_id — same parent passage → one citation.
#     This eliminates the multiple-child-same-parent duplication and reduces
#     noise significantly.  The page shown is still the child's page (the
#     start of the parent passage) which is correct for the majority of cases.
#
# ── BUG 3 FIX — Offline mode shows unreadable 300-char child fragments ───
#   PROBLEM:
#     The offline branch of stream() built OfflineChunk objects using
#     c.get("content", "") which is the 300-character child fragment.
#     c["parent_content"] — the full 1500-char readable passage — was ignored.
#
#   FIX:
#     Use  c.get("parent_content") or c.get("content", "")
#     The `or` fallback handles atomic chunks (tables, images) where
#     parent_content == content, so there is no regression for those types.
# ──────────────────────────────────────────────────────────────────────────

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
        """
        Return one citation per unique parent passage (deduplicated on parent_id).

        BUG 2 FIX:
        Previously every chunk in the retrieval result produced its own
        citation, so a single parent passage retrieved via 3 different child
        chunks would appear three times.  Now we track seen parent_ids and
        only emit the first citation for each unique parent.

        Fallback key: if parent_id is absent (e.g. naive / atomic chunks),
        we fall back to (source, page) to avoid duplication there too.
        """
        citations: list[dict] = []
        seen: set[str]        = set()

        for chunk in self.retrieval.get_chunks():
            # Build a deduplication key — prefer parent_id for hierarchical
            # chunks; fall back to source+page for other chunker strategies.
            dedup_key = chunk.get("parent_id") or (
                f"{chunk.get('source', '')}|{chunk.get('page', '')}"
            )

            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            citations.append({
                "source"      : chunk.get("source",       "unknown"),
                "page"        : chunk.get("page",          "?"),
                "heading"     : chunk.get("heading",       ""),
                "section_path": chunk.get("section_path",  ""),
                "type"        : chunk.get("type",          "text"),
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
        Retrieve children → Rerank children → Expand to parents → LLM stream
    Offline (is_online=False):
        Retrieve children (no rerank, no expand) → OfflineQueryResponse with chunk cards
        No LLM call, no SSE — just JSON with precise manual excerpts.

    A4 KEY CHANGE:
        _retrieve() now separates reranking from parent expansion.
        Reranker runs on 300-token children (precise), then expansion
        gives the LLM 1500-token parent passages (context). Both win.
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
        print(f"  [RAG CHAIN] Reranker  : {'✅ children first (A4)' if use_reranker else '❌'}")
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
        Run retrieval with the correct child-first / parent-expand order.
 
        OFFLINE mode:
            retrieve() → child chunks (300 tok) → return directly
            No reranking (CPU-intensive), no parent expansion.
 
        ONLINE mode (A4 fix):
            retrieve() → child chunks (300 tok)
            [optional] rerank children → precise cross-encoder signal
            expand_to_parents()        → 1500-tok passages for LLM context
 
        PHASE 4 CHANGE:
            store=rag_service.get_vector_store() passed to retriever.retrieve()
            so online calls use the cloud store and offline calls use local,
            without rebuilding the chain on network state changes.
        """
        # ── Phase 4: resolve active store on every call ────────────────────
        # Imported inside the method to avoid circular import
        # (rag_service imports RAGChain at module level).
        import services.rag_service as _rag_svc
        active_store = _rag_svc.get_vector_store()
 
        retrieval = self.retriever.retrieve(
            question,
            filter_field = "source" if self._source_filter else None,
            filter_value = self._source_filter,
            is_offline   = is_offline,
            store        = active_store,              # ← the only new argument
        )
 
        # ── OFFLINE: return child chunks directly ──────────────────────────
        if is_offline:
            print(f"  [RAG CHAIN] Offline — returning {len(retrieval)} child chunks")
            return retrieval
 
        # ── ONLINE: rerank children first (A4) ────────────────────────────
        if self.use_reranker and self.reranker and len(retrieval) > 0:
            print(f"  [RAG CHAIN] Reranking {len(retrieval)} children...")
            retrieval = self.reranker.rerank(
                query     = question,
                retrieval = retrieval,
                top_k     = self.rerank_top_k,
            )
 
        # ── Expand top-N children to parent passages ───────────────────────
        if hasattr(self.retriever, "expand_to_parents"):
            retrieval = self.retriever.expand_to_parents(retrieval)
        else:
            print("  [RAG CHAIN] expand_to_parents not available — skipping")
 
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
            # _retrieve with is_offline=True returns child chunks directly
            retrieval = self._retrieve(question, is_offline=True)
            chunks    = retrieval.get_chunks()[:offline_top_k]

            offline_chunks = [
                OfflineChunk(
                    source       = c.get("source", "unknown"),
                    page         = c.get("page"),
                    heading      = c.get("heading", ""),
                    section_path = c.get("section_path", ""),
                    # ── BUG 3 FIX ─────────────────────────────────────────
                    # Previously used c.get("content") which is the raw 300-char
                    # child fragment — unreadable mid-sentence snippets.
                    # parent_content is the full 1500-char readable passage that
                    # the hierarchical chunker stored inline on every child dict.
                    # The `or` fallback handles atomic chunks (tables/images)
                    # where parent_content == content — no regression there.
                    content      = c.get("parent_content") or c.get("content", ""),
                    score        = round(float(c.get("score", 0.0)), 4),
                    chunk_type   = c.get("type", "text"),
                    bbox         = c.get("bbox"),
                    page_width   = c.get("page_width"),
                    page_height  = c.get("page_height"),
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

        # Retrieve + rerank children + expand to parents (online — A4)
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