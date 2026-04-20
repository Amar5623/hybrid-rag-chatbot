# ingestion/chunker.py
#
# CHANGES vs previous version:
#   - HierarchicalChunker.chunk_hierarchical() embeds parent_content
#     directly on each child chunk dict (no separate parents dict / SQLite).
#   - Return signature of chunk_hierarchical() changed:
#       OLD: (children: list[dict], parents: dict)
#       NEW: children: list[dict]   (parents embedded inline)
#
# ── BUG 1 FIX ─────────────────────────────────────────────────────────────
#   _group_by_section() now also starts a new group whenever the PAGE NUMBER
#   changes, in addition to the existing section_path / heading triggers.
#
#   ROOT CAUSE:
#     When PDF heading detection fails (very common for ship manuals that use
#     consistent font sizes with no clear size jumps), every block comes out
#     with section_path="" and type="text".  Neither the heading nor the
#     section_path trigger ever fires, so the ENTIRE document landed in one
#     giant group.  That group's meta_base was stamped with group[0]["page"]
#     (= page 1), so every child chunk across the whole PDF reported page 1.
#
#   FIX:
#     Add a third OR-condition: `block.get("page") != current[-1].get("page")`
#     This guarantees that blocks from different pages are always placed in
#     separate groups, even if heading detection completely fails.
#     Result: each chunk now carries the correct page number for its content.
# ──────────────────────────────────────────────────────────────────────────

import os
import sys
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
)

# Atomic types — must never be re-chunked (splitting destroys their meaning)
_ATOMIC_TYPES = {"table", "image"}


# ─────────────────────────────────────────────────────────
# BASE CHUNKER
# ─────────────────────────────────────────────────────────

class BaseChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = "base"

    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement chunk()")

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        result: list[dict] = []
        for doc in docs:
            if doc.get("type") in _ATOMIC_TYPES:
                doc["chunk_index"]  = 0
                doc["total_chunks"] = 1
                doc["strategy"]     = "none"
                result.append(doc)
                continue

            sub_chunks = self.chunk(doc["content"])
            total      = len(sub_chunks)

            for i, sub in enumerate(sub_chunks):
                new_doc                 = doc.copy()
                new_doc["content"]      = sub
                new_doc["chunk_index"]  = i
                new_doc["total_chunks"] = total
                new_doc["strategy"]     = self.strategy_name
                result.append(new_doc)

        return result

    def get_stats(self, chunks: list[str]) -> dict:
        if not chunks:
            return {}
        lengths = [len(c) for c in chunks]
        return {
            "strategy"    : self.strategy_name,
            "total_chunks": len(chunks),
            "avg_length"  : round(sum(lengths) / len(lengths)),
            "min_length"  : min(lengths),
            "max_length"  : max(lengths),
        }


# ─────────────────────────────────────────────────────────
# STRATEGY 1 — FIXED SIZE
# ─────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "fixed_size"
        self._splitter = CharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separator     = "\n"
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────
# STRATEGY 2 — RECURSIVE
# ─────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "recursive"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────
# STRATEGY 3 — HIERARCHICAL PARENT-CHILD  (RECOMMENDED)
# ─────────────────────────────────────────────────────────

class HierarchicalChunker(BaseChunker):
    """
    Small-to-big retrieval: child chunks (300 chars) embedded into Qdrant
    for precise retrieval; parent text (1200 chars) stored directly on the
    child as parent_content metadata field.

    CHANGE vs original:
      chunk_hierarchical() now returns ONLY children (list[dict]).
      Each child carries parent_content inline — no separate parents dict,
      no SQLite parent store needed.

      At retrieval time HybridRetriever reads chunk["parent_content"]
      directly from the Qdrant payload — zero extra DB round-trip.
    """

    def __init__(
        self,
        child_size    : int = CHILD_CHUNK_SIZE,
        child_overlap : int = CHILD_CHUNK_OVERLAP,
        parent_size   : int = PARENT_CHUNK_SIZE,
        parent_overlap: int = PARENT_CHUNK_OVERLAP,
    ):
        super().__init__(child_size, child_overlap)
        self.strategy_name  = "hierarchical"
        self.child_size     = child_size
        self.child_overlap  = child_overlap
        self.parent_size    = parent_size
        self.parent_overlap = parent_overlap

        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = child_size,
            chunk_overlap = child_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = parent_size,
            chunk_overlap = parent_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )

    def chunk(self, text: str) -> list[str]:
        return [c.strip() for c in self._child_splitter.split_text(text) if c.strip()]

    def chunk_hierarchical(self, blocks: list[dict]) -> list[dict]:
        """
        Full hierarchical chunking pipeline.

        Args:
            blocks : structured blocks from any loader

        Returns:
            children : list[dict]
                Each child dict has all metadata PLUS:
                  - parent_content : str  — the larger parent passage
                  - parent_id      : str  — stable hash ID (kept for reference)
        """
        children: list[dict] = []

        text_blocks   = [b for b in blocks if b.get("type") not in _ATOMIC_TYPES]
        atomic_blocks = [b for b in blocks if b.get("type")     in _ATOMIC_TYPES]

        # ── 1. Text / heading / bullet blocks ─────────────
        groups = self._group_by_section(text_blocks)

        for g_idx, group in enumerate(groups):
            combined  = "\n\n".join(b["content"] for b in group)
            meta_base = {
                "source"      : group[0]["source"],
                "page"        : group[0]["page"],
                "type"        : group[0].get("type", "text"),
                "heading"     : group[0].get("heading", ""),
                "section_path": group[0].get("section_path", ""),
            }

            parent_texts = self._parent_splitter.split_text(combined)

            for p_idx, parent_text in enumerate(parent_texts):
                parent_id   = self._make_parent_id(
                    meta_base["source"],
                    meta_base["page"],
                    meta_base["section_path"],
                    g_idx * 1000 + p_idx,
                )
                child_texts = self._child_splitter.split_text(parent_text)
                total_c     = len(child_texts)

                for c_idx, child_text in enumerate(child_texts):
                    if not child_text.strip():
                        continue
                    children.append({
                        **meta_base,
                        "content"       : child_text,
                        "parent_content": parent_text,   # ← inline parent
                        "parent_id"     : parent_id,
                        "chunk_index"   : c_idx,
                        "total_chunks"  : total_c,
                        "strategy"      : self.strategy_name,
                    })

        # ── 2. Atomic blocks — self-contained ─────────────
        for a_idx, block in enumerate(atomic_blocks):
            content = block.get("content", "").strip()
            if not content:
                continue

            parent_id = self._make_parent_id(
                block["source"],
                block["page"],
                block.get("section_path", ""),
                100_000 + a_idx,
            )

            children.append({
                **block,
                "parent_content": content,   # atomic: parent == child
                "parent_id"     : parent_id,
                "chunk_index"   : 0,
                "total_chunks"  : 1,
                "strategy"      : self.strategy_name,
            })

        print(
            f"  [CHUNKER] {len(children)} children "
            f"(parent_content embedded inline) from {len(blocks)} blocks"
        )
        return children

    # ── helpers ───────────────────────────────────────────

    @staticmethod
    def _make_parent_id(source: str, page: int, section: str, idx: int) -> str:
        raw = f"{source}|p{page}|{section}|{idx}"
        return "par_" + hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _group_by_section(blocks: list[dict]) -> list[list[dict]]:
        """
        Split a flat list of blocks into groups that will become separate
        parent passages.

        A new group is started when ANY of the following is true:
          1. The block is a heading  — explicit section start
          2. The section_path changes — heading breadcrumb changed
          3. The PAGE NUMBER changes  — ← BUG 1 FIX

        Condition 3 is the critical addition.  When PDF heading detection
        fails (no clear font-size jumps), conditions 1 and 2 never fire,
        causing the entire document to collapse into a single group.  All
        children of that group then inherit group[0]["page"] = 1.

        By also breaking on page changes we guarantee that blocks from
        different pages always end up in different groups, so each child
        chunk carries the correct page number even when section detection
        is completely unavailable.
        """
        if not blocks:
            return []

        groups: list[list[dict]] = []
        current: list[dict]      = [blocks[0]]

        for block in blocks[1:]:
            new_section = (
                block.get("type") == "heading"
                or block.get("section_path") != current[-1].get("section_path")
                or block.get("page")         != current[-1].get("page")   # ← BUG 1 FIX
            )
            if new_section:
                groups.append(current)
                current = [block]
            else:
                current.append(block)

        if current:
            groups.append(current)
        return groups


# ─────────────────────────────────────────────────────────
# CHUNKER FACTORY
# ─────────────────────────────────────────────────────────

class ChunkerFactory:
    STRATEGIES: dict[str, type[BaseChunker]] = {
        "hierarchical": HierarchicalChunker,
        "recursive"   : RecursiveChunker,
        "fixed"       : FixedSizeChunker,
    }

    @staticmethod
    def get(strategy: str = "hierarchical", **kwargs) -> BaseChunker:
        strategy = strategy.lower()
        if strategy not in ChunkerFactory.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(ChunkerFactory.STRATEGIES.keys())}"
            )
        return ChunkerFactory.STRATEGIES[strategy](**kwargs)

    @staticmethod
    def available_strategies() -> list[str]:
        return list(ChunkerFactory.STRATEGIES.keys())


__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "HierarchicalChunker",
    "ChunkerFactory",
]