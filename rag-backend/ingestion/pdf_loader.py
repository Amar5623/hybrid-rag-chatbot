# ingestion/pdf_loader.py
#
# CHANGES vs original:
#   - Tesseract OCR removed entirely (was too heavy, fragile on low-end machines)
#   - Images now use page context as semantic description — same search quality
#   - Heading detection upgraded: full section_path breadcrumb
#     e.g. "Chapter 3 > 3.1 Methods > Results" instead of just last heading
#   - Bullet / numbered list detection — avoids merging list items into blob text
#   - pdfplumber still used for tables only (it's genuinely better for this)
#   - _make_chunk signature unchanged — all other loaders inherit BaseLoader unmodified

import os
import re
import sys
import fitz          # PyMuPDF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGES_DIR

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

# ── Heading scale factor ──────────────────────────────────
# A span whose font size ≥ body_size * this is treated as heading
_HEADING_SCALE = 1.15

# ── Bullet patterns ───────────────────────────────────────
_BULLET_RE = re.compile(
    r"^(\s*[•·‣▸▶◦▪▫●○◆◇►]\s+"    # unicode bullets
    r"|\s*[-*+]\s+"                   # markdown-style
    r"|\s*\d{1,2}[.)]\s+"            # numbered  1. 2) 10.
    r"|\s*[a-zA-Z][.)]\s+"           # alpha     a. b)
    r")"
)

# Minimum image size in bytes — skip icons / decorative elements
_MIN_IMAGE_BYTES = 5_000


# ─────────────────────────────────────────────────────────
# BASE LOADER  (unchanged — all other loaders inherit this)
# ─────────────────────────────────────────────────────────

class BaseLoader:
    """
    Abstract base for all document loaders.
    Provides _make_chunk() helper used by every subclass.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.chunks    = []

    def load(self) -> list[dict]:
        raise NotImplementedError("Subclasses must implement load()")

    def _make_chunk(self, content: str, page: int, chunk_type: str, **extra) -> dict:
        """
        Standard chunk format used across all loaders.
        Extra keyword args are merged directly — lets subclasses inject
        heading, section_path, image_path, row_range, sheet_name, etc.
        """
        chunk = {
            "content"     : content,
            "page"        : page,
            "type"        : chunk_type,
            "source"      : self.file_name,
            "heading"     : "",          # overridden by PDF loader
            "section_path": "",          # NEW: full breadcrumb
        }
        chunk.update(extra)
        return chunk

    def get_summary(self) -> str:
        types = {}
        for c in self.chunks:
            types[c["type"]] = types.get(c["type"], 0) + 1
        return f"📄 {self.file_name} → {len(self.chunks)} chunks {types}"


# ─────────────────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────────────────

class PDFLoader(BaseLoader):
    """
    Loads PDF files and extracts:
      - Text   (via PyMuPDF) with full section breadcrumb + bullet detection
      - Tables (via pdfplumber) — converted to markdown
      - Images (via PyMuPDF)   — page context used as semantic description
                                  No Tesseract OCR required

    SECTION BREADCRUMB (section_path):
    ───────────────────────────────────
    Instead of storing just the last heading, we maintain a stack of
    (level, heading_text) pairs as we walk the document.
    section_path = "Chapter 3 > 3.1 Methods > Results"
    This lets the LLM and UI provide richer source attribution.

    HEADING LEVEL DETECTION:
    ─────────────────────────
    We map font-size ratios to heading levels 1-3:
        ratio ≥ 1.8  →  H1
        ratio ≥ 1.4  →  H2
        ratio ≥ 1.15 →  H3  (body * HEADING_SCALE)

    BULLET DETECTION:
    ──────────────────
    Lines matching _BULLET_RE are tagged type="bullet" instead of "text".
    This prevents bullet lists from being merged into prose blobs.

    IMAGE SEMANTICS WITHOUT OCR:
    ──────────────────────────────
    Each image chunk is assigned a content string built from:
      1. Page context  (first 300 chars of surrounding page text)
      2. Generic label (fallback if page has no text)
    This gives semantic meaning to diagrams without Tesseract.
    """

    HEADING_SCALE_FACTOR = _HEADING_SCALE

    def __init__(
        self,
        file_path        : str,
        extract_images   : bool = True,
        image_output_dir : str  = IMAGES_DIR,
    ):
        super().__init__(file_path)
        self.extract_images   = extract_images
        self.image_output_dir = image_output_dir

    # ── PUBLIC ────────────────────────────────────────────

    def load(self) -> list[dict]:
        """Master method — runs all extractors and returns unified chunks."""
        print(f"\n📄 Loading PDF: {self.file_name}")

        self.chunks = []
        self.chunks.extend(self._extract_text_with_structure())
        self.chunks.extend(self._extract_tables())

        if self.extract_images:
            self.chunks.extend(self._extract_images())

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    # ── TEXT + HEADINGS + BULLETS ─────────────────────────

    def _extract_text_with_structure(self) -> list[dict]:
        """
        Extract text with full section breadcrumb and bullet detection.

        Section stack: [(level, heading_text), ...]
        When a new heading is found, all stack entries at the same or deeper
        level are popped, then the new heading is pushed.
        """
        results: list[dict] = []
        doc = fitz.open(self.file_path)

        # Running section stack across pages
        section_stack: list[tuple[int, str]] = []

        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            blocks    = page_dict.get("blocks", [])

            # Collect all non-empty font sizes to compute median (body) size
            all_sizes: list[float] = []
            for blk in blocks:
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            all_sizes.append(span["size"])

            if not all_sizes:
                continue

            all_sizes.sort()
            body_size         = all_sizes[len(all_sizes) // 2]
            heading_threshold = body_size * self.HEADING_SCALE_FACTOR

            for blk in blocks:
                if blk.get("type") != 0:   # 0 = text block
                    continue

                block_lines: list[str] = []
                max_span_size: float   = 0.0
                has_bullet             = False

                for line in blk.get("lines", []):
                    line_text  = ""
                    line_max   = 0.0

                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t.strip():
                            line_text += t
                            if span["size"] > line_max:
                                line_max = span["size"]

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    if _BULLET_RE.match(line_text):
                        has_bullet = True

                    block_lines.append(line_text)
                    if line_max > max_span_size:
                        max_span_size = line_max

                content = "\n".join(block_lines).strip()
                if not content:
                    continue

                # ── Classify block ──────────────────────────
                if max_span_size >= heading_threshold and len(content) < 200:
                    # Heading — update the section stack
                    block_type = "heading"
                    level      = self._heading_level(max_span_size, body_size)
                    # Pop stack entries at same or deeper level
                    section_stack = [
                        (lvl, txt)
                        for lvl, txt in section_stack
                        if lvl < level
                    ]
                    section_stack.append((level, content))
                elif has_bullet:
                    block_type = "bullet"
                else:
                    block_type = "text"

                section_path = " > ".join(txt for _, txt in section_stack)
                heading      = section_stack[-1][1] if section_stack else ""

                results.append(
                    self._make_chunk(
                        content      = content,
                        page         = page_num + 1,
                        chunk_type   = block_type,
                        heading      = heading,
                        section_path = section_path,
                    )
                )

        doc.close()
        print(f"  [TEXT]  {len(results)} text/heading/bullet blocks")
        return results

    # ── TABLES ────────────────────────────────────────────

    def _extract_tables(self) -> list[dict]:
        """Extract tables using pdfplumber → convert to markdown."""
        results: list[dict] = []

        if not _HAS_PDFPLUMBER:
            print("  [TABLES] pdfplumber not installed — skipping tables")
            return results

        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for t_idx, table in enumerate(page.extract_tables()):
                        md = self._table_to_markdown(table)
                        if md:
                            content = f"[TABLE {t_idx+1} — Page {page_num+1}]\n{md}"
                            results.append(
                                self._make_chunk(
                                    content    = content,
                                    page       = page_num + 1,
                                    chunk_type = "table",
                                )
                            )
        except Exception as e:
            print(f"  [TABLES] Extraction error: {e}")

        print(f"  [TABLES] {len(results)} tables extracted")
        return results

    # ── IMAGES ────────────────────────────────────────────

    def _extract_images(self) -> list[dict]:
        """
        Extract images and create searchable chunks — NO Tesseract.

        Content priority:
          1. Page context (surrounding text as semantic description)
          2. Generic label (absolute fallback)

        Why no OCR?
          Tesseract is ~400MB, needs system install, is slow on low-end hardware,
          and returns empty text for diagrams/figures anyway. Page context gives
          equal or better semantic retrieval for most use cases.
        """
        os.makedirs(self.image_output_dir, exist_ok=True)
        results: list[dict] = []
        doc     = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            # Page text used as semantic description for figures on this page
            page_text    = page.get_text("text").strip()
            page_context = page_text[:300].replace("\n", " ").strip()

            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]
                    img_ext    = base_image.get("ext", "png")

                    if len(img_bytes) < _MIN_IMAGE_BYTES:
                        continue   # skip tiny icons / decorative elements

                    stem         = os.path.splitext(self.file_name)[0]
                    img_filename = f"{stem}_p{page_num+1}_i{img_idx+1}.{img_ext}"
                    img_path     = os.path.abspath(
                        os.path.join(self.image_output_dir, img_filename)
                    )
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    if page_context:
                        content = (
                            f"[IMAGE — Page {page_num+1}, Figure {img_idx+1}] "
                            f"Figure on this page. Page context: {page_context}"
                        )
                    else:
                        content = (
                            f"[IMAGE — Page {page_num+1}, Figure {img_idx+1}] "
                            f"Visual figure or diagram from {self.file_name}"
                        )

                    results.append(
                        self._make_chunk(
                            content    = content,
                            page       = page_num + 1,
                            chunk_type = "image",
                            image_path = img_path,
                        )
                    )

                except Exception as e:
                    print(f"  [IMAGE]  Skipped p{page_num+1} i{img_idx+1}: {e}")

        doc.close()
        print(f"  [IMAGES] {len(results)} images extracted (no OCR)")
        return results

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _heading_level(font_size: float, body_size: float) -> int:
        """Estimate heading depth from font-size ratio."""
        ratio = font_size / body_size
        if ratio >= 1.8:
            return 1
        if ratio >= 1.4:
            return 2
        return 3

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        """Convert pdfplumber raw table (list of lists) → markdown string."""
        if not table:
            return ""
        cleaned = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]
        cleaned = [row for row in cleaned if any(c for c in row)]
        if len(cleaned) < 2:
            return ""
        header    = "| " + " | ".join(cleaned[0]) + " |"
        separator = "|" + "|".join(["---"] * len(cleaned[0])) + "|"
        rows      = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
        return "\n".join([header, separator] + rows)


__all__ = ["BaseLoader", "PDFLoader"]