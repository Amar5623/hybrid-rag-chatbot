# ingestion/pdf_loader.py
#
# CHANGES vs previous version:
#   - bbox [x0, y0, x1, y1] now captured and stored on every text chunk.
#   - page_width and page_height stored on every text chunk.
#   - Table chunks get bbox from pdfplumber's find_tables() (replaces extract_tables()).
#   - Image chunks get bbox from PyMuPDF get_image_rects() — falls back to None.
#   - _make_chunk() docstring updated to document new fields.
#
#   WHY bbox MATTERS:
#     At ingestion time, PyMuPDF gives us the exact (x0, y0, x1, y1) rectangle
#     of every text block on the page — in PDF point coordinates (1pt = 1/72 inch).
#     If we don't store this NOW, we can never recover it without re-parsing the PDF.
#     The frontend PDF viewer (Person B, Day 2) will use these coordinates to draw
#     a highlight rectangle over the exact source text when a worker clicks
#     "Open in manual". This is the entire foundation of the PDF deep-link feature.
#
#   COORDINATE SYSTEM:
#     PDF coordinate origin is top-left (0, 0).
#     x0 = left edge of block
#     y0 = top edge of block
#     x1 = right edge of block
#     y1 = bottom edge of block
#     page_width / page_height: dimensions of the page in the same units.
#     The frontend uses (bbox / page_dimensions) to position highlights at any zoom.
#
#   TABLE CHANGE:
#     extract_tables() → find_tables() + table.extract()
#     find_tables() returns Table objects which have a .bbox property.
#     extract_tables() returns only the raw cell data with no position info.
#     Functionally identical for the text content, now also yields position.
#
#   BACKWARD COMPATIBILITY:
#     bbox, page_width, page_height default to None in _make_chunk() via chunk.update(extra).
#     Chunks that don't pass these (image chunks where bbox can't be recovered) simply
#     have None — the frontend handles this gracefully by navigating to the page
#     without a highlight overlay.

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
        heading, section_path, image_path, bbox, page_width, page_height, etc.

        Standard fields always present:
            content      : str   — the text content of this chunk
            page         : int   — 1-indexed page number
            type         : str   — "text" | "heading" | "bullet" | "table" | "image"
            source       : str   — filename of the source document
            heading      : str   — last heading seen before this block (may be "")
            section_path : str   — full breadcrumb e.g. "Ch3 > 3.1 > Results"

        Optional fields (set by PDFLoader, None if unavailable):
            bbox         : list[float] | None — [x0, y0, x1, y1] in PDF points
            page_width   : float | None       — page width in PDF points
            page_height  : float | None       — page height in PDF points

            These three are the foundation of the PDF deep-link highlight feature.
            Frontend uses:  highlight_x = bbox[0] / page_width * canvas_width
                            highlight_y = bbox[1] / page_height * canvas_height

        Other optional fields injected by specific loaders:
            image_path   : str  — absolute path to extracted image file
        """
        chunk = {
            "content"     : content,
            "page"        : page,
            "type"        : chunk_type,
            "source"      : self.file_name,
            "heading"     : "",
            "section_path": "",
            # NEW: bbox fields — default None, overridden by PDFLoader
            "bbox"        : None,
            "page_width"  : None,
            "page_height" : None,
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
      - Text   (via PyMuPDF) with full section breadcrumb + bullet detection + BBOX
      - Tables (via pdfplumber find_tables) — converted to markdown + BBOX
      - Images (via PyMuPDF)   — page context used as semantic description + BBOX attempt

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

    BBOX STORAGE (NEW):
    ────────────────────
    Every text block's PyMuPDF bbox → stored as [x0, y0, x1, y1].
    page_width and page_height stored alongside so the frontend can
    compute normalized positions at any zoom level.
    Coordinate system: origin at top-left, units in PDF points (1pt = 1/72 inch).

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

    # ── TEXT + HEADINGS + BULLETS + BBOX ─────────────────

    def _extract_text_with_structure(self) -> list[dict]:
        """
        Extract text blocks with:
          - Full section breadcrumb (section_stack)
          - Heading level detection (font-size ratio)
          - Bullet/list detection (_BULLET_RE)
          - BBOX [x0, y0, x1, y1] for each block  ← NEW
          - page_width, page_height for each block ← NEW

        The bbox is taken directly from blk["bbox"] in PyMuPDF's block dict.
        It's a fitz.Rect-compatible tuple (x0, y0, x1, y1) in PDF points.
        We round to 2 decimal places to keep payload size reasonable.

        page_width / page_height come from page.rect.width / .height.
        These are stored once per block (they're constant per page) so the
        frontend always has what it needs without a separate page-info lookup.
        """
        results: list[dict] = []
        doc = fitz.open(self.file_path)

        # Running section stack across pages
        section_stack: list[tuple[int, str]] = []

        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            blocks    = page_dict.get("blocks", [])

            # ── NEW: page dimensions ──────────────────────
            # Stored on every chunk from this page.
            # page.rect is a fitz.Rect(x0, y0, x1, y1) — for a standard page,
            # x0=0, y0=0, so width = rect.x1, height = rect.y1.
            pg_width  = round(page.rect.width,  2)
            pg_height = round(page.rect.height, 2)

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

                block_lines  : list[str] = []
                max_span_size: float     = 0.0
                has_bullet               = False

                for line in blk.get("lines", []):
                    line_text = ""
                    line_max  = 0.0

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

                # ── NEW: extract block bbox ───────────────
                # blk["bbox"] is (x0, y0, x1, y1) in PDF points.
                # Round to 2 dp — full float precision is unnecessary overhead.
                raw_bbox  = blk.get("bbox")
                block_bbox = (
                    [round(v, 2) for v in raw_bbox]
                    if raw_bbox and len(raw_bbox) == 4
                    else None
                )

                # ── Classify block ──────────────────────────
                if max_span_size >= heading_threshold and len(content) < 200:
                    block_type = "heading"
                    level      = self._heading_level(max_span_size, body_size)
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
                        # ── NEW ──────────────────────────
                        bbox         = block_bbox,
                        page_width   = pg_width,
                        page_height  = pg_height,
                    )
                )

        doc.close()
        print(f"  [TEXT]  {len(results)} text/heading/bullet blocks (bbox captured)")
        return results

    # ── TABLES + BBOX ─────────────────────────────────────

    def _extract_tables(self) -> list[dict]:
        """
        Extract tables using pdfplumber.

        CHANGED: extract_tables() → find_tables() + table.extract()
        find_tables() returns Table objects which expose .bbox.
        extract_tables() only returns raw cell data — no position info.

        Table.bbox is (x0, top, x1, bottom) in pdfplumber's coordinate system
        (same as PDF points, origin top-left) — directly compatible with our
        chunk bbox format [x0, y0, x1, y1].

        If pdfplumber is not installed or find_tables() fails, falls back
        gracefully (bbox = None, content extraction still attempted).
        """
        results: list[dict] = []

        if not _HAS_PDFPLUMBER:
            print("  [TABLES] pdfplumber not installed — skipping tables")
            return results

        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):

                    # CHANGED: find_tables() gives us Table objects with .bbox
                    # table.extract() gives the same cell data as extract_tables()
                    try:
                        tables = page.find_tables()
                    except Exception:
                        tables = []

                    for t_idx, table in enumerate(tables):
                        try:
                            raw_data = table.extract()
                        except Exception:
                            continue

                        md = self._table_to_markdown(raw_data)
                        if not md:
                            continue

                        content = f"[TABLE {t_idx+1} — Page {page_num+1}]\n{md}"

                        # ── NEW: table bbox ───────────────────────────
                        # table.bbox → (x0, top, x1, bottom) in PDF points
                        # Directly maps to our [x0, y0, x1, y1] format.
                        table_bbox = None
                        if hasattr(table, "bbox") and table.bbox:
                            try:
                                table_bbox = [round(v, 2) for v in table.bbox]
                            except (TypeError, ValueError):
                                table_bbox = None

                        # page dimensions from pdfplumber
                        pg_width  = round(float(page.width),  2) if page.width  else None
                        pg_height = round(float(page.height), 2) if page.height else None

                        results.append(
                            self._make_chunk(
                                content     = content,
                                page        = page_num + 1,
                                chunk_type  = "table",
                                bbox        = table_bbox,   # NEW
                                page_width  = pg_width,     # NEW
                                page_height = pg_height,    # NEW
                            )
                        )

        except Exception as e:
            print(f"  [TABLES] Extraction error: {e}")

        print(f"  [TABLES] {len(results)} tables extracted (bbox captured where available)")
        return results

    # ── IMAGES + BBOX ─────────────────────────────────────

    def _extract_images(self) -> list[dict]:
        """
        Extract images and create searchable chunks — NO Tesseract.

        CHANGED: now attempts to recover the on-page bbox for each image
        via page.get_image_rects(xref). This returns the list of rectangles
        where the image is drawn on the page. We take the first (largest) one.

        If get_image_rects() fails or returns nothing, bbox stays None.
        The frontend handles None bbox by navigating to the page without
        drawing a highlight — still useful (worker goes to right page).

        Content priority (unchanged):
          1. Page context  (first 300 chars of surrounding page text)
          2. Generic label (absolute fallback)
        """
        os.makedirs(self.image_output_dir, exist_ok=True)
        results: list[dict] = []
        doc     = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            page_text    = page.get_text("text").strip()
            page_context = page_text[:300].replace("\n", " ").strip()

            # Page dimensions — same as text extractor
            pg_width  = round(page.rect.width,  2)
            pg_height = round(page.rect.height, 2)

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

                    # ── NEW: image bbox ───────────────────────────────
                    # get_image_rects(xref) returns list of fitz.Rect where
                    # this image xref is placed on the page.
                    # We take the first one (images rarely appear multiple times).
                    # Falls back to None if unavailable (older PyMuPDF versions
                    # or embedded images with complex placement transforms).
                    img_bbox = None
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            r        = rects[0]
                            img_bbox = [round(r.x0, 2), round(r.y0, 2),
                                        round(r.x1, 2), round(r.y1, 2)]
                    except Exception:
                        img_bbox = None

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
                            content     = content,
                            page        = page_num + 1,
                            chunk_type  = "image",
                            image_path  = img_path,
                            bbox        = img_bbox,     # NEW
                            page_width  = pg_width,     # NEW
                            page_height = pg_height,    # NEW
                        )
                    )

                except Exception as e:
                    print(f"  [IMAGE]  Skipped p{page_num+1} i{img_idx+1}: {e}")

        doc.close()
        print(f"  [IMAGES] {len(results)} images extracted (bbox attempted for all)")
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