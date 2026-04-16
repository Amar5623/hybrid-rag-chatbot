# utils/image_captioner.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import pytesseract
from config import TESSERACT_PATH


# ── Point pytesseract at the local Tesseract install ──────────────────────────
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ─────────────────────────────────────────
# IMAGE CAPTIONER
# ─────────────────────────────────────────

class ImageCaptioner:
    """
    Extracts text from images using Tesseract OCR.

    In a RAG pipeline, images (charts, scanned pages, diagrams with labels)
    often contain useful text that would otherwise be lost. This class
    converts them into plain text chunks that can be embedded and retrieved.

    ✅ Fully offline  ✅ No API key  ✅ Works on screenshots, scans, figures
    ❌ Cannot describe visual content (shapes, colors) — text extraction only

    Requires:
        - Tesseract installed at TESSERACT_PATH (config.py)
        - pip install pytesseract pillow
    """

    # Tesseract config strings
    CONFIGS = {
        "default"   : "",                        # standard mixed content
        "single_col": "--psm 6",                 # single block of text
        "sparse"    : "--psm 11",                # sparse text, no order assumed
        "digits"    : "--psm 6 outputbase digits", # numbers only
    }

    def __init__(self, lang: str = "eng", mode: str = "default"):
        """
        Args:
            lang : Tesseract language code (default: 'eng')
            mode : one of 'default', 'single_col', 'sparse', 'digits'
        """
        self.lang   = lang
        self.config = self.CONFIGS.get(mode, "")
        self._verify_tesseract()
        print(f"  [CAPTIONER] ImageCaptioner ready. lang={lang} | mode={mode}")

    def _verify_tesseract(self) -> None:
        """Check Tesseract is accessible, warn if not."""
        try:
            version = pytesseract.get_tesseract_version()
            print(f"  [CAPTIONER] Tesseract version: {version}")
        except Exception as e:
            print(f"  [CAPTIONER] ⚠️  Tesseract not found: {e}")
            print(f"  [CAPTIONER]    Check TESSERACT_PATH in config.py")

    # ── CORE ─────────────────────────────────

    def extract_text(self, image_path: str) -> str:
        """
        Extract all text from an image file.

        Args:
            image_path : path to image (.png, .jpg, .jpeg, .tiff, .bmp)

        Returns:
            Extracted text string (stripped). Empty string if no text found.
        """
        if not os.path.exists(image_path):
            print(f"  [CAPTIONER] ⚠️  File not found: {image_path}")
            return ""

        try:
            image = Image.open(image_path)
            image = self._preprocess(image)
            text  = pytesseract.image_to_string(
                image,
                lang   = self.lang,
                config = self.config,
            )
            return text.strip()
        except Exception as e:
            print(f"  [CAPTIONER] ⚠️  Error processing {image_path}: {e}")
            return ""

    def extract_text_from_pil(self, image: Image.Image) -> str:
        """
        Extract text from an already-loaded PIL Image object.
        Useful when images are extracted from PDFs in memory.
        """
        try:
            image = self._preprocess(image)
            return pytesseract.image_to_string(
                image,
                lang   = self.lang,
                config = self.config,
            ).strip()
        except Exception as e:
            print(f"  [CAPTIONER] ⚠️  Error processing PIL image: {e}")
            return ""

    def extract_with_confidence(self, image_path: str) -> dict:
        """
        Extract text with per-word confidence scores.

        Returns:
            {
                "text"      : str,           ← full extracted text
                "words"     : list[str],     ← individual words
                "confidences": list[float],  ← per-word confidence (0-100)
                "avg_confidence": float,     ← mean confidence
            }
        """
        if not os.path.exists(image_path):
            return {"text": "", "words": [], "confidences": [], "avg_confidence": 0.0}

        try:
            image = Image.open(image_path)
            image = self._preprocess(image)
            data  = pytesseract.image_to_data(
                image,
                lang        = self.lang,
                config      = self.config,
                output_type = pytesseract.Output.DICT,
            )

            words       = []
            confidences = []
            for word, conf in zip(data["text"], data["conf"]):
                word = word.strip()
                if word and int(conf) > 0:
                    words.append(word)
                    confidences.append(float(conf))

            full_text      = " ".join(words)
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            return {
                "text"          : full_text,
                "words"         : words,
                "confidences"   : confidences,
                "avg_confidence": avg_confidence,
            }
        except Exception as e:
            print(f"  [CAPTIONER] ⚠️  Error: {e}")
            return {"text": "", "words": [], "confidences": [], "avg_confidence": 0.0}

    def to_chunk(self, image_path: str, min_chars: int = 20) -> dict | None:
        """
        Convert image text into a RAG-ready chunk dict.
        Returns None if extracted text is too short to be useful.

        Args:
            image_path : path to image file
            min_chars  : minimum character count to keep the chunk

        Returns:
            {
                "content": str,
                "source" : str,   ← filename
                "page"   : None,
                "type"   : "image"
            }
            or None if text too short.
        """
        text = self.extract_text(image_path)

        if len(text) < min_chars:
            print(f"  [CAPTIONER] Skipping {os.path.basename(image_path)} — too little text ({len(text)} chars)")
            return None

        return {
            "content": text,
            "source" : os.path.basename(image_path),
            "page"   : None,
            "type"   : "image",
        }

    def batch_to_chunks(
        self,
        image_paths : list[str],
        min_chars   : int = 20,
    ) -> list[dict]:
        """
        Process multiple images and return a list of RAG-ready chunks.
        Skips images with too little text automatically.
        """
        chunks = []
        for path in image_paths:
            chunk = self.to_chunk(path, min_chars=min_chars)
            if chunk:
                chunks.append(chunk)
        print(f"  [CAPTIONER] Processed {len(image_paths)} images → {len(chunks)} chunks")
        return chunks

    # ── PREPROCESSING ────────────────────────

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Convert to grayscale and upscale small images.
        Simple but effective for improving OCR accuracy.
        """
        # Convert to grayscale (Tesseract works best with it)
        image = image.convert("L")

        # Upscale if image is very small — helps Tesseract accuracy
        w, h = image.size
        if w < 300 or h < 100:
            scale = max(300 / w, 100 / h, 2.0)
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

        return image

    def get_info(self) -> dict:
        return {
            "type"  : "ImageCaptioner",
            "backend": "tesseract-ocr",
            "lang"  : self.lang,
            "config": self.config,
        }


__all__ = ["ImageCaptioner"]