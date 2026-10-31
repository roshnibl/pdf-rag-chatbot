"""PDF text extraction with OCR fallback.

Uses PyMuPDF for native text extraction and falls back to Tesseract OCR for
scanned pages or embedded images. Returns per-page records so that downstream
chunking can attach accurate page-level provenance to every passage.
"""
from __future__ import annotations

import functools
import io
import os
from dataclasses import dataclass, field

import pymupdf  # PyMuPDF 2.x (the old `fitz` module name is deprecated)
import pytesseract
from PIL import Image

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


@dataclass
class PageRecord:
    """One extracted page with provenance."""

    pdf_id: str
    filename: str
    page_number: int          # 1-indexed, human-friendly
    text: str
    ocr_used: bool = False


@dataclass
class DocumentRecord:
    """All pages extracted from a single PDF."""

    pdf_id: str
    filename: str
    pages: list[PageRecord] = field(default_factory=list)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.ocr_used)


@functools.cache
def _ocr_available() -> bool:
    """Check once whether Tesseract is callable; warn (once) if not."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print(
            "  [warn] Tesseract not found - skipping OCR, using native text only. "
            "Install Tesseract and set TESSERACT_CMD in .env to enable OCR for "
            "scanned pages."
        )
        return False


def _ocr_page(page: pymupdf.Page) -> str:
    """Render a page to an image and run Tesseract OCR on it."""
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract is not installed or not on PATH. Install it and/or set "
            "TESSERACT_CMD in your .env. See README for instructions."
        )


def load_pdf(path: str) -> DocumentRecord:
    """Extract every page of a single PDF, using OCR where native text is sparse.

    A page is sent to OCR when PyMuPDF returns fewer than OCR_CHAR_THRESHOLD
    characters, which reliably catches scanned pages and image-only pages.
    """
    filename = os.path.basename(path)
    pdf_id = os.path.splitext(filename)[0]
    doc = pymupdf.open(path)
    record = DocumentRecord(pdf_id=pdf_id, filename=filename)

    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text").strip()
            ocr_used = False

            # OCR only when native text is sparse AND Tesseract is available.
            if len(text) < config.OCR_CHAR_THRESHOLD and _ocr_available():
                ocr_text = _ocr_page(page).strip()
                # Keep whichever extraction yielded more content.
                if len(ocr_text) > len(text):
                    text = ocr_text
                    ocr_used = True

            record.pages.append(
                PageRecord(
                    pdf_id=pdf_id,
                    filename=filename,
                    page_number=index + 1,
                    text=text,
                    ocr_used=ocr_used,
                )
            )
    finally:
        doc.close()

    return record


def load_all_pdfs(pdf_dir: str = config.PDF_DIR) -> list[DocumentRecord]:
    """Load every *.pdf in a directory (sorted for deterministic ordering)."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    paths = sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )
    if not paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Add at least 10 PDFs (each >= 200 pages)."
        )

    return [load_pdf(p) for p in paths]
