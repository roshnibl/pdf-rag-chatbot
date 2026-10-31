"""Deterministic chunking with per-chunk metadata.

Cleans page text (strips repeated headers/footers, normalizes whitespace),
detects language, then splits into token-sized passages with overlap. Each
chunk carries full provenance: pdf_id, filename, page_number, chunk_index.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import LangDetectException, detect

import config
from pipeline.pdf_loader import DocumentRecord

_ENCODER = tiktoken.get_encoding(config.TOKENIZER_ENCODING)


@dataclass
class Chunk:
    """A single retrievable passage with provenance metadata."""

    id: str
    text: str
    pdf_id: str
    filename: str
    page_number: int
    chunk_index: int
    language: str

    def metadata(self) -> dict:
        """Metadata dict stored alongside the embedding in Chroma."""
        return {
            "pdf_id": self.pdf_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "language": self.language,
        }


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def _detect_header_footer_lines(pages: list[str]) -> set[str]:
    """Find lines that repeat across many pages (likely headers/footers).

    A line appearing on more than ~40% of pages is almost certainly boilerplate
    such as a running header, footer, or page-number band.
    """
    if len(pages) < 5:
        return set()

    first_last = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        # Only the top-2 and bottom-2 lines are header/footer candidates.
        for ln in lines[:2] + lines[-2:]:
            # Ignore pure page numbers; those are handled by the digit regex.
            if not ln.isdigit():
                first_last[ln] += 1

    threshold = max(3, int(0.4 * len(pages)))
    return {ln for ln, count in first_last.items() if count >= threshold}


def _clean_text(text: str, boilerplate: set[str]) -> str:
    """Remove boilerplate lines, stray page numbers, and normalize whitespace."""
    # NFKC folds compatibility characters (en/em-spaces, ligatures, full-width
    # forms) to canonical equivalents, improving embedding + display quality.
    text = unicodedata.normalize("NFKC", text)

    cleaned_lines = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped in boilerplate:
            continue
        # Drop lines that are just a page number.
        if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", stripped):
            continue
        cleaned_lines.append(stripped)

    joined = "\n".join(cleaned_lines)
    # Collapse 3+ newlines to 2, and runs of spaces to one.
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    return joined.strip()


def _detect_language(text: str) -> str:
    sample = text[:1000]
    try:
        return detect(sample) if sample.strip() else "unknown"
    except LangDetectException:
        return "unknown"


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Token-aware recursive splitter (deterministic for reproducibility)."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.TOKENIZER_ENCODING,
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def chunk_document(doc: DocumentRecord) -> list[Chunk]:
    """Clean, then chunk every page of one document, preserving page numbers."""
    page_texts = [p.text for p in doc.pages]
    boilerplate = _detect_header_footer_lines(page_texts)
    splitter = _build_splitter()

    chunks: list[Chunk] = []
    for page in doc.pages:
        cleaned = _clean_text(page.text, boilerplate)
        if not cleaned:
            continue

        language = _detect_language(cleaned)
        for local_idx, passage in enumerate(splitter.split_text(cleaned)):
            passage = passage.strip()
            # Skip trivially short fragments that carry no retrievable meaning.
            if _token_len(passage) < 20:
                continue

            chunk_index = len(chunks)
            chunk_id = f"{doc.pdf_id}_p{page.page_number}_c{local_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=passage,
                    pdf_id=doc.pdf_id,
                    filename=doc.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    language=language,
                )
            )

    return chunks


def chunk_documents(docs: list[DocumentRecord]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
