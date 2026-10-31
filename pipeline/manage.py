"""Add or remove a single PDF to/from the live vector DB (UI-driven).

Unlike ingest.py (which resets and rebuilds the whole collection), these
operations are INCREMENTAL: add appends one PDF's chunks, remove deletes one
PDF's chunks by metadata — so the chatbot stays available throughout.
"""
from __future__ import annotations

import os

import config
from pipeline.chunker import chunk_document
from pipeline.embedder import embed_passages, get_chroma_collection
from pipeline.pdf_loader import load_pdf


def list_pdf_files() -> list[str]:
    """Raw PDF filenames currently in the corpus folder."""
    if not os.path.isdir(config.PDF_DIR):
        return []
    return sorted(f for f in os.listdir(config.PDF_DIR) if f.lower().endswith(".pdf"))


def ingest_pdf_file(path: str, batch_size: int = 256) -> tuple[int, int]:
    """Chunk + embed one PDF and append it to the collection.

    If a PDF with the same filename is already indexed, its old chunks are
    removed first so re-uploading cleanly replaces it. Returns (pages, chunks).
    """
    filename = os.path.basename(path)
    doc = load_pdf(path)
    chunks = chunk_document(doc)
    collection = get_chroma_collection()

    # Replace-on-reupload: clear any prior chunks for this file.
    try:
        collection.delete(where={"filename": filename})
    except Exception:
        pass

    if not chunks:
        return len(doc.pages), 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed_passages([c.text for c in batch]),
            metadatas=[c.metadata() for c in batch],
        )
    return len(doc.pages), len(chunks)


def remove_pdf(filename: str) -> None:
    """Delete a PDF's chunks from the collection and the file from disk."""
    collection = get_chroma_collection()
    collection.delete(where={"filename": filename})
    path = os.path.join(config.PDF_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
