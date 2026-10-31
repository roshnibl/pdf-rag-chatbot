"""One-shot ingestion: PDFs -> text/OCR -> chunks -> embeddings -> Chroma.

Run this once (or whenever the corpus changes) before launching the app:

    python ingest.py

It is deterministic and idempotent: the Chroma collection is reset and rebuilt
from scratch each run, so embeddings are always consistent with the chunking
rules in config.py.
"""
from __future__ import annotations

import time

import config
from pipeline.chunker import chunk_documents
from pipeline.embedder import persist_chunks
from pipeline.pdf_loader import load_all_pdfs


def main():
    start = time.time()

    print(f"[1/3] Loading PDFs from {config.PDF_DIR} ...")
    docs = load_all_pdfs()
    total_pages = sum(len(d.pages) for d in docs)
    total_ocr = sum(d.ocr_page_count for d in docs)
    print(f"      {len(docs)} PDFs, {total_pages} pages ({total_ocr} via OCR)")
    for d in docs:
        print(f"        - {d.filename}: {len(d.pages)} pages")

    print("[2/3] Cleaning + chunking ...")
    chunks = chunk_documents(docs)
    print(f"      {len(chunks)} chunks created")

    print(f"[3/3] Embedding with {config.EMBEDDING_MODEL} + persisting to Chroma ...")
    count = persist_chunks(chunks)

    elapsed = time.time() - start
    print("\nIngestion complete.")
    print(f"  Chunks indexed : {count}")
    print(f"  Vector DB path : {config.CHROMA_DIR}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print("\nLaunch the chatbot with:  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
