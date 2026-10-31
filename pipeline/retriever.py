"""Retrieval + optional cross-encoder reranking.

Embeds the query, pulls TOP_K_RETRIEVE candidates from Chroma via the HNSW
index, then optionally reranks them with a lightweight cross-encoder before
returning the TOP_K_FINAL passages that get sent to the LLM.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

import config
from pipeline.embedder import embed_query, get_chroma_collection

_reranker_lock = threading.Lock()
_reranker: CrossEncoder | None = None


@dataclass
class RetrievedChunk:
    """A retrieved passage with its scores and provenance."""

    text: str
    filename: str
    page_number: int
    pdf_id: str
    vector_score: float          # cosine similarity (1 = identical)
    rerank_score: float | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                # Let PyTorch use all CPU cores for faster rerank at query time.
                try:
                    import torch

                    torch.set_num_threads(config.EMBEDDING_THREADS)
                except Exception:
                    pass
                _reranker = CrossEncoder(
                    config.RERANKER_MODEL, device=config.EMBEDDING_DEVICE
                )
    return _reranker


def retrieve(query: str, top_k_final: int | None = None) -> list[RetrievedChunk]:
    """Return the most relevant passages for a query.

    Pipeline: embed query -> HNSW search (TOP_K_RETRIEVE) -> optional rerank ->
    truncate to top_k_final.
    """
    top_k_final = top_k_final or config.TOP_K_FINAL
    collection = get_chroma_collection()

    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=config.TOP_K_RETRIEVE,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates: list[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, distances):
        candidates.append(
            RetrievedChunk(
                text=text,
                filename=meta.get("filename", "unknown"),
                page_number=int(meta.get("page_number", 0)),
                pdf_id=meta.get("pdf_id", "unknown"),
                # Chroma cosine distance -> similarity.
                vector_score=round(1.0 - dist, 4),
            )
        )

    if not candidates:
        return []

    if config.USE_RERANKER:
        reranker = _get_reranker()
        pairs = [(query, c.text) for c in candidates]
        scores = reranker.predict(pairs)
        for cand, score in zip(candidates, scores):
            cand.rerank_score = round(float(score), 4)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    else:
        candidates.sort(key=lambda c: c.vector_score, reverse=True)

    return candidates[:top_k_final]
