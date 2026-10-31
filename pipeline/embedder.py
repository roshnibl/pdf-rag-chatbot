"""Embedding + vector-DB persistence.

Embeddings use fastembed (ONNX, int8-quantized bge-small) rather than the
PyTorch sentence-transformers backend — same model weights, 3-5x faster on CPU.
Vectors are L2-normalized and stored in a persistent Chroma collection with an
HNSW index (cosine space). The reranker (query-time) still uses
sentence-transformers; see retriever.py.
"""
from __future__ import annotations

import threading

import chromadb
import numpy as np
from chromadb.config import Settings
from fastembed import TextEmbedding

import config
from pipeline.chunker import Chunk

_model_lock = threading.Lock()
_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Lazily load and cache the fastembed model (thread-safe singleton)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(
                    model_name=config.EMBEDDING_MODEL,
                    threads=config.EMBEDDING_THREADS,
                )
    return _model


def _l2_normalize(vectors) -> np.ndarray:
    """Row-wise L2 normalization so Chroma cosine distance is well-defined."""
    arr = np.asarray(list(vectors), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_passages(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Embed document passages (no query instruction prefix)."""
    model = get_embedding_model()
    vectors = model.embed(texts, batch_size=batch_size)
    return _l2_normalize(vectors).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query, applying the bge query instruction prefix."""
    model = get_embedding_model()
    vectors = model.embed([config.QUERY_INSTRUCTION + query])
    return _l2_normalize(vectors)[0].tolist()


def get_chroma_collection():
    """Open (or create) the persistent Chroma collection with an HNSW index."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Delete and recreate the collection for a clean re-ingest."""
    client = chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def persist_chunks(chunks: list[Chunk], batch_size: int = 512) -> int:
    """Embed and write all chunks to Chroma. Returns the number persisted."""
    collection = reset_collection()

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embed_passages([c.text for c in batch])
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in batch],
        )
        total += len(batch)
        print(f"  persisted {total}/{len(chunks)} chunks")

    return total
