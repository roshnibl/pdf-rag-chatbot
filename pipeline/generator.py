"""Answer generation with Gemini, grounded on retrieved passages.

Builds a numbered-context prompt that forces the model to cite sources as
[filename p.N], synthesizes the answer, and returns both the answer text and
the structured source list for the UI to render.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from pipeline.retriever import RetrievedChunk, retrieve

_client: genai.Client | None = None

# Transient server-side statuses worth retrying with a short backoff.
# NOTE: 429 (quota/rate limit) is deliberately NOT here — retrying it instantly
# just consumes more quota; it needs a longer wait, surfaced to the user instead.
_RETRYABLE_STATUS = {500, 502, 503, 504}


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]


SYSTEM_INSTRUCTION = (
    "You are a precise research assistant answering questions strictly from a "
    "private PDF knowledge base. Follow these rules without exception:\n"
    "1. Use ONLY the information in the provided context passages. Do not rely "
    "on outside knowledge.\n"
    "2. Every factual sentence must cite its source inline using the exact "
    "format [filename p.PAGE], e.g. [Biology2e.pdf p.214].\n"
    "3. Answer as fully as the context allows, synthesizing across the passages "
    "even if the relevant details are spread across several of them. Only if the "
    "passages are genuinely unrelated to the question should you say: 'I could "
    "not find this in the provided documents.' Do not give up when partial but "
    "on-topic information is present.\n"
    "4. Never invent facts, numbers, or names beyond what the context states.\n"
    "5. Be concise and factual. Do not speculate."
)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your free key from https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved passages as a numbered, labelled context block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (source: {c.filename} p.{c.page_number})\n{c.text}"
        )
    return "\n\n".join(lines)


def _generate_with_retry(client: genai.Client, prompt: str, max_attempts: int = 2):
    """Call Gemini, retrying transient 429/5xx errors with a short backoff.

    Kept deliberately shallow (2 attempts, ~1.5s) because the google-genai SDK
    already retries internally; deep retries here would stack on top of the
    SDK's and make an interactive query hang for over a minute on an outage.
    """
    delay = 1.5
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=config.GEMINI_TEMPERATURE,
                    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
                ),
            )
        except genai_errors.APIError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            last_exc = e
            if status in _RETRYABLE_STATUS and attempt < max_attempts:
                time.sleep(delay)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def generate_answer(query: str, chunks: list[RetrievedChunk] | None = None) -> Answer:
    """Retrieve (if needed) and synthesize a grounded, cited answer."""
    client = _get_client()

    if chunks is None:
        chunks = retrieve(query)

    if not chunks:
        return Answer(
            text="I could not find this in the provided documents.",
            sources=[],
        )

    context = _build_context_block(chunks)
    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above, citing each fact as "
        "[filename p.PAGE]."
    )

    try:
        response = _generate_with_retry(client, prompt)
    except genai_errors.APIError as e:
        status = getattr(e, "code", None) or getattr(e, "status_code", None)
        if status == 429:
            note = (
                "Gemini free-tier quota/rate limit reached. Wait ~a minute and "
                "ask again; if it persists, the daily free quota is used up — it "
                "resets in 24h, or use a key with a paid plan."
            )
        elif status in _RETRYABLE_STATUS:
            note = (
                "Gemini is temporarily busy (high demand). Try the question "
                "again in a moment."
            )
        else:
            note = f"Gemini call failed ({e})."
        # Retrieval succeeded; show the note and still return sources so the UI
        # stays useful instead of erroring out.
        return Answer(
            text=f"{note}\n\n(Retrieval worked — the relevant sources are listed below.)",
            sources=chunks,
        )

    return Answer(text=response.text, sources=chunks)
