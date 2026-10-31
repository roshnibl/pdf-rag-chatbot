"""Retrieval + answer evaluation harness.

Implements the metrics required by the challenge:
  - Latency p50 / p95 (end-to-end)
  - Recall@k and MRR (retrieval relevance, using a labelled query set)
  - Citation accuracy (fraction of answers whose citations point to pages that
    were actually retrieved)

Provide a JSON eval set at eval_set.json shaped as:
[
  {
    "query": "What is photosynthesis?",
    "relevant": [{"filename": "Biology2e.pdf", "page_number": 214}]
  },
  ...
]

Run:  python evaluate.py
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time

import config
from pipeline.generator import generate_answer
from pipeline.retriever import retrieve

EVAL_SET_PATH = os.path.join(config.BASE_DIR, "eval_set.json")
CITATION_RE = re.compile(r"\[([^\[\]]+?\.pdf)\s+p\.(\d+)\]", re.IGNORECASE)


def _is_relevant(chunk, relevant: list[dict]) -> bool:
    for r in relevant:
        if (
            chunk.filename.lower() == r["filename"].lower()
            and chunk.page_number == int(r["page_number"])
        ):
            return True
    return False


def recall_at_k(retrieved, relevant, k) -> float:
    top = retrieved[:k]
    if not relevant:
        return 0.0
    hits = sum(1 for r in relevant if any(_is_relevant(c, [r]) for c in top))
    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if _is_relevant(chunk, relevant):
            return 1.0 / rank
    return 0.0


def citation_accuracy(answer_text: str, retrieved) -> float | None:
    """Fraction of inline citations that match a retrieved passage's page.

    Returns None when the answer contains no citations (not penalized here).
    """
    cites = CITATION_RE.findall(answer_text)
    if not cites:
        return None
    retrieved_pages = {(c.filename.lower(), c.page_number) for c in retrieved}
    valid = sum(
        1 for fn, pg in cites if (fn.lower(), int(pg)) in retrieved_pages
    )
    return valid / len(cites)


def main():
    if not os.path.exists(EVAL_SET_PATH):
        print(
            f"No eval set found at {EVAL_SET_PATH}.\n"
            "Create eval_set.json (see the module docstring for the format)."
        )
        return

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    latencies, rr_scores, cite_scores = [], [], []
    r_at_1, r_at_3, r_at_5 = [], [], []

    for item in eval_set:
        query = item["query"]
        relevant = item.get("relevant", [])

        t0 = time.time()
        retrieved = retrieve(query, top_k_final=config.TOP_K_FINAL)
        answer = generate_answer(query, chunks=retrieved)
        latencies.append(time.time() - t0)

        if relevant:
            r_at_1.append(recall_at_k(retrieved, relevant, 1))
            r_at_3.append(recall_at_k(retrieved, relevant, 3))
            r_at_5.append(recall_at_k(retrieved, relevant, 5))
            rr_scores.append(reciprocal_rank(retrieved, relevant))

        ca = citation_accuracy(answer.text, retrieved)
        if ca is not None:
            cite_scores.append(ca)

    def _mean(xs):
        return statistics.mean(xs) if xs else 0.0

    def _p95(xs):
        if not xs:
            return 0.0
        ordered = sorted(xs)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]

    print("\n=== Evaluation results ===")
    print(f"Queries evaluated : {len(eval_set)}")
    print(f"Latency p50       : {statistics.median(latencies):.2f}s")
    print(f"Latency p95       : {_p95(latencies):.2f}s")
    print(f"Latency max       : {max(latencies):.2f}s")
    print(f"Recall@1          : {_mean(r_at_1):.3f}")
    print(f"Recall@3          : {_mean(r_at_3):.3f}")
    print(f"Recall@5          : {_mean(r_at_5):.3f}")
    print(f"MRR               : {_mean(rr_scores):.3f}")
    print(f"Citation accuracy : {_mean(cite_scores):.3f}")


if __name__ == "__main__":
    main()
