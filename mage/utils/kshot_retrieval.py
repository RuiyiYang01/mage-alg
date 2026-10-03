"""Medprompt-style kNN exemplar retrieval over benchmark train splits.

For each test question, retrieve the top-K most semantically similar
training questions (with their gold answers) to inject as few-shot
demonstrations. This is the core trick that gives Medprompt 74-78% on
Llama-3.1-8B for MedQA-USMLE.

Reference:
- Nori, H. et al. "Can Generalist Foundation Models Outcompete
  Special-Purpose Tuning?" arXiv 2311.16452 (2023).
- Wang, X. et al. "Self-Consistency Improves Chain of Thought Reasoning."
  arXiv 2203.11171 (2022).

Design:
- One-time: load train split, embed all questions, save to disk cache
  (keyed by benchmark name + embedding model + train hash)
- Per-query: encode the test question, cosine-similarity top-K, return
  formatted exemplars

We do NOT generate Claude CoT rationales here (that's a separate offline
pre-pass; this util just retrieves the raw (Q, A, options) tuples). The
caller decides how to format them into a prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("experiments/_kshot_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


class KShotIndex:
    """In-memory kNN index over a train split.

    Lazily loads sentence-transformers model. Embeddings are cached on
    disk per (benchmark, model, train_size) tuple, so the second
    instantiation is near-instant.
    """

    def __init__(
        self,
        benchmark: str,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.benchmark = benchmark
        self.model_name = model_name
        self._model: Any = None
        self._questions: list[str] = []
        self._records: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)

    def _cache_key(self, n_records: int) -> str:
        return f"{self.benchmark}_{self.model_name.replace('/', '_')}_n{n_records}"

    def build(self, records: list[dict[str, Any]], q_field: str = "question") -> None:
        """Build the index from a list of records.

        Each record must have a `q_field` (the question text used for
        embedding) and arbitrary other fields preserved.
        """
        self._records = records
        self._questions = [str(r.get(q_field, ""))[:500] for r in records]
        n = len(self._questions)
        if n == 0:
            self._embeddings = None
            return

        cache_path = CACHE_ROOT / f"{self._cache_key(n)}.npy"
        if cache_path.exists():
            try:
                self._embeddings = np.load(cache_path)
                logger.info(f"Loaded {n} kNN embeddings from cache: {cache_path}")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}; rebuilding")

        self._load_model()
        logger.info(f"Embedding {n} {self.benchmark} train questions")
        self._embeddings = self._model.encode(
            self._questions, normalize_embeddings=True, batch_size=64, show_progress_bar=False,
        )
        np.save(cache_path, self._embeddings)
        logger.info(f"Saved kNN cache: {cache_path}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-K most similar records (by cosine sim)."""
        if self._embeddings is None or not self._records:
            return []
        self._load_model()
        q_emb = self._model.encode([str(query)[:500]], normalize_embeddings=True)
        scores = (q_emb @ self._embeddings.T)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in top_idx:
            r = dict(self._records[i])
            r["_kshot_score"] = float(scores[i])
            out.append(r)
        return out


# -- Benchmark-specific loaders + formatters --

def load_finqa_train_index(max_records: int = 6251) -> KShotIndex:
    """Load FinQA train split as a KShotIndex.

    Records have: question (str), answer (str), context (str).
    """
    from datasets import load_dataset
    from mage.environments.finqa_env import FinQAEnvironment

    ds = load_dataset("ChanceFocus/flare-finqa", split=f"train[:{max_records}]")
    records = []
    for ex in ds:
        # Reuse env's split logic
        ctx, _ = FinQAEnvironment._split_query(ex["query"])
        records.append({
            "question": ex["text"],
            "answer": str(ex["answer"]).strip(),
            "context": ctx[:1500],  # truncate to keep prompt short
        })
    idx = KShotIndex(benchmark="finqa")
    idx.build(records, q_field="question")
    return idx


def load_medqa_train_index(max_records: int = 10178) -> KShotIndex:
    """Load MedQA train split as a KShotIndex.

    Records have: question (str), answer_letter (str), options (dict).
    """
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=f"train[:{max_records}]")
    records = []
    for ex in ds:
        records.append({
            "question": ex["question"],
            "answer_letter": ex.get("answer_idx", ""),
            "answer_text": ex.get("answer", ""),
            "options": dict(ex.get("options", {})),
        })
    idx = KShotIndex(benchmark="medqa")
    idx.build(records, q_field="question")
    return idx


def format_finqa_kshot(records: list[dict], max_chars_per: int = 1200, with_reasoning: bool = True) -> str:
    """Format FinQA exemplars as a few-shot block.

    If `with_reasoning` and the record has a `reasoning` field (Medprompt-style
    pre-generated CoT), include it. Otherwise fall back to (Q, A) only.
    """
    if not records:
        return ""
    parts = []
    for i, r in enumerate(records):
        q = r.get("question", "")[:300]
        a = r.get("answer", "")
        reasoning = r.get("reasoning", "") if with_reasoning else ""
        if reasoning:
            block = f"Example {i+1}:\nQuestion: {q}\nReasoning: {reasoning[:1000]}\nAnswer: {a}"
        else:
            block = f"Example {i+1}:\nQuestion: {q}\nAnswer: {a}"
        if len(block) > max_chars_per:
            block = block[:max_chars_per]
        parts.append(block)
    return "\n\n".join(parts) + "\n\n"


def format_medqa_kshot(records: list[dict], max_chars_per: int = 1600, with_reasoning: bool = True) -> str:
    """Format MedQA exemplars as a few-shot block.

    If `with_reasoning` and the record has a `reasoning` field (Medprompt-style
    pre-generated CoT), include it as the demonstrated solving process.
    """
    if not records:
        return ""
    parts = []
    for i, r in enumerate(records):
        q = r.get("question", "")[:600]
        opts = r.get("options", {})
        opt_str = "\n".join(f"{k}. {opts[k]}" for k in "ABCD" if k in opts)
        a_letter = r.get("answer_letter") or r.get("answer", "")
        reasoning = r.get("reasoning", "") if with_reasoning else ""
        if reasoning:
            block = f"Example {i+1}:\nQuestion: {q}\nOptions:\n{opt_str}\nReasoning: {reasoning[:1000]}\nAnswer: {a_letter}"
        else:
            block = f"Example {i+1}:\nQuestion: {q}\nOptions:\n{opt_str}\nAnswer: {a_letter}"
        if len(block) > max_chars_per:
            block = block[:max_chars_per]
        parts.append(block)
    return "\n\n".join(parts) + "\n\n"


# -- Rationale pool loaders --

RATIONALE_DIR = Path("experiments/_kshot_rationales")


def load_rationale_index(benchmark: str, teacher: str) -> KShotIndex | None:
    """Load a teacher-generated rationale pool as a kNN index.

    Pool file: experiments/_kshot_rationales/<bench>_<teacher>.json
    Each record has (question, reasoning, answer, options/context).
    """
    path = RATIONALE_DIR / f"{benchmark}_{teacher}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            records = json.load(f)
        if not records:
            return None
        # Normalize answer field name across benches (medqa pool uses "answer"
        # which is the LETTER; finqa pool also uses "answer" which is numeric)
        for r in records:
            if "answer_letter" not in r and benchmark == "medqa":
                r["answer_letter"] = r.get("answer", "")
        idx = KShotIndex(benchmark=f"{benchmark}_rationale_{teacher}")
        idx.build(records, q_field="question")
        logger.info(f"Loaded rationale pool: {benchmark}_{teacher} ({len(records)} records)")
        return idx
    except Exception as e:
        logger.warning(f"Failed to load rationale pool {path}: {e}")
        return None


# -- Module-level singletons (lazy) --

_INDEX_CACHE: dict[str, KShotIndex] = {}


def get_index(benchmark: str, teacher: str = "") -> KShotIndex | None:
    """Return a singleton KShotIndex for the benchmark.

    If `teacher` is provided, prefer the rationale pool (Medprompt full
    recipe); fall back to raw train-split kNN.
    """
    cache_key = f"{benchmark}_{teacher}"
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]
    idx = None
    if teacher:
        idx = load_rationale_index(benchmark, teacher)
    if idx is None:
        try:
            if benchmark == "finqa":
                idx = load_finqa_train_index()
            elif benchmark == "medqa":
                idx = load_medqa_train_index()
        except Exception as e:
            logger.warning(f"Failed to load raw kshot index for {benchmark}: {e}")
            return None
    if idx is not None:
        _INDEX_CACHE[cache_key] = idx
    return idx


def retrieve_kshot(benchmark: str, query: str, top_k: int = 5, teacher: str = "") -> list[dict]:
    idx = get_index(benchmark, teacher=teacher)
    if idx is None:
        return []
    return idx.retrieve(query, top_k=top_k)
