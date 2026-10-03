"""LLM-as-Judge evaluation for QA benchmarks.

v15: Uses Claude (teacher agent) to judge whether a predicted answer is
semantically correct given the gold answer(s). This handles:
  - Paraphrases: "US dollar" ≈ "United States dollar"
  - Abbreviations: "NYC" ≈ "New York City"
  - Partial matches: "theoretical astrophysics" ≈ "Physics"
  - Different granularity: "Barbados" ≈ "Saint Michael Parish, Barbados"

Much more accurate than F1 token overlap, which rejects many correct
answers due to verbosity or different wording.

Usage:
    from mage.utils.llm_judge import LLMJudge
    judge = LLMJudge(llm_backend)
    score = judge.evaluate("US dollar", "United States dollar",
                           question="what currency does PR use?")
    # Returns 1.0 (correct)

The judge caches results to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Disk cache for judge results (avoid re-calling Claude for the same Q/A pair)
JUDGE_CACHE_DIR = Path(__file__).parent.parent.parent / "experiments" / "_judge_cache"


class LLMJudge:
    """Use Claude as a semantic answer judge.

    For each (question, prediction, gold) triple, asks Claude:
    "Is the predicted answer correct?" → Yes/No.

    Results are cached in memory and on disk so repeated calls
    (same question across iterations) are free.
    """

    def __init__(self, llm_backend: Any = None):
        """
        Args:
            llm_backend: A strong LLM backend (Claude) with a `generate`
                method. If None, falls back to F1 scoring.
        """
        self._llm = llm_backend
        self._cache: dict[str, float] = {}
        self._disk_cache_path = JUDGE_CACHE_DIR / "judge_cache.json"
        self._load_disk_cache()
        self._call_count = 0

    def _load_disk_cache(self) -> None:
        if self._disk_cache_path.exists():
            try:
                with open(self._disk_cache_path) as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached judge results")
            except Exception:
                self._cache = {}

    def _save_disk_cache(self) -> None:
        try:
            self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._disk_cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.debug(f"Failed to save judge cache: {e}")

    def _cache_key(self, question: str, predicted: str, expected: str) -> str:
        raw = f"{question[:200]}|{predicted[:100]}|{expected[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def evaluate(
        self,
        predicted: str,
        expected: str,
        question: str = "",
        task_type: str = "",
    ) -> float:
        """Judge whether the predicted answer is correct.

        Returns 1.0 if correct, 0.0 if wrong.
        Falls back to F1 if Claude is unavailable.
        """
        if not predicted or not expected:
            return 0.0

        # Quick exact match first (no API call needed)
        if self._quick_match(predicted, expected):
            return 1.0

        # Check cache
        key = self._cache_key(question, predicted, expected)
        if key in self._cache:
            return self._cache[key]

        # Call Claude
        if self._llm is None:
            # Fallback to F1
            return self._f1_fallback(predicted, expected)

        score = self._call_judge(predicted, expected, question)
        self._cache[key] = score
        self._call_count += 1

        # Periodic disk save (every 50 calls)
        if self._call_count % 50 == 0:
            self._save_disk_cache()

        return score

    def flush_cache(self) -> None:
        """Save the in-memory cache to disk."""
        self._save_disk_cache()

    def _quick_match(self, predicted: str, expected: str) -> bool:
        """Fast exact-match check (case-insensitive, normalized).

        Tries three layers in order:
        1. Numeric equivalence (with percent <-> decimal swap and 1% rel /
           0.05 abs tolerance) — covers GSM8K, RealMath, FinQA, FinanceMath.
           This MUST run before semantic norm because '14.1' and '14.09032'
           normalize-equal as strings only by chance, while '0.65' vs '65'
           need numeric awareness.
        2. Letter / option match (single A-D, A-E) — covers MedQA, MMLU.
        3. String normalization (case, articles, punctuation) — covers
           HotpotQA, WebQA.
        """
        import re

        # Layer 1: numeric equivalence
        pred_num = self._to_float(predicted)
        if pred_num is not None:
            for ans in expected.split("|"):
                gold_num = self._to_float(ans)
                if gold_num is None:
                    continue
                # Direct, percent->decimal, decimal->percent
                for cand in (pred_num, pred_num / 100.0, pred_num * 100.0):
                    if self._numeric_close(cand, gold_num):
                        return True

        # Layer 2: single-letter MCQ match (A-E)
        def first_letter(s: str) -> str:
            m = re.search(r"\b([A-Ea-e])\b", s)
            return m.group(1).upper() if m else ""
        pred_l = first_letter(predicted)
        if pred_l:
            for ans in expected.split("|"):
                if first_letter(ans) == pred_l:
                    # Only accept letter-match if BOTH look like letter answers
                    # (i.e. expected is short, A-E only). Avoid false positives
                    # on long sentences that happen to contain a stray letter.
                    if len(ans.strip()) <= 3:
                        return True

        # Layer 3: string normalization
        def norm(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = re.sub(r"[^\w\s]", "", s)
            return " ".join(s.split())

        pred_norm = norm(predicted)
        if not pred_norm:
            return False
        for ans in expected.split("|"):
            ans_norm = norm(ans)
            if not ans_norm:
                continue
            if ans_norm == pred_norm:
                return True
            # Substring: if gold is contained in pred or vice versa
            if ans_norm in pred_norm or pred_norm in ans_norm:
                return True
        return False

    @staticmethod
    def _to_float(s: str) -> float | None:
        import re
        s = re.sub(r"[,$\s%]", "", str(s).strip())
        if not s:
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _numeric_close(p: float, e: float) -> bool:
        if e == 0:
            return abs(p) < 0.01
        if abs(p - e) / max(abs(e), 1e-9) <= 0.01:
            return True
        if abs(p - e) <= 0.05:
            return True
        return False

    def _call_judge(
        self, predicted: str, expected: str, question: str,
    ) -> float:
        """Ask Claude to judge the answer."""
        expected_list = expected.split("|")
        gold_str = ", ".join(f'"{a}"' for a in expected_list[:5])

        prompt = (
            f"You are judging whether a QA model's answer is correct.\n\n"
            f"Question: {question[:300]}\n"
            f"Accepted answers: {gold_str}\n"
            f"Model's answer: \"{predicted[:200]}\"\n\n"
            f"Is the model's answer semantically correct? Consider:\n"
            f"- Paraphrases and abbreviations count as correct "
            f'("US dollar" = "United States dollar")\n'
            f"- More specific answers are correct "
            f'("Mila Kunis" is correct for "actress")\n'
            f"- Extra detail is OK if the core answer is right\n"
            f"- Completely wrong or unrelated answers are incorrect\n\n"
            f"Reply with ONLY 'correct' or 'incorrect'."
        )

        try:
            response = self._llm.generate(
                prompt, temperature=0.0, max_tokens=10,
            ).strip().lower()

            if "correct" in response and "incorrect" not in response:
                return 1.0
            return 0.0
        except Exception as e:
            logger.debug(f"Judge API call failed: {e}")
            return self._f1_fallback(predicted, expected)

    @staticmethod
    def _f1_fallback(predicted: str, expected: str) -> float:
        """F1 token overlap as fallback when Claude is unavailable."""
        import re
        from collections import Counter

        def norm(s):
            s = s.lower().strip()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = re.sub(r"[^\w\s]", "", s)
            return " ".join(s.split())

        pred_tokens = norm(predicted).split()
        best_f1 = 0.0
        for ans in expected.split("|"):
            gold_tokens = norm(ans).split()
            if not pred_tokens or not gold_tokens:
                continue
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            p = num_same / len(pred_tokens)
            r = num_same / len(gold_tokens)
            f1 = 2 * p * r / (p + r)
            best_f1 = max(best_f1, f1)
        return best_f1
