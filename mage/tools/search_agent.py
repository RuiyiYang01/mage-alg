"""Hierarchical Search Agent — Article → Paragraph → Sentence retrieval.

v15: Core RAG module for MAGE. Given a question, the search agent:
  1. Identifies the target entity (from Freebase URL or question text)
  2. Fetches the FULL Wikipedia article (~30K chars of clean text)
  3. Splits into paragraphs and sentences
  4. Searches hierarchically: BM25 to filter paragraphs, then dense
     retrieval (sentence-transformers) to rank sentences
  5. Returns the top-K most relevant SENTENCES as context for the learner

Also searches the EvoKG's failure memory bank for similar past solved
questions (history search).

The search strategy is selected by a Thompson Sampling bandit that
evolves based on the learner's success/failure feedback.

Usage:
    from mage.tools.search_agent import SearchAgent
    agent = SearchAgent()
    results = agent.search(
        question="What currency does Puerto Rico use?",
        freebase_url="http://www.freebase.com/view/en/puerto_rico",
    )
    # Returns: ["The currency of Puerto Rico is the United States dollar (USD).", ...]
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ARTICLE_CACHE_DIR = Path(__file__).parent / "wikipedia_articles"
ARTICLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Text processing utilities
# ============================================================

def _strip_wikitext(wikitext: str) -> str:
    """Convert wikitext markup to clean plain text."""
    text = wikitext
    # Remove infobox / template blocks {{...}} (multi-pass for nesting)
    for _ in range(5):
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if text == prev:
            break
    # Remove [[File:...]], [[Image:...]]
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text, flags=re.I)
    # [[link|display]] → display; [[link]] → link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    # [http://...] external links
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)
    # HTML tags + ref tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    # Section headers == Title == → \nTitle\n
    text = re.sub(r"={2,}\s*([^=]+?)\s*={2,}", r"\n\1\n", text)
    # Bold/italic
    text = re.sub(r"'{2,}", "", text)
    # Tables {| ... |}
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Remove lines that are just pipe-separated template remnants
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") or stripped.startswith("!"):
            continue
        if len(stripped) < 10 and not any(c.isalpha() for c in stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _fetch_full_article(title: str, max_chars: int = 30000) -> str:
    """Fetch full Wikipedia article via the parse API (wikitext → clean text)."""
    import requests

    safe_name = re.sub(r"[^a-z0-9_]", "_", title.lower())[:120]
    cache_path = ARTICLE_CACHE_DIR / f"{safe_name}.txt"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if len(text) > 500:
            return text[:max_chars]

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "format": "json",
            },
            timeout=15,
            headers={"User-Agent": "MAGE-Research/1.0 (research)"},
        )
        wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext:
            return ""
        text = _strip_wikitext(wikitext)
        if text:
            cache_path.write_text(text[:max_chars], encoding="utf-8")
        return text[:max_chars]
    except Exception as e:
        logger.debug(f"Failed to fetch article '{title}': {e}")
        return ""


def _split_paragraphs(text: str, min_len: int = 40) -> list[str]:
    """Split text into paragraphs (non-empty, min length)."""
    paragraphs = []
    for block in text.split("\n\n"):
        block = block.strip()
        if len(block) >= min_len:
            paragraphs.append(block)
    return paragraphs


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on period/question/exclamation followed by space and capital
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 15:  # skip tiny fragments
            sentences.append(s)
    return sentences


def _tokenize_simple(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 relevance score."""
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        f = tf.get(qt, 0)
        if f == 0:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


# ============================================================
# Hierarchical Index
# ============================================================

class HierarchicalIndex:
    """Article → Paragraph → Sentence index with dense retrieval.

    For each article, stores pre-computed sentence embeddings. Search
    is two-stage: BM25 filters paragraphs, then dense similarity ranks
    sentences within those paragraphs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        # Cache: title → list of (sentence_text, paragraph_idx, embedding)
        self._indexed: dict[str, list[tuple[str, int, np.ndarray]]] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)

    def index_article(self, title: str, text: str) -> int:
        """Index an article: split into paragraphs + sentences, embed.

        Returns the number of sentences indexed.
        """
        if title in self._indexed:
            return len(self._indexed[title])

        self._load_model()
        paragraphs = _split_paragraphs(text)
        entries: list[tuple[str, int, np.ndarray]] = []
        all_sentences: list[str] = []
        para_ids: list[int] = []

        for p_idx, para in enumerate(paragraphs):
            for sent in _split_sentences(para):
                all_sentences.append(sent)
                para_ids.append(p_idx)

        if not all_sentences:
            self._indexed[title] = []
            return 0

        # Batch embed
        embeddings = self._model.encode(
            all_sentences, normalize_embeddings=True, show_progress_bar=False,
        )
        for sent, p_idx, emb in zip(all_sentences, para_ids, embeddings):
            entries.append((sent, p_idx, emb))

        self._indexed[title] = entries
        return len(entries)

    def search_sentences(
        self,
        question: str,
        title: str,
        top_k: int = 5,
        min_score: float = 0.25,
        paragraph_prefilter: int = 10,
    ) -> list[tuple[str, float]]:
        """Hierarchical search: BM25 paragraph filter → dense sentence ranking.

        Args:
            question: The question to find answers for.
            title: Wikipedia article title (must be indexed first).
            top_k: Number of sentences to return.
            min_score: Minimum cosine similarity threshold.
            paragraph_prefilter: Number of paragraphs to consider (BM25).

        Returns:
            List of (sentence, score) tuples, most relevant first.
        """
        if title not in self._indexed or not self._indexed[title]:
            return []

        self._load_model()
        entries = self._indexed[title]

        # Stage 1: BM25 paragraph prefilter
        # Group sentences by paragraph
        para_sentences: dict[int, list[str]] = {}
        for sent, p_idx, _emb in entries:
            para_sentences.setdefault(p_idx, []).append(sent)

        q_tokens = _tokenize_simple(question)
        para_scores = []
        for p_idx, sents in para_sentences.items():
            para_text = " ".join(sents)
            doc_tokens = _tokenize_simple(para_text)
            avg_dl = sum(
                len(_tokenize_simple(" ".join(s)))
                for s in para_sentences.values()
            ) / max(len(para_sentences), 1)
            score = _bm25_score(q_tokens, doc_tokens, avg_dl)
            para_scores.append((score, p_idx))
        para_scores.sort(key=lambda x: -x[0])
        top_paras = {p_idx for _, p_idx in para_scores[:paragraph_prefilter]}

        # Stage 2: Dense sentence ranking within top paragraphs
        q_emb = self._model.encode([question], normalize_embeddings=True)
        scored: list[tuple[float, str]] = []
        for sent, p_idx, emb in entries:
            if p_idx not in top_paras:
                continue
            sim = float(q_emb @ emb.T)
            if sim >= min_score:
                scored.append((sim, sent))

        scored.sort(key=lambda x: -x[0])
        return [(sent, score) for score, sent in scored[:top_k]]


# ============================================================
# Search Agent
# ============================================================

SEARCH_STRATEGIES = [
    "entity_direct",   # entity URL → article → hierarchical search
    "question_search", # use question as Wikipedia search query
    "hybrid",          # both entity + question search, merge results
]


class SearchAgent:
    """Evolving hierarchical search agent.

    Searches TWO sources:
    1. External KB (Wikipedia) — via HierarchicalIndex
    2. Internal history (EvoKG failure memories) — via FailureMemoryIndex

    The search strategy is selected by a Thompson Sampling bandit that
    evolves based on the learner's success/failure feedback.
    """

    def __init__(self, top_k: int = 5, min_pulls_per_arm: int = 20,
                 ensemble_all_strategies: bool = False):
        self.top_k = top_k
        self._index = HierarchicalIndex()
        # Per-task-type bandit: {task_type: {strategy: {"alpha": float, "beta": float}}}
        self._bandit: dict[str, dict[str, dict[str, float]]] = {}
        # v15.3: during warm-up, round-robin across arms until each has at
        # least `min_pulls_per_arm` reward updates. Prevents the bandit from
        # locking onto a single arm (e.g. "hybrid") before the other arms
        # have any signal. Counted as pulls = (alpha + beta - 2) per arm.
        self.min_pulls_per_arm: int = int(min_pulls_per_arm)
        # Per-eval-pass round-robin counter so three arms get roughly equal
        # pulls within a single 200-question eval during warm-up.
        self._rr_counter: dict[str, int] = {}
        # v19.B — when True, run all 3 strategies and union results
        # (instead of bandit-selecting one). Higher recall at the cost of
        # 3× retrieval latency. Useful for benchmarks where the bandit
        # underexplores certain arms.
        self.ensemble_all_strategies: bool = bool(ensemble_all_strategies)

    def search(
        self,
        question: str,
        entity: str | None = None,
        freebase_url: str | None = None,
        task_type: str | None = None,
        evokg: Any | None = None,
        deterministic: bool = False,
        return_strategy: bool = False,
    ) -> list[str] | tuple[list[str], str]:
        """Main entry: retrieve relevant sentences from KB + history.

        Returns list of sentence strings, most relevant first. If
        ``return_strategy=True`` (v15.2), also returns the strategy arm
        that was used so the caller can credit the correct bandit arm
        with the learner's downstream success/failure.
        """
        # Select search strategy (or ensemble all)
        if self.ensemble_all_strategies:
            strategy = "ensemble_all"  # marker, not used in conditionals below
        else:
            strategy = self._select_strategy(task_type, deterministic)

        # 1. External KB search (Wikipedia)
        kb_sentences: list[tuple[str, float]] = []
        if self.ensemble_all_strategies or strategy in ("entity_direct", "hybrid"):
            kb_sentences.extend(
                self._search_kb_by_entity(question, entity, freebase_url)
            )
        if self.ensemble_all_strategies or strategy in ("question_search", "hybrid"):
            kb_sentences.extend(
                self._search_kb_by_question(question)
            )

        # 2. Internal history search (EvoKG memories)
        history_sentences: list[tuple[str, float]] = []
        if evokg is not None:
            history_sentences = self._search_history(question, task_type, evokg)

        # 3. Merge, deduplicate, return top-K
        all_results = kb_sentences + history_sentences
        sentences = self._merge_and_dedup(all_results, self.top_k)
        if return_strategy:
            return sentences, strategy
        return sentences

    def _search_kb_by_entity(
        self,
        question: str,
        entity: str | None,
        freebase_url: str | None,
    ) -> list[tuple[str, float]]:
        """Search Wikipedia by entity: fetch article → hierarchical retrieval."""
        if not entity and freebase_url:
            from mage.tools.wikipedia_tool import freebase_url_to_entity
            entity = freebase_url_to_entity(freebase_url)
        if not entity:
            entity = self._extract_entity(question)
        if not entity:
            return []

        title = self._resolve_title(entity)
        if not title:
            return []

        # Fetch and index the article
        article = _fetch_full_article(title)
        if not article:
            return []
        self._index.index_article(title, article)

        # Hierarchical search
        return self._index.search_sentences(question, title, top_k=self.top_k)

    def _search_kb_by_question(self, question: str) -> list[tuple[str, float]]:
        """Search Wikipedia using the question as a search query."""
        from mage.tools.wikipedia_tool import search_wikipedia

        titles = search_wikipedia(question, max_results=2)
        results: list[tuple[str, float]] = []
        for title in titles:
            article = _fetch_full_article(title)
            if not article:
                continue
            self._index.index_article(title, article)
            results.extend(
                self._index.search_sentences(question, title, top_k=3)
            )
        return results

    def _search_history(
        self,
        question: str,
        task_type: str | None,
        evokg: Any,
    ) -> list[tuple[str, float]]:
        """Search EvoKG for similar past solved questions (both success + failure memories).

        v15: Searches BOTH success memories (correct answers from past
        iterations) and failure memories (worked examples from Claude).
        Success memories are prioritized because they are PROVEN correct.
        """
        scored: list[tuple[str, float]] = []

        # 1. Search SUCCESS memories first (proven correct, highest value)
        try:
            success_mems = evokg.get_success_memories(task_type=task_type)
        except Exception:
            success_mems = []

        # 2. Search FAILURE memories (worked examples from Claude)
        try:
            failure_mems = evokg.get_failure_memories(task_type=task_type)
        except Exception:
            failure_mems = []

        all_mems = [(m, "success") for m in success_mems] + [(m, "failure") for m in failure_mems]
        if not all_mems:
            return []

        self._index._load_model()
        q_emb = self._index._model.encode([question], normalize_embeddings=True)

        for m, mem_type in all_mems:
            past_q = m.get("question", "")
            if not past_q:
                continue
            m_emb = self._index._model.encode([past_q], normalize_embeddings=True)
            sim = float(q_emb @ m_emb.T)
            if sim > 0.4:
                if mem_type == "success":
                    answer = m.get("correct_answer", "")
                    hint = f"Similar solved question: {past_q[:100]} → Answer: {answer}"
                    scored.append((hint, sim))  # full weight for proven answers
                else:
                    expected = m.get("expected", "")
                    reasoning = m.get("reasoning", "")[:100]
                    hint = f"Similar question: {past_q[:100]} → {reasoning} → Answer: {expected}"
                    scored.append((hint, sim * 0.8))  # slight discount

        scored.sort(key=lambda x: -x[1])
        return scored[:3]

    def _resolve_title(self, entity: str) -> str | None:
        """Resolve entity name to Wikipedia title."""
        from mage.tools.wikipedia_tool import (
            fetch_wikipedia_summary,
            search_wikipedia,
            _candidate_titles,
        )
        # Try direct title variants
        for candidate in _candidate_titles(entity):
            summary = fetch_wikipedia_summary(candidate, max_chars=100)
            if summary:
                return candidate.replace("_", " ")
        # Fall back to search
        titles = search_wikipedia(entity, max_results=1)
        return titles[0] if titles else None

    def _extract_entity(self, question: str) -> str | None:
        """Extract main entity from question via simple heuristics."""
        q = question.strip().rstrip("?")
        patterns = [
            r"(?:what|which|where|who|when|how).*(?:of|in|for|at|from)\s+(.+?)$",
            r"(?:where|when) (?:was|is|did|does) (.+?)(?:\s+(?:born|located|live|die|go|play|win))",
            r"(?:who) (?:plays?|is|was|wrote|directed|founded) (.+?)$",
        ]
        for pat in patterns:
            m = re.search(pat, q, flags=re.I)
            if m:
                ent = m.group(1).strip()
                if len(ent) > 2:
                    return ent
        # Fallback: longest capitalized phrase
        caps = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
        return max(caps, key=len) if caps else None

    # ---- Strategy bandit (RL-like evolution) ----

    def _select_strategy(
        self, task_type: str | None, deterministic: bool,
    ) -> str:
        """Pick a search strategy.

        Warm-up: if any arm has fewer than ``min_pulls_per_arm`` reward
        updates, force exploration by picking the least-pulled arm (ties
        broken by a round-robin counter). This guarantees each strategy
        gets real signal before the bandit locks onto a winner.

        Steady state: deterministic → argmax of α/(α+β) (reproducible eval);
        non-deterministic → Thompson sampling (exploration in training).
        """
        tt = task_type or "_default"
        if tt not in self._bandit:
            self._bandit[tt] = {
                s: {"alpha": 1.0, "beta": 1.0} for s in SEARCH_STRATEGIES
            }

        arms = self._bandit[tt]

        # v15.3 — warm-up phase
        def pulls(arm: dict[str, float]) -> float:
            # Each reward update adds 1 to (alpha + beta) regardless of reward,
            # so (alpha + beta - 2) counts the pulls since init at (1,1).
            return arm["alpha"] + arm["beta"] - 2.0

        min_pulls = min(pulls(a) for a in arms.values())
        if min_pulls < self.min_pulls_per_arm:
            # Round-robin across under-pulled arms. Picking the *least-pulled*
            # arm guarantees even coverage; the counter breaks ties in a
            # reproducible way across a single eval pass.
            underpulled = [
                s for s, a in arms.items()
                if pulls(a) < self.min_pulls_per_arm
            ]
            # Within under-pulled arms, pick the one with the fewest pulls,
            # breaking ties by the per-task round-robin counter.
            underpulled.sort(
                key=lambda s: (pulls(arms[s]), SEARCH_STRATEGIES.index(s))
            )
            idx = self._rr_counter.get(tt, 0) % len(underpulled)
            self._rr_counter[tt] = self._rr_counter.get(tt, 0) + 1
            return underpulled[idx]

        # Steady state
        if deterministic:
            # Expected value: alpha / (alpha + beta)
            best = max(
                arms.items(),
                key=lambda x: x[1]["alpha"] / (x[1]["alpha"] + x[1]["beta"]),
            )
            return best[0]
        else:
            import random
            samples = {
                s: random.betavariate(a["alpha"], a["beta"])
                for s, a in arms.items()
            }
            return max(samples, key=samples.get)

    def update(
        self,
        task_type: str | None,
        strategy: str,
        reward: float,
    ) -> None:
        """Update bandit arm with success/failure signal.

        v15.2 — auto-initialize the task-type bandit and the strategy arm
        if they don't exist yet (e.g. the very first call for a task_type
        or a strategy that's never been selected). This ensures every real
        reward signal is recorded, rather than silently dropped.
        """
        tt = task_type or "_default"
        if strategy not in SEARCH_STRATEGIES:
            return
        if tt not in self._bandit:
            self._bandit[tt] = {
                s: {"alpha": 1.0, "beta": 1.0} for s in SEARCH_STRATEGIES
            }
        elif strategy not in self._bandit[tt]:
            self._bandit[tt][strategy] = {"alpha": 1.0, "beta": 1.0}
        self._bandit[tt][strategy]["alpha"] += reward
        self._bandit[tt][strategy]["beta"] += 1.0 - reward

    def save(self, path: Path) -> None:
        """Persist bandit state to JSON."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._bandit, f, indent=2)

    def load(self, path: Path) -> None:
        """Load bandit state from JSON.

        v15.3 — backfill any missing strategy arms with fresh (1,1) priors
        so old bandit files (which only had 'hybrid' updated) still
        trigger the warm-up exploration phase for the other arms.
        """
        import json
        if path.exists():
            with open(path) as f:
                self._bandit = json.load(f)
            for tt, arms in self._bandit.items():
                for s in SEARCH_STRATEGIES:
                    if s not in arms:
                        arms[s] = {"alpha": 1.0, "beta": 1.0}

    # ---- Helpers ----

    @staticmethod
    def _merge_and_dedup(
        results: list[tuple[str, float]], top_k: int,
    ) -> list[str]:
        """Merge results from multiple sources, deduplicate, return top-K."""
        seen: set[str] = set()
        merged: list[tuple[float, str]] = []
        for text, score in results:
            # Simple dedup by first 50 chars
            key = text[:50].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append((score, text))
        merged.sort(key=lambda x: -x[0])
        return [text for _, text in merged[:top_k]]


# ============================================================
# Convenience function
# ============================================================

def search_for_question(
    question: str,
    freebase_url: str = "",
    task_type: str = "",
    evokg: Any = None,
    top_k: int = 5,
) -> list[str]:
    """One-shot search: create agent, search, return sentences."""
    agent = SearchAgent(top_k=top_k)
    return agent.search(
        question=question,
        freebase_url=freebase_url or None,
        task_type=task_type or None,
        evokg=evokg,
    )
