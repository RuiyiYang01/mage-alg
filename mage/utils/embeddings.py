"""Embedding-based skill matching for MAGE strategy router.

Lazy-loads a sentence-transformer model to match questions to skills
via cosine similarity. Replaces brittle keyword matching.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SkillEmbeddingIndex:
    """Semantic similarity index for matching questions to skills."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._skill_ids: list[str] = []
        self._skill_embeddings: np.ndarray | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded")

    def register_skills(self, skills: list[dict[str, Any]]) -> None:
        """Pre-compute embeddings for all skill descriptions."""
        self._load_model()
        self._skill_ids = []
        texts = []
        for skill in skills:
            sid = skill.get("node_id", skill.get("skill_id", ""))
            self._skill_ids.append(sid)
            # Rich text: name + description + prompt_template
            text = f"{skill.get('name', sid)}: {skill.get('description', '')}. {skill.get('prompt_template', '')}"
            texts.append(text[:512])

        if texts:
            self._skill_embeddings = self._model.encode(texts, normalize_embeddings=True)
            logger.info(f"Indexed {len(texts)} skills for embedding matching")
        else:
            self._skill_embeddings = None

    def find_skill(self, question: str, top_k: int = 1) -> list[tuple[str, float]]:
        """Find best matching skill(s) via cosine similarity."""
        if self._skill_embeddings is None or not self._skill_ids:
            return []

        self._load_model()
        q_emb = self._model.encode([question[:512]], normalize_embeddings=True)
        scores = (q_emb @ self._skill_embeddings.T)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._skill_ids[i], float(scores[i])) for i in top_indices]

    def invalidate(self) -> None:
        """Clear cached embeddings when skills change."""
        self._skill_ids = []
        self._skill_embeddings = None


class FailureMemoryIndex:
    """Retrieval index over the failure memory bank.

    Each memory is a concrete worked example from a past failure. At eval
    time we retrieve the top-K most similar past examples and inject them
    as few-shot demonstrations in the prompt.

    The index is rebuilt from EvoKG.get_failure_memories() whenever the
    memory bank grows. Embeddings are computed once per memory and cached.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._memory_ids: list[str] = []
        self._memory_payloads: list[dict] = []
        self._memory_embeddings: np.ndarray | None = None
        # Cached question-text → embedding to avoid recomputing on rebuild
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model for failure memory: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)

    def rebuild(self, memories: list[dict]) -> None:
        """Rebuild the index from a list of failure-memory dicts.

        Each memory dict should have at least 'node_id' and 'question'.
        Other fields ('expected', 'wrong_answer', 'reasoning', 'task_type',
        'skill_id') are kept as the payload returned by retrieve().
        """
        if not memories:
            self._memory_ids = []
            self._memory_payloads = []
            self._memory_embeddings = None
            return

        self._load_model()
        ids = []
        payloads = []
        texts_to_encode = []  # list of (index, question_text)
        cached_embs = []      # list of (index, embedding)
        for m in memories:
            mid = m.get("node_id", "")
            q = m.get("question", "")[:512]
            ids.append(mid)
            payloads.append(m)
            if q in self._embedding_cache:
                cached_embs.append((len(ids) - 1, self._embedding_cache[q]))
            else:
                texts_to_encode.append((len(ids) - 1, q))

        # Encode the new ones in a single batch
        new_embs = None
        if texts_to_encode:
            new_texts = [t[1] for t in texts_to_encode]
            new_embs = self._model.encode(new_texts, normalize_embeddings=True)
            for (_, q), emb in zip(texts_to_encode, new_embs):
                self._embedding_cache[q] = emb

        # Determine embedding dimension from any available source
        dim = None
        if cached_embs:
            dim = cached_embs[0][1].shape[0]
        elif new_embs is not None and len(new_embs) > 0:
            dim = new_embs.shape[1]
        if dim is None:
            self._memory_ids = []
            self._memory_payloads = []
            self._memory_embeddings = None
            return

        n = len(ids)
        embs = np.zeros((n, dim), dtype=np.float32)
        for idx, emb in cached_embs:
            embs[idx] = emb
        if new_embs is not None:
            for (idx, _q), emb in zip(texts_to_encode, new_embs):
                embs[idx] = emb

        self._memory_ids = ids
        self._memory_payloads = payloads
        self._memory_embeddings = embs

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        task_type_filter: str | None = None,
        skill_id_filter: str | None = None,
        min_score: float = 0.4,
    ) -> list[dict]:
        """Retrieve top-K similar past failure memories.

        Returns full payload dicts (including reasoning, expected answer).

        Filters (applied in combination):
        - ``task_type_filter``: keep only memories whose payload
          ``task_type`` equals the filter.
        - ``skill_id_filter`` (v13): keep only memories whose payload
          ``skill_id`` equals the filter. Used by multi-step chain
          reasoning so each chain step retrieves memories for ITS
          sub-skill, not the full target skill.

        If a filter removes all memories, it is dropped and the
        unfiltered pool is used (graceful degradation).
        """
        if self._memory_embeddings is None or not self._memory_ids:
            return []

        self._load_model()
        q_emb = self._model.encode([query[:512]], normalize_embeddings=True)
        scores = (q_emb @ self._memory_embeddings.T)[0]

        # Apply task_type filter
        valid = list(range(len(self._memory_ids)))
        if task_type_filter:
            filtered = [
                i for i in valid
                if self._memory_payloads[i].get("task_type") == task_type_filter
            ]
            if filtered:  # only apply if it doesn't wipe out the pool
                valid = filtered

        # v13: Apply skill_id filter (per-step chain retrieval)
        if skill_id_filter:
            filtered = [
                i for i in valid
                if self._memory_payloads[i].get("skill_id") == skill_id_filter
            ]
            if filtered:
                valid = filtered

        valid_scored = [(i, scores[i]) for i in valid if scores[i] >= min_score]
        valid_scored.sort(key=lambda x: -x[1])

        return [
            {**self._memory_payloads[i], "_score": float(s)}
            for i, s in valid_scored[:top_k]
        ]

    def size(self) -> int:
        return len(self._memory_ids)


class SuccessMemoryIndex(FailureMemoryIndex):
    """Retrieval index over success memories (v15.5).

    Semantically the mirror of FailureMemoryIndex: each indexed memory is
    a past question the learner answered correctly, together with:

    * ``correct_answer`` — the final answer that was judged correct.
    * ``reasoning_trace`` — the learner's own CoT for that answer.
    * ``decomposition`` — list of (step_name, step_output) if the answer
      came from a multi-step ``_run_skill_chain`` traversal. Empty for
      single-shot answers.

    Retrieval uses the same embedding + cosine-similarity logic inherited
    from ``FailureMemoryIndex``; the only override is the log message so
    operators can distinguish the two caches in diagnostics.

    The payload format is DIFFERENT from failure memories — consumers
    (the learner's ``_format_memories_as_examples``) should dispatch on a
    ``kind`` field that downstream callers inject after retrieve().
    """

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(
            f"Loading embedding model for success memory: {self._model_name}"
        )
        self._model = SentenceTransformer(self._model_name)
