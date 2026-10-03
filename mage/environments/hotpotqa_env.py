"""HotpotQA environment adapter for MAGE.

Multi-hop text QA over Wikipedia paragraphs. Two question types:
- Bridge: requires chaining information across documents
- Comparison: requires comparing entities from different documents

Skills map naturally to reasoning capabilities:
- fact_retrieval: identify relevant supporting paragraphs
- bridge_reasoning: chain facts across documents (A→B→answer)
- comparison_reasoning: compare attributes from multiple documents
- multi_step_decomposition: break complex questions into sub-questions
- answer_extraction: extract precise answer from reasoning chain
"""

from __future__ import annotations

import logging
import random
import re
import string
from collections import Counter
from typing import Any

from mage.core.evokg import EvoKG
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    RelationType,
    TaskNode,
)
from mage.environments.base import (
    EpisodeResult,
    EvaluationInstance,
    ReasoningEnvironment,
    SkillDefinition,
)

logger = logging.getLogger(__name__)

HOTPOTQA_SKILLS = {
    # Reasoning skills
    "fact_retrieval": {
        "name": "Fact Retrieval",
        "category": "retrieval",
        "prereqs": [],
        "description": "Identify and retrieve relevant supporting paragraphs from context",
    },
    "single_hop_reasoning": {
        "name": "Single-Hop Reasoning",
        "category": "reasoning",
        "prereqs": ["fact_retrieval", "tool_search"],
        "description": "Answer questions requiring one retrieval step, using search tools",
    },
    "bridge_reasoning": {
        "name": "Bridge Reasoning",
        "category": "reasoning",
        "prereqs": ["fact_retrieval", "single_hop_reasoning"],
        "description": "Chain facts across documents to answer bridge questions",
    },
    "comparison_reasoning": {
        "name": "Comparison Reasoning",
        "category": "reasoning",
        "prereqs": ["fact_retrieval", "single_hop_reasoning", "tool_compare"],
        "description": "Compare attributes from multiple documents using comparison tools",
    },
    "answer_extraction": {
        "name": "Answer Extraction",
        "category": "extraction",
        "prereqs": ["fact_retrieval"],
        "description": "Extract precise answer spans from reasoning chain",
    },
    "multi_step_decomposition": {
        "name": "Multi-Step Decomposition",
        "category": "reasoning",
        "prereqs": ["single_hop_reasoning", "bridge_reasoning"],
        "description": "Break complex multi-hop questions into sub-questions",
    },
    # Tool skills
    "tool_search": {
        "name": "Paragraph Search",
        "category": "tool",
        "prereqs": [],
        "description": "Use search_paragraphs and extract_entities to find relevant facts in context",
    },
    "tool_compare": {
        "name": "Value Comparison",
        "category": "tool",
        "prereqs": [],
        "description": "Use compare_values to compare numbers, dates, and attributes",
    },
}

# Reasoning step types (action space for this environment)
REASONING_ACTIONS = [
    "retrieve",      # retrieve relevant paragraphs
    "decompose",     # break question into sub-questions
    "reason",        # apply reasoning over retrieved facts
    "compare",       # compare entities/attributes
    "answer",        # produce final answer
]


class HotpotQAEnvironment(ReasoningEnvironment):
    """HotpotQA multi-hop QA environment for MAGE.

    Uses the distractor setting: 10 paragraphs provided, 2 are gold supporting.
    Agent must identify supporting facts and reason to the answer.
    """

    def __init__(
        self,
        split: str = "validation",
        max_instances: int = 500,
        held_out_size: int = 0,
        seed: int = 42,
        select_paragraphs: bool = False,
        top_k_paragraphs: int = 5,
    ):
        self._split = split
        self._max_instances = max_instances
        self._held_out_size = held_out_size
        self._seed = seed
        self._data: list[dict] = []
        self._test_data: list[dict] = []
        self._loaded = False
        # Paragraph selection: filter the 10 distractor paragraphs to top-K.
        # DISABLED by default — keyword-based filtering hurts bridge questions
        # where the second-hop paragraph has low surface overlap with the
        # question. The unfiltered 10-paragraph context is the safer baseline.
        self._select_paragraphs = select_paragraphs
        self._top_k_paragraphs = top_k_paragraphs

    def _load_data(self) -> None:
        if self._loaded:
            return

        from datasets import load_dataset

        total_needed = self._max_instances + self._held_out_size
        logger.info(
            f"Loading HotpotQA ({self._split}, max {self._max_instances}, "
            f"held_out {self._held_out_size})"
        )
        ds = load_dataset(
            "hotpotqa/hotpot_qa", "distractor",
            split=f"{self._split}[:{total_needed}]",
        )

        all_items: list[dict] = []
        for ex in ds:
            # Store paragraphs as a list of (title, text) so we can filter
            paragraphs = []
            for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
                para = " ".join(sentences)
                paragraphs.append((title, para))

            # Build full context (all 10 paragraphs)
            full_context = "\n\n".join(f"[{t}]\n{p}" for t, p in paragraphs)

            # Optionally filter to the top-K most question-relevant paragraphs
            question = ex["question"]
            if self._select_paragraphs:
                selected = self._select_top_paragraphs(
                    question, paragraphs, top_k=self._top_k_paragraphs,
                )
                context = "\n\n".join(f"[{t}]\n{p}" for t, p in selected)
            else:
                context = full_context

            # Determine skill type based on question type
            q_type = ex.get("type", "bridge")
            if q_type == "comparison":
                skill_id = "comparison_reasoning"
            else:
                skill_id = "bridge_reasoning"

            level = ex.get("level", "medium")

            all_items.append({
                "id": ex["id"],
                "question": question,
                "answer": ex["answer"],
                "type": q_type,
                "level": level,
                "context": context,
                "full_context": full_context,
                "paragraphs": paragraphs,
                "supporting_facts": list(zip(
                    ex["supporting_facts"]["title"],
                    ex["supporting_facts"]["sent_id"],
                )),
                "skill_id": skill_id,
            })

        random.Random(self._seed).shuffle(all_items)
        self._data = all_items[: self._max_instances]
        self._test_data = (
            all_items[self._max_instances : self._max_instances + self._held_out_size]
            if self._held_out_size > 0
            else []
        )
        self._loaded = True
        logger.info(
            f"Loaded {len(self._data)} HotpotQA train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
        )
        logger.info(
            f"Loaded {len(self._data)} HotpotQA instances "
            f"(select_paragraphs={self._select_paragraphs}, top_k={self._top_k_paragraphs})"
        )

    @staticmethod
    def _select_top_paragraphs(
        question: str,
        paragraphs: list[tuple[str, str]],
        top_k: int = 3,
    ) -> list[tuple[str, str]]:
        """Score paragraphs by relevance to the question and return top-K.

        Scoring: word-overlap (lemma-ish) + entity match (capitalized words
        from the question that appear in the paragraph or its title).
        Pure deterministic — no LLM call.
        """
        import re

        def normalize(text: str) -> set[str]:
            words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
            stop = {
                "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
                "or", "but", "is", "are", "was", "were", "be", "been", "being",
                "what", "which", "who", "whom", "whose", "where", "when", "why",
                "how", "did", "does", "do", "this", "that", "these", "those",
                "with", "from", "by", "as", "it", "its",
            }
            return {w for w in words if len(w) > 2 and w not in stop}

        def entities(text: str) -> set[str]:
            # Capitalized multi-word phrases (proper nouns)
            return set(re.findall(r"[A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+)*", text))

        q_words = normalize(question)
        q_entities = {e.lower() for e in entities(question)}

        scored = []
        for title, text in paragraphs:
            p_words = normalize(text)
            p_words.update(normalize(title))
            word_overlap = len(q_words & p_words)

            t_lower = title.lower()
            entity_overlap = sum(
                1 for e in q_entities
                if e in t_lower or e in text.lower()
            )

            # Title match is strongest signal in HotpotQA
            title_in_q = 1 if t_lower in question.lower() else 0
            score = word_overlap + 3 * entity_overlap + 5 * title_in_q
            scored.append((score, title, text))

        scored.sort(key=lambda x: -x[0])
        return [(t, p) for _, t, p in scored[:top_k]]

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def domain(self) -> str:
        return "qa"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in HOTPOTQA_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return list(REASONING_ACTIONS)

    def get_question(self, instance_id: str) -> tuple[str, str, str]:
        self._load_data()
        for item in self._data:
            if item["id"] == instance_id:
                return item["question"], item["context"], item["answer"]
        raise KeyError(f"Instance {instance_id} not found")

    def get_evaluation_instances(
        self, skill_id: str | None = None, num_instances: int = 10,
    ) -> list[EvaluationInstance]:
        self._load_data()

        candidates = self._data
        if skill_id:
            candidates = [d for d in self._data if d["skill_id"] == skill_id]

        selected = candidates[:num_instances]

        return [
            EvaluationInstance(
                instance_id=item["id"],
                instruction=item["question"],
                target_skill=item["skill_id"],
                context=item["context"],
                expected_answer=item["answer"],
                metadata={"type": item["type"], "level": item["level"], "task_type": item["type"]},
            )
            for item in selected
        ]

    def get_test_instances(self, num_instances: int = 100) -> list[EvaluationInstance]:
        """Held-out test instances (disjoint from train pool; see base class)."""
        self._load_data()
        selected = self._test_data[:num_instances]
        return [
            EvaluationInstance(
                instance_id=item["id"],
                instruction=item["question"],
                target_skill=item["skill_id"],
                context=item["context"],
                expected_answer=item["answer"],
                metadata={
                    "type": item["type"],
                    "level": item["level"],
                    "task_type": item["type"],
                    "held_out": True,
                },
            )
            for item in selected
        ]

    def run_episode(
        self,
        agent_fn: Any,
        instruction: str,
        seed: int | None = None,
        max_steps: int = 5,
        context: str = "",
    ) -> EpisodeResult:
        """Run a QA episode. Clean single-shot — don't interfere with model ability."""
        observations = [f"Question: {instruction}\n\nContext:\n{context}"]
        actions = []

        prompt = (
            f"Answer the following question based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {instruction}\n\n"
            f"Give your answer directly. Answer:"
        )

        response = agent_fn(prompt, instruction)
        actions.append(response)
        final_answer = self._extract_answer_from_response(response)
        observations.append(f"Final answer: {final_answer}")

        return EpisodeResult(
            observations=observations,
            actions=actions,
            reward=0.0,
            success=False,
            metadata={"final_answer": final_answer, "num_steps": 1},
        )

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from LLM response."""
        # Try "Answer: X" pattern
        for pattern in [
            r"(?:Answer|ANSWER|answer)\s*:\s*(.+?)(?:\n|$)",
            r"(?:Final answer|FINAL ANSWER)\s*:\s*(.+?)(?:\n|$)",
            r"(?:The answer is|the answer is)\s+(.+?)(?:\.|$)",
        ]:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()

        # Try to find a short answer at the end (after reasoning)
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            # Take the shortest line from the last 3 — likely the actual answer
            candidates = lines[-3:] if len(lines) >= 3 else lines
            shortest = min(candidates, key=len)
            # If it's very short (< 50 chars), it's likely the answer
            if len(shortest) < 50:
                return shortest
            return lines[-1]
        return response.strip()

    def evaluate_answer(self, predicted: str, expected: str) -> float:
        """F1-based evaluation (standard HotpotQA metric)."""
        return _compute_f1(predicted, expected)

    def build_evokg(self) -> EvoKG:
        return build_hotpotqa_evokg()


def build_hotpotqa_evokg() -> EvoKG:
    """Build an EvoKG for HotpotQA with reasoning skill prerequisites."""
    evokg = EvoKG()

    for sid, sdef in HOTPOTQA_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="qa",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))

    for sid, sdef in HOTPOTQA_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    # Environment nodes (document types)
    for entity in ["wikipedia_paragraph", "supporting_fact", "bridge_entity", "comparison_entity"]:
        evokg.add_node(EnvironmentNode(
            node_id=f"env_{entity}", name=entity.replace("_", " ").title(),
            entity_type="document_element", domain="qa",
        ))

    evokg.add_node(TaskNode(
        "task_hotpotqa", "Answer multi-hop questions over Wikipedia",
        difficulty=0.7, domain="qa",
    ))

    logger.info(f"Built HotpotQA EvoKG: {evokg}")
    return evokg


# ---- Standard QA metrics ----

def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
