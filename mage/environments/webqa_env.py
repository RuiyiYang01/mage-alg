"""WebQuestions environment adapter for MAGE.

KG-based factoid QA over Freebase. Questions require entity recognition
and relation traversal to find answers.

Skills:
- entity_recognition: identify the topic entity in the question
- relation_identification: determine which KG relation to traverse
- answer_retrieval: extract the final answer from KG traversal
- complex_question_parsing: handle questions with constraints or multi-part answers
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

WEBQA_SKILLS = {
    "entity_recognition": {
        "name": "Entity Recognition",
        "category": "retrieval",
        "prereqs": [],
        "description": "Identify the topic entity mentioned in the question",
    },
    "relation_identification": {
        "name": "Relation Identification",
        "category": "reasoning",
        "prereqs": ["entity_recognition"],
        "description": "Determine which knowledge graph relation to follow",
    },
    "answer_retrieval": {
        "name": "Answer Retrieval",
        "category": "extraction",
        "prereqs": ["entity_recognition", "relation_identification"],
        "description": "Extract the answer by traversing the knowledge graph",
    },
    "multi_answer_handling": {
        "name": "Multi-Answer Handling",
        "category": "reasoning",
        "prereqs": ["answer_retrieval"],
        "description": "Handle questions with multiple valid answers",
    },
    "complex_question_parsing": {
        "name": "Complex Question Parsing",
        "category": "reasoning",
        "prereqs": ["entity_recognition", "relation_identification"],
        "description": "Parse questions with constraints, superlatives, or temporal conditions",
    },
}

REASONING_ACTIONS = ["identify_entity", "find_relation", "traverse", "answer"]


class WebQuestionsEnvironment(ReasoningEnvironment):
    """WebQuestions KG-QA environment for MAGE.

    Uses the Google WebQuestions dataset (Freebase-based factoid QA).
    """

    def __init__(
        self,
        split: str = "test",
        max_instances: int = 500,
        held_out_size: int = 0,
        seed: int = 42,
        use_wikipedia: bool = False,  # OPT-IN: caused -7pp regression in v11 testing
    ):
        self._split = split
        self._max_instances = max_instances
        # v15.5-large: held-out pool for generalization eval. Loaded alongside
        # the training pool, never consumed by the evolution loop. Populated
        # only when held_out_size > 0.
        self._held_out_size = held_out_size
        self._seed = seed
        self._data: list[dict] = []
        self._test_data: list[dict] = []
        self._loaded = False
        self._use_wikipedia = use_wikipedia

    def _load_data(self) -> None:
        if self._loaded:
            return

        from datasets import load_dataset

        total_needed = self._max_instances + self._held_out_size
        logger.info(
            f"Loading WebQuestions ({self._split}, max {self._max_instances}, "
            f"held_out {self._held_out_size})"
        )
        ds = load_dataset("web_questions", split=f"{self._split}[:{total_needed}]")

        all_items: list[dict] = []
        for i, ex in enumerate(ds):
            # Classify question complexity
            question = ex["question"]
            answers = ex["answers"]
            skill_id = self._classify_question(question, answers)

            all_items.append({
                "id": f"webq_{i}",
                "question": question,
                "answers": answers,
                "context": "",  # populated by Wikipedia retrieval below
                "skill_id": skill_id,
                "url": ex.get("url", ""),
            })

        # v15.5-large: deterministic shuffle → split into train + held-out.
        random.Random(self._seed).shuffle(all_items)
        self._data = all_items[: self._max_instances]
        self._test_data = (
            all_items[self._max_instances : self._max_instances + self._held_out_size]
            if self._held_out_size > 0
            else []
        )

        # Populate Wikipedia context for each instance using the dataset's
        # Freebase URL field. The dataset *includes* this URL, so retrieval
        # is fair-game (this is the WebQuestionsSP-style setting).
        if self._use_wikipedia:
            from mage.tools.wikipedia_tool import fetch_for_freebase_url
            n_with_context = 0
            for item in self._data:
                ctx = fetch_for_freebase_url(item["url"], max_chars=600)
                if ctx:
                    item["context"] = ctx
                    n_with_context += 1
            logger.info(
                f"Populated Wikipedia context for {n_with_context}/{len(self._data)} instances"
            )

        self._loaded = True
        logger.info(
            f"Loaded {len(self._data)} WebQuestions train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
        )

    def _classify_question(self, question: str, answers: list[str]) -> str:
        """Classify question into skill type based on heuristics."""
        q_lower = question.lower()

        # Multi-answer
        if len(answers) > 3:
            return "multi_answer_handling"

        # Complex patterns
        complex_patterns = ["most", "first", "last", "biggest", "smallest",
                           "before", "after", "how many", "when did"]
        if any(p in q_lower for p in complex_patterns):
            return "complex_question_parsing"

        # Default: answer retrieval (simple entity-relation lookup)
        return "answer_retrieval"

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "webquestions"

    @property
    def domain(self) -> str:
        return "kgqa"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in WEBQA_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return list(REASONING_ACTIONS)

    def get_question(self, instance_id: str) -> tuple[str, str, str]:
        self._load_data()
        for item in self._data:
            if item["id"] == instance_id:
                return item["question"], item["context"], "|".join(item["answers"])
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
                expected_answer="|".join(item["answers"]),
                metadata={"num_answers": len(item["answers"]), "task_type": item["skill_id"]},
            )
            for item in selected
        ]

    def get_test_instances(self, num_instances: int = 100) -> list[EvaluationInstance]:
        """Return held-out test instances (disjoint from the training pool).

        Populated only if the environment was constructed with
        ``held_out_size > 0``. Returns an empty list otherwise.
        """
        self._load_data()
        selected = self._test_data[:num_instances]
        return [
            EvaluationInstance(
                instance_id=item["id"],
                instruction=item["question"],
                target_skill=item["skill_id"],
                context=item["context"],
                expected_answer="|".join(item["answers"]),
                metadata={
                    "num_answers": len(item["answers"]),
                    "task_type": item["skill_id"],
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
        max_steps: int = 3,
        context: str = "",
    ) -> EpisodeResult:
        """Run a QA episode for WebQuestions."""
        prompt = (
            f"Answer the following factoid question. Give a short, precise answer.\n\n"
            f"Question: {instruction}\n\n"
            f"Answer:"
        )

        response = agent_fn(prompt, instruction)
        final_answer = response.strip().split("\n")[0].strip()

        return EpisodeResult(
            observations=[f"Question: {instruction}", f"Response: {response}"],
            actions=[response],
            reward=0.0,
            success=False,
            metadata={"final_answer": final_answer},
        )

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from response — handles CoT output.

        v14: With CoT prompting, the model reasons first then writes
        'Answer: X'. Scan for that pattern before falling back to
        first-line extraction.
        """
        import re
        # Scan for "Answer: X" pattern (CoT output)
        for pat in [
            r"(?:Answer|ANSWER|Final answer|final answer)\s*:\s*(.+?)(?:\n|$)",
        ]:
            m = re.search(pat, response)
            if m:
                return m.group(1).strip()
        # Fallback: first line (legacy single-shot format)
        return response.strip().split("\n")[0].strip()

    def evaluate_answer(self, predicted: str, expected: str) -> float:
        """Check if prediction matches any of the accepted answers.

        Expected format: "answer1|answer2|answer3"
        """
        expected_answers = expected.split("|")
        pred_norm = _normalize(predicted)

        # Check exact match against any answer
        for ans in expected_answers:
            if _normalize(ans) == pred_norm:
                return 1.0

        # Check F1 against best matching answer
        best_f1 = 0.0
        for ans in expected_answers:
            f1 = _compute_f1(predicted, ans)
            best_f1 = max(best_f1, f1)

        return best_f1

    def build_evokg(self) -> EvoKG:
        return build_webqa_evokg()


def build_webqa_evokg() -> EvoKG:
    """Build an EvoKG for WebQuestions with KG reasoning skills."""
    evokg = EvoKG()

    for sid, sdef in WEBQA_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="kgqa",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))

    for sid, sdef in WEBQA_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    evokg.add_node(EnvironmentNode(
        "env_freebase", "Freebase Knowledge Graph",
        entity_type="knowledge_base", domain="kgqa",
    ))

    evokg.add_node(TaskNode(
        "task_webqa", "Answer factoid questions using knowledge graph reasoning",
        difficulty=0.6, domain="kgqa",
    ))

    logger.info(f"Built WebQuestions EvoKG: {evokg}")
    return evokg


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _compute_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)
