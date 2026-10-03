"""MedQA-USMLE environment adapter for MAGE.

USMLE-style medical multiple-choice QA (Jin et al., Applied Sciences 2021).
1,273 4-option MCQs sourced from USMLE Step 1/2/3 practice. Each question
is a clinical vignette with 4 options labelled A-D and a single correct
letter. Auto-eval = exact-match on the chosen letter.

Skills mirror the implicit medical-reasoning hierarchy:
- clinical_reading: parse the vignette (patient history, exam, labs)
- differential_diagnosis: shortlist candidate diagnoses from findings
- mechanism_reasoning: reason about pathophysiology / pharmacology mechanism
- treatment_selection: choose appropriate management or next step
- option_elimination: eliminate distractors and commit to the best letter
"""

from __future__ import annotations

import logging
import random
import re
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

MEDQA_SKILLS = {
    "clinical_reading": {
        "name": "Clinical Vignette Reading",
        "category": "extraction",
        "prereqs": [],
        "description": "Identify the salient patient history, physical exam, and lab findings from a USMLE clinical vignette",
    },
    "differential_diagnosis": {
        "name": "Differential Diagnosis",
        "category": "reasoning",
        "prereqs": ["clinical_reading"],
        "description": "Shortlist candidate diagnoses consistent with the vignette's signs and symptoms",
    },
    "mechanism_reasoning": {
        "name": "Pathophysiology and Pharmacology Reasoning",
        "category": "reasoning",
        "prereqs": ["clinical_reading"],
        "description": "Reason about underlying disease mechanisms or drug pharmacology when the question asks why or how",
    },
    "treatment_selection": {
        "name": "Treatment / Next-Step Selection",
        "category": "reasoning",
        "prereqs": ["differential_diagnosis"],
        "description": "Choose the appropriate management, next diagnostic test, or treatment for the diagnosed condition",
    },
    "option_elimination": {
        "name": "Option Elimination",
        "category": "extraction",
        "prereqs": ["differential_diagnosis", "mechanism_reasoning", "treatment_selection"],
        "description": "Eliminate distractor options and commit to the single best answer letter (A/B/C/D)",
    },
}

LETTERS = ["A", "B", "C", "D"]


class MedQAEnvironment(ReasoningEnvironment):
    """MedQA-USMLE 4-option MCQ environment for MAGE.

    Uses GBaker/MedQA-USMLE-4-options on HuggingFace (1,273 test items).
    The dataset provides `question`, `options` (dict A-D), `answer` (full
    text), and `answer_idx` (letter). We score by letter match.
    """

    HF_PATH = "GBaker/MedQA-USMLE-4-options"

    def __init__(
        self,
        split: str = "test",
        max_instances: int = 500,
        held_out_size: int = 0,
        seed: int = 42,
    ):
        self._split = split
        self._max_instances = max_instances
        self._held_out_size = held_out_size
        self._seed = seed
        self._data: list[dict] = []
        self._test_data: list[dict] = []
        self._loaded = False

    def _load_data(self) -> None:
        if self._loaded:
            return

        from datasets import load_dataset

        total_needed = self._max_instances + self._held_out_size
        logger.info(
            f"Loading MedQA-USMLE ({self._split}, max {self._max_instances}, "
            f"held_out {self._held_out_size})"
        )
        ds = load_dataset(
            self.HF_PATH,
            split=f"{self._split}[:{total_needed}]",
        )

        all_items: list[dict] = []
        for i, ex in enumerate(ds):
            qid = f"medqa_{i}"
            question = ex["question"]
            options = ex["options"]
            answer_text = ex["answer"]
            answer_idx = str(ex.get("answer_idx", "")).strip().upper()
            meta_info = ex.get("meta_info", "step1")

            options_str = "\n".join(
                f"{k}. {options[k]}" for k in LETTERS if k in options
            )

            skill_id = self._classify(question)

            all_items.append({
                "id": qid,
                "question": question,
                "options": options,
                "options_str": options_str,
                "answer_letter": answer_idx,
                "answer_text": answer_text,
                "step": meta_info,
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
            f"Loaded {len(self._data)} MedQA train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
        )

    @staticmethod
    def _classify(question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["mechanism", "responsible for", "pharmacolog",
                                  "best explains", "due to which", "caused by"]):
            return "mechanism_reasoning"
        if any(k in q for k in ["next step", "best treatment", "management",
                                  "appropriate therapy", "most appropriate"]):
            return "treatment_selection"
        if any(k in q for k in ["most likely diagnosis", "diagnosis"]):
            return "differential_diagnosis"
        return "option_elimination"

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "medqa"

    @property
    def domain(self) -> str:
        return "medical"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in MEDQA_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return ["A", "B", "C", "D"]

    def _format_context(self, item: dict) -> str:
        return f"Options:\n{item['options_str']}"

    def get_question(self, instance_id: str) -> tuple[str, str, str]:
        self._load_data()
        for item in self._data:
            if item["id"] == instance_id:
                return item["question"], self._format_context(item), item["answer_letter"]
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
                context=self._format_context(item),
                expected_answer=item["answer_letter"],
                metadata={
                    "task_type": item["skill_id"],
                    "step": item["step"],
                    "answer_text": item["answer_text"],
                },
            )
            for item in selected
        ]

    def get_test_instances(self, num_instances: int = 100) -> list[EvaluationInstance]:
        self._load_data()
        selected = self._test_data[:num_instances]
        return [
            EvaluationInstance(
                instance_id=item["id"],
                instruction=item["question"],
                target_skill=item["skill_id"],
                context=self._format_context(item),
                expected_answer=item["answer_letter"],
                metadata={
                    "task_type": item["skill_id"],
                    "step": item["step"],
                    "answer_text": item["answer_text"],
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
        prompt = (
            "You are a medical expert taking the USMLE. Read the clinical "
            "vignette and choose the single best option (A, B, C, or D). "
            "Reason step by step, then give the final letter after 'Answer:'.\n\n"
            f"Question: {instruction}\n\n"
            f"{context}\n\n"
            "Let's think step by step."
        )
        response = agent_fn(prompt, instruction)
        final_answer = self._extract_answer_from_response(response)
        return EpisodeResult(
            observations=[f"Question: {instruction}", f"Response: {response}"],
            actions=[response],
            reward=0.0,
            success=False,
            metadata={"final_answer": final_answer},
        )

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract a letter A/B/C/D from the LLM response."""
        # Pattern: "Answer: B" or "answer is B"
        for pat in [
            r"(?:Answer|ANSWER|Final answer|final answer|answer is)\s*[:\s]\s*\(?([A-D])\)?",
            r"\boption\s+\(?([A-D])\)?\s+is\s+(?:the\s+)?(?:correct|best|right)",
            r"\b(?:choose|pick|select)\s+\(?([A-D])\)?",
        ]:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Last standalone A/B/C/D in the response
        letters = re.findall(r"\b([A-D])\b", response)
        if letters:
            return letters[-1].upper()
        return response.strip().split("\n")[-1].strip().upper()[:1]

    def evaluate_answer(self, predicted: str, expected: str, **kwargs) -> float:
        """Letter exact-match (case-insensitive)."""
        p = str(predicted).strip().upper()[:1]
        e = str(expected).strip().upper()[:1]
        return 1.0 if p == e and p in ("A", "B", "C", "D") else 0.0

    def build_evokg(self) -> EvoKG:
        return build_medqa_evokg()


def build_medqa_evokg() -> EvoKG:
    evokg = EvoKG()
    for sid, sdef in MEDQA_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="medical",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))
    for sid, sdef in MEDQA_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))
    evokg.add_node(EnvironmentNode(
        "env_usmle", "USMLE Medical Knowledge Base",
        entity_type="exam_corpus", domain="medical",
    ))
    evokg.add_node(TaskNode(
        "task_medqa", "Answer USMLE-style 4-option clinical MCQs",
        difficulty=0.65, domain="medical",
    ))
    logger.info(f"Built MedQA EvoKG: {evokg}")
    return evokg
