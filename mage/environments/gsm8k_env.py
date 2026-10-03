"""GSM8K environment adapter for MAGE.

Grade-school math word problems requiring multi-step arithmetic reasoning.
Each question decomposes into 2-5 computation steps — ideal for demonstrating
v13 chain reasoning where the EvoKG's skill chains guide step-by-step solving.

Skills:
- arithmetic_extraction: identify the numbers and operations in the problem
- multi_step_computation: chain arithmetic operations in the correct order
- unit_tracking: keep track of units (dollars, meters, items) across steps
- answer_formatting: extract the final numeric answer
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

GSM8K_SKILLS = {
    "arithmetic_extraction": {
        "name": "Arithmetic Extraction",
        "category": "extraction",
        "prereqs": [],
        "description": "Identify the key numbers, quantities, and arithmetic operations stated in a math word problem",
    },
    "multi_step_computation": {
        "name": "Multi-Step Computation",
        "category": "reasoning",
        "prereqs": ["arithmetic_extraction"],
        "description": "Chain multiple arithmetic operations (add, subtract, multiply, divide) in the correct order to solve the problem",
    },
    "unit_tracking": {
        "name": "Unit Tracking",
        "category": "reasoning",
        "prereqs": ["arithmetic_extraction"],
        "description": "Keep track of units (dollars, meters, items, hours) across computation steps to avoid unit errors",
    },
    "answer_formatting": {
        "name": "Answer Formatting",
        "category": "extraction",
        "prereqs": ["multi_step_computation", "unit_tracking"],
        "description": "Extract the final numeric answer from the reasoning chain and output it in the expected format",
    },
}


class GSM8KEnvironment(ReasoningEnvironment):
    """GSM8K math reasoning environment for MAGE.

    Uses the OpenAI GSM8K dataset (1319 test problems). Each problem
    requires 2-5 arithmetic steps. The ground truth includes step-by-step
    reasoning with calculator annotations.
    """

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
            f"Loading GSM8K ({self._split}, max {self._max_instances}, "
            f"held_out {self._held_out_size})"
        )
        ds = load_dataset(
            "openai/gsm8k", "main",
            split=f"{self._split}[:{total_needed}]",
        )

        all_items: list[dict] = []
        for i, ex in enumerate(ds):
            question = ex["question"]
            answer_text = ex["answer"]

            # Extract final numeric answer after ####
            final_answer = self._extract_gold_answer(answer_text)

            # Count reasoning steps (lines with <<...>> calculations)
            n_steps = len(re.findall(r"<<.+?>>", answer_text))

            # Classify by complexity
            if n_steps <= 2:
                skill_id = "multi_step_computation"
            elif n_steps <= 4:
                skill_id = "answer_formatting"
            else:
                skill_id = "answer_formatting"

            all_items.append({
                "id": f"gsm8k_{i}",
                "question": question,
                "answer": str(final_answer),
                "full_solution": answer_text,
                "n_steps": n_steps,
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
            f"Loaded {len(self._data)} GSM8K train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
        )

    @staticmethod
    def _extract_gold_answer(answer_text: str) -> str:
        """Extract the numeric answer after #### marker."""
        m = re.search(r"####\s*(.+)", answer_text)
        if m:
            return m.group(1).strip().replace(",", "")
        return answer_text.strip().split("\n")[-1].strip()

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def domain(self) -> str:
        return "math"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in GSM8K_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return ["extract", "compute", "format", "answer"]

    def get_question(self, instance_id: str) -> tuple[str, str, str]:
        self._load_data()
        for item in self._data:
            if item["id"] == instance_id:
                return item["question"], "", item["answer"]
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
                context="",  # no external context — pure reasoning
                expected_answer=item["answer"],
                metadata={
                    "n_steps": item["n_steps"],
                    "task_type": f"gsm8k_{item['n_steps']}step",
                },
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
                context="",
                expected_answer=item["answer"],
                metadata={
                    "n_steps": item["n_steps"],
                    "task_type": f"gsm8k_{item['n_steps']}step",
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
        """Run a math QA episode."""
        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"Problem: {instruction}\n\n"
            f"Show your work, then give the final numeric answer after 'Answer:'."
        )

        response = agent_fn(prompt, instruction)
        final_answer = self._extract_answer_from_response(response)

        return EpisodeResult(
            observations=[f"Problem: {instruction}", f"Response: {response}"],
            actions=[response],
            reward=0.0,
            success=False,
            metadata={"final_answer": final_answer},
        )

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract numeric answer from LLM response."""
        # Try "Answer: X" pattern
        m = re.search(r"(?:Answer|ANSWER|answer|Final answer)\s*[:\s]\s*\$?\s*([-\d,]+(?:\.\d+)?)", response)
        if m:
            return m.group(1).strip().replace(",", "")

        # Try #### pattern (if model mimics GSM8K format)
        m = re.search(r"####\s*([-\d,]+(?:\.\d+)?)", response)
        if m:
            return m.group(1).strip().replace(",", "")

        # Last number in the response
        nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", response)
        if nums:
            return nums[-1].replace(",", "")

        return response.strip().split("\n")[-1].strip()

    def evaluate_answer(self, predicted: str, expected: str, **kwargs) -> float:
        """Numeric comparison with tolerance."""
        pred_clean = re.sub(r"[,$\s]", "", str(predicted).strip())
        exp_clean = re.sub(r"[,$\s]", "", str(expected).strip())

        # Exact string match
        if pred_clean == exp_clean:
            return 1.0

        # Numeric comparison
        try:
            p = float(pred_clean)
            e = float(exp_clean)
            if e == 0:
                return 1.0 if p == 0 else 0.0
            if abs(p - e) / max(abs(e), 1e-9) <= 0.01:  # 1% tolerance
                return 1.0
            if abs(p - e) <= 0.5:  # absolute tolerance for small numbers
                return 1.0
        except (ValueError, TypeError):
            pass

        return 0.0

    def build_evokg(self) -> EvoKG:
        return build_gsm8k_evokg()


def build_gsm8k_evokg() -> EvoKG:
    """Build an EvoKG for GSM8K with math reasoning skill prerequisites."""
    evokg = EvoKG()

    for sid, sdef in GSM8K_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="math",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))

    for sid, sdef in GSM8K_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    evokg.add_node(EnvironmentNode(
        "env_arithmetic", "Arithmetic Operations",
        entity_type="operation", domain="math",
    ))

    evokg.add_node(TaskNode(
        "task_gsm8k", "Solve grade-school math word problems",
        difficulty=0.6, domain="math",
    ))

    logger.info(f"Built GSM8K EvoKG: {evokg}")
    return evokg
