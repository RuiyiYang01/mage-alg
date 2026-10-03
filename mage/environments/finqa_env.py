"""FinQA environment adapter for MAGE.

Numerical financial reasoning over S&P 500 10-K filings (Chen et al., EMNLP 2021).
Each question requires extracting numbers from a financial passage (often
tabular) and chaining 1-5 arithmetic operations to compute a percentage,
ratio, or growth rate. Loaded via the ChanceFocus/flare-finqa HuggingFace
mirror — the original ibm-research/finqa uses a deprecated loading script.

Skills mirror MAGE's reasoning hierarchy:
- table_extraction: read the relevant numbers from the financial context
- numerical_reasoning: chain arithmetic (subtraction, division, ratio, growth)
- percentage_computation: compute percentage change / share / margin
- multi_step_finance: combine multiple operations across rows/years
- answer_formatting: emit the final number in the expected format
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

FINQA_SKILLS = {
    "table_extraction": {
        "name": "Financial Table Extraction",
        "category": "extraction",
        "prereqs": [],
        "description": "Identify the relevant financial figures (revenue, expenses, share counts, dates) referenced by the question from a 10-K filing context",
    },
    "numerical_reasoning": {
        "name": "Numerical Reasoning",
        "category": "reasoning",
        "prereqs": ["table_extraction"],
        "description": "Chain arithmetic operations (subtract, divide, ratio) to compute a single derived quantity from extracted figures",
    },
    "percentage_computation": {
        "name": "Percentage Computation",
        "category": "reasoning",
        "prereqs": ["numerical_reasoning"],
        "description": "Compute percentage change, percentage share, or margin from two or more financial quantities and emit a decimal or percent",
    },
    "multi_step_finance": {
        "name": "Multi-Step Financial Computation",
        "category": "reasoning",
        "prereqs": ["numerical_reasoning"],
        "description": "Combine multiple arithmetic steps across rows/years (e.g. growth rate over multi-year window, weighted average) to derive the answer",
    },
    "answer_formatting": {
        "name": "Numeric Answer Formatting",
        "category": "extraction",
        "prereqs": ["percentage_computation", "multi_step_finance"],
        "description": "Emit the final numeric answer in the dataset's expected format (decimal vs percent, rounding) and strip stray units",
    },
}


class FinQAEnvironment(ReasoningEnvironment):
    """FinQA financial-numerical-reasoning environment for MAGE.

    Uses ChanceFocus/flare-finqa (HF mirror of the original FinQA dataset).
    1,147 test questions; each pairs a 10-K passage with a question and a
    single numeric gold answer. We follow GSM8K-style auto-eval: numeric
    comparison with relative + absolute tolerance.
    """

    HF_PATH = "ChanceFocus/flare-finqa"

    def __init__(
        self,
        split: str = "test",
        max_instances: int = 500,
        held_out_size: int = 0,
        seed: int = 42,
        max_context_chars: int = 4000,
    ):
        self._split = split
        self._max_instances = max_instances
        self._held_out_size = held_out_size
        self._seed = seed
        self._max_context_chars = max_context_chars
        self._data: list[dict] = []
        self._test_data: list[dict] = []
        self._loaded = False

    def _load_data(self) -> None:
        if self._loaded:
            return

        from datasets import load_dataset

        total_needed = self._max_instances + self._held_out_size
        logger.info(
            f"Loading FinQA ({self._split}, max {self._max_instances}, "
            f"held_out {self._held_out_size})"
        )
        ds = load_dataset(
            self.HF_PATH,
            split=f"{self._split}[:{total_needed}]",
        )

        all_items: list[dict] = []
        for i, ex in enumerate(ds):
            qid = ex.get("id", f"finqa_{i}")
            query = ex["query"]
            answer = str(ex["answer"]).strip()
            question = ex["text"]

            context, _ = self._split_query(query)
            if len(context) > self._max_context_chars:
                context = context[: self._max_context_chars]

            skill_id = self._classify(question, answer)

            all_items.append({
                "id": qid,
                "question": question,
                "context": context,
                "answer": answer,
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
            f"Loaded {len(self._data)} FinQA train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
        )

    @staticmethod
    def _split_query(query: str) -> tuple[str, str]:
        """Split flare-finqa's combined `query` into (context, instruction).

        flare-finqa packs both into one string. We strip the standard prefix
        and split off the trailing question if present.
        """
        text = query
        prefix = "Please answer the given financial question based on the context."
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
        if text.startswith("Context:"):
            text = text[len("Context:"):].strip()
        m = re.search(r"\n?Question:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if m:
            return text[: m.start()].strip(), m.group(1).strip()
        return text.strip(), ""

    @staticmethod
    def _classify(question: str, answer: str) -> str:
        q = question.lower()
        if "%" in q or "percent" in q or "percentage" in q:
            return "percentage_computation"
        if "growth" in q or "change" in q or "increase" in q or "decrease" in q:
            return "multi_step_finance"
        # Pure ratio / division (single step) → numerical_reasoning
        if any(k in q for k in ["ratio", "per share", "fraction"]):
            return "numerical_reasoning"
        # Default: numerical_reasoning bucket
        return "numerical_reasoning"

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "finqa"

    @property
    def domain(self) -> str:
        return "finance"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in FINQA_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return ["extract", "compute", "format", "answer"]

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
                metadata={"task_type": item["skill_id"]},
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
                context=item["context"],
                expected_answer=item["answer"],
                metadata={"task_type": item["skill_id"], "held_out": True},
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
            "You are a financial analyst. Use the context (which may include "
            "tables) to answer the question. Show your computation, then give "
            "the final numeric answer after 'Answer:'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {instruction}\n\n"
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
        """Extract a numeric answer from the LLM response.

        Order of attempts:
        1. "Answer: X" / "Final answer: X" pattern
        2. "####" GSM8K-style marker
        3. Last number in the response
        """
        for pat in [
            r"(?:Answer|ANSWER|Final answer|final answer)\s*[:\s]\s*\$?\s*([-\d,]+(?:\.\d+)?\s*%?)",
        ]:
            m = re.search(pat, response)
            if m:
                return m.group(1).strip().replace(",", "").rstrip("%").strip()
        m = re.search(r"####\s*([-\d,]+(?:\.\d+)?)", response)
        if m:
            return m.group(1).strip().replace(",", "")
        nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", response)
        if nums:
            return nums[-1].replace(",", "")
        return response.strip().split("\n")[-1].strip()

    def evaluate_answer(self, predicted: str, expected: str, **kwargs) -> float:
        """Numeric comparison with FinQA-style tolerance.

        FinQA gold answers are mixed: percentage values may appear as decimal
        (0.14464) or percent (14.464). We try both interpretations.
        """
        def _to_float(s: str) -> float | None:
            s = re.sub(r"[,$\s%]", "", str(s).strip())
            if not s:
                return None
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        p = _to_float(predicted)
        e = _to_float(expected)
        if p is None or e is None:
            return 1.0 if str(predicted).strip() == str(expected).strip() else 0.0

        # Direct comparison
        if self._numeric_match(p, e):
            return 1.0
        # Percent ↔ decimal swap (FinQA gold often stores 0.14 for 14%)
        if self._numeric_match(p / 100.0, e):
            return 1.0
        if self._numeric_match(p * 100.0, e):
            return 1.0
        return 0.0

    @staticmethod
    def _numeric_match(p: float, e: float) -> bool:
        if e == 0:
            return abs(p) < 0.01
        if abs(p - e) / max(abs(e), 1e-9) <= 0.01:
            return True
        if abs(p - e) <= 0.05:
            return True
        return False

    def build_evokg(self) -> EvoKG:
        return build_finqa_evokg()


def build_finqa_evokg() -> EvoKG:
    evokg = EvoKG()
    for sid, sdef in FINQA_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="finance",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))
    for sid, sdef in FINQA_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))
    evokg.add_node(EnvironmentNode(
        "env_filings", "S&P 500 10-K Filings",
        entity_type="document_corpus", domain="finance",
    ))
    evokg.add_node(TaskNode(
        "task_finqa", "Answer numerical questions about financial filings",
        difficulty=0.7, domain="finance",
    ))
    logger.info(f"Built FinQA EvoKG: {evokg}")
    return evokg
