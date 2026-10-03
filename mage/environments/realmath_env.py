"""RealMath environment adapter for MAGE.

Research-level mathematics benchmark drawn from arXiv papers and Math
StackExchange posts. Each problem gives a theorem statement + surrounding
paper/post context, asks a specific mathematical question, and expects a
verifiable short-form answer (typically a LaTeX expression).

Source: https://github.com/ethz-spylab/RealMath (NeurIPS 2025 D&B)
Dataset: ``ethz-spylab/realmath`` on HuggingFace Hub with 3 splits:
  - ``Math_arXiv``       — 633 samples from math.arXiv
  - ``CS_arXiv``         — 111 samples from cs.arXiv
  - ``Math_StackExchange`` — 542 samples from Math StackExchange

Design choices:
  * Context = the paper/post LaTeX truncated to ~2000 chars (HotpotQA-style).
  * Answer = a LaTeX expression or phrase. Exact-match (after normalization)
    OR LLM-judge handles semantic equivalence (the critic in v15.5 wires
    the judge automatically).
  * Skills cover research-level mathematical reasoning. Multi-step derivations
    benefit from v13 chain reasoning → v15.5 stores the decomposition.
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

REALMATH_SKILLS = {
    "theorem_comprehension": {
        "name": "Theorem Comprehension",
        "category": "extraction",
        "prereqs": [],
        "description": (
            "Read a theorem statement together with its surrounding paper "
            "or post context, identify the quantities, operators, and "
            "constraints involved, and isolate what the question is asking"
        ),
    },
    "symbolic_manipulation": {
        "name": "Symbolic Manipulation",
        "category": "reasoning",
        "prereqs": ["theorem_comprehension"],
        "description": (
            "Perform algebraic/analytic manipulation of symbolic expressions "
            "— substitution, simplification, series expansion, use of known "
            "identities — to transform one expression into another"
        ),
    },
    "mathematical_derivation": {
        "name": "Mathematical Derivation",
        "category": "reasoning",
        "prereqs": ["symbolic_manipulation"],
        "description": (
            "Chain multiple symbolic manipulations into a complete derivation, "
            "using definitions, lemmas, and intermediate results to arrive at "
            "the final expression the theorem claims"
        ),
    },
    "expression_identification": {
        "name": "Expression Identification",
        "category": "extraction",
        "prereqs": ["mathematical_derivation"],
        "description": (
            "Recognize the expected form of the final answer (closed-form, "
            "piecewise, asymptotic, numeric) and extract it from the derivation "
            "in a comparable format"
        ),
    },
}

# Which HuggingFace splits to pull from. By default we mix all three.
_DEFAULT_SPLITS = ("Math_arXiv", "CS_arXiv", "Math_StackExchange")

# Cap on context length — the raw LaTeX can be 20k+ chars; inject ~2000 chars
# so the learner has the theorem's surrounding paragraphs without blowing up
# the prompt budget.
_MAX_CONTEXT_CHARS = 2000


class RealMathEnvironment(ReasoningEnvironment):
    """RealMath reasoning environment for MAGE.

    Uses the ``ethz-spylab/realmath`` HuggingFace dataset. Each instance
    carries the question, a truncated context window from the source paper
    or post, and the expected LaTeX answer.
    """

    def __init__(
        self,
        split: str = "all",
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

        logger.info(
            f"Loading RealMath ({self._split}, max {self._max_instances})"
        )

        if self._split == "all":
            splits_to_load = _DEFAULT_SPLITS
        elif self._split in _DEFAULT_SPLITS:
            splits_to_load = (self._split,)
        else:
            raise ValueError(
                f"Unknown RealMath split {self._split!r}; "
                f"expected one of {_DEFAULT_SPLITS + ('all',)}"
            )

        raw_records: list[dict[str, Any]] = []
        for subset in splits_to_load:
            ds = load_dataset("ethz-spylab/realmath", split=subset)
            for i, ex in enumerate(ds):
                raw_records.append({
                    "id": f"realmath_{subset}_{i}",
                    "subset": subset,
                    "theorem": ex.get("theorem", "") or "",
                    "question": ex.get("question", "") or "",
                    "answer": ex.get("answer", "") or "",
                    "context": ex.get("context", "") or "",
                    "link": ex.get("link", "") or "",
                })

        # Stable shuffle; then split train pool + held-out test pool.
        random.Random(self._seed).shuffle(raw_records)
        train_slice = raw_records[: self._max_instances]
        test_slice = (
            raw_records[self._max_instances : self._max_instances + self._held_out_size]
            if self._held_out_size > 0
            else []
        )

        def _build(item: dict[str, Any]) -> dict[str, Any]:
            q_lower = item["question"].lower()
            a_text = item["answer"]
            if len(a_text) > 250 or "prove" in q_lower or "show that" in q_lower:
                skill_id = "mathematical_derivation"
            elif any(kw in a_text for kw in ("\\begin{cases}", "\\frac", "\\sum", "\\int")):
                skill_id = "expression_identification"
            else:
                skill_id = "symbolic_manipulation"

            ctx = item["context"] or ""
            if len(ctx) > _MAX_CONTEXT_CHARS:
                ctx = "...\n" + ctx[-_MAX_CONTEXT_CHARS:]
            if item["theorem"]:
                theorem_blk = (
                    "Theorem statement: "
                    + item["theorem"].strip()
                    + "\n\n"
                )
                budget = _MAX_CONTEXT_CHARS - len(theorem_blk)
                ctx = theorem_blk + (ctx[-budget:] if budget > 0 else "")

            return {
                "id": item["id"],
                "subset": item["subset"],
                "question": item["question"],
                "context": ctx,
                "answer": a_text,
                "skill_id": skill_id,
            }

        self._data = [_build(item) for item in train_slice]
        self._test_data = [_build(item) for item in test_slice]

        self._loaded = True
        logger.info(
            f"Loaded {len(self._data)} RealMath train instances"
            + (f" + {len(self._test_data)} held-out test instances"
               if self._test_data else "")
            + f" (subsets: {', '.join(splits_to_load)})"
        )

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "realmath"

    @property
    def domain(self) -> str:
        return "research_math"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in REALMATH_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return ["read", "derive", "simplify", "identify", "answer"]

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
                metadata={
                    "subset": item["subset"],
                    # task_type mirrors the skill_id so the critic's round-robin
                    # and the search-agent bandit can condition on it.
                    "task_type": item["skill_id"],
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
                context=item["context"],
                expected_answer=item["answer"],
                metadata={
                    "subset": item["subset"],
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
        """Run a single research-math QA episode."""
        prompt = (
            f"You are answering a research-mathematics question based on the "
            f"following theorem and paper excerpt.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {instruction}\n\n"
            f"Think step by step through the derivation, then give your "
            f"final answer after 'Answer:' as a LaTeX expression."
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

    @staticmethod
    def _extract_answer_from_response(response: str) -> str:
        """Pull the LaTeX expression following an 'Answer:' marker."""
        # First: last line beginning with Answer:
        for line in reversed(response.strip().split("\n")[-12:]):
            m = re.match(
                r"\s*(?:Final\s+)?(?:Answer|ANSWER|answer)\s*[:\s]\s*(.+)",
                line,
            )
            if m:
                return m.group(1).strip()

        # Second: anywhere in response
        m = re.search(
            r"(?:Final\s+)?(?:Answer|ANSWER|answer)\s*[:\s]\s*(.+?)(?:\n|$)",
            response,
        )
        if m:
            return m.group(1).strip()

        # Third: last block-math \[...\]
        m = re.search(r"\\\[([^\]]+?)\\\]", response)
        if m:
            return f"\\[{m.group(1).strip()}\\]"

        # Fallback: last non-empty line
        lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
        return lines[-1] if lines else response.strip()

    def evaluate_answer(self, predicted: str, expected: str, **kwargs) -> float:
        """Exact-match after LaTeX normalization.

        The critic in v15.5 uses an LLMJudge (Claude) for semantic
        equivalence, which is what we really want for math expressions.
        This fallback only fires if the judge is disabled.
        """
        pred_n = _normalize_latex(predicted)
        exp_n = _normalize_latex(expected)
        if not pred_n or not exp_n:
            return 0.0
        if pred_n == exp_n:
            return 1.0
        # Cheap substring check — catches "answer is X" prefixes.
        if exp_n in pred_n or pred_n in exp_n:
            return 1.0
        return 0.0

    def build_evokg(self) -> EvoKG:
        return build_realmath_evokg()


def _normalize_latex(s: str) -> str:
    """Light-weight LaTeX normalization for exact-match comparison.

    * strips outer display-math delimiters (``\\[...\\]``, ``$$...$$``,
      ``$...$``)
    * collapses whitespace
    * lowercases non-command text
    """
    if not s:
        return ""
    t = s.strip()
    # Outer display math
    t = re.sub(r"^\\\[", "", t)
    t = re.sub(r"\\\]$", "", t)
    t = re.sub(r"^\$\$", "", t)
    t = re.sub(r"\$\$$", "", t)
    t = re.sub(r"^\$", "", t)
    t = re.sub(r"\$$", "", t)
    # Remove math environment wrappers
    t = re.sub(r"\\begin\{(?:equation|align|gather)\*?\}", "", t)
    t = re.sub(r"\\end\{(?:equation|align|gather)\*?\}", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def build_realmath_evokg() -> EvoKG:
    """Seed EvoKG with RealMath skill prerequisites + env/task nodes."""
    evokg = EvoKG()

    for sid, sdef in REALMATH_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="research_math",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))

    for sid, sdef in REALMATH_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    evokg.add_node(EnvironmentNode(
        "env_latex", "LaTeX Expressions",
        entity_type="notation", domain="research_math",
    ))

    evokg.add_node(TaskNode(
        "task_realmath",
        "Answer research-level mathematical questions from paper/post context",
        difficulty=0.85, domain="research_math",
    ))

    logger.info(f"Built RealMath EvoKG: {evokg}")
    return evokg
