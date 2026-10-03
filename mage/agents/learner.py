"""Real Learner Agent: trains via SFT with LoRA, acts via the trained model.

This is the agent that evolves — its LoRA adapter is updated each iteration
while the base model stays frozen.

Supports both sequential (games/web) and reasoning (QA) interaction modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mage.agents.base import LearnerAgent, SkillDataset
from mage.training.sft_trainer import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


class SFTLearner(LearnerAgent):
    """Learner agent backed by a real LLM with LoRA SFT training.

    In the evolution loop:
    - train(): receives SkillDataset from Explorer, runs SFT
    - act(): generates actions using the trained model (used by Critic for evaluation)

    The LoRA adapter is reset each iteration (train from scratch with
    cumulative data), following EXIF's approach.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        super().__init__("SFTLearner", config)

        sft_config_dict = config or {}
        self.sft_config = SFTConfig.from_dict(sft_config_dict)
        self.trainer = SFTTrainer(self.sft_config)
        self.interaction_mode = sft_config_dict.get("interaction_mode", "sequential")
        self.reward_weighted = sft_config_dict.get("reward_weighted", False)
        self.output_dir = Path(sft_config_dict.get("output_dir", "experiments/default"))

        # Cumulative dataset across iterations
        self._cumulative_data: list[Any] = []
        self._training_iterations = 0

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """Train on the skill dataset via SFT.

        If config.cumulative=True, trains on all data from all iterations.
        LoRA is reset each iteration (fresh adapter, cumulative data).
        """
        self._training_iterations += 1

        # Accumulate data
        if self.sft_config.cumulative:
            self._cumulative_data.extend(dataset.trajectories)
            train_data = SkillDataset(
                trajectories=list(self._cumulative_data),
                iteration=iteration,
            )
            logger.info(
                f"Cumulative training: {len(dataset)} new + "
                f"{len(self._cumulative_data) - len(dataset)} previous = "
                f"{len(train_data)} total trajectories"
            )
        else:
            train_data = dataset

        # Filter to high-quality trajectories only
        min_reward = self.config.get("min_reward_threshold", 0.0)
        successful = [t for t in train_data.trajectories if t.reward > min_reward]
        if len(successful) >= 5:
            train_data = SkillDataset(trajectories=successful, iteration=iteration)
            logger.info(f"Filtered to {len(train_data)} trajectories (reward > {min_reward})")
        else:
            logger.warning(f"Only {len(successful)} trajectories above threshold {min_reward}, using all {len(train_data)}")

        checkpoint_dir = self.output_dir / f"iter_{iteration}" / "checkpoint"

        metrics = self.trainer.train(
            train_data,
            iteration=iteration,
            output_dir=checkpoint_dir,
            interaction_mode=self.interaction_mode,
            reward_weighted=self.reward_weighted,
        )

        metrics["cumulative_trajectories"] = len(self._cumulative_data)
        return metrics

    def act(self, observation: str, instruction: str) -> str:
        """Generate an action given observation and instruction.

        Used by Critic during evaluation.
        For sequential envs: returns a single action word.
        For reasoning envs: returns full text response.
        """
        try:
            if self.interaction_mode == "reasoning":
                response = self.trainer.generate(observation, instruction, max_new_tokens=512)
                return response.strip()
            else:
                response = self.trainer.generate(observation, instruction, max_new_tokens=32)
                action = response.strip().split("\n")[0].strip().lower().replace(" ", "_")
                return action
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return "noop"

    def save(self, path: str | Path) -> None:
        """Save learner state (LoRA checkpoint)."""
        self.trainer.save_checkpoint(path)

    def load(self, path: str | Path) -> None:
        """Load learner state (LoRA checkpoint)."""
        self.trainer.load_checkpoint(path)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))


class ICLLearner(LearnerAgent):
    """Learner that evolves via In-Context Learning — no parameter updates.

    Instead of training LoRA, this learner:
    - Stores high-quality reasoning traces in an example bank (organized by skill)
    - At inference, retrieves the best examples as few-shot demonstrations
    - The base model is NEVER modified — evolution is purely in the example bank

    This directly tests the paper's thesis: structured self-knowledge (EvoKG +
    curated examples) matters more than the training mechanism (LoRA/GRPO).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        environment: Any = None,
    ):
        super().__init__("ICLLearner", config)
        from mage.utils.llm import LLMBackend

        self.environment = environment
        self.output_dir = Path((config or {}).get("output_dir", "experiments/default"))
        self.max_examples = (config or {}).get("max_examples_per_skill", 3)
        self.min_reward = (config or {}).get("min_reward_threshold", 0.5)

        # Example bank: skill_id → list of (instruction, response, reward)
        self._example_bank: dict[str, list[dict[str, Any]]] = {}
        self._llm: LLMBackend | None = None

    def set_environment(self, environment: Any) -> None:
        self.environment = environment

    def set_llm(self, llm: Any) -> None:
        """Inject the LLM backend (shared with Explorer/Navigator)."""
        self._llm = llm

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """'Training' = store high-quality trajectories in the example bank.

        For each trajectory with reward above threshold:
        - Store the (instruction, reasoning trace) pair indexed by skill
        - Keep only the top-K examples per skill (ranked by reward)
        """
        added = 0
        for traj in dataset.trajectories:
            if traj.reward < self.min_reward:
                continue
            if not traj.skill_id:
                continue

            skill = traj.skill_id
            if skill not in self._example_bank:
                self._example_bank[skill] = []

            example = {
                "instruction": traj.instruction,
                "response": "\n".join(traj.actions) if traj.actions else "",
                "reward": traj.reward,
                "context": traj.metadata.get("context", ""),
                "expected": traj.metadata.get("expected", ""),
                "predicted": traj.metadata.get("predicted", ""),
            }
            self._example_bank[skill].append(example)
            added += 1

            # Keep only top-K per skill
            self._example_bank[skill].sort(key=lambda x: x["reward"], reverse=True)
            self._example_bank[skill] = self._example_bank[skill][:self.max_examples]

        total_examples = sum(len(v) for v in self._example_bank.values())
        skills_with_examples = len(self._example_bank)

        # Save example bank
        self._save_bank(iteration)

        metrics = {
            "iteration": iteration,
            "status": "completed",
            "added_examples": added,
            "total_examples": total_examples,
            "skills_with_examples": skills_with_examples,
            "bank_summary": {
                sk: len(exs) for sk, exs in self._example_bank.items()
            },
        }
        logger.info(
            f"ICL bank updated: +{added} examples, "
            f"{total_examples} total across {skills_with_examples} skills"
        )
        return metrics

    def act(self, observation: str, instruction: str) -> str:
        """Generate with few-shot examples from the example bank."""
        if self._llm is None:
            return "noop"

        # Build few-shot prompt with retrieved examples
        examples = self._retrieve_examples(instruction)
        prompt = self._build_icl_prompt(observation, instruction, examples)

        try:
            response = self._llm.generate(prompt, temperature=0.3, max_tokens=512)
            return response.strip()
        except Exception as e:
            logger.debug(f"ICL generation failed: {e}")
            return "noop"

    def _retrieve_examples(self, instruction: str) -> list[dict[str, Any]]:
        """Retrieve relevant examples from the bank.

        Simple strategy: collect top examples from all skills.
        The EvoKG bandit handles smarter selection at the Explorer level.
        """
        all_examples = []
        for skill, exs in self._example_bank.items():
            all_examples.extend(exs)

        # Sort by reward, take top K
        all_examples.sort(key=lambda x: x["reward"], reverse=True)
        return all_examples[:self.max_examples]

    def _build_icl_prompt(
        self,
        observation: str,
        instruction: str,
        examples: list[dict[str, Any]],
    ) -> str:
        """Build prompt with few-shot examples."""
        parts = []

        if examples:
            parts.append("Here are examples of successful reasoning:\n")
            for i, ex in enumerate(examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Question: {ex['instruction']}")
                if ex.get("response"):
                    parts.append(f"Reasoning: {ex['response'][:300]}")
                if ex.get("predicted"):
                    parts.append(f"Answer: {ex['predicted']}")
                parts.append("")

            parts.append("Now answer the following question using similar reasoning:\n")

        parts.append(observation)

        return "\n".join(parts)

    def _save_bank(self, iteration: int) -> None:
        """Save example bank as JSON."""
        import json
        save_dir = self.output_dir / f"iter_{iteration}"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "example_bank.json"
        with open(path, "w") as f:
            json.dump(self._example_bank, f, indent=2)
        logger.info(f"Saved example bank to {path}")

    def save(self, path: str | Path) -> None:
        import json
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "example_bank.json", "w") as f:
            json.dump(self._example_bank, f, indent=2)

    def load(self, path: str | Path) -> None:
        import json
        path = Path(path) / "example_bank.json"
        if path.exists():
            self._example_bank = json.load(open(path))
            logger.info(f"Loaded example bank: {sum(len(v) for v in self._example_bank.values())} examples")

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))


class RoutedLearner(LearnerAgent):
    """Learner that evolves via strategy routing — no parameter updates.

    The LLM stays frozen. Evolution happens in:
    1. StrategyRouter (bandit learns which strategy per skill)
    2. Principles (Claude-extracted rules injected into prompts)
    3. Hybrid verification (tools compute first for computational skills)
    4. Embedding-based skill matching (replaces keyword matching)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        environment: Any = None,
    ):
        super().__init__("RoutedLearner", config)
        from mage.core.strategy_router import StrategyRouter

        cfg = config or {}
        self.environment = environment
        self.output_dir = Path(cfg.get("output_dir", "experiments/default"))
        self._llm: Any = None
        self._router = StrategyRouter()
        self._evokg: Any = None
        self._embedding_index: Any = None
        self._memory_index: Any = None  # FailureMemoryIndex (lazy)
        self._success_index: Any = None  # v15.5: SuccessMemoryIndex (lazy)
        self._search_agent: Any = None  # v15: SearchAgent (set externally)
        # v15.2: metadata from the most recent act()/_run_question call.
        # The Critic reads `search_strategy` off this to credit the right
        # bandit arm with the question's correct/wrong reward.
        self._last_call_metadata: dict[str, Any] = {}

        # v13.2 — Ablation flags. Default: all features ON.
        self.use_chain_reasoning: bool = bool(cfg.get("use_chain_reasoning", True))
        self.max_chain_depth: int = int(cfg.get("max_chain_depth", -1))
        self.use_memory_bank: bool = bool(cfg.get("use_memory_bank", True))
        self.use_retrieval_recipes: bool = bool(cfg.get("use_retrieval_recipes", True))
        self.use_search_agent: bool = bool(cfg.get("use_search_agent", True))
        # v15.6 ablation: disable success-memory retrieval while keeping
        # failure memories. Isolates the v15.5 success-memory contribution.
        self.use_success_memories: bool = bool(cfg.get("use_success_memories", True))
        # v16: temperature for sequential-env action sampling (Crafter,
        # WebShop). 0.0 = deterministic (default); 0.3-1.0 injects sampling
        # diversity to escape local optima — useful when the agent loops on
        # the same suboptimal action sequence under temperature=0.
        self.sequential_temperature: float = float(cfg.get("sequential_temperature", 0.0))
        # v17 Phase 1: HippoRAG-style PageRank rerank over the EvoKG.
        # When True, dense retrieval pulls a wider candidate pool (3× top_k)
        # then PR through the KG promotes nodes whose neighbours (sibling
        # memories, prerequisite skills) are also relevant. Closes some of
        # the gap to RetroAgent (their SimUtil-UCB does similar exploration
        # weighting). Default False — opt-in via config so behaviour is
        # backwards-compatible with v15.6/v16 result tables.
        self.use_kg_retrieval: bool = bool(cfg.get("use_kg_retrieval", False))
        # 2026-05-01: Program-of-Thought path. When enabled AND task_type is
        # numeric (finqa, gsm8k, realmath), the learner asks for Python code
        # rather than natural-language CoT and executes it in a restricted
        # environment. Eliminates arithmetic compounding errors that plateau
        # FinQA at ~58% under pure CoT. Default OFF for backwards-compat.
        self.use_program_of_thought: bool = bool(cfg.get("use_program_of_thought", False))
        # 2026-05-02: Medprompt-style choice-shuffle ensemble for MCQ tasks.
        # Generates `ensemble_n` answers with options permuted under different
        # rotations; final answer = majority vote over rotated answers.
        # Defends against position bias and improves robustness on MedQA / MMLU.
        # Recipe from Nori et al. 2023 (arXiv 2311.16452) — gets 74-78% on
        # Llama-3.1-8B for MedQA-USMLE.
        self.use_choice_shuffle_ensemble: bool = bool(
            cfg.get("use_choice_shuffle_ensemble", False)
        )
        self.ensemble_n: int = int(cfg.get("ensemble_n", 5))
        self.mcq_task_types: set[str] = set(cfg.get("mcq_task_types", [
            "clinical_reading", "differential_diagnosis", "mechanism_reasoning",
            "treatment_selection", "option_elimination",  # MedQA full set
        ]))
        # 2026-05-02: Self-consistency on numeric tasks. For FinQA / GSM8K /
        # RealMath: generate `sc_n` answers at non-zero temperature, vote on
        # numeric values within tolerance. Wang et al. 2023 — adds 2-5pp on
        # numeric reasoning when base CoT is decent.
        self.use_self_consistency: bool = bool(cfg.get("use_self_consistency", False))
        self.sc_n: int = int(cfg.get("sc_n", 5))
        self.sc_temperature: float = float(cfg.get("sc_temperature", 0.3))
        # 2026-05-03: Medprompt-style kNN exemplar retrieval. For each test
        # question, retrieve top-K most similar training questions and inject
        # them as few-shot demonstrations. The HIGHEST-leverage feature on
        # frozen 8B for MedQA (74-78% per Medprompt paper). Compatible with
        # existing memory bank — kshot exemplars come from training pool,
        # success memories come from past evolution iterations.
        self.use_kshot_retrieval: bool = bool(cfg.get("use_kshot_retrieval", False))
        self.kshot_n: int = int(cfg.get("kshot_n", 5))
        # benchmark name for kshot lookup (set by env adapter or config)
        self.kshot_benchmark: str = cfg.get("kshot_benchmark", "")
        # Teacher to draw rationales from: "opus", "sonnet", "haiku", "" (no teacher)
        # When set + rationale pool exists, kshot uses Medprompt full recipe
        # (CoT-rationale-augmented exemplars). Otherwise falls back to raw kNN.
        self.kshot_teacher: str = cfg.get("kshot_teacher", "")
        # Task-types that benefit from PoT (override via config if needed)
        self.pot_task_types: set[str] = set(cfg.get("pot_task_types", [
            "numerical_reasoning", "percentage_computation", "multi_step_finance",
            "table_extraction", "answer_formatting",  # FinQA full set
            "financial_data_extraction", "percentage_change_computation",
            "ratio_computation", "proportion_of_total", "multi_period_average",
            "midpoint_interpolation", "difference_computation",
            "investment_return_computation", "multi_step_financial_reasoning",
            "conditional_comparison",  # FinQA Claude-discovered set
        ]))
        # v19.A: Claude-as-guider escalation for uncertain answers. When the
        # frozen learner emits a hedging or unparseable response, escalate
        # the question to Claude with full KG context (retrieved memories,
        # principles, search hits + the learner's own attempt). Tracks
        # escalation rate per skill in EvoKG so future similar questions
        # can pre-emptively escalate. Default False — opt-in.
        self.use_claude_escalation: bool = bool(cfg.get("use_claude_escalation", False))
        self._claude_escalator: Any = None  # set externally
        # v20.A — cascade principles from prerequisite skills
        self.use_principles_cascade: bool = bool(cfg.get("use_principles_cascade", True))
        # v20.B — inject past successful action sequences as recipes
        self.use_action_recipes: bool = bool(cfg.get("use_action_recipes", True))
        # v20.C — inject goal-decomposition skill lattice (planning backbone)
        self.use_skill_lattice: bool = bool(cfg.get("use_skill_lattice", True))
        # v20.D — override skill_id with curriculum frontier when current is mastered
        self.use_curriculum_retrieval: bool = bool(cfg.get("use_curriculum_retrieval", True))
        # v21.B+D — env-side inference hints (Crafter survival + craft prereqs)
        self.use_inference_hints: bool = bool(cfg.get("use_inference_hints", True))
        # v21.C — cap principles cascade size (tighter for Crafter to avoid
        # late-iter prompt bloat regression)
        self.principles_cascade_max: int = int(cfg.get("principles_cascade_max", 8))

    def set_environment(self, env: Any) -> None:
        self.environment = env

    def set_llm(self, llm: Any) -> None:
        self._llm = llm

    def set_search_agent(self, search_agent: Any) -> None:
        """v15: Inject the evolving search agent."""
        self._search_agent = search_agent

    def set_claude_escalator(self, claude_llm: Any) -> None:
        """v19.A: Inject a Claude backend used for escalating uncertain
        answers. Caller passes any LLMBackend that exposes ``.generate``."""
        self._claude_escalator = claude_llm

    @staticmethod
    def _is_uncertain(response: str, predicted: str, question: str,
                      strict: bool = True) -> bool:
        """v19.A: heuristic uncertainty detector for the frozen learner.

        ``strict=True`` (default in v19.A2): only trigger on catastrophic
        failure signals — empty answer, truncated think-block leak, or
        explicit "I don't know" / "I'm not sure" in the response. Filters
        out legitimate hedging that's normal in CoT reasoning.

        ``strict=False`` (v19.A1): the broader heuristic triggers on any
        2+ hedging markers. Tested on WebQA → escalated 55% of questions
        and regressed accuracy by 5pp. Don't use.
        """
        if not predicted or not predicted.strip() or len(predicted.strip()) < 2:
            return True
        r_low = response.lower()
        if "<think>" in r_low and "</think>" not in r_low:
            return True  # truncated reasoning leak
        if strict:
            # Only the strongest "I really don't know" signals
            strong_signals = (
                "i don't know", "i'm not sure", "i cannot determine",
                "no information available", "cannot be determined from",
                "insufficient information",
            )
            if any(s in r_low for s in strong_signals):
                return True
            # Predicted is the literal question (severe parroting)
            p = predicted.strip().lower()
            if len(p) > 30 and p == question.strip().lower():
                return True
            return False
        # legacy non-strict (v19.A1) path — broader hedging
        hedges = (
            "based on the context", "i'm not sure", "i don't know",
            "uncertain", "unclear", "wait,", "let me reconsider",
            "actually,", "it could be", "i think it might",
            "hmm,", "but maybe", "or possibly",
        )
        hedge_count = sum(1 for h in hedges if h in r_low)
        if hedge_count >= 2:
            return True
        p = predicted.strip().lower()
        if len(p) > 15 and p in question.lower():
            return True
        return False

    def _claude_escalate(
        self,
        question: str,
        context: str,
        learner_response: str,
        principles: list[str],
        memories: list[dict],
    ) -> str:
        """v19.A: re-prompt Claude with full KG context."""
        if self._claude_escalator is None:
            return ""
        parts: list[str] = []
        if context:
            parts.append(f"Reference context:\n{context[:2000]}")
        if memories:
            mem_texts: list[str] = []
            for m in memories[:3]:
                t = (m.get("action_pattern") or
                     m.get("metadata", {}).get("reasoning_trace") or
                     "").strip()
                if t:
                    mem_texts.append(f"- {t[:240]}")
            if mem_texts:
                parts.append("Related past examples:\n" + "\n".join(mem_texts))
        if principles:
            parts.append("Skill principles:\n" + "\n".join(f"- {p[:200]}" for p in principles[:3]))
        parts.append(f"A small model attempted: {learner_response[-500:]}")
        parts.append(f"Question: {question}")
        parts.append("Provide ONLY the correct final answer (concise — one word/number/phrase).")
        prompt = "\n\n".join(parts)
        try:
            resp = self._claude_escalator.generate(
                prompt, temperature=0.0, max_tokens=80,
            ).strip()
            if resp.lower().startswith("answer:"):
                resp = resp.split(":", 1)[1].strip()
            # Take first line only — guard against Claude adding reasoning
            return resp.split("\n")[0].strip()
        except Exception:
            return ""

    def set_evokg(self, evokg: Any) -> None:
        self._evokg = evokg
        self._register_strategies_from_evokg()
        # Refresh embedding index
        if self._embedding_index:
            self._embedding_index.invalidate()
        self._ensure_embedding_index()
        # Refresh memory index from the new EvoKG state
        self._rebuild_memory_index()

    def _ensure_embedding_index(self) -> None:
        if self._embedding_index is None:
            from mage.utils.embeddings import SkillEmbeddingIndex
            self._embedding_index = SkillEmbeddingIndex()
        # ALWAYS rebuild — skills may have been added/split during EVOLVE
        if self._evokg:
            from mage.core.schemas import NodeType
            skills = self._evokg.get_nodes_by_type(NodeType.CAPABILITY)
            self._embedding_index.register_skills(skills)

    def _rebuild_memory_index(self) -> None:
        """Rebuild the retrieval indices from the EvoKG.

        v15.5 — rebuild BOTH failure and success indices. Previously only
        failure memories were indexed, which meant the ~100 success
        memories/iter stored by v15.1 were never retrieved as few-shot
        demonstrations (Bug A in the v15.5 plan).
        """
        from mage.utils.embeddings import FailureMemoryIndex, SuccessMemoryIndex
        if self._memory_index is None:
            self._memory_index = FailureMemoryIndex()
        if self._success_index is None:
            self._success_index = SuccessMemoryIndex()
        if self._evokg:
            fail_mems = self._evokg.get_failure_memories()
            succ_mems = self._evokg.get_success_memories()
            self._memory_index.rebuild(fail_mems)
            self._success_index.rebuild(succ_mems)
            if fail_mems or succ_mems:
                logger.debug(
                    f"Rebuilt memory indices: "
                    f"{len(fail_mems)} failures, {len(succ_mems)} successes"
                )

    def _register_strategies_from_evokg(self) -> None:
        """Register strategies including prerequisite cascade."""
        if not self._evokg:
            return
        from mage.core.schemas import NodeType
        skills = self._evokg.get_nodes_by_type(NodeType.CAPABILITY)

        for skill in skills:
            sid = skill["node_id"]
            candidates = [{"type": "direct"}]

            # Skill's own strategy — only tool type (CoT removed, it hurts QA accuracy)
            strategy = skill.get("strategy", {})
            if isinstance(strategy, dict) and strategy.get("type") == "tool":
                candidates.append(strategy)

            # PREREQUISITE CASCADE: inherit tool strategies from prereqs
            prereqs = self._evokg.get_prerequisites(sid)
            for prereq_id in prereqs:
                prereq_data = self._evokg.get_node(prereq_id)
                if prereq_data:
                    prereq_strategy = prereq_data.get("strategy", {})
                    if isinstance(prereq_strategy, dict) and prereq_strategy.get("type") == "tool":
                        cascaded = dict(prereq_strategy)
                        cascaded["_cascaded_from"] = prereq_id
                        candidates.append(cascaded)

            self._router.register_strategies(sid, candidates)

    def _find_skill_for_task(self, task_type: str) -> str | None:
        """Map task_type to the best matching skill that has a tool for it."""
        if not task_type or not self._evokg:
            return None
        from mage.core.schemas import NodeType
        skills = self._evokg.get_nodes_by_type(NodeType.CAPABILITY)

        # Find skills with tool strategies whose tool matches the task_type
        for skill in skills:
            strategy = skill.get("strategy", {})
            if isinstance(strategy, dict) and strategy.get("type") == "tool":
                # Check if this skill's description/name relates to the task type
                skill_name = skill.get("name", "").lower()
                skill_desc = skill.get("description", "").lower()
                task_lower = task_type.lower().replace("_", " ")

                # Match keywords from task_type to skill
                task_words = set(task_lower.split())
                skill_words = set(skill_name.split() + skill_desc.split())
                overlap = task_words & skill_words
                if len(overlap) >= 2:
                    return skill["node_id"]

        # Fallback: use STBench TASK_TO_SKILL mapping
        from mage.environments.stbench_env import TASK_TO_SKILL
        mapped = TASK_TO_SKILL.get(task_type)
        if mapped and self._evokg.has_node(mapped):
            return mapped

        return None

    def _find_skill(self, instruction: str) -> str:
        """Find best matching skill via embedding similarity."""
        self._ensure_embedding_index()
        if self._embedding_index:
            matches = self._embedding_index.find_skill(instruction, top_k=1)
            if matches:
                return matches[0][0]
        return "unknown"

    def _get_principles(self, skill_id: str) -> list[str]:
        """Retrieve principles from EvoKG for a skill."""
        if not self._evokg:
            return []
        skill_data = self._evokg.get_node(skill_id)
        if skill_data and skill_data.get("principles"):
            return skill_data["principles"][:5]
        return []

    def _get_principles_cascade(self, skill_id: str, max_depth: int = 2,
                                  max_total: int = 8) -> list[tuple[str, str]]:
        """v20.A — collect principles from this skill AND its prerequisite
        chain (up to ``max_depth`` hops). Useful for sequential envs where
        a target skill (e.g. make_stone_pickaxe) requires knowing the
        prereq tactics (collect_wood, collect_stone, place_table).

        Returns list of (skill_name, principle_text) tuples, ordered by:
          1. Target skill principles first
          2. Then prereq principles in BFS order
        Capped at ``max_total`` overall.
        """
        if not self._evokg or not skill_id:
            return []
        seen: set[str] = set()
        out: list[tuple[str, str]] = []

        # BFS over prereq edges
        frontier = [(skill_id, 0)]
        while frontier and len(out) < max_total:
            sid, depth = frontier.pop(0)
            if sid in seen:
                continue
            seen.add(sid)
            sdata = self._evokg.get_node(sid) or {}
            sname = sdata.get("name", sid)
            for p in (sdata.get("principles") or [])[:3]:
                if isinstance(p, dict):
                    p = p.get("text", "")
                if p and len(out) < max_total:
                    out.append((sname, p))
            if depth < max_depth:
                try:
                    prereqs = self._evokg.get_prerequisites(sid) or []
                except Exception:
                    prereqs = []
                for pr in prereqs:
                    frontier.append((pr, depth + 1))
        return out

    def _extract_answer(self, response: str) -> str:
        """Extract answer using environment-specific method.

        v13: For multi-step chain reasoning, the learner's response may
        contain intermediate reasoning steps followed by a final answer.
        v15: Handles Qwen3's <think>...</think> tags — strip reasoning
        block to get the actual answer output.
        """
        import re

        # v15: Strip Qwen3-style <think>...</think> reasoning blocks
        clean_response = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL,
        ).strip()
        # If stripping removed everything, use original
        if not clean_response:
            clean_response = response.strip()

        # 1. Environment-specific extractor (highest priority)
        if hasattr(self.environment, '_extract_answer_from_response'):
            return self.environment._extract_answer_from_response(clean_response)
        if hasattr(self.environment, '_extract_answer'):
            return self.environment._extract_answer(clean_response)

        # 2. v13: Scan the last 5 lines for a final answer marker. This
        #    handles chain outputs like "Step 1: ... Step 2: ... Answer: X".
        lines = [l for l in response.strip().split("\n") if l.strip()]
        tail = lines[-5:] if len(lines) >= 5 else lines
        for line in reversed(tail):
            for pat in (
                r"(?:Answer|ANSWER|Final answer|FINAL ANSWER)\s*:\s*(.+)",
                r"<answer>(.*?)</answer>",
            ):
                m = re.search(pat, line)
                if m:
                    return m.group(1).strip()

        # 3. Full-response regex (original behaviour)
        for pat in [r"(?:Answer|ANSWER)\s*:\s*(.+?)(?:\n|$)", r"<answer>(.*?)</answer>"]:
            m = re.search(pat, response)
            if m:
                return m.group(1).strip()

        # 4. Fallback: last non-empty line
        return lines[-1].strip() if lines else response.strip()

    # Threshold (in characters) above which the question already has rich
    # context. When context is long, type-strategy injection adds noise.
    LONG_CONTEXT_THRESHOLD = 500
    # Minimum cosine score for a type strategy to be injected. High threshold
    # ensures we only inject strategies that are CLEARLY relevant — generic
    # advice does more harm than good when only marginally relevant.
    TYPE_STRATEGY_MIN_SCORE = 0.55

    @staticmethod
    def _is_format_constrained(question: str, task_type: str | None) -> bool:
        """Detect questions whose answer format is already constrained.

        For these tasks, generic type strategies add noise. Examples:
          * STBench: option lists like '(1) ... (2) ...'
          * HotpotQA: long gold context — model has the info
          * Anything with 'Answer in 1-3 words' or 'Answer:' framing already

        WebQA factoid questions are NOT format-constrained — the model has
        free-form output, and a type strategy ('output the city/country')
        actively helps it pick the right format.
        """
        import re
        if not question:
            return False
        # Multiple-choice option markers (STBench style)
        if re.search(r"\((?:[1-9]|10|[A-D])\)", question):
            return True
        # Explicit instructions present (most benchmarks have these)
        # but only treat as constrained if the task_type indicates a
        # benchmark with structured questions
        if task_type:
            tt = task_type.upper()
            if tt.startswith("STBENCH") or tt.startswith("STARK"):
                return True
            if tt in ("BRIDGE", "COMPARISON"):  # HotpotQA
                return True
        return False

    def _execute_retrieval_recipe(
        self,
        recipe: dict,
        question: str,
    ) -> str:
        """v12.3 — Execute a retrieval recipe to fetch context for a question.

        The recipe's ``query_template`` may contain a ``{question}`` placeholder.
        We format it, run a Wikipedia summary fetch, and return the result.
        Returns empty string on failure.
        """
        try:
            from mage.tools.wikipedia_tool import (
                fetch_wikipedia_summary,
                search_wikipedia,
            )
        except ImportError:
            return ""

        template = recipe.get("query_template", "")
        if not template:
            return ""
        try:
            query = template.format(question=question)
        except Exception:
            query = question  # template format failed, use raw question

        # Two-stage: search → fetch top match
        try:
            titles = search_wikipedia(query, max_results=1)
            if titles:
                return fetch_wikipedia_summary(titles[0], max_chars=500)
        except Exception:
            pass

        # Fall back: try fetching the entity directly from the question
        try:
            return fetch_wikipedia_summary(question, max_chars=500)
        except Exception:
            return ""

    def _retrieve_memories(
        self,
        question: str,
        task_type: str | None,
        top_k: int = 3,
        context_len: int = 0,
        skill_id_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve memories conditioned on whether the question is
        format-constrained (HotpotQA/STBench) or open (WebQA factoid).

        v11.2 — conditional retrieval:
          * Long context OR format-constrained question: SPECIFIC memories
            only. Type strategies are noise here.
          * Open factoid question: mix of specific + type strategies, BUT
            only inject the type strategy if its similarity score is high
            enough (>= 0.55). Marginal type strategies hurt more than help.

        v13 — per-step chain retrieval:
          * ``skill_id_filter`` (optional) restricts retrieval to memories
            written for that specific sub-skill. Used inside
            ``_run_skill_chain`` so each step pulls its own sub-skill's
            worked examples, not the whole target skill's.

        v13.2 — ablation flag:
          * If ``self.use_memory_bank`` is False, returns [] immediately
            so ablation conditions can isolate the memory bank's contribution.

        v15.5 — retrieves from BOTH success and failure indices:
          * Short-context (WebQA factoid, STBench short): 2 success + 1 failure
          * Long-context (HotpotQA, STBench with context): 1 success + 2 failures
            (richer context → rely on correctives more than demos)
          * Success memories are tagged ``kind="success"``; failure memories
            retain their existing ``kind`` (``specific`` / ``type_strategy``).
          * _format_memories_as_examples dispatches on ``kind``.
        """
        if not getattr(self, "use_memory_bank", True):
            return []

        fail_size = self._memory_index.size() if self._memory_index else 0
        succ_size = (
            self._success_index.size()
            if self._success_index and getattr(self, "use_success_memories", True)
            else 0
        )
        if fail_size == 0 and succ_size == 0:
            return []

        long_context = context_len >= self.LONG_CONTEXT_THRESHOLD
        format_constrained = self._is_format_constrained(question, task_type)
        specifics_only = long_context or format_constrained

        # v15.5 — dual-index allocation.
        # Short-context: lean on success demonstrations; small correction.
        # Long-context: rich context already helps; lean on corrections.
        if long_context:
            n_succ_want = max(1, top_k - 2)
            n_fail_want = top_k - n_succ_want  # e.g. 2 for top_k=3
        else:
            n_succ_want = min(top_k, max(2, top_k - 1))
            n_fail_want = max(0, top_k - n_succ_want)  # e.g. 1 for top_k=3

        # -- Success memories (positive demonstrations) ------------------
        succ_results: list[dict] = []
        if succ_size > 0 and n_succ_want > 0:
            raw = self._success_index.retrieve(
                query=question,
                top_k=n_succ_want,
                task_type_filter=task_type or None,
                skill_id_filter=skill_id_filter,
            )
            for m in raw:
                m["kind"] = "success"
                succ_results.append(m)

        # -- Failure memories (corrections) ------------------------------
        fail_results: list[dict] = []
        if fail_size > 0 and n_fail_want > 0:
            if specifics_only:
                candidates = self._memory_index.retrieve(
                    query=question,
                    top_k=n_fail_want * 2,
                    task_type_filter=task_type or None,
                    skill_id_filter=skill_id_filter,
                )
                fail_results = [
                    m for m in candidates
                    if m.get("kind", "specific") == "specific"
                ][:n_fail_want]
            else:
                top_fail = self._memory_index.retrieve(
                    query=question,
                    top_k=n_fail_want,
                    task_type_filter=task_type or None,
                    skill_id_filter=skill_id_filter,
                )
                # Ensure at least one type_strategy if a strong one matches
                if not any(m.get("kind") == "type_strategy" for m in top_fail):
                    extras = self._memory_index.retrieve(
                        query=question,
                        top_k=n_fail_want * 2,
                        task_type_filter=task_type or None,
                        skill_id_filter=skill_id_filter,
                        min_score=self.TYPE_STRATEGY_MIN_SCORE,
                    )
                    for m in extras:
                        if m.get("kind") == "type_strategy":
                            top_fail.insert(0, m)
                            break
                fail_results = top_fail[:n_fail_want]

        # Order: success demonstrations first (what to imitate), then
        # failure corrections (what not to do).
        top = succ_results + fail_results

        # v17 Phase 1: HippoRAG-style PageRank rerank (opt-in).
        # The dense retrieval above is good but ignores graph structure.
        # If the EvoKG is wired and the flag is on, we re-rank the
        # candidate pool by personalized PageRank seeded with the dense
        # candidates. Memory nodes whose attached skills are *adjacent in
        # the prereq graph* to a high-scoring seed get promoted — the
        # "right answer is one prereq away" signal.
        if (
            getattr(self, "use_kg_retrieval", False)
            and self._evokg is not None
            and len(top) > 0
        ):
            try:
                top = self._pagerank_rerank(top, top_k=top_k + 1)
            except Exception as e:
                logger.debug(f"PageRank rerank skipped ({e}); using dense order")
        return top[:top_k + 1]

    def _pagerank_rerank(
        self,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """v17 Phase 1 (HippoRAG-faithful): rerank candidates by a HYBRID
        score that fuses the dense embedding similarity with personalized
        PageRank through the EvoKG.

        Faithful to HippoRAG (Gutiérrez et al., NeurIPS 2024):
          * **alpha = 0.5** (their default; the canonical web-PR 0.85
            walks too far for our small KGs and was the cause of the
            v17-first-pass HotpotQA regression).
          * **Hybrid score** ``= dense_weight * normalised_dense_score +
            (1 - dense_weight) * normalised_pr_score`` rather than
            replacing the dense order. ``dense_weight = 0.7`` keeps the
            strong direct-similarity signal as the primary driver and
            uses PR as a secondary boost from graph-adjacent memories.
          * **Result is a SUPERSET-then-rerank**: PR may add memory IDs
            that weren't in the dense top-K (they get dense_score = 0,
            so promotion only happens when graph relevance is high).
        """
        seeds: dict[str, float] = {}
        cand_by_id: dict[str, dict] = {}
        for m in candidates:
            nid = m.get("node_id") or m.get("id")
            if not nid:
                continue
            score = float(m.get("_score", 1.0))
            seeds[nid] = max(seeds.get(nid, 0.0), score)
            cand_by_id.setdefault(nid, m)
        if not seeds:
            return candidates

        def _is_memory(nid: str, nd: dict) -> bool:
            return nd.get("outcome", "") in ("success_memory", "failure_memory")

        ranked = self._evokg.retrieve_via_pagerank(
            seed_weights=seeds,
            top_k=max(top_k * 3, len(seeds) * 2),
            alpha=0.5,
            node_filter=_is_memory,
        )
        if not ranked:
            return candidates

        # Normalise dense and PR scores to [0,1] before fusion so the
        # weights are meaningful regardless of absolute scales.
        dense_max = max((float(m.get("_score", 0.0)) for m in candidates), default=1.0) or 1.0
        pr_max = max((s for _, s, _ in ranked), default=1.0) or 1.0
        dense_weight = 0.7

        # Build the union of (dense candidates) ∪ (PR-ranked memory nodes)
        all_ids: dict[str, dict] = dict(cand_by_id)
        pr_score_by_id: dict[str, float] = {nid: s for nid, s, _ in ranked}
        for nid, _s, nd in ranked:
            if nid not in all_ids:
                all_ids[nid] = {
                    "node_id": nid,
                    "question": nd.get("question", "") or nd.get("action_pattern", ""),
                    "reasoning_trace": nd.get("reasoning_trace", ""),
                    "correct_answer": nd.get("correct_answer", ""),
                    "expected": nd.get("expected", "") or nd.get("expected_answer", ""),
                    "kind": "success" if nd.get("outcome") == "success_memory" else "specific",
                    "_score": 0.0,
                }

        scored: list[tuple[float, dict]] = []
        for nid, base in all_ids.items():
            d_norm = float(base.get("_score", 0.0)) / dense_max
            p_norm = pr_score_by_id.get(nid, 0.0) / pr_max
            fused = dense_weight * d_norm + (1.0 - dense_weight) * p_norm
            entry = dict(base)
            entry["_pr_score"] = pr_score_by_id.get(nid, 0.0)
            entry["_fused_score"] = fused
            scored.append((fused, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]] or candidates

    def _format_memories_as_examples(
        self,
        memories: list[dict],
        long_context: bool = False,
    ) -> str:
        """Format retrieved memories as a few-shot block.

        v11.2 — conditional format:
          * Long context: minimal format, just specific examples (matches v10
            which got 73% on HotpotQA).
          * Short context: full format with type strategies + worked examples
            (helps WebQA generalize across factoid questions).

        v15.5 — successes are now first-class demonstrations:
          * ``kind="success"`` memories (from SuccessMemoryIndex) are
            formatted as PROVEN worked examples using the learner's own
            reasoning trace, with the chain decomposition if available.
          * ``kind="specific"`` / ``kind="type_strategy"`` (from
            FailureMemoryIndex) still render as before — corrections from
            Claude-written reasoning. They become the "avoid this mistake"
            half of the prompt.
        """
        if not memories:
            return ""

        successes = [m for m in memories if m.get("kind") == "success"]
        type_strategies = [m for m in memories if m.get("kind") == "type_strategy"]
        specifics = [
            m for m in memories
            if m.get("kind", "specific") == "specific"
        ]

        parts: list[str] = []

        # v15.5 — positive demonstrations first (what to imitate)
        if successes:
            parts.append(
                "Examples of similar questions you have solved CORRECTLY "
                "in the past (use these as templates):"
            )
            for i, m in enumerate(successes, 1):
                q = m.get("question", "")[:300]
                reasoning = (m.get("reasoning_trace") or "").strip()
                # Fall back to a compact Q→A summary when no CoT was stored
                if not reasoning:
                    reasoning = f"Answer: {m.get('correct_answer', '')[:100]}"
                else:
                    reasoning = reasoning[:500]
                a = m.get("correct_answer", "")[:150]
                decomp = m.get("decomposition") or []
                step_lines = ""
                if decomp:
                    step_lines = "\n  Steps: " + " ; ".join(
                        f"[{s.get('step', '?')}] {s.get('output', '')[:100]}"
                        for s in decomp[:4]
                    )
                parts.append(
                    f"\nSolved example {i}:\n"
                    f"  Question: {q}\n"
                    f"  Reasoning: {reasoning}"
                    f"{step_lines}\n"
                    f"  Answer: {a}"
                )
            parts.append("")

        # Corrections from failure memories (older path)
        if long_context:
            if specifics:
                parts.append(
                    "Corrections — previous mistakes on similar questions:"
                )
                for i, m in enumerate(specifics, 1):
                    q = m.get("question", "")[:300]
                    r = m.get("reasoning", "")[:400]
                    a = m.get("expected", "")[:100]
                    parts.append(
                        f"\nCorrection {i}:\n"
                        f"  Question: {q}\n"
                        f"  Reasoning: {r}\n"
                        f"  Correct answer: {a}"
                    )
                parts.append("")
        else:
            if type_strategies:
                parts.append("Strategy for this kind of question:")
                for s in type_strategies[:2]:
                    parts.append(f"  - {s.get('reasoning', '')[:400]}")
                parts.append("")

            if specifics:
                parts.append("Corrections from past mistakes:")
                for i, m in enumerate(specifics, 1):
                    q = m.get("question", "")[:250]
                    r = m.get("reasoning", "")[:350]
                    a = m.get("expected", "")[:100]
                    parts.append(
                        f"\nCorrection {i}:\n"
                        f"  Question: {q}\n"
                        f"  Reasoning: {r}\n"
                        f"  Correct answer: {a}"
                    )
                parts.append("")

        if not parts:
            return ""
        parts.append(
            "Now answer the new question using the same approach "
            "demonstrated above:\n"
        )
        return "\n".join(parts)

    def _run_question(self, question: str, context: str, skill_id: str, task_type: str | None = None, deterministic: bool = False) -> tuple[str, dict]:
        """Run a single question through the UNIFIED routing pipeline.

        ONE system, not two. The router selects the strategy, then:
        - If tool: try pre-built extractor → dynamic extractor → LLM extraction
        - If direct: use LLM with principles + retrieved failure memories

        The router's bandit learns from the result — tools that work get reused.
        Retrieved failure memories are injected as few-shot examples in the
        prompt — this is the cumulative learning loop.
        """
        strategy = self._router.select_strategy(skill_id, deterministic=deterministic)
        skey = self._router._strategy_key(strategy)
        # v20.A — use prereq-cascade principles if enabled; falls back to
        # single-skill lookup when v20.A is off.
        if getattr(self, "use_principles_cascade", True):
            try:
                cascade = self._get_principles_cascade(skill_id, max_depth=2, max_total=6)
                principles = [p for _, p in cascade] if cascade else self._get_principles(skill_id)
            except Exception:
                principles = self._get_principles(skill_id)
        else:
            principles = self._get_principles(skill_id)

        # If router selected a tool strategy, try the full tool chain
        if strategy.get("type") == "tool":
            tool_answer = self._try_tool_chain(question, task_type, strategy)
            if tool_answer is not None:
                return tool_answer, {
                    "strategy": skey,
                    "used_tool": True,
                    "deterministic": True,
                    "had_principles": len(principles) > 0,
                    "memories_used": 0,
                    "_strategy_dict": strategy,
                }
            # Tool failed — fall through to direct LLM

        # 2026-05-02 — Medprompt-style choice-shuffle ensemble for MCQ.
        # Generate N answers under permuted option labels; majority vote.
        # Fires only for MCQ task types (MedQA), via opt-in flag.
        if (
            getattr(self, "use_choice_shuffle_ensemble", False)
            and (task_type in getattr(self, "mcq_task_types", set()))
            and "Options:" in (context or "")
        ):
            try:
                ans, meta = self._run_mcq_ensemble(
                    question, context, skill_id, task_type, principles, deterministic,
                )
                if ans is not None:
                    return ans, meta
            except Exception as e:
                logger.debug(f"MCQ ensemble raised; falling back: {e}")

        # 2026-05-02 — Self-consistency on numeric tasks. Run N times with
        # nonzero temperature, vote on the most common numeric answer.
        # If pot_task_types is empty (config opted out of filtering), fire
        # for every task; this covers benchmarks like GSM8K/RealMath/STBench
        # whose task_types weren't pre-listed.
        _pot_set = getattr(self, "pot_task_types", set())
        if (
            getattr(self, "use_self_consistency", False)
            and (not _pot_set or task_type in _pot_set)
        ):
            try:
                ans, meta = self._run_self_consistency(
                    question, context, skill_id, task_type, principles, deterministic,
                )
                if ans is not None:
                    return ans, meta
            except Exception as e:
                logger.debug(f"Self-consistency raised; falling back: {e}")

        # 2026-05-01 — Program-of-Thought dispatch. Numeric tasks (FinQA,
        # GSM8K, RealMath) benefit from generating + executing Python rather
        # than natural-language CoT — arithmetic is exact, no compounding
        # error. Fires only when the flag is set AND the task_type is in
        # the PoT-friendly set; otherwise falls through to chain/CoT.
        if (
            getattr(self, "use_program_of_thought", False)
            and (task_type in getattr(self, "pot_task_types", set()))
        ):
            try:
                pot_answer, pot_meta = self._run_pot(
                    question, context, skill_id, task_type, principles, deterministic,
                )
                if pot_answer is not None:
                    return pot_answer, pot_meta
            except Exception as e:
                logger.debug(f"PoT path raised; falling back to CoT: {e}")
            # PoT failed — fall through to chain/CoT

        # v13 — Skill-chain dispatch.
        # If the KG has a multi-hop prerequisite chain for this skill AND
        # chain reasoning is enabled, traverse the chain step-by-step. Each
        # step runs the frozen Llama with its own sub-skill's memories and
        # principles. This is the reasoning-depth-evolves layer.
        if getattr(self, "use_chain_reasoning", True) and self._evokg is not None:
            try:
                chain = self._evokg.get_skill_chain(skill_id)
            except Exception:
                chain = []
            max_depth = int(getattr(self, "max_chain_depth", -1) or -1)
            if max_depth > 0 and len(chain) > max_depth:
                # Keep the most recent/specific prereqs; target is last
                chain = chain[-max_depth:]
            if len(chain) > 1:
                return self._run_skill_chain(
                    question, context, chain, task_type, deterministic,
                )

        # v15 — Search Agent: retrieve knowledge from external KB + history.
        # The search agent runs BEFORE memories and principles, providing
        # grounding context that the learner reads + reasons over.
        # Only fires for questions WITHOUT existing context (factoid QA).
        search_context = ""
        search_strategy_used: str = ""
        if (getattr(self, "use_search_agent", True)
                and self._search_agent is not None
                and not context):
            try:
                # Get metadata for entity resolution
                freebase_url = None
                if hasattr(self.environment, '_data'):
                    for item in getattr(self.environment, '_data', []):
                        if item.get("question", "")[:100] == question[:100]:
                            freebase_url = item.get("url")
                            break

                # v15.2: ask the search agent for the strategy arm it picked
                # so downstream callers can credit the correct bandit arm
                # with the eventual correct/wrong reward.
                retrieved, search_strategy_used = self._search_agent.search(
                    question=question,
                    freebase_url=freebase_url,
                    task_type=task_type,
                    evokg=self._evokg,
                    deterministic=deterministic,
                    return_strategy=True,
                )
                if retrieved:
                    search_context = "\n".join(f"- {s[:200]}" for s in retrieved[:3])
            except Exception as e:
                logger.debug(f"Search agent failed: {e}")

        # Detect whether the question already has rich context (HotpotQA) or
        # is format-constrained (STBench multi-choice). Both should use the
        # specific-memories-only format. Otherwise (WebQA factoid) use the
        # full hierarchical format with type strategies.
        effective_context = search_context or context
        context_len = len(question) + len(effective_context)
        long_context = context_len >= self.LONG_CONTEXT_THRESHOLD
        format_constrained = self._is_format_constrained(question, task_type)
        use_minimal_format = long_context or format_constrained

        # Retrieve relevant past failure memories (the cumulative learning bank)
        memories = self._retrieve_memories(
            question, task_type, top_k=3, context_len=context_len,
        )

        memory_block = self._format_memories_as_examples(
            memories, long_context=use_minimal_format,
        )

        # Build prompt with search context + memories + principles
        base_prompt = self._router.build_prompt_for_strategy(
            question, effective_context, {"type": "direct"},
            principles=principles,
        )
        # Memories go BEFORE the question/principles so they act like few-shot
        full_prompt = (memory_block + "\n" + base_prompt) if memory_block else base_prompt

        # Eval pipeline (deterministic=True) uses temperature=0.0 to remove
        # ±2% per-iteration sampling noise. Training uses 0.3 for some
        # exploration in the bandit reward signal.
        gen_temp = 0.0 if deterministic else 0.3
        response = self._llm.generate(full_prompt, temperature=gen_temp, max_tokens=1024)
        predicted = self._extract_answer(response)

        # v19.A — Claude escalation for hedged / unparseable answers
        escalated = False
        if (self.use_claude_escalation
                and self._claude_escalator is not None
                and self._is_uncertain(response, predicted, question)):
            claude_pred = self._claude_escalate(
                question, effective_context, response, principles, memories,
            )
            if claude_pred:
                predicted = claude_pred
                escalated = True

        return predicted, {
            "strategy": "direct" if strategy.get("type") != "tool" else f"{skey}:fallback",
            "used_tool": False,
            "deterministic": False,
            "had_principles": len(principles) > 0,
            "memories_used": len(memories),
            "search_strategy": search_strategy_used,  # v15.2: which arm was used
            # v15.5: raw LLM response (full CoT) for success-memory harvest.
            # Critic captures this into per_question_results; on correct answer
            # the evolution loop stores it as reasoning_trace in the KG.
            "raw_response": response,
            "decomposition": [],  # single-shot path has no chain steps
            "escalated": escalated,  # v19.A
            "_strategy_dict": strategy,
        }

    def _run_mcq_ensemble(
        self,
        question: str,
        context: str,
        skill_id: str,
        task_type: str | None,
        principles: list[str],
        deterministic: bool,
    ) -> tuple[str, dict]:
        """Medprompt-style choice-shuffle ensemble for MCQ tasks.

        Runs the LLM `ensemble_n` times under different option-label
        permutations. The model can't shortcut on letter position because the
        correct content moves between labels. Final answer = the LETTER that
        the most permutations chose (after un-permuting back to canonical
        labels). Recipe from Nori et al. 2023 (arXiv 2311.16452).

        For deterministic eval we still set temperature=0 each call but the
        permutations themselves provide the diversity.
        """
        import re, random
        from collections import Counter

        # Parse "Options:\nA. ...\nB. ...\nC. ...\nD. ..." from context
        opt_block = re.search(r"Options:\s*\n(.*)", context, re.DOTALL)
        if not opt_block:
            return None, {"strategy": "mcq_ensemble_failed", "reason": "no_options"}
        lines = [l.strip() for l in opt_block.group(1).splitlines() if l.strip()]
        opts: dict[str, str] = {}
        for ln in lines:
            m = re.match(r"^([A-D])[.)]\s*(.+)$", ln)
            if m:
                opts[m.group(1)] = m.group(2).strip()
        if len(opts) != 4:
            return None, {"strategy": "mcq_ensemble_failed", "reason": "bad_options"}

        # 5 fixed permutations: identity + 4 cyclic rotations
        # perm[orig_letter] = new_letter
        perms = [
            {"A": "A", "B": "B", "C": "C", "D": "D"},  # identity
            {"A": "B", "B": "C", "C": "D", "D": "A"},
            {"A": "C", "B": "D", "C": "A", "D": "B"},
            {"A": "D", "B": "A", "C": "B", "D": "C"},
            # 5th: reverse
            {"A": "D", "B": "C", "C": "B", "D": "A"},
        ][: max(1, self.ensemble_n)]

        memories: list[dict] = []
        if self.use_memory_bank:
            try:
                memories = self._retrieve_memories(
                    question, task_type, top_k=2,
                    context_len=len(question) + len(context),
                    skill_id=skill_id,
                )
            except Exception:
                memories = []
        few_shot = self._format_memories_as_examples(memories, long_context=False)

        principle_block = ""
        if principles:
            principle_block = (
                "Principles:\n" + "\n".join(f"- {p[:240]}" for p in principles[:3]) + "\n\n"
            )

        # 2026-05-03 — Medprompt-style kNN exemplars from train split
        kshot_block = ""
        if self.use_kshot_retrieval and self.kshot_benchmark:
            try:
                from mage.utils.kshot_retrieval import retrieve_kshot, format_medqa_kshot
                kshot_recs = retrieve_kshot(
                    self.kshot_benchmark, question,
                    top_k=self.kshot_n, teacher=self.kshot_teacher,
                )
                if kshot_recs:
                    kshot_block = format_medqa_kshot(kshot_recs)
            except Exception as e:
                logger.debug(f"kshot retrieval failed: {e}")

        votes: Counter = Counter()
        per_run_answers = []
        all_raw = []
        for perm in perms:
            # Build permuted options: new_letter -> opts[old_letter]
            inv = {v: k for k, v in perm.items()}  # new -> old
            permuted_opts = "\n".join(f"{nl}. {opts[inv[nl]]}" for nl in "ABCD")
            prompt = (
                "You are a medical expert taking the USMLE. Read the clinical "
                "vignette and choose the single best option (A, B, C, or D). "
                "Reason briefly, then give the final letter after 'Answer:'.\n\n"
                f"{kshot_block}"
                f"{principle_block}"
                f"{few_shot}"
                f"Question: {question}\n\n"
                f"Options:\n{permuted_opts}\n\n"
                "Reason briefly then answer:"
            )
            raw = self._llm.generate(
                prompt, temperature=0.0 if deterministic else 0.3, max_tokens=400,
            )
            all_raw.append(raw[:300])
            # Extract the letter from the response
            picked = ""
            for pat in (r"(?:Answer|ANSWER|answer is)\s*[:\s]\s*\(?([A-D])\)?",
                        r"\boption\s+\(?([A-D])\)?\s+is\s+(?:correct|best|right)"):
                m = re.search(pat, raw, re.IGNORECASE)
                if m:
                    picked = m.group(1).upper()
                    break
            if not picked:
                # Fallback: last A/B/C/D in response
                letters = re.findall(r"\b([A-D])\b", raw)
                if letters:
                    picked = letters[-1].upper()
            per_run_answers.append(picked)
            if picked in perm:
                # Un-permute: the model picked `picked` under perm; the original
                # letter that maps to picked is inv[picked].
                original = inv[picked]
                votes[original] += 1

        if not votes:
            return None, {"strategy": "mcq_ensemble_failed", "reason": "no_votes"}
        # Majority vote
        winner, n_votes = votes.most_common(1)[0]

        return winner, {
            "strategy": "mcq_ensemble",
            "used_tool": False,
            "deterministic": deterministic,
            "had_principles": len(principles) > 0,
            "memories_used": len(memories),
            "chain_length": 1,
            "chain_steps": ["mcq_ensemble"],
            "search_strategy": "",
            "raw_response": " || ".join(all_raw[:3]),
            "decomposition": [
                {"step": f"perm_{i}", "output": f"picked={per_run_answers[i] or 'none'}"}
                for i in range(len(perms))
            ],
            "ensemble_votes": dict(votes),
            "ensemble_n": len(perms),
            "_strategy_dict": {"type": "mcq_ensemble"},
        }

    def _run_self_consistency(
        self,
        question: str,
        context: str,
        skill_id: str,
        task_type: str | None,
        principles: list[str],
        deterministic: bool,
    ) -> tuple[str, dict]:
        """Self-consistency for numeric reasoning. Run N CoT samples at
        nonzero temperature; majority-vote the numeric answer (within 1%
        relative tolerance). Wang et al. 2023.
        """
        import re
        from collections import Counter

        memories: list[dict] = []
        if self.use_memory_bank:
            try:
                memories = self._retrieve_memories(
                    question, task_type, top_k=2,
                    context_len=len(question) + len(context),
                    skill_id=skill_id,
                )
            except Exception:
                memories = []
        few_shot = self._format_memories_as_examples(memories, long_context=False)

        principle_block = ""
        if principles:
            principle_block = (
                "Principles:\n" + "\n".join(f"- {p[:240]}" for p in principles[:3]) + "\n\n"
            )

        # 2026-05-03 — kNN exemplars from FinQA train split
        kshot_block = ""
        if self.use_kshot_retrieval and self.kshot_benchmark:
            try:
                from mage.utils.kshot_retrieval import retrieve_kshot, format_finqa_kshot
                kshot_recs = retrieve_kshot(
                    self.kshot_benchmark, question,
                    top_k=self.kshot_n, teacher=self.kshot_teacher,
                )
                if kshot_recs:
                    kshot_block = format_finqa_kshot(kshot_recs)
            except Exception as e:
                logger.debug(f"kshot retrieval failed: {e}")

        prompt = (
            "You are a financial analyst. Use the context (which may include "
            "tables) to answer the question. Show your computation step by "
            "step, then give the final numeric answer after 'Answer:'.\n\n"
            f"{kshot_block}"
            f"{principle_block}"
            f"{few_shot}"
            f"Context:\n{context[:2500]}\n\n"
            f"Question: {question}\n\n"
            "Let's think step by step:"
        )

        def _to_float(s: str) -> float | None:
            s = re.sub(r"[,$\s%]", "", str(s).strip())
            if not s:
                return None
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        def _extract(resp: str) -> str:
            for pat in (
                r"(?:Answer|Final answer|ANSWER)\s*[:\s]\s*\$?\s*([-\d,]+(?:\.\d+)?\s*%?)",
            ):
                m = re.search(pat, resp)
                if m:
                    return m.group(1).strip().replace(",", "").rstrip("%").strip()
            nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", resp)
            return nums[-1].replace(",", "") if nums else ""

        # Generate N samples at temperature
        samples: list[str] = []
        all_raw: list[str] = []
        n_runs = max(1, self.sc_n)
        for _ in range(n_runs):
            raw = self._llm.generate(
                prompt, temperature=self.sc_temperature, max_tokens=600,
            )
            all_raw.append(raw[:200])
            samples.append(_extract(raw))

        # Bucket numeric answers within 1% relative tolerance
        nums = [_to_float(s) for s in samples]
        valid = [(s, n) for s, n in zip(samples, nums) if n is not None]
        if not valid:
            # Fallback: most common string
            c = Counter(samples)
            winner = c.most_common(1)[0][0] if c else ""
        else:
            buckets: list[tuple[float, list[str], int]] = []
            for s, n in valid:
                placed = False
                for i, (rep, members, count) in enumerate(buckets):
                    if rep == 0:
                        eq = abs(n) < 0.01
                    else:
                        eq = abs(n - rep) / max(abs(rep), 1e-9) <= 0.01
                    if eq:
                        members.append(s)
                        buckets[i] = (rep, members, count + 1)
                        placed = True
                        break
                if not placed:
                    buckets.append((n, [s], 1))
            buckets.sort(key=lambda b: -b[2])
            winner = buckets[0][1][0]

        return winner, {
            "strategy": "self_consistency",
            "used_tool": False,
            "deterministic": deterministic,
            "had_principles": len(principles) > 0,
            "memories_used": len(memories),
            "chain_length": 1,
            "chain_steps": ["sc"],
            "search_strategy": "",
            "raw_response": " || ".join(all_raw[:3]),
            "decomposition": [
                {"step": f"sc_{i}", "output": s} for i, s in enumerate(samples)
            ],
            "sc_n": len(samples),
            "_strategy_dict": {"type": "self_consistency"},
        }

    def _run_pot(
        self,
        question: str,
        context: str,
        skill_id: str,
        task_type: str | None,
        principles: list[str],
        deterministic: bool,
    ) -> tuple[str, dict]:
        """Program-of-Thought path for numeric tasks.

        Generates Python code, executes it in a restricted sandbox, returns
        the result as the answer. Single shot — no chain (chain compounds
        Llama errors on numeric tasks). Few-shot from success memories
        (formatted as Python solutions when present).

        Returns (answer_str, metadata) on success, (None, _) on failure
        so the caller can fall through to CoT.
        """
        from mage.utils.safe_eval import run_pot

        # Retrieve up to 3 similar past success memories for few-shot,
        # but extract just the question + answer (we'll show them as Python
        # patterns rather than NL CoT).
        memories: list[dict] = []
        if self.use_memory_bank:
            memories = self._retrieve_memories(
                question, task_type, top_k=3,
                context_len=len(question) + len(context),
                skill_id=skill_id,
            )

        few_shot = ""
        for m in memories[:3]:
            mq = (m.get("question") or "")[:200]
            ma = m.get("correct_answer") or m.get("expected_answer") or ""
            if not mq or not ma:
                continue
            few_shot += (
                f"# Example:\n"
                f"# Question: {mq}\n"
                f"# Solution:\n"
                f"# (compute answer)\n"
                f"result = {ma}\n\n"
            )

        principle_block = ""
        if principles:
            principle_block = (
                "Principles for this skill:\n" +
                "\n".join(f"- {p[:240]}" for p in principles[:3]) + "\n\n"
            )

        # CoT-then-PoT: brief reasoning identifies which table values to
        # use, then Python code does the arithmetic exactly. The Fino1 /
        # FinanceMath / ProgramFC recipe. Output is a SINGLE response with
        # both reasoning AND a fenced Python block.
        prompt = (
            "You are a financial analyst. Solve the question by writing "
            "Python code that computes the answer exactly.\n\n"
            "Output format (you MUST follow this):\n"
            "1. A 2-3 line reasoning paragraph identifying which specific "
            "numbers from the context you will use.\n"
            "2. A fenced Python code block ```python ... ``` that:\n"
            "   - extracts each number into a clearly-named variable\n"
            "   - performs the computation with arithmetic operators\n"
            "   - assigns the final answer to `result`\n"
            "   - DOES NOT use imports or print()\n\n"
            "FinQA scale conventions (CRITICAL):\n"
            "- Percentages: gold answers are usually DECIMALS, so divide "
            "your final percentage by 100. Example: '14%' → result = 0.14\n"
            "- Money in millions: if the question or context speaks in "
            "millions, return the number IN MILLIONS, not the raw value. "
            "Example: '94 million' → result = 94, not 94000000\n"
            "- Negatives in parentheses: '(60.8)' means -60.8\n\n"
            f"{principle_block}"
            f"{few_shot}"
            f"Context (table + paragraphs):\n{context[:2500]}\n\n"
            f"Question: {question}\n\n"
            "Now write your reasoning then the Python block:\n"
        )

        gen_temp = 0.0 if deterministic else 0.3
        raw = self._llm.generate(prompt, temperature=gen_temp, max_tokens=900)
        full_text = raw

        result, error, code = run_pot(full_text)
        if result is None or error is not None:
            return None, {"strategy": "pot_failed", "pot_error": error or "no_result"}

        # Format result as a clean string (FinQA gold answers are decimals)
        if isinstance(result, bool):
            answer = "true" if result else "false"
        elif isinstance(result, (int, float)):
            try:
                f = float(result)
                # Strip trailing zeros / .0 for cleaner extraction
                if f == int(f):
                    answer = str(int(f))
                else:
                    answer = f"{f:.6f}".rstrip("0").rstrip(".")
            except Exception:
                answer = str(result)
        else:
            answer = str(result)[:120]

        return answer, {
            "strategy": "pot",
            "used_tool": True,  # Python eval IS a tool
            "deterministic": deterministic,
            "had_principles": len(principles) > 0,
            "memories_used": len(memories),
            "chain_length": 1,
            "chain_steps": ["pot"],
            "search_strategy": "",
            "raw_response": raw[:1000],
            "decomposition": [{"step": "pot", "output": code[:300]}],
            "_strategy_dict": {"type": "pot"},
        }

    def _run_skill_chain(
        self,
        question: str,
        context: str,
        chain: list[str],
        task_type: str | None,
        deterministic: bool,
    ) -> tuple[str, dict]:
        """v13 — Multi-step reasoning by traversing a prerequisite chain.

        Given a chain ``[skill_1, skill_2, ..., skill_N]`` in topological
        order (root prereq → target skill), run the frozen LLM once per
        step. Each step:
          * uses its own sub-skill's principles
          * retrieves its own sub-skill's failure memories (via
            ``skill_id_filter``) for targeted few-shot examples
          * sees the prior steps' outputs in its prompt
          * produces one intermediate output

        The last step's output is extracted as the final answer. This
        gives the learner genuine multi-step reasoning without any
        parameter updates: as the KG deepens its chains over iterations,
        the number of reasoning steps per question grows.

        Error propagation: if step 1 is wrong, downstream steps see the
        wrong intermediate in their prompts. We accept this; the per-step
        memories + principles are the main mitigation. Alternative
        approaches (self-correction, multiple step framings) are too
        expensive for the inner eval loop.
        """
        accumulated: list[tuple[str, str]] = []  # [(skill_name, step_output), ...]
        total_memories_used = 0
        gen_temp = 0.0 if deterministic else 0.3

        # Context length is fixed across steps (the question + original ctx)
        context_len = len(question) + len(context)

        for step_idx, step_skill_id in enumerate(chain):
            skill_data = self._evokg.get_node(step_skill_id) or {}
            step_name = skill_data.get("name", step_skill_id)

            # Per-step principles + memories (sub-skill filtered)
            step_principles = self._get_principles(step_skill_id)
            step_memories = self._retrieve_memories(
                question,
                task_type,
                top_k=3,
                context_len=context_len,
                skill_id_filter=step_skill_id,
            )
            total_memories_used += len(step_memories)

            # Step instruction: prefer the skill's prompt_template, else name
            raw_tmpl = (skill_data.get("prompt_template", "") or "").strip()
            step_instruction = (
                raw_tmpl if raw_tmpl
                else f"Apply the skill '{step_name}' to the question."
            )

            # Assemble prior-steps block (becomes the reasoning trace visible
            # to this step). Each prior step is truncated to 200 chars to
            # keep the prompt compact.
            prior_block = ""
            if accumulated:
                prior_lines = [
                    f"  [{name}] {out[:200]}" for name, out in accumulated
                ]
                prior_block = (
                    "Prior reasoning steps (do not repeat — build on them):\n"
                    + "\n".join(prior_lines)
                    + "\n\n"
                )

            # Memory few-shot block (same format as single-shot path)
            memory_block = self._format_memories_as_examples(
                step_memories, long_context=False,
            )

            # Context block (only if present; HotpotQA has gold paragraphs,
            # WebQA/STBench usually don't)
            context_block = f"Context:\n{context}\n\n" if context else ""

            # Principles block reuses the router's prompt builder logic
            # but with step-local principles
            principles_block = ""
            if step_principles:
                principles_block = (
                    "IMPORTANT RULES for this sub-skill:\n"
                    + "\n".join(
                        f"  {i + 1}. {p}" for i, p in enumerate(step_principles)
                    )
                    + "\n\n"
                )

            # Per-step prompt
            is_last = (step_idx == len(chain) - 1)
            output_instruction = (
                "Output ONLY the final answer (concise, in the expected format):"
                if is_last
                else "Output ONLY the result of this sub-step (concise):"
            )
            step_prompt = (
                f"{memory_block}"
                f"{principles_block}"
                f"{prior_block}"
                f"{context_block}"
                f"Question: {question}\n\n"
                f"Current step [{step_idx + 1}/{len(chain)}]: {step_instruction}\n"
                f"{output_instruction}"
            )

            try:
                step_output = self._llm.generate(
                    step_prompt, temperature=gen_temp, max_tokens=1024,
                ).strip()
            except Exception as e:
                logger.debug(f"Chain step {step_idx} ({step_name}) LLM call failed: {e}")
                step_output = ""

            accumulated.append((step_name, step_output))

        # Final answer: extract from the last step's output
        final_raw = accumulated[-1][1] if accumulated else ""
        final_answer = self._extract_answer(final_raw)

        # v19.A — Claude escalation also for chain path
        escalated = False
        if (self.use_claude_escalation
                and self._claude_escalator is not None
                and self._is_uncertain(final_raw, final_answer, question)):
            chain_summary = "\n".join(
                f"step {i+1} ({n}): {o[:150]}"
                for i, (n, o) in enumerate(accumulated)
            )
            claude_pred = self._claude_escalate(
                question, context, chain_summary, principles=[],
                memories=[],
            )
            if claude_pred:
                final_answer = claude_pred
                escalated = True

        return final_answer, {
            "strategy": "chain",
            "used_tool": False,
            "deterministic": deterministic,
            "had_principles": False,  # per-step, not aggregated
            "memories_used": total_memories_used,
            "chain_length": len(chain),
            "chain_steps": [a[0] for a in accumulated],
            "search_strategy": "",  # v15.2: chain path bypasses search agent
            # v15.5 — raw last-step response as reasoning trace, and
            # per-step decomposition from the chain accumulator so future
            # similar multi-hop questions can see the decomposition pattern.
            "raw_response": final_raw,
            "decomposition": [
                {"step": name, "output": out} for name, out in accumulated
            ],
            "escalated": escalated,  # v19.A
            "_strategy_dict": {"type": "chain"},
        }

    def _try_tool_chain(self, question: str, task_type: str | None, strategy: dict) -> str | None:
        """Unified tool chain: pre-built → dynamic → LLM extraction.

        Tries all available extraction methods for the tool.
        """
        tool_name = strategy.get("tool", "")

        # 1. Pre-built extractor (fastest, most reliable)
        if task_type:
            from mage.tools.tool_pipeline import run_tool_pipeline
            answer = run_tool_pipeline(question, task_type, evokg=self._evokg)
            if answer is not None:
                return answer

        # 2. Router's hybrid verification (regex + LLM extraction)
        tool_result = self._router.execute_hybrid_verification(
            question, strategy, self._llm,
        )
        if tool_result:
            mapped = self._router._map_tool_result_to_option(question, tool_result)
            if mapped is not None:
                return mapped

        return None

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """Run questions through router, score, update bandit."""
        if not self._llm or not self.environment:
            return {"iteration": iteration, "status": "skipped", "reason": "no_llm_or_env"}

        self._register_strategies_from_evokg()
        self._ensure_embedding_index()

        # Deduplicate questions
        seen = set()
        questions = []
        for traj in dataset.trajectories:
            if traj.instruction not in seen:
                seen.add(traj.instruction)
                questions.append({
                    "instruction": traj.instruction,
                    "context": traj.metadata.get("context", ""),
                    "expected": traj.metadata.get("expected", ""),
                    "skill_id": self._find_skill_for_task(traj.metadata.get("task_type", "")) or traj.skill_id or self._find_skill(traj.instruction),
                    "task_type": traj.metadata.get("task_type", ""),
                })

        if not questions:
            return {"iteration": iteration, "status": "skipped", "reason": "no_questions"}

        total_reward = 0.0
        strategy_usage: dict[str, int] = {}
        tool_usage = {"attempted": 0, "succeeded": 0}

        for q in questions:
            skill_id = q["skill_id"]

            predicted, meta = self._run_question(
                q["instruction"], q["context"], skill_id,
                task_type=q.get("task_type"),
            )

            skey = meta.get("strategy", "direct")
            strategy_usage[skey] = strategy_usage.get(skey, 0) + 1

            if meta.get("used_tool"):
                tool_usage["attempted"] += 1

            reward = 0.0
            if q["expected"] and predicted:
                reward = self.environment.evaluate_answer(predicted, q["expected"])
                if meta["used_tool"] and reward > 0:
                    tool_usage["succeeded"] += 1

            # Update router with the strategy that was ACTUALLY used
            self._router.update(skill_id, meta["_strategy_dict"], reward)
            total_reward += reward

        mean_reward = total_reward / len(questions) if questions else 0.0

        self._router.save(self.output_dir / f"iter_{iteration}" / "strategy_router.json")

        metrics = {
            "iteration": iteration,
            "status": "completed",
            "num_questions": len(questions),
            "mean_reward": round(mean_reward, 4),
            "strategy_usage": strategy_usage,
            "tool_usage": tool_usage,
        }
        logger.info(
            f"Router training: {len(questions)} questions, mean_reward={mean_reward:.3f}, "
            f"strategies={strategy_usage}, tools={tool_usage}"
        )
        return metrics

    def act(self, observation: str, instruction: str, task_type: str = "") -> str:
        """Generate answer using full routing pipeline (for Critic evaluation).

        Two modes:
        - Sequential env (Crafter, WebShop): the *instruction* is a complete
          agent prompt (action vocabulary, achievements, history, current obs).
          Use it directly with optional KG augmentation (principles, success
          memories). Without this branch, the QA pipeline below would discard
          the system prompt and the agent would lose all game context.
        - QA env (HotpotQA, WebQA, GSM8K, STBench, RealMath): observation is
          the question. Run the QA pipeline (memories, principles, search).

        Uses DETERMINISTIC strategy selection — same state always produces
        same answer. This makes evaluation reproducible and rollback meaningful.
        """
        if not self._llm:
            self._last_call_metadata = {"search_strategy": ""}
            return "noop"

        # Detect sequential-env mode: SOME caller is passing a complete agent
        # prompt (action vocab, page/state, achievements, etc.). Crafter puts
        # this in `instruction`; WebShop puts it in `observation`. We detect
        # in EITHER slot and dispatch with the long one as the system prompt.
        SEQ_MARKERS = (
            "Current observation",
            "Valid actions",
            "ACTIONS:",
            "achievements you can get",
            "WebShop",
            "search[",
            "click[",
            "Available actions:",
            "Current page:",
        )
        instr_is_seq = len(instruction) > 400 and any(m in instruction for m in SEQ_MARKERS)
        obs_is_seq = len(observation) > 400 and any(m in observation for m in SEQ_MARKERS)
        is_sequential = instr_is_seq or obs_is_seq

        try:
            if is_sequential:
                # Use whichever slot carries the long agent prompt as the
                # system prompt; pass the other as a short context hint.
                if instr_is_seq:
                    sys_prompt, ctx = instruction, observation
                else:
                    sys_prompt, ctx = observation, instruction
                return self._act_sequential(ctx, sys_prompt)
            skill_id = self._find_skill(instruction)
            predicted, meta = self._run_question(
                observation, "", skill_id, task_type=task_type,
                deterministic=True,  # reproducible evaluation
            )
            # Expose metadata to the caller (Critic reads this on every question)
            self._last_call_metadata = meta or {}
            return predicted
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            self._last_call_metadata = {"search_strategy": ""}
            return "noop"

    def _act_sequential(self, observation: str, instruction: str) -> str:
        """Sequential-env path. Pass the system prompt through verbatim and
        inject KG-derived hints (principles + success memories) without
        clobbering the env's carefully-built prompt.

        Critical for Crafter / WebShop where the instruction contains the
        action vocabulary, goal, and history. The QA pipeline would discard
        all of this.
        """
        # Find the skill the agent appears to be working toward, using the
        # tail of the instruction (which carries the most recent context).
        skill_id = self._find_skill((observation or "") + " " + instruction[-1500:])

        # v20.D — Curriculum-aware retrieval: if the inferred current skill
        # is already MASTERED (mastery >= threshold), override with the first
        # learnable-frontier skill (unmastered with mastered prereqs). This
        # prevents the agent from cycling on already-solved early-game tasks
        # (collect_wood, wake_up) instead of advancing to make_pickaxe etc.
        if (skill_id and getattr(self, "use_curriculum_retrieval", True)
                and self._evokg):
            try:
                cur_node = self._evokg.get_node(skill_id) or {}
                cur_mastery = cur_node.get("mastery", 0.0)
                MASTERY_THRESH = 0.5
                if cur_mastery >= MASTERY_THRESH:
                    frontier = self._evokg.get_learnable_frontier(
                        mastery_threshold=MASTERY_THRESH,
                    )
                    if frontier:
                        # First frontier item = least-mastered next target
                        next_target = frontier[0].get("node_id", skill_id)
                        if next_target != skill_id:
                            skill_id = next_target
            except Exception:
                pass

        # v20.A — Cascade principles from this skill + its prerequisites
        # (up to 2 hops). For compound Crafter tasks (e.g. make_stone_pickaxe
        # needs collect_wood + collect_stone + place_table tactics) the
        # single-skill lookup misses prereq knowledge.
        # v21.C — `max_total` configurable (default 8 for backward-compat;
        # Crafter benefits from tighter 4 to avoid late-iter prompt bloat).
        principles_cascade: list[tuple[str, str]] = []
        if skill_id and getattr(self, "use_principles_cascade", True):
            try:
                principles_cascade = self._get_principles_cascade(
                    skill_id, max_depth=2,
                    max_total=int(getattr(self, "principles_cascade_max", 8)),
                )
            except Exception:
                principles_cascade = []
        # Backward-compat: keep the flat list too for legacy callers
        principles = [p for _, p in principles_cascade]

        # v20.B — Action recipes (sequential-env-only): past action sequences
        # that successfully unlocked this skill. Far more actionable than text
        # principles for game action selection.
        recipes: list[dict] = []
        if skill_id and getattr(self, "use_action_recipes", True) and self._evokg:
            try:
                if hasattr(self._evokg, "get_action_recipes"):
                    recipes = self._evokg.get_action_recipes(skill_id, top_k=3)
                    # Also include recipes for prerequisite skills (composite tasks)
                    if hasattr(self._evokg, "get_prerequisites"):
                        prereqs = self._evokg.get_prerequisites(skill_id) or []
                        for pr in prereqs[:3]:
                            recipes.extend(self._evokg.get_action_recipes(pr, top_k=2))
            except Exception:
                recipes = []

        # v20.C — Goal-decomposition skill lattice: structured plan that
        # shows the agent the topologically-ordered sequence of skills
        # leading to the current target. More planful than just
        # cascade-principles. Each lattice node includes its top principles
        # and any captured action recipes.
        lattice: list[dict] = []
        if skill_id and getattr(self, "use_skill_lattice", True) and self._evokg:
            try:
                if hasattr(self._evokg, "get_skill_lattice"):
                    lattice = self._evokg.get_skill_lattice(
                        skill_id, max_depth=4, max_total=8,
                    )
            except Exception:
                lattice = []

        # Retrieve memories (success + failure) for similar situations.
        memories: list[dict] = []
        if getattr(self, "use_memory_bank", True):
            try:
                memories = self._retrieve_memories(
                    observation or instruction[-500:],
                    task_type="",
                    top_k=5,  # v20.A — bumped from 3 to 5
                    context_len=len(instruction),
                    skill_id=skill_id or None,
                )
            except Exception:
                memories = []

        # v20.A — separate success vs failure for clearer signal
        succ_mems = [m for m in memories if m.get("kind") == "success"]
        fail_mems = [m for m in memories if m.get("kind") != "success"]

        # Build the augmentation block.
        augment_lines: list[str] = []
        # v21.B + v21.D — Env-specific INFERENCE HINTS at TOP priority.
        # For Crafter: critical survival warnings + available/blocked craft
        # actions + next-craftable suggestion. Other envs can opt-in by
        # implementing `get_inference_hints(obs_text)`.
        if (getattr(self, "use_inference_hints", True)
                and self.environment is not None
                and hasattr(self.environment, "get_inference_hints")):
            try:
                hints = self.environment.get_inference_hints(observation or instruction[-2000:])
            except Exception:
                hints = None
            if hints:
                if hints.get("critical"):
                    augment_lines.append(hints["critical"])
                    augment_lines.append("")
                avail = hints.get("available_crafts") or []
                blocked = hints.get("blocked_crafts") or {}
                next_c = hints.get("next_craftable")
                if avail or blocked:
                    augment_lines.append("CRAFTING STATUS (based on current inventory):")
                    if next_c:
                        augment_lines.append(f"  ✅ NEXT BEST CRAFT: {next_c}")
                    if avail:
                        augment_lines.append(f"  available NOW: {', '.join(avail[:6])}")
                    if blocked:
                        for act, reason in list(blocked.items())[:4]:
                            augment_lines.append(f"  ❌ {act} blocked — needs {reason}")
                    augment_lines.append("")
        # v20.C — show the goal-decomposition lattice FIRST (it's the planning
        # backbone). When v20.C is on, we omit the v20.A flat-cascade because
        # the lattice already includes per-skill principles.
        if lattice:
            current_skill_name = (self._evokg.get_node(skill_id) or {}).get("name", skill_id) if self._evokg and skill_id else skill_id
            augment_lines.append(f"PLAN — to reach '{current_skill_name}', complete these in order:")
            for i, node in enumerate(lattice, 1):
                marker = " ← CURRENT" if node["skill_id"] == skill_id else ""
                ms = node.get("mastery", 0.0)
                aug = f"  [{i}] {node['name']}{marker}"
                if ms >= 0.5:
                    aug += " (✓ mastered)"
                augment_lines.append(aug)
                # Show top principle
                for p in node.get("principles", [])[:1]:
                    augment_lines.append(f"      • {p[:240]}")
                # Show top recipe
                for rec in node.get("recipes", [])[:1]:
                    if rec:
                        augment_lines.append(f"      • recipe: {' → '.join(rec[:8])}")
            augment_lines.append("")
        if recipes:
            augment_lines.append("PROVEN ACTION RECIPES (sequences that worked in past episodes):")
            seen_keys: set[str] = set()
            for r in recipes[:5]:
                key = "|".join(r.get("actions", []))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                sid = r.get("skill_id", "?")
                acts = r.get("actions", [])
                ctx = r.get("ctx_summary", "")[:120]
                augment_lines.append(
                    f"  [{sid}] {' → '.join(acts)}"
                    + (f" (when: {ctx})" if ctx else "")
                )
            augment_lines.append("")
        if principles_cascade:
            augment_lines.append("GAME TACTICS (current skill + prerequisites):")
            current_skill_name = (self._evokg.get_node(skill_id) or {}).get("name", skill_id) if self._evokg and skill_id else skill_id
            for i, (sname, p) in enumerate(principles_cascade, 1):
                # Mark which skill the principle is from
                tag = "" if sname == current_skill_name else f" [{sname}]"
                augment_lines.append(f"  {i}.{tag} {p[:300]}")
        if succ_mems:
            augment_lines.append("")
            augment_lines.append("PAST SUCCESSFUL APPROACHES IN SIMILAR SITUATIONS:")
            for m in succ_mems[:3]:
                snippet = (
                    m.get("reasoning_trace")
                    or m.get("correct_answer")
                    or m.get("question")
                    or ""
                )
                snippet = str(snippet)[:200].replace("\n", " ")
                if snippet:
                    augment_lines.append(f"  + {snippet}")
        if fail_mems:
            augment_lines.append("")
            augment_lines.append("AVOID — common mistakes from past attempts:")
            for m in fail_mems[:2]:
                snippet = (
                    m.get("action_pattern")
                    or m.get("question")
                    or ""
                )
                snippet = str(snippet)[:200].replace("\n", " ")
                if snippet:
                    augment_lines.append(f"  - {snippet}")
        augment_block = "\n".join(augment_lines).strip()

        # Insert the augmentation BEFORE "Current observation" / "Recent history"
        # so the LLM sees the system prompt → tactics → history → current obs.
        full_prompt = instruction
        if augment_block:
            for marker in ("Recent history (most recent last):",
                           "Current observation:",
                           "Reason briefly,"):
                if marker in instruction:
                    parts = instruction.split(marker, 1)
                    full_prompt = (
                        parts[0].rstrip()
                        + "\n\n" + augment_block + "\n\n"
                        + marker + parts[1]
                    )
                    break
            else:
                # No known marker → append at the end as a fallback.
                full_prompt = instruction + "\n\n" + augment_block

        # Sequential envs need short outputs (one action per call), but we
        # allow some reasoning room so a model can think briefly.
        # Temperature is configurable (default 0.0 for reproducible eval; set
        # higher to inject sampling diversity for game environments where
        # local optima trap the deterministic agent — BALROG uses 1.0).
        seq_temp = float(getattr(self, "sequential_temperature", 0.0))
        try:
            response = self._llm.generate(
                full_prompt, temperature=seq_temp, max_tokens=256,
            )
        except Exception as e:
            logger.debug(f"Sequential LLM call failed: {e}")
            response = ""

        self._last_call_metadata = {
            "strategy": "sequential",
            "search_strategy": "",
            "had_principles": len(principles) > 0,
            "memories_used": len(memories),
            "skill_id": skill_id or "",
            "raw_response": response,
        }
        return (response or "").strip()

    def save(self, path: str | Path) -> None:
        self._router.save(Path(path) / "strategy_router.json")

    def load(self, path: str | Path) -> None:
        from mage.core.strategy_router import StrategyRouter
        p = Path(path) / "strategy_router.json"
        if p.exists():
            self._router = StrategyRouter.load(p)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))


class GRPOLearner(LearnerAgent):
    """Learner that trains via GRPO (Group Relative Policy Optimization).

    For reasoning environments, this agent generates K answers per question
    from its own policy, scores each, and trains via policy gradient.
    No Explorer-generated answers needed — the Learner explores itself.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        environment: Any = None,
    ):
        super().__init__("GRPOLearner", config)
        from mage.training.grpo_trainer import GRPOConfig, GRPOTrainer

        grpo_config_dict = config or {}
        self.grpo_config = GRPOConfig.from_dict(grpo_config_dict)
        self.trainer = GRPOTrainer(self.grpo_config)
        self.environment = environment
        self.output_dir = Path(grpo_config_dict.get("output_dir", "experiments/default"))

    def set_environment(self, environment: Any) -> None:
        self.environment = environment

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """Train via GRPO on questions from the Explorer."""
        # Deduplicate questions
        seen = set()
        questions = []
        for traj in dataset.trajectories:
            if traj.instruction not in seen:
                seen.add(traj.instruction)
                questions.append({
                    "instruction": traj.instruction,
                    "context": traj.metadata.get("context", ""),
                    "expected_answer": traj.metadata.get("expected", ""),
                    "skill_id": traj.skill_id,
                })

        if not questions:
            logger.warning("No questions to train on")
            return {"iteration": iteration, "status": "skipped", "reason": "no_questions"}

        logger.info(f"GRPO: Sampling {self.grpo_config.num_samples} completions for {len(questions)} questions")

        all_completions = []
        all_rewards = []

        for q_idx, q in enumerate(questions):
            completions = self.trainer.sample_completions(
                instruction=q["instruction"],
                context=q["context"],
                num_samples=self.grpo_config.num_samples,
            )
            rewards = []
            for c in completions:
                answer = self._extract_answer(c)
                if q["expected_answer"] and answer and self.environment:
                    score = self.environment.evaluate_answer(answer, q["expected_answer"])
                else:
                    score = 0.0
                rewards.append(score)

            all_completions.append(completions)
            all_rewards.append(rewards)

            if (q_idx + 1) % 10 == 0:
                mean_r = sum(rewards) / len(rewards)
                logger.info(f"  [{q_idx+1}/{len(questions)}] mean_reward={mean_r:.3f}")

        checkpoint_dir = self.output_dir / f"iter_{iteration}" / "checkpoint"
        metrics = self.trainer.train(
            questions=questions, completions=all_completions,
            rewards=all_rewards, iteration=iteration, output_dir=checkpoint_dir,
        )
        metrics["num_questions"] = len(questions)
        return metrics

    def _extract_answer(self, response: str) -> str:
        import re
        for pattern in [
            r"(?:Answer|ANSWER|answer)\s*:\s*(.+?)(?:\n|$)",
            r"(?:Final answer|FINAL ANSWER)\s*:\s*(.+?)(?:\n|$)",
        ]:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else response.strip()

    def act(self, observation: str, instruction: str) -> str:
        try:
            return self.trainer.generate(observation, instruction, max_new_tokens=512).strip()
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return "noop"

    def save(self, path: str | Path) -> None:
        self.trainer.save_checkpoint(path)

    def load(self, path: str | Path) -> None:
        self.trainer.load_checkpoint(path)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))


class APILearner(LearnerAgent):
    """Learner that uses an API backend (no local training).

    Useful for:
    - Evaluating API models (GPT-4o, Claude) as baselines
    - Explorer=Learner self-evolving mode with API models
    - Testing the pipeline without GPU

    Training is simulated (no actual weight updates).
    """

    def __init__(
        self,
        llm_backend: Any = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__("APILearner", config)
        self.llm = llm_backend
        self._training_log: list[dict] = []

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """No real training for API models — log the data for tracking."""
        metrics = {
            "iteration": iteration,
            "status": "api_model_no_training",
            "num_trajectories": len(dataset),
            "skills_seen": list({t.skill_id for t in dataset.trajectories if t.skill_id}),
        }
        self._training_log.append(metrics)
        return metrics

    def act(self, observation: str, instruction: str) -> str:
        """Generate action via API call."""
        if self.llm is None:
            return "noop"

        prompt = f"Task: {instruction}\n\n{observation}\n\nChoose your next action:"
        system = "You are an intelligent agent. Respond with ONLY a valid action name."

        try:
            response = self.llm.generate(prompt, system, temperature=0.3, max_tokens=32)
            return response.strip().split("\n")[0].strip().lower().replace(" ", "_")
        except Exception as e:
            logger.debug(f"API generation failed: {e}")
            return "noop"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))
