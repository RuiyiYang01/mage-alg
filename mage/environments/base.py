"""Environment abstraction layer for MAGE.

Provides a unified interface across all evaluation domains:
- Interactive environments (Crafter, Webshop): multi-step, stateful
- QA environments (multi-hop QA, ST-Bench): question → reasoning → answer
- Meta environments (cross-domain, continual learning): wrappers over base envs

All environments must produce:
1. Skills (capability nodes for EvoKG)
2. Episodes/instances that can be run
3. Success signals for evaluation

The key abstraction is the InteractionMode:
- SEQUENTIAL: agent takes actions in a stateful environment (games, web)
- REASONING: agent processes a question through reasoning steps (QA, ST)
Both produce Trajectories with (instruction, observations, actions, reward).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mage.core.evokg import EvoKG


class InteractionMode(str, Enum):
    """How the agent interacts with this environment."""
    SEQUENTIAL = "sequential"   # Crafter, Webshop: obs → action → obs → ...
    REASONING = "reasoning"     # QA, ST-Bench: question → think → retrieve → answer


@dataclass
class SkillDefinition:
    """Definition of a skill/capability in an environment."""
    skill_id: str
    name: str
    category: str = "general"
    prerequisites: list[str] = field(default_factory=list)
    description: str = ""
    difficulty: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of running one episode/instance in an environment."""
    observations: list[str]
    actions: list[str]
    reward: float
    success: bool
    skills_achieved: list[str] = field(default_factory=list)
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationInstance:
    """A single evaluation instance (task, question, episode seed, etc.)."""
    instance_id: str
    instruction: str                    # task description or question
    target_skill: str | None = None     # which skill this tests
    context: str = ""                   # additional context (documents, scene, etc.)
    expected_answer: str | None = None  # for QA: ground truth answer
    seed: int | None = None             # for interactive: environment seed
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Abstract environment interface for MAGE.

    All environments — games, web, QA, reasoning — implement this interface.
    The evolution loop, Explorer, Critic, and Learner interact only through this.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name (e.g., 'crafter', 'webshop', 'hotpotqa')."""
        ...

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain tag for EvoKG nodes (e.g., 'crafter', 'web', 'qa')."""
        ...

    @property
    @abstractmethod
    def interaction_mode(self) -> InteractionMode:
        """How agents interact with this environment."""
        ...

    @abstractmethod
    def get_skill_definitions(self) -> list[SkillDefinition]:
        """Return all skills/capabilities in this environment.

        These become CapabilityNode entries in EvoKG's G_cap.
        """
        ...

    @abstractmethod
    def get_action_space(self) -> list[str]:
        """Return valid action names.

        For SEQUENTIAL: discrete actions (move_left, click[...], etc.)
        For REASONING: reasoning step types (retrieve, reason, answer, etc.)
        """
        ...

    @abstractmethod
    def run_episode(
        self,
        agent_fn: Any,
        instruction: str,
        seed: int | None = None,
        max_steps: int = 100,
        context: str = "",
    ) -> EpisodeResult:
        """Run a single episode with the agent.

        Args:
            agent_fn: Callable(observation: str, instruction: str) → action: str
            instruction: Task description or question
            seed: Random seed for reproducibility
            max_steps: Maximum interaction steps
            context: Additional context (documents, scene description, etc.)

        Returns:
            EpisodeResult with observations, actions, reward, success, etc.
        """
        ...

    @abstractmethod
    def get_evaluation_instances(
        self,
        skill_id: str | None = None,
        num_instances: int = 10,
    ) -> list[EvaluationInstance]:
        """Get evaluation instances for a skill (or all skills).

        For games: returns seeds + task instructions
        For QA: returns questions + contexts + expected answers
        """
        ...

    @abstractmethod
    def build_evokg(self) -> EvoKG:
        """Build an EvoKG seeded with this environment's skill tree.

        Returns an EvoKG with:
        - G_cap: capability nodes for each skill with prerequisite edges
        - G_env: environment entity nodes
        - G_task: top-level task nodes
        """
        ...

    def evaluate_answer(self, predicted: str, expected: str) -> float:
        """Score a predicted answer against expected (for QA environments).

        Default: exact match. Override for F1, BLEU, etc.
        Returns float in [0, 1].
        """
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0

    def get_prerequisite_items(self, skill_id: str) -> dict[str, Any]:
        """Get prerequisite inventory/state for evaluating a specific skill.

        For Crafter: returns items needed (e.g., stone_pickaxe for collect_iron)
        For QA: returns context documents needed
        Default: empty (no prerequisites needed in environment state).
        """
        return {}


class SequentialEnvironment(BaseEnvironment):
    """Base for stateful, multi-step environments (Crafter, Webshop).

    Agents interact through: reset() → step(action) → step(action) → ...
    """

    @property
    def interaction_mode(self) -> InteractionMode:
        return InteractionMode.SEQUENTIAL

    @abstractmethod
    def reset(self, seed: int | None = None, **kwargs: Any) -> str:
        """Reset environment and return initial observation as text."""
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Take an action, return (observation_text, reward, done, info)."""
        ...

    def run_episode(
        self,
        agent_fn: Any,
        instruction: str,
        seed: int | None = None,
        max_steps: int = 100,
        context: str = "",
    ) -> EpisodeResult:
        """Default episode runner for sequential environments."""
        obs = self.reset(seed=seed)
        observations = [obs]
        actions = []
        total_reward = 0.0
        skills_achieved = []

        for step_num in range(max_steps):
            action = agent_fn(obs, instruction)
            obs, reward, done, info = self.step(action)
            observations.append(obs)
            actions.append(action)
            total_reward += reward

            # Track achievements if available
            new_skills = info.get("new_achievements", {})
            if new_skills:
                skills_achieved.extend(new_skills.keys())

            if done:
                break

        return EpisodeResult(
            observations=observations,
            actions=actions,
            reward=total_reward,
            success=total_reward > 0 or len(skills_achieved) > 0,
            skills_achieved=skills_achieved,
            metadata={"steps": len(actions), "seed": seed},
        )


class ReasoningEnvironment(BaseEnvironment):
    """Base for QA / reasoning environments (multi-hop QA, ST-Bench).

    Agents interact through: question + context → reasoning steps → answer.
    A "trajectory" here is a reasoning trace, not a game trajectory.
    """

    @property
    def interaction_mode(self) -> InteractionMode:
        return InteractionMode.REASONING

    @abstractmethod
    def get_question(self, instance_id: str) -> tuple[str, str, str]:
        """Get a question instance.

        Returns (question, context, expected_answer).
        """
        ...

    def get_test_instances(self, num_instances: int = 100) -> list[EvaluationInstance]:
        """Get HELD-OUT test instances — disjoint from evolution training set.

        v15.5-large: this pool is loaded alongside the training pool but
        never consumed by the evolution loop. It's evaluated once at the
        end of training (via scripts/eval_held_out.py) to measure true
        generalization — accuracy on questions the learner has never
        seen during evolution, where stored success memories cannot be
        keyed by question.

        Subclasses that don't implement this return an empty list,
        meaning no held-out evaluation is possible for them.
        """
        return []

    def run_episode(
        self,
        agent_fn: Any,
        instruction: str,
        seed: int | None = None,
        max_steps: int = 10,
        context: str = "",
    ) -> EpisodeResult:
        """Default episode runner for reasoning environments.

        The agent produces reasoning steps as "actions" and a final answer.
        Each reasoning step is one "action" in the trajectory.
        """
        observations = [f"Question: {instruction}\n\nContext: {context}"]
        actions = []
        final_answer = ""

        # Multi-step reasoning loop
        current_context = context
        for step_num in range(max_steps):
            prompt = self._build_reasoning_prompt(instruction, current_context, actions)
            response = agent_fn(prompt, instruction)

            # Parse response: is it a final answer or an intermediate step?
            if self._is_final_answer(response):
                final_answer = self._extract_answer(response)
                actions.append(f"answer: {final_answer}")
                observations.append(f"Final answer: {final_answer}")
                break
            else:
                # Intermediate reasoning step (retrieval, decomposition, etc.)
                actions.append(response)
                step_result = self._execute_reasoning_step(response, current_context)
                observations.append(step_result)
                current_context = f"{current_context}\n\n{step_result}"

        return EpisodeResult(
            observations=observations,
            actions=actions,
            reward=0.0,  # reward computed separately via evaluate_answer
            success=False,  # determined by answer evaluation
            metadata={"final_answer": final_answer, "num_steps": len(actions)},
        )

    def _build_reasoning_prompt(
        self, question: str, context: str, steps_so_far: list[str],
    ) -> str:
        """Build prompt for next reasoning step."""
        prompt = f"Question: {question}\n\nContext:\n{context}"
        if steps_so_far:
            prompt += f"\n\nReasoning so far:\n" + "\n".join(
                f"Step {i+1}: {s}" for i, s in enumerate(steps_so_far)
            )
        prompt += "\n\nProvide your next reasoning step, or 'ANSWER: <your answer>' if ready."
        return prompt

    def _is_final_answer(self, response: str) -> bool:
        """Check if response contains a final answer."""
        response_lower = response.strip().lower()
        return response_lower.startswith("answer:") or response_lower.startswith("final answer:")

    def _extract_answer(self, response: str) -> str:
        """Extract the answer from a response string."""
        for prefix in ["ANSWER:", "Answer:", "answer:", "Final Answer:", "final answer:"]:
            if response.strip().startswith(prefix):
                return response.strip()[len(prefix):].strip()
        return response.strip()

    def _execute_reasoning_step(self, step: str, context: str) -> str:
        """Execute an intermediate reasoning step.

        Override in subclasses for retrieval, tool use, etc.
        Default: return the step as-is (pure chain-of-thought).
        """
        return f"[Reasoning step executed: {step[:100]}]"


class MetaEnvironment(BaseEnvironment):
    """Wrapper for cross-domain transfer and continual learning experiments.

    Composes multiple base environments and manages the transfer/sequential protocol.
    """

    def __init__(self, environments: list[BaseEnvironment]):
        self.environments = environments
        self._active_idx = 0

    @property
    def name(self) -> str:
        names = [e.name for e in self.environments]
        return f"meta_{'_'.join(names)}"

    @property
    def domain(self) -> str:
        return "meta"

    @property
    def interaction_mode(self) -> InteractionMode:
        return self.active_env.interaction_mode

    @property
    def active_env(self) -> BaseEnvironment:
        return self.environments[self._active_idx]

    def switch_environment(self, idx: int) -> None:
        """Switch to a different environment (for cross-domain / continual)."""
        self._active_idx = idx

    def get_skill_definitions(self) -> list[SkillDefinition]:
        """Aggregate skills from all environments with domain tags."""
        all_skills = []
        for env in self.environments:
            for skill in env.get_skill_definitions():
                # Prefix skill IDs with domain to avoid collisions
                tagged = SkillDefinition(
                    skill_id=f"{env.domain}_{skill.skill_id}",
                    name=f"[{env.domain}] {skill.name}",
                    category=skill.category,
                    prerequisites=[f"{env.domain}_{p}" for p in skill.prerequisites],
                    description=skill.description,
                    difficulty=skill.difficulty,
                    metadata={**skill.metadata, "source_domain": env.domain},
                )
                all_skills.append(tagged)
        return all_skills

    def get_action_space(self) -> list[str]:
        return self.active_env.get_action_space()

    def run_episode(self, agent_fn, instruction, seed=None, max_steps=100, context=""):
        return self.active_env.run_episode(agent_fn, instruction, seed, max_steps, context)

    def get_evaluation_instances(self, skill_id=None, num_instances=10):
        return self.active_env.get_evaluation_instances(skill_id, num_instances)

    def build_evokg(self) -> EvoKG:
        """Build a merged EvoKG from all environments."""
        from mage.core.schemas import CapabilityNode, Edge, RelationType

        evokg = EvoKG()
        for env in self.environments:
            sub_kg = env.build_evokg()
            # Merge nodes with domain prefix
            for node_id, data in sub_kg.graph.nodes(data=True):
                prefixed_id = f"{env.domain}_{node_id}"
                data_copy = dict(data)
                data_copy["domain"] = env.domain
                evokg.graph.add_node(prefixed_id, **data_copy)

            # Merge edges with domain prefix
            for src, dst, data in sub_kg.graph.edges(data=True):
                evokg.graph.add_edge(f"{env.domain}_{src}", f"{env.domain}_{dst}", **data)

        return evokg
