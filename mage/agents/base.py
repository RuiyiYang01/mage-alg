"""Base agent interfaces for MAGE framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mage.core.evokg import EvoKG
from mage.core.schemas import EvolutionPlan, StructuredEvaluation

# Use string annotation to avoid circular imports with environments.base
# At runtime, environment parameter accepts any BaseEnvironment instance


@dataclass
class Trajectory:
    """A single (instruction, action_sequence) pair collected by Explorer."""
    instruction: str
    observations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    reward: float = 0.0
    skill_id: str | None = None     # which skill this trajectory demonstrates
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "observations": self.observations,
            "actions": self.actions,
            "reward": self.reward,
            "skill_id": self.skill_id,
            "metadata": self.metadata,
        }


@dataclass
class SkillDataset:
    """Collection of trajectories for training the Learner."""
    trajectories: list[Trajectory] = field(default_factory=list)
    iteration: int = 0

    def __len__(self) -> int:
        return len(self.trajectories)

    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def filter_by_skill(self, skill_id: str) -> list[Trajectory]:
        return [t for t in self.trajectories if t.skill_id == skill_id]


class BaseAgent(ABC):
    """Abstract base for all MAGE agents."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        ...


class NavigatorAgent(BaseAgent):
    """Plans the evolution curriculum by querying EvoKG.

    Stage 1: Graph algorithm computes learnable frontier and priorities.
    Stage 2: (Optional) LLM selects and orders from candidates.
    """

    @abstractmethod
    def plan(self, evokg: EvoKG, iteration: int) -> EvolutionPlan:
        """Produce an evolution plan for the next iteration."""
        ...

    def run(self, **kwargs: Any) -> EvolutionPlan:
        return self.plan(kwargs["evokg"], kwargs.get("iteration", 0))


class ExplorerAgent(BaseAgent):
    """Collects trajectories by interacting with an environment.

    Guided by the evolution plan from Navigator and past experience from EvoKG.
    """

    @abstractmethod
    def explore(
        self,
        plan: EvolutionPlan,
        evokg: EvoKG,
        environment: Any,
        num_episodes: int = 50,
    ) -> SkillDataset:
        """Explore the environment and collect trajectories."""
        ...

    def run(self, **kwargs: Any) -> SkillDataset:
        return self.explore(
            kwargs["plan"],
            kwargs["evokg"],
            kwargs["environment"],
            kwargs.get("num_episodes", 50),
        )


class LearnerAgent(BaseAgent):
    """The agent being evolved — trains on skill datasets.

    Training mechanism is pluggable (SFT, DPO, RL).
    """

    @abstractmethod
    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        """Train on the skill dataset. Returns training metrics."""
        ...

    @abstractmethod
    def act(self, observation: str, instruction: str) -> str:
        """Generate an action given observation and instruction."""
        ...

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return self.train(kwargs["dataset"], kwargs.get("iteration", 0))


class CriticAgent(BaseAgent):
    """Evaluates the Learner and produces structured feedback.

    Outputs structured evaluations with per-skill success rates and
    failure mode analysis — directly writable to EvoKG.
    """

    @abstractmethod
    def evaluate(
        self,
        learner: LearnerAgent,
        evokg: EvoKG,
        environment: Any,
        iteration: int,
    ) -> StructuredEvaluation:
        """Evaluate learner on skills from EvoKG."""
        ...

    def run(self, **kwargs: Any) -> StructuredEvaluation:
        return self.evaluate(
            kwargs["learner"],
            kwargs["evokg"],
            kwargs["environment"],
            kwargs.get("iteration", 0),
        )


class CuratorAgent(BaseAgent):
    """Maintains and evolves the EvoKG.

    Updates mastery levels, adds experience patterns, discovers new tasks,
    prunes stale knowledge, identifies cross-domain analogies.
    """

    @abstractmethod
    def update(
        self,
        evokg: EvoKG,
        evaluation: StructuredEvaluation,
        dataset: SkillDataset,
        iteration: int,
    ) -> EvoKG:
        """Update EvoKG based on evaluation results and collected data."""
        ...

    def run(self, **kwargs: Any) -> EvoKG:
        return self.update(
            kwargs["evokg"],
            kwargs["evaluation"],
            kwargs["dataset"],
            kwargs.get("iteration", 0),
        )
