"""Stub agent implementations for toy validation of the evolution loop.

These agents use simple heuristics instead of LLMs — enough to verify
that the evolution loop, EvoKG updates, and snapshot saving work correctly.
"""

from __future__ import annotations

import random
from typing import Any

from mage.agents.base import (
    CriticAgent,
    CuratorAgent,
    ExplorerAgent,
    LearnerAgent,
    NavigatorAgent,
    SkillDataset,
    Trajectory,
)
from mage.core.evokg import EvoKG
from mage.core.schemas import (
    Edge,
    EvolutionPlan,
    EvolutionTarget,
    ExperienceNode,
    FailureMode,
    RelationType,
    SkillEvaluation,
    StructuredEvaluation,
)
from mage.environments.toy_gridworld import ToyGridWorld


class StubNavigator(NavigatorAgent):
    """Navigator that uses graph algorithm only (no LLM).

    Computes learnable frontier from EvoKG and ranks by priority.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("StubNavigator", config)
        self.max_targets = (config or {}).get("max_targets", 3)

    def plan(self, evokg: EvoKG, iteration: int) -> EvolutionPlan:
        frontier = evokg.get_learnable_frontier(
            mastery_threshold=self.config.get("mastery_threshold", 0.5),
        )

        targets = []
        for node in frontier[: self.max_targets]:
            priority = 1.0 - node.get("mastery", 0.0)
            targets.append(EvolutionTarget(
                skill_id=node["node_id"],
                priority=priority,
                reason=f"In learnable frontier, mastery={node.get('mastery', 0.0):.2f}",
                exploration_brief=f"Practice skill: {node.get('name', node['node_id'])}",
            ))

        return EvolutionPlan(
            iteration=iteration,
            targets=targets,
            rationale=f"Selected {len(targets)} skills from frontier (iter {iteration})",
        )


class StubExplorer(ExplorerAgent):
    """Explorer that runs episodes in the toy grid world."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("StubExplorer", config)

    def explore(
        self,
        plan: EvolutionPlan,
        evokg: EvoKG,
        environment: Any,
        num_episodes: int = 50,
    ) -> SkillDataset:
        dataset = SkillDataset(iteration=plan.iteration)

        if not isinstance(environment, ToyGridWorld):
            return dataset

        episodes_per_skill = max(1, num_episodes // max(len(plan.targets), 1))

        for target in plan.targets:
            mastery = evokg.get_mastery(target.skill_id)
            # Simulate exploration with gradually improving success
            exploration_mastery = min(1.0, mastery + 0.3 + random.uniform(0, 0.2))

            for _ in range(episodes_per_skill):
                success, obs, actions = environment.attempt_skill(
                    target.skill_id, exploration_mastery,
                )
                dataset.add(Trajectory(
                    instruction=target.exploration_brief,
                    observations=obs,
                    actions=actions,
                    reward=1.0 if success else 0.0,
                    skill_id=target.skill_id,
                ))

        return dataset


class StubLearner(LearnerAgent):
    """Learner that simulates training by tracking exposure to skills.

    Mastery improves proportionally to the number of successful trajectories seen.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("StubLearner", config)
        self._skill_exposure: dict[str, float] = {}
        self._training_iterations = 0

    def train(self, dataset: SkillDataset, iteration: int) -> dict[str, Any]:
        self._training_iterations += 1
        skills_trained = set()

        for traj in dataset.trajectories:
            if traj.skill_id:
                skills_trained.add(traj.skill_id)
                current = self._skill_exposure.get(traj.skill_id, 0.0)
                # Successful trajectories contribute more
                gain = 0.05 if traj.reward > 0 else 0.01
                self._skill_exposure[traj.skill_id] = min(1.0, current + gain)

        return {
            "iteration": iteration,
            "num_trajectories": len(dataset),
            "skills_trained": list(skills_trained),
            "training_step": self._training_iterations,
        }

    def act(self, observation: str, instruction: str) -> str:
        return "do"  # stub

    def get_skill_mastery(self, skill_id: str) -> float:
        """Get simulated mastery level for a skill."""
        return self._skill_exposure.get(skill_id, 0.0)


class StubCritic(CriticAgent):
    """Critic that evaluates the Learner by running test episodes."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("StubCritic", config)
        self.trials_per_skill = (config or {}).get("trials_per_skill", 10)

    def evaluate(
        self,
        learner: LearnerAgent,
        evokg: EvoKG,
        environment: Any,
        iteration: int,
    ) -> StructuredEvaluation:
        if not isinstance(environment, ToyGridWorld) or not isinstance(learner, StubLearner):
            return StructuredEvaluation(iteration=iteration)

        evaluations = []
        total_success = 0
        total_attempts = 0

        for skill_id in environment.get_skill_ids():
            mastery = learner.get_skill_mastery(skill_id)
            successes = 0

            for _ in range(self.trials_per_skill):
                success, _, _ = environment.attempt_skill(skill_id, mastery)
                if success:
                    successes += 1

            success_rate = successes / self.trials_per_skill
            total_success += successes
            total_attempts += self.trials_per_skill

            failure_modes = []
            if success_rate < 0.5:
                prereqs = evokg.get_prerequisites(skill_id)
                unmastered_prereqs = [
                    p for p in prereqs
                    if evokg.get_mastery(p) < 0.5
                ]
                if unmastered_prereqs:
                    failure_modes.append(FailureMode(
                        "missing_prerequisite",
                        f"Prerequisites not mastered: {unmastered_prereqs}",
                        0.7,
                    ))
                else:
                    failure_modes.append(FailureMode(
                        "insufficient_practice",
                        f"Mastery too low: {mastery:.2f}",
                        1.0 - success_rate,
                    ))

            evaluations.append(SkillEvaluation(
                skill_id=skill_id,
                success_rate=success_rate,
                attempts=self.trials_per_skill,
                failure_modes=failure_modes,
                recommended_focus=[skill_id] if success_rate < 0.5 else [],
            ))

        overall = total_success / total_attempts if total_attempts > 0 else 0.0

        return StructuredEvaluation(
            iteration=iteration,
            evaluations=evaluations,
            overall_success_rate=overall,
            summary=f"Evaluated {len(evaluations)} skills, overall success: {overall:.2%}",
        )


class StubCurator(CuratorAgent):
    """Curator that updates EvoKG based on Critic's structured evaluation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("StubCurator", config)
        self._exp_counter = 0

    def update(
        self,
        evokg: EvoKG,
        evaluation: StructuredEvaluation,
        dataset: SkillDataset,
        iteration: int,
    ) -> EvoKG:
        iter_tag = f"iter_{iteration}"

        # Update mastery levels from evaluation
        for skill_eval in evaluation.evaluations:
            if evokg.has_node(skill_eval.skill_id):
                evokg.update_mastery(
                    skill_eval.skill_id,
                    skill_eval.success_rate,
                    iter_tag,
                )

        # Add experience patterns from trajectories
        skill_outcomes: dict[str, dict[str, int]] = {}
        for traj in dataset.trajectories:
            if traj.skill_id:
                if traj.skill_id not in skill_outcomes:
                    skill_outcomes[traj.skill_id] = {"success": 0, "failure": 0}
                if traj.reward > 0:
                    skill_outcomes[traj.skill_id]["success"] += 1
                else:
                    skill_outcomes[traj.skill_id]["failure"] += 1

        for skill_id, outcomes in skill_outcomes.items():
            total = outcomes["success"] + outcomes["failure"]
            if total > 0:
                self._exp_counter += 1
                exp_id = f"exp_{iteration}_{self._exp_counter}"
                outcome = "success" if outcomes["success"] > outcomes["failure"] else "failure"
                confidence = outcomes["success"] / total

                exp_node = ExperienceNode(
                    node_id=exp_id,
                    action_pattern=f"Attempted {skill_id}: {outcomes['success']}/{total} success",
                    outcome=outcome,
                    context_conditions=f"iteration_{iteration}",
                    confidence=confidence,
                    domain_origin="toy_gridworld",
                    frequency=total,
                )
                evokg.add_experience(exp_node)
                evokg.add_edge(Edge(
                    exp_id, skill_id, RelationType.EXPERIENCE_OF_SKILL,
                ))

        # Prune low-confidence experiences
        evokg.prune_low_confidence(threshold=0.2)

        return evokg
