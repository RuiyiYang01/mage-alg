"""KG Curator Agent: Maintains and evolves the EvoKG.

Updates mastery levels from Critic evaluations, adds experience patterns
from trajectories, discovers new task dependencies, and prunes stale knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

from mage.agents.base import CuratorAgent, SkillDataset
from mage.core.evokg import EvoKG
from mage.core.schemas import (
    Edge,
    ExperienceNode,
    RelationType,
    StructuredEvaluation,
)
from mage.utils.llm import LLMBackend

logger = logging.getLogger(__name__)

ABSTRACTION_PROMPT = """\
You are analyzing agent trajectories in a game environment. Given the trajectory below, \
extract a reusable experience pattern.

Skill: {skill_id}
Outcome: {outcome}
Actions: {actions}
Final observation summary: {observation}

Describe the pattern in one concise sentence (e.g., "navigate to resource, collect using pickaxe, \
return to crafting table"). Output ONLY the pattern, nothing else."""


class EvoKGCurator(CuratorAgent):
    """Curator that updates EvoKG based on evaluation and exploration data.

    Operations:
    1. Update G_cap mastery levels from Critic's evaluation
    2. Add experience patterns to G_exp (LLM-abstracted from trajectories)
    3. Discover new tasks for G_task
    4. Prune low-confidence experience nodes
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__("EvoKGCurator", config)
        self.llm = llm
        self.prune_threshold = self.config.get("prune_threshold", 0.2)
        self.max_experiences_per_skill = self.config.get("max_experiences_per_skill", 5)
        self._exp_counter = 0

    def update(
        self,
        evokg: EvoKG,
        evaluation: StructuredEvaluation,
        dataset: SkillDataset,
        iteration: int,
    ) -> EvoKG:
        iter_tag = f"iter_{iteration}"

        # Step 1: Update mastery levels
        updated_skills = self._update_mastery(evokg, evaluation, iter_tag)
        logger.info(f"  Updated mastery for {updated_skills} skills")

        # Step 2: Add experience patterns
        new_experiences = self._add_experiences(evokg, evaluation, dataset, iteration)
        logger.info(f"  Added {new_experiences} experience patterns")

        # Step 3: Prune stale experiences
        pruned = evokg.prune_low_confidence(self.prune_threshold)
        if pruned:
            logger.info(f"  Pruned {pruned} low-confidence experiences")

        return evokg

    def _update_mastery(
        self,
        evokg: EvoKG,
        evaluation: StructuredEvaluation,
        iter_tag: str,
    ) -> int:
        """Update G_cap mastery levels from Critic's evaluation.

        Uses a ratchet mechanism: mastery can only decrease slowly (decay_rate)
        but can increase freely. This prevents catastrophic forgetting of
        skills that aren't currently being targeted.

        IMPORTANT: The Critic uses task_type as skill_id (e.g.
        ``STBENCH_DIRECTION_DETERMINATION``, ``bridge``), but the EvoKG stores
        Claude-discovered names (e.g. ``coordinate_parsing``). We use
        ``evokg.map_task_to_skill`` to bridge the two ID spaces — without this
        mapping, ``has_node`` returns False for every skill and mastery never
        moves from 0.0.
        """
        count = 0
        alpha = self.config.get("mastery_ema_alpha", 0.6)
        decay_rate = self.config.get("mastery_decay_rate", 0.1)

        for skill_eval in evaluation.evaluations:
            # v12.1d: Always update per-task-type mastery (parallel signal,
            # independent of any task_type → skill mapping). The Navigator
            # and round-robin selector can read this directly.
            raw_task_type = skill_eval.skill_id
            old_task_mastery = evokg.get_task_mastery(raw_task_type)
            new_eval = skill_eval.success_rate
            if new_eval >= old_task_mastery:
                new_task_mastery = alpha * new_eval + (1 - alpha) * old_task_mastery
            else:
                new_task_mastery = (
                    old_task_mastery - decay_rate * (old_task_mastery - new_eval)
                )
            evokg.update_task_mastery(raw_task_type, new_task_mastery, iter_tag)

            # Per-skill mastery (legacy path) — requires task_type → skill map
            skill_id = raw_task_type
            if not evokg.has_node(skill_id):
                mapped = evokg.map_task_to_skill(skill_id)
                if mapped:
                    skill_id = mapped
            if not evokg.has_node(skill_id):
                logger.debug(
                    f"  Curator: no EvoKG skill matches '{raw_task_type}', "
                    f"skipping per-skill mastery update (task_mastery still tracked)"
                )
                continue

            old_mastery = evokg.get_mastery(skill_id)
            if new_eval >= old_mastery:
                new_mastery = alpha * new_eval + (1 - alpha) * old_mastery
            else:
                new_mastery = old_mastery - decay_rate * (old_mastery - new_eval)

            evokg.update_mastery(skill_id, new_mastery, iter_tag)
            count += 1
        return count

    def _add_experiences(
        self,
        evokg: EvoKG,
        evaluation: StructuredEvaluation,
        dataset: SkillDataset,
        iteration: int,
    ) -> int:
        """Add abstracted experience patterns to G_exp."""
        count = 0

        # Add failure patterns from evaluation (capped to avoid overwhelming the KG)
        max_failures_per_skill = self.config.get("max_failures_per_skill", 3)
        for skill_eval in evaluation.evaluations:
            # Resolve task_type to EvoKG skill node
            target_skill = skill_eval.skill_id
            if not evokg.has_node(target_skill):
                mapped = evokg.map_task_to_skill(target_skill)
                if mapped:
                    target_skill = mapped
            if not evokg.has_node(target_skill):
                continue  # no matching skill in EvoKG

            # Check how many failure experiences already exist for this skill
            existing_failures = [
                e for e in evokg.get_experiences_for_skill(target_skill)
                if e.get("outcome") == "failure"
            ]
            if len(existing_failures) >= max_failures_per_skill:
                continue  # enough failure patterns already recorded

            for fm in skill_eval.failure_modes[:1]:  # only top failure mode
                if fm.frequency < 0.3:
                    continue

                self._exp_counter += 1
                exp_id = f"exp_{iteration}_{self._exp_counter}"

                exp_node = ExperienceNode(
                    node_id=exp_id,
                    action_pattern=f"Failure: {fm.detail}",
                    outcome="failure",
                    context_conditions=f"skill={target_skill}, iter={iteration}",
                    confidence=fm.frequency,
                    domain_origin=evokg.graph.nodes.get(target_skill, {}).get("domain", "unknown"),
                    frequency=int(fm.frequency * skill_eval.attempts),
                )
                evokg.add_experience(exp_node)
                evokg.add_edge(Edge(
                    exp_id, target_skill,
                    RelationType.EXPERIENCE_OF_SKILL,
                ))
                count += 1

        # Add success patterns from trajectories
        successful_by_skill: dict[str, list] = {}
        for traj in dataset.trajectories:
            if traj.reward > 0 and traj.skill_id:
                if traj.skill_id not in successful_by_skill:
                    successful_by_skill[traj.skill_id] = []
                successful_by_skill[traj.skill_id].append(traj)

        for skill_id, trajectories in successful_by_skill.items():
            # Limit experiences per skill
            existing = evokg.get_experiences_for_skill(skill_id)
            if len(existing) >= self.max_experiences_per_skill:
                continue

            # Abstract the pattern
            pattern = self._abstract_pattern(skill_id, trajectories, "success")

            self._exp_counter += 1
            exp_id = f"exp_{iteration}_{self._exp_counter}"

            exp_node = ExperienceNode(
                node_id=exp_id,
                action_pattern=pattern,
                outcome="success",
                context_conditions=f"skill={skill_id}, iter={iteration}",
                confidence=len(trajectories) / max(len(dataset), 1),
                domain_origin=evokg.graph.nodes.get(skill_id, {}).get("domain", "unknown"),
                frequency=len(trajectories),
            )
            evokg.add_experience(exp_node)

            if evokg.has_node(skill_id):
                evokg.add_edge(Edge(
                    exp_id, skill_id,
                    RelationType.EXPERIENCE_OF_SKILL,
                ))
            count += 1

        return count

    def _abstract_pattern(
        self,
        skill_id: str,
        trajectories: list,
        outcome: str,
    ) -> str:
        """Abstract raw trajectories into a reusable pattern.

        Uses LLM if available, otherwise falls back to action sequence summary.
        """
        # Pick a representative trajectory
        traj = trajectories[0]

        if self.llm is not None:
            try:
                obs_summary = traj.observations[-1][:200] if traj.observations else "N/A"
                prompt = ABSTRACTION_PROMPT.format(
                    skill_id=skill_id,
                    outcome=outcome,
                    actions=" → ".join(traj.actions),
                    observation=obs_summary,
                )
                pattern = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
                return pattern.strip()
            except Exception as e:
                logger.debug(f"LLM abstraction failed: {e}")

        # Fallback: summarize action sequence
        action_str = " → ".join(traj.actions[:8])
        return f"{outcome.title()} pattern for {skill_id}: {action_str}"
