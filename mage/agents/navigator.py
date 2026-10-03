"""Navigator Agent: Hybrid graph algorithm + LLM curriculum planner.

Stage 1 (Graph Algorithm — deterministic):
    - Topological sort on G_cap prerequisite edges
    - Filter to learnable frontier (all prereqs mastered, own mastery < threshold)
    - Rank by priority = (1 - mastery) * task_importance * transfer_potential
    - Output top-K candidates

Stage 2 (LLM Reasoning — contextual):
    - LLM receives candidate list + relevant G_exp subgraph
    - Selects and orders based on contextual factors
    - Generates exploration brief for Explorer
    - Fallback: if LLM fails, use Stage 1 ranking directly
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mage.agents.base import NavigatorAgent
from mage.core.evokg import EvoKG
from mage.core.schemas import (
    EvolutionPlan,
    EvolutionTarget,
    NodeType,
)
from mage.utils.llm import LLMBackend

logger = logging.getLogger(__name__)

NAVIGATOR_SYSTEM_PROMPT = """\
You are the Navigator agent in the MAGE framework. Your role is to plan the evolution \
curriculum — deciding which skills the Learner agent should focus on next.

You will receive:
1. A list of candidate skills from the learnable frontier (skills whose prerequisites \
are mastered but that the agent hasn't mastered yet)
2. Past experience patterns related to these skills (successes and failures)
3. Current mastery levels

Your job is to:
1. Select and prioritize which skills to focus on in the next evolution round
2. Generate a brief exploration guide for each selected skill
3. Consider which skills can be practiced together efficiently

Output a JSON object with this structure:
{
  "targets": [
    {
      "skill_id": "...",
      "priority": 0.0-1.0,
      "reason": "why this skill was selected",
      "exploration_brief": "specific guidance for the Explorer agent"
    }
  ],
  "rationale": "overall reasoning for this plan"
}"""


class HybridNavigator(NavigatorAgent):
    """Two-stage Navigator: graph algorithm proposes, LLM selects."""

    def __init__(
        self,
        llm: LLMBackend | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__("HybridNavigator", config)
        self.llm = llm
        self.max_targets = self.config.get("max_targets", 3)
        self.mastery_threshold = self.config.get("mastery_threshold", 0.5)
        self.use_llm = self.config.get("use_llm", True) and llm is not None
        self.stagnation_patience = self.config.get("stagnation_patience", 2)
        # Track how many consecutive iterations each skill has been targeted without progress
        self._target_history: dict[str, int] = {}

    def plan(self, evokg: EvoKG, iteration: int) -> EvolutionPlan:
        # Stage 1: Graph algorithm
        candidates = self._graph_stage(evokg)

        if not candidates:
            logger.warning("No candidates in learnable frontier — returning empty plan")
            return EvolutionPlan(iteration=iteration, rationale="No learnable skills available")

        # Stage 2: LLM refinement (optional)
        if self.use_llm and self.llm is not None:
            try:
                plan = self._llm_stage(evokg, candidates, iteration)
                if plan.targets:
                    return plan
                logger.warning("LLM stage returned empty targets, falling back to graph stage")
            except Exception as e:
                logger.warning(f"LLM stage failed ({e}), falling back to graph stage")

        # Fallback: use graph algorithm ranking directly
        targets = [
            EvolutionTarget(
                skill_id=c["node_id"],
                priority=c["priority"],
                reason=f"Graph-ranked: mastery={c['mastery']:.2f}, priority={c['priority']:.2f}",
                exploration_brief=f"Practice skill: {c.get('name', c['node_id'])}. "
                                  f"Current mastery: {c['mastery']:.0%}.",
                # v12.2: include the multi-hop prerequisite chain
                skill_chain=evokg.get_skill_chain(c["node_id"]),
            )
            for c in candidates[: self.max_targets]
        ]

        plan = EvolutionPlan(
            iteration=iteration,
            targets=targets,
            rationale=f"Graph-only: selected {len(targets)} from {len(candidates)} frontier candidates",
        )
        self._update_stagnation_tracking(plan, evokg)
        return plan

    def _graph_stage(self, evokg: EvoKG) -> list[dict[str, Any]]:
        """Stage 1: Compute learnable frontier and rank by priority.

        Improvements over naive frontier:
        - Penalizes skills that have been targeted repeatedly without progress (stagnation)
        - Includes partially-mastered skills for reinforcement (prevent forgetting)
        - Prioritizes skills with successful experience patterns (we know how to improve)
        """
        frontier = evokg.get_learnable_frontier(mastery_threshold=self.mastery_threshold)

        for node in frontier:
            mastery = node.get("mastery", 0.0)
            skill_id = node["node_id"]

            # Base priority: how much improvement is needed
            need = 1.0 - mastery

            # Bonus for skills that unlock many downstream skills
            downstream_count = len(evokg.get_edges_from(skill_id))
            unlock_bonus = min(downstream_count * 0.1, 0.3)

            # Bonus for skills with SUCCESS experience (we know what works)
            experiences = evokg.get_experiences_for_skill(skill_id)
            success_exp = [e for e in experiences if e.get("outcome") == "success"]
            success_bonus = min(len(success_exp) * 0.15, 0.3)

            # Penalty for stagnation: targeted N times without mastery improvement
            stagnation_count = self._target_history.get(skill_id, 0)
            stagnation_penalty = min(stagnation_count * 0.2, 0.6)

            # Bonus for skills with SOME progress (0 < mastery < threshold)
            progress_bonus = 0.2 if 0.1 < mastery < self.mastery_threshold else 0.0

            node["priority"] = max(0.05, need + unlock_bonus + success_bonus + progress_bonus - stagnation_penalty)

        # Also include partially-mastered skills that need reinforcement
        # (skills near threshold that might decay without practice)
        reinforcement_candidates = []
        for node_id, data in evokg.graph.nodes(data=True):
            if data.get("node_type") != "capability":
                continue
            mastery = data.get("mastery", 0.0)
            # Skills between 0.3 and 0.6 — near threshold, need reinforcement
            if 0.3 <= mastery <= 0.6:
                nd = dict(data)
                nd["node_id"] = node_id
                nd["priority"] = 0.5  # moderate priority for reinforcement
                nd["_reinforcement"] = True
                reinforcement_candidates.append(nd)

        all_candidates = frontier + reinforcement_candidates
        all_candidates.sort(key=lambda x: x["priority"], reverse=True)
        return all_candidates

    def _update_stagnation_tracking(self, plan: EvolutionPlan, evokg: EvoKG) -> None:
        """Track which skills are being targeted repeatedly without progress."""
        for target in plan.targets:
            mastery = evokg.get_mastery(target.skill_id) if evokg.has_node(target.skill_id) else 0
            if mastery < 0.1:  # no meaningful progress
                self._target_history[target.skill_id] = self._target_history.get(target.skill_id, 0) + 1
            else:
                self._target_history[target.skill_id] = 0  # reset if progress made

    def _llm_stage(
        self, evokg: EvoKG, candidates: list[dict[str, Any]], iteration: int,
    ) -> EvolutionPlan:
        """Stage 2: LLM refines candidate selection and generates exploration briefs."""
        # Build context for LLM
        candidate_info = []
        for c in candidates[: self.max_targets * 2]:  # give LLM more candidates than needed
            prereqs = evokg.get_prerequisites(c["node_id"])
            prereq_info = [
                {"id": p, "mastery": evokg.get_mastery(p)}
                for p in prereqs
            ]
            experiences = evokg.get_experiences_for_skill(c["node_id"])
            exp_summaries = [
                {"pattern": e.get("action_pattern", ""), "outcome": e.get("outcome", "")}
                for e in experiences[:3]
            ]

            candidate_info.append({
                "skill_id": c["node_id"],
                "name": c.get("name", c["node_id"]),
                "mastery": c.get("mastery", 0.0),
                "graph_priority": c["priority"],
                "description": c.get("description", ""),
                "prerequisites": prereq_info,
                "past_experiences": exp_summaries,
            })

        prompt = f"""Iteration {iteration}. Select up to {self.max_targets} skills to focus on.

## Candidates from learnable frontier:
{json.dumps(candidate_info, indent=2)}

## Current EvoKG stats:
{json.dumps(evokg.stats(), indent=2)}

Select the best skills to focus on and provide exploration guidance for each."""

        result = self.llm.generate_json(prompt, NAVIGATOR_SYSTEM_PROMPT, temperature=0.3)

        targets = []
        for t in result.get("targets", []):
            sid = t["skill_id"]
            targets.append(EvolutionTarget(
                skill_id=sid,
                priority=float(t.get("priority", 0.5)),
                reason=t.get("reason", "LLM selected"),
                exploration_brief=t.get("exploration_brief", ""),
                # v12.2: include the multi-hop prerequisite chain
                skill_chain=evokg.get_skill_chain(sid),
            ))

        return EvolutionPlan(
            iteration=iteration,
            targets=targets,
            rationale=result.get("rationale", "LLM-guided selection"),
        )


class GraphOnlyNavigator(NavigatorAgent):
    """Navigator using only graph algorithm — no LLM. For ablation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("GraphOnlyNavigator", config)
        self.max_targets = self.config.get("max_targets", 3)
        self.mastery_threshold = self.config.get("mastery_threshold", 0.5)

    def plan(self, evokg: EvoKG, iteration: int) -> EvolutionPlan:
        frontier = evokg.get_learnable_frontier(mastery_threshold=self.mastery_threshold)
        frontier.sort(key=lambda x: x.get("mastery", 0.0))

        targets = [
            EvolutionTarget(
                skill_id=n["node_id"],
                priority=1.0 - n.get("mastery", 0.0),
                reason=f"Lowest mastery in frontier: {n.get('mastery', 0.0):.2f}",
                exploration_brief=f"Practice: {n.get('name', n['node_id'])}",
            )
            for n in frontier[: self.max_targets]
        ]

        return EvolutionPlan(
            iteration=iteration,
            targets=targets,
            rationale=f"Graph-only: {len(targets)} targets from frontier",
        )


class RandomNavigator(NavigatorAgent):
    """Random skill selection — baseline for ablation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("RandomNavigator", config)
        self.max_targets = self.config.get("max_targets", 3)

    def plan(self, evokg: EvoKG, iteration: int) -> EvolutionPlan:
        import random
        all_skills = evokg.get_nodes_by_type(NodeType.CAPABILITY)
        unmastered = [s for s in all_skills if s.get("mastery", 0.0) < 0.5]

        if not unmastered:
            return EvolutionPlan(iteration=iteration, rationale="All skills mastered")

        selected = random.sample(unmastered, min(self.max_targets, len(unmastered)))
        targets = [
            EvolutionTarget(
                skill_id=s["node_id"],
                priority=0.5,
                reason="Random selection",
                exploration_brief=f"Practice: {s.get('name', s['node_id'])}",
            )
            for s in selected
        ]

        return EvolutionPlan(
            iteration=iteration,
            targets=targets,
            rationale=f"Random: {len(targets)} targets selected randomly",
        )
