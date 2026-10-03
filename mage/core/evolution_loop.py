"""The MAGE evolution loop: PLAN → EXPLORE → LEARN → EVALUATE → UPDATE."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mage.agents.base import (
    CriticAgent,
    CuratorAgent,
    ExplorerAgent,
    LearnerAgent,
    NavigatorAgent,
)
from mage.core.evokg import EvoKG

logger = logging.getLogger(__name__)


class EvolutionLoop:
    """Orchestrates the MAGE evolution process.

    Each iteration:
      1. PLAN     — Navigator queries EvoKG → evolution plan
      2. EXPLORE  — Explorer collects trajectories guided by plan
      3. LEARN    — Learner trains on collected trajectories
      4. EVALUATE — Critic evaluates Learner → structured evaluation
      5. UPDATE   — Curator updates EvoKG with evaluation results
      6. SNAPSHOT — Save EvoKG and metrics
    """

    def __init__(
        self,
        evokg: EvoKG,
        navigator: NavigatorAgent,
        explorer: ExplorerAgent,
        learner: LearnerAgent,
        critic: CriticAgent,
        curator: CuratorAgent,
        environment: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.evokg = evokg
        self.navigator = navigator
        self.explorer = explorer
        self.learner = learner
        self.critic = critic
        self.curator = curator
        self.environment = environment
        self.config = config or {}

        self.output_dir = Path(self.config.get("output_dir", "experiments/default"))
        self.num_episodes = self.config.get("num_episodes", 50)
        self.history: list[dict[str, Any]] = []

        # Inject environment reference into GRPO learner (for scoring completions)
        if hasattr(self.learner, 'set_environment'):
            self.learner.set_environment(self.environment)

    def run(self, num_iterations: int = 4) -> list[dict[str, Any]]:
        """Run the full evolution loop for N iterations."""
        logger.info(f"Starting MAGE evolution loop for {num_iterations} iterations")
        logger.info(f"Initial EvoKG: {self.evokg}")

        for k in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {k}")
            logger.info(f"{'='*60}")

            metrics = self._run_iteration(k)
            self.history.append(metrics)

            logger.info(f"Iteration {k} complete: {metrics}")

        self._save_history()
        logger.info(f"Evolution complete. {num_iterations} iterations done.")
        return self.history

    def _run_iteration(self, iteration: int) -> dict[str, Any]:
        """Execute one iteration of the evolution loop."""
        self.evokg.iteration = iteration
        metrics: dict[str, Any] = {"iteration": iteration}

        # Step 1: PLAN
        logger.info(f"[Iter {iteration}] Step 1: PLAN — Navigator planning curriculum")
        plan = self.navigator.plan(self.evokg, iteration)
        metrics["plan"] = plan.to_dict()
        metrics["num_targets"] = len(plan.targets)
        logger.info(f"  Plan: {len(plan.targets)} targets — {[t.skill_id for t in plan.targets]}")

        # Load bandit state if explorer supports it
        if hasattr(self.explorer, 'use_bandit') and self.explorer.use_bandit:
            from mage.core.experience_bandit import ExperienceBandit
            bandit_path = self.output_dir / "bandit_state.json"
            bandit = ExperienceBandit.load(bandit_path)
            bandit.k = getattr(self.explorer, '_bandit', ExperienceBandit()).k
            self.explorer.set_bandit(bandit)

        # Step 2: EXPLORE
        logger.info(f"[Iter {iteration}] Step 2: EXPLORE — Collecting trajectories")
        dataset = self.explorer.explore(
            plan, self.evokg, self.environment, self.num_episodes,
        )
        metrics["num_trajectories"] = len(dataset)
        logger.info(f"  Collected {len(dataset)} trajectories")

        # Step 3: LEARN
        logger.info(f"[Iter {iteration}] Step 3: LEARN — Training Learner")
        train_metrics = self.learner.train(dataset, iteration)
        metrics["training"] = train_metrics
        logger.info(f"  Training metrics: {train_metrics}")

        # Step 4: EVALUATE
        logger.info(f"[Iter {iteration}] Step 4: EVALUATE — Critic evaluating Learner")
        evaluation = self.critic.evaluate(
            self.learner, self.evokg, self.environment, iteration,
        )
        metrics["evaluation"] = evaluation.to_dict()
        metrics["overall_success_rate"] = evaluation.overall_success_rate
        logger.info(f"  Overall success rate: {evaluation.overall_success_rate:.2%}")

        # Step 5: UPDATE
        logger.info(f"[Iter {iteration}] Step 5: UPDATE — Curator updating EvoKG")
        self.evokg = self.curator.update(self.evokg, evaluation, dataset, iteration)
        metrics["evokg_stats"] = self.evokg.stats()
        logger.info(f"  EvoKG: {self.evokg}")

        # Step 6: SNAPSHOT
        snapshot_dir = self.output_dir / f"iter_{iteration}"
        self.evokg.save_snapshot(snapshot_dir / "evokg.json")

        eval_path = snapshot_dir / "evaluation.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        # Save bandit state (cumulative, single file)
        if (hasattr(self.explorer, '_bandit')
                and self.explorer._bandit is not None):
            self.explorer._bandit.save(self.output_dir / "bandit_state.json")

        return metrics

    def _save_history(self) -> None:
        """Save full evolution history."""
        history_path = self.output_dir / "evolution_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Saved evolution history to {history_path}")
