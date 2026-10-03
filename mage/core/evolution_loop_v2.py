"""MAGE v2 Evolution Loop: DISCOVER → PLAN → EXPLORE → EVALUATE → EVOLVE.

Key difference from v1: the EvoKG skill structure itself evolves.
Uses a strong LLM (Claude) for skill discovery and failure analysis.
Uses a weak LLM (local) for execution (Explorer/Learner).
"""

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
from mage.agents.discovery import SkillDiscovery
from mage.core.evokg import EvoKG
from mage.utils.llm import LLMBackend

logger = logging.getLogger(__name__)


class EvolutionLoopV2:
    """MAGE v2: Co-evolutionary loop with dynamic skill discovery.

    Two-tier LLM architecture:
      - Strong LLM (Claude): skill discovery, failure analysis, graph evolution
      - Weak LLM (local): exploration, execution

    Loop phases:
      0. DISCOVER  — (iter 0 only) Discover initial skills from question batch
      1. PLAN      — Navigator selects skills to practice
      2. EXPLORE   — Explorer collects trajectories following skill strategies
      3. EVALUATE  — Critic evaluates + collects failure details
      4. EVOLVE    — Strong LLM analyzes failures, evolves graph structure
      5. SNAPSHOT  — Save EvoKG + metrics
    """

    def __init__(
        self,
        evokg: EvoKG | None,
        navigator: NavigatorAgent,
        explorer: ExplorerAgent,
        learner: LearnerAgent,
        critic: CriticAgent,
        curator: CuratorAgent,
        environment: Any,
        discovery: SkillDiscovery,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.evokg = evokg
        self.navigator = navigator
        self.explorer = explorer
        self.learner = learner
        self.critic = critic
        self.curator = curator
        self.environment = environment
        self.discovery = discovery
        self.config = config or {}

        self.output_dir = Path(self.config.get("output_dir", "experiments/default"))
        self.num_episodes = self.config.get("num_episodes", 50)
        # v12.0: bypass DISCOVER cache when set (--no-discover-cache)
        self.force_discover = bool(self.config.get("force_discover", False))
        self.history: list[dict[str, Any]] = []

        # Inject environment into learner if needed
        if hasattr(self.learner, 'set_environment'):
            self.learner.set_environment(self.environment)
        if hasattr(self.learner, 'set_llm'):
            # For ICL learner, share the explorer's LLM
            if hasattr(self.explorer, 'llm'):
                self.learner.set_llm(self.explorer.llm)

        # v15: Initialize the evolving search agent
        if hasattr(self.learner, 'set_search_agent'):
            from mage.tools.search_agent import SearchAgent
            sa_cfg = (self.config or {}).get("search_agent", {}) or {}
            self._search_agent = SearchAgent(
                top_k=int(sa_cfg.get("top_k", 5)),
                min_pulls_per_arm=int(sa_cfg.get("min_pulls_per_arm", 20)),
                # v19.B — opt-in: union all 3 retrieval strategies
                ensemble_all_strategies=bool(sa_cfg.get("ensemble_all_strategies", False)),
            )
            # Load saved bandit state if available
            sa_path = self.output_dir / "search_agent_bandit.json"
            self._search_agent.load(sa_path)
            self.learner.set_search_agent(self._search_agent)

        # Failure collection for evolution phase
        self._iteration_failures: dict[str, list[dict]] = {}

        # Best checkpoint — restored only on catastrophic drops
        self._best_accuracy = 0.0
        self._best_evokg_snapshot: str | None = None
        self._best_router_snapshot: str | None = None

        # Pre-EVOLVE snapshot for per-iteration delta-guard revert
        self._pre_evolve_acc: float | None = None
        self._pre_evolve_snapshot: str | None = None
        self._pre_evolve_router: str | None = None

        # v15.6 — Adaptive curriculum config. When present, the loop
        # samples per-iter eval batches as (n_new fresh + ALL accumulated
        # failures) and runs a held-out final exam after the last iter
        # instead of repeating the same fixed set every iteration.
        cur_cfg = (self.config or {}).get("curriculum", {}) or {}
        self.curriculum_enabled: bool = bool(cur_cfg)
        self.cur_n_new: int = int(cur_cfg.get("n_new", 100))
        self.cur_demote_after: int = int(cur_cfg.get("demote_after_n_fails", 5))
        self.cur_demote_every_k: int = int(cur_cfg.get("demote_revisit_every_k", 3))
        self.cur_final_exam_size: int = int(cur_cfg.get("final_exam_size", 200))
        self._curriculum_state: dict[str, Any] = {}
        # Map of instance_id → EvaluationInstance for fast revisit lookup.
        # Populated by `_setup_curriculum`.
        self._instance_index: dict[str, Any] = {}

    def run(self, num_iterations: int = 5) -> list[dict[str, Any]]:
        """Run the full v2 evolution loop.

        v15.5-resume: if ``self.output_dir`` already contains ``iter_N/evokg.json``
        snapshots, automatically continue from iter N+1. The latest snapshot
        replaces the DISCOVER step (and the search-agent bandit auto-loads
        from ``search_agent_bandit.json`` in __init__). To force a fresh
        run, point ``output_dir`` to a clean directory.
        """
        logger.info(f"Starting MAGE v2 evolution loop for {num_iterations} iterations")

        # v15.5: resume support — detect existing iter snapshots
        resume_from = self._detect_resume_iter()
        if resume_from > 0:
            iter_path = self.output_dir / f"iter_{resume_from - 1}" / "evokg.json"
            self.evokg = EvoKG.load_snapshot(str(iter_path))
            logger.info(
                f"[RESUME] Loaded EvoKG from {iter_path}; "
                f"continuing at iter {resume_from} "
                f"({self.evokg.graph.number_of_nodes()} nodes, "
                f"{self.evokg.graph.number_of_edges()} edges)"
            )
            # Also load the prior strategy router state so the learner's bandit
            # continues from where it left off.
            router_path = (
                self.output_dir / f"iter_{resume_from - 1}" / "strategy_router.json"
            )
            if router_path.exists() and hasattr(self.learner, "_router"):
                from mage.core.strategy_router import StrategyRouter
                self.learner._router = StrategyRouter.load(router_path)
                logger.info(f"[RESUME] Loaded strategy router from {router_path}")
            # Load the prior history file so future iterations append to it.
            self._load_existing_history()
        else:
            # Phase 0: DISCOVER (iter 0 only, if no EvoKG provided)
            if self.evokg is None or self.evokg.graph.number_of_nodes() == 0:
                self.evokg = self._discover_skills()

        logger.info(f"Initial EvoKG: {self.evokg}")

        # Load any previously saved dynamic tools
        self._load_dynamic_tools()

        # v15.6 — set up curriculum pools (no-op if curriculum disabled).
        # Resume-safe: loads from curriculum_state.json if present.
        self._setup_curriculum()

        for k in range(resume_from, resume_from + num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {k}")
            logger.info(f"{'='*60}")

            metrics = self._run_iteration(k)
            self.history.append(metrics)
            logger.info(f"Iteration {k} complete")

        self._save_history()

        # v15.6 — held-out generalisation eval, after the last training iter
        if self.curriculum_enabled:
            try:
                self._run_final_exam()
            except Exception as e:
                logger.warning(f"Final exam failed: {e}")

        logger.info(
            f"Evolution complete. {num_iterations} iterations done "
            f"(iters {resume_from} .. {resume_from + num_iterations - 1})."
        )
        return self.history

    def _detect_resume_iter(self) -> int:
        """Return the next iteration index to run (0 if no prior state)."""
        if not self.output_dir.exists():
            return 0
        max_iter = -1
        for p in self.output_dir.glob("iter_*"):
            if not (p / "evokg.json").exists():
                continue
            try:
                idx = int(p.name.split("_", 1)[1])
                max_iter = max(max_iter, idx)
            except (IndexError, ValueError):
                continue
        return max_iter + 1

    def _load_existing_history(self) -> None:
        """If we resumed, also load the prior evolution_history.json so
        subsequent iters APPEND rather than overwrite."""
        path = self.output_dir / "evolution_history.json"
        if not path.exists():
            return
        try:
            import json as _json
            with open(path) as f:
                self.history = _json.load(f)
            logger.info(
                f"[RESUME] Loaded {len(self.history)} prior history entries from {path}"
            )
        except Exception as e:
            logger.warning(f"[RESUME] Failed to load prior history: {e}")
            self.history = []

    def _discover_skills(self) -> EvoKG:
        """Phase 0: Discover skills from the environment's questions.

        v12.0: Uses a per-experiment disk cache so the same Claude-discovered
        skills are reused across runs of the same config. This makes results
        reproducible. Set `force_discover=True` (or pass `--no-discover-cache`
        to the CLI) to bypass the cache.
        """
        logger.info("[DISCOVER] Analyzing questions to discover skills...")

        # Get sample questions from environment
        instances = self.environment.get_evaluation_instances(
            skill_id=None, num_instances=50,
        )

        questions = []
        for inst in instances:
            questions.append({
                "question": inst.instruction,
                "answer": inst.expected_answer or "",
                "task_type": inst.metadata.get("task_type", "unknown"),
            })

        # Get available tools
        from mage.tools.st_tools import TOOL_REGISTRY
        available_tools = list(TOOL_REGISTRY.keys())

        # Per-experiment cache path
        discovery_dir = self.output_dir / "discovery"
        cache_path = discovery_dir / "discovery_cache.json"
        force = bool(getattr(self, "force_discover", False))

        # Use strong LLM to discover skills (with cache)
        evokg = self.discovery.build_initial_evokg(
            questions=questions,
            domain=self.environment.domain,
            available_tools=available_tools,
            cache_path=cache_path,
            force_discover=force,
        )

        logger.info(f"[DISCOVER] Discovered skills: {evokg}")

        # Save discovery result snapshot (separate from the cache)
        evokg.save_snapshot(discovery_dir / "initial_evokg.json")

        return evokg

    # ------------------------------------------------------------------
    # v15.6 — Adaptive curriculum
    # ------------------------------------------------------------------

    def _setup_curriculum(self) -> None:
        """Reserve a held-out final-exam pool, then build the per-iter
        sampling pools.

        Called once before the iteration loop begins. Pulls a large slab
        of instances from the env, carves off ``cur_final_exam_size`` for
        the held-out pool, and initialises the unseen / failed / solved
        sets used by ``_select_iter_batch``.

        Resume support: if ``curriculum_state.json`` already exists in
        the output dir, load it instead of re-initialising. This keeps
        a continuation run on the same train/test split as its origin.
        """
        if not self.curriculum_enabled:
            return

        state_path = self.output_dir / "curriculum_state.json"
        if state_path.exists():
            self._load_curriculum_state(state_path)
            return

        # Pull as many instances as possible (cap at 10k for memory)
        pool = self.environment.get_evaluation_instances(
            skill_id=None, num_instances=10_000,
        )
        if not pool:
            logger.warning(
                "Curriculum: env returned 0 instances; disabling curriculum."
            )
            self.curriculum_enabled = False
            return

        # Build instance_id → instance index for revisit lookup.
        self._instance_index = {
            getattr(inst, "instance_id", f"_idx_{i}"): inst
            for i, inst in enumerate(pool)
        }

        # Final exam: take the LAST `final_exam_size` from the shuffled pool
        # so it is disjoint from anything sampled for training.
        if self.cur_final_exam_size > 0 and len(pool) > self.cur_final_exam_size:
            curriculum = pool[: -self.cur_final_exam_size]
            final_exam_ids = [
                getattr(p, "instance_id", "") for p in pool[-self.cur_final_exam_size :]
            ]
        else:
            curriculum = pool
            final_exam_ids = []

        curriculum_ids = [getattr(p, "instance_id", "") for p in curriculum]
        self._curriculum_state = {
            "unseen_ids": curriculum_ids,
            "final_exam_ids": final_exam_ids,
            "failed_pool": {},   # qid → {n_consec_fails, last_iter, demoted}
            "solved_pool": [],   # list of qids ever solved
            "iter_metrics": [],  # per-iter [{iteration, new_acc, revisit_acc, recovered, ...}]
        }
        self._save_curriculum_state()
        logger.info(
            f"  Curriculum: {len(curriculum_ids)} train + "
            f"{len(final_exam_ids)} final-exam instances"
        )

    def _select_iter_batch(self, iteration: int) -> tuple[list[Any], set[str], set[str]]:
        """Build this iteration's evaluation batch.

        Returns ``(instances, fresh_ids, revisit_ids)``.

        - ``fresh_ids``: ``cur_n_new`` ids popped from the unseen pool.
        - ``revisit_ids``: every active failure (``demoted=False``) plus
          demoted failures whose turn it is on the every-Kth schedule.

        Each instance is tagged with ``metadata["_curriculum_tag"]`` so
        the critic can record whether each per-question result was a
        fresh probe or a revisit.
        """
        state = self._curriculum_state
        # 1. Fresh sample
        n_take = min(self.cur_n_new, len(state["unseen_ids"]))
        fresh_ids = list(state["unseen_ids"][:n_take])
        state["unseen_ids"] = state["unseen_ids"][n_take:]

        # 2. Revisit ALL accumulated failures (with demote schedule)
        revisit_ids: list[str] = []
        for qid, info in state["failed_pool"].items():
            if info.get("demoted"):
                # Demoted Q: revisit only every K iters
                if iteration % max(self.cur_demote_every_k, 1) != 0:
                    continue
            revisit_ids.append(qid)

        # 3. Build instance list with tags
        batch: list[Any] = []
        for qid in fresh_ids:
            inst = self._instance_index.get(qid)
            if inst is None:
                continue
            inst.metadata["_curriculum_tag"] = "fresh"
            batch.append(inst)
        for qid in revisit_ids:
            inst = self._instance_index.get(qid)
            if inst is None:
                continue
            inst.metadata["_curriculum_tag"] = "revisit"
            batch.append(inst)

        logger.info(
            f"  Curriculum batch [iter {iteration}]: "
            f"{len(fresh_ids)} fresh + {len(revisit_ids)} revisit "
            f"(failed_pool={len(state['failed_pool'])}, "
            f"solved={len(state['solved_pool'])}, "
            f"unseen={len(state['unseen_ids'])})"
        )
        return batch, set(fresh_ids), set(revisit_ids)

    def _update_curriculum_state(
        self,
        per_q_results: list[dict],
        iteration: int,
        fresh_ids: set[str],
        revisit_ids: set[str],
    ) -> dict[str, Any]:
        """Move questions between unseen / failed / solved pools.

        Demotion rule: a question that is wrong ``cur_demote_after``
        consecutive times is marked ``demoted=True``; future iters
        revisit it only every ``cur_demote_every_k`` iters.

        Returns a small dict of curriculum-specific metrics.
        """
        state = self._curriculum_state
        failed = state["failed_pool"]
        solved_set = set(state["solved_pool"])

        recovered = 0      # failed -> solved this iter
        new_failed = 0     # fresh wrong this iter
        still_failed = 0   # revisited and still wrong
        new_solved = 0     # fresh correct this iter

        n_fresh_correct = 0
        n_revisit_correct = 0

        for rec in per_q_results:
            qid = rec.get("instance_id", "")
            if not qid:
                continue
            tag = rec.get("curriculum_tag", "")
            correct = bool(rec.get("correct"))

            if tag == "fresh":
                if correct:
                    n_fresh_correct += 1
                    solved_set.add(qid)
                    new_solved += 1
                else:
                    new_failed += 1
                    failed[qid] = {
                        "n_consec_fails": 1,
                        "last_iter": iteration,
                        "demoted": False,
                    }
            elif tag == "revisit":
                if correct:
                    n_revisit_correct += 1
                    if qid in failed:
                        del failed[qid]
                        recovered += 1
                    solved_set.add(qid)
                else:
                    still_failed += 1
                    if qid in failed:
                        failed[qid]["n_consec_fails"] += 1
                        failed[qid]["last_iter"] = iteration
                        if failed[qid]["n_consec_fails"] >= self.cur_demote_after:
                            failed[qid]["demoted"] = True

        state["solved_pool"] = sorted(solved_set)
        n_fresh = sum(1 for r in per_q_results if r.get("curriculum_tag") == "fresh")
        n_revisit = sum(1 for r in per_q_results if r.get("curriculum_tag") == "revisit")

        metrics = {
            "iteration": iteration,
            "n_fresh": n_fresh,
            "n_revisit": n_revisit,
            "new_acc": n_fresh_correct / max(n_fresh, 1),
            "revisit_acc": n_revisit_correct / max(n_revisit, 1) if n_revisit else None,
            "recovered_count": recovered,
            "new_failed": new_failed,
            "still_failed": still_failed,
            "new_solved": new_solved,
            "failed_pool_size": len(failed),
            "solved_pool_size": len(solved_set),
            "unseen_remaining": len(state["unseen_ids"]),
            "demoted_count": sum(1 for v in failed.values() if v.get("demoted")),
        }
        state["iter_metrics"].append(metrics)
        self._save_curriculum_state()

        logger.info(
            f"  Curriculum metrics: new_acc={metrics['new_acc']:.1%} "
            f"({n_fresh_correct}/{n_fresh}), "
            f"revisit_acc="
            + (f"{metrics['revisit_acc']:.1%} ({n_revisit_correct}/{n_revisit})"
               if n_revisit else "n/a")
            + f", RECOVERED={recovered}, failed={metrics['failed_pool_size']}, "
            f"solved={metrics['solved_pool_size']}, "
            f"unseen={metrics['unseen_remaining']}, "
            f"demoted={metrics['demoted_count']}"
        )
        return metrics

    def _save_curriculum_state(self) -> None:
        if not self.curriculum_enabled:
            return
        path = self.output_dir / "curriculum_state.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._curriculum_state, f, indent=2)
        except Exception as e:
            logger.warning(f"  Failed to save curriculum_state: {e}")

    def _load_curriculum_state(self, path: Path) -> None:
        try:
            with open(path) as f:
                self._curriculum_state = json.load(f)
            logger.info(
                f"  [RESUME] Loaded curriculum state: "
                f"{len(self._curriculum_state.get('unseen_ids', []))} unseen, "
                f"{len(self._curriculum_state.get('failed_pool', {}))} failed, "
                f"{len(self._curriculum_state.get('solved_pool', []))} solved, "
                f"{len(self._curriculum_state.get('final_exam_ids', []))} final-exam"
            )
            # Rebuild the instance index from the env so revisit lookup works
            pool = self.environment.get_evaluation_instances(
                skill_id=None, num_instances=10_000,
            )
            self._instance_index = {
                getattr(inst, "instance_id", f"_idx_{i}"): inst
                for i, inst in enumerate(pool)
            }
        except Exception as e:
            logger.warning(f"  Failed to load curriculum_state: {e}; starting fresh")
            self._curriculum_state = {}

    def _run_final_exam(self) -> dict[str, Any]:
        """Post-loop held-out evaluation. No EVOLVE, no memory writes."""
        if not self.curriculum_enabled:
            return {}
        state = self._curriculum_state
        if not state.get("final_exam_ids"):
            logger.info("  Final exam: no held-out instances reserved; skipping.")
            return {}

        instances = [
            self._instance_index[qid]
            for qid in state["final_exam_ids"]
            if qid in self._instance_index
        ]
        # Strip any stale curriculum tag to avoid contaminating the metric.
        for inst in instances:
            inst.metadata.pop("_curriculum_tag", None)

        logger.info(
            f"\n{'='*60}\nFINAL EXAM ({len(instances)} held-out questions)\n{'='*60}"
        )
        eval_obj = self.critic.evaluate_instances(
            self.learner, self.evokg, self.environment, instances, iteration=-1,
        )
        per_task = {}
        for sk in eval_obj.evaluations:
            per_task[sk.skill_id] = {
                "accuracy": sk.success_rate,
                "attempts": sk.attempts,
                "success": int(sk.success_rate * sk.attempts),
            }
        summary = {
            "n_questions": len(instances),
            "overall_accuracy": eval_obj.overall_success_rate,
            "per_task_accuracy": per_task,
            "per_question_results": eval_obj.per_question_results,
        }
        out_path = self.output_dir / "final_exam_results.json"
        out_path.write_text(json.dumps(summary, indent=2))
        logger.info(
            f"FINAL EXAM accuracy: "
            f"{eval_obj.overall_success_rate*100:.2f}% "
            f"({int(eval_obj.overall_success_rate*len(instances))}/{len(instances)})"
        )
        for tn, tv in per_task.items():
            logger.info(
                f"  {tn:30s}  {tv['success']:3d}/{tv['attempts']:3d}  "
                f"{tv['accuracy']*100:5.1f}%"
            )
        return summary

    def _run_iteration(self, iteration: int) -> dict[str, Any]:
        """Execute one iteration of the v2 evolution loop."""
        self.evokg.iteration = iteration
        metrics: dict[str, Any] = {"iteration": iteration}
        self._iteration_failures = {}

        # Load dynamic tools created in previous iterations
        self._load_dynamic_tools()

        # Load router state from previous iteration (preserve bandit memory)
        if iteration > 0 and hasattr(self.learner, 'load'):
            prev_path = self.output_dir / f"iter_{iteration - 1}"
            if (prev_path / "strategy_router.json").exists():
                self.learner.load(prev_path)
                logger.info(f"  Loaded router state from iter {iteration - 1}")

        # Inject updated EvoKG into routed learner
        if hasattr(self.learner, 'set_evokg'):
            self.learner.set_evokg(self.evokg)

        # Step 1: PLAN
        logger.info(f"[Iter {iteration}] Step 1: PLAN")
        plan = self.navigator.plan(self.evokg, iteration)
        metrics["plan"] = plan.to_dict()
        metrics["num_targets"] = len(plan.targets)
        logger.info(f"  Plan: {len(plan.targets)} targets — {[t.skill_id for t in plan.targets]}")

        # Load bandit if available
        if hasattr(self.explorer, 'use_bandit') and self.explorer.use_bandit:
            from mage.core.experience_bandit import ExperienceBandit
            bandit_path = self.output_dir / "bandit_state.json"
            bandit = ExperienceBandit.load(bandit_path)
            self.explorer.set_bandit(bandit)

        # Step 2: EXPLORE
        logger.info(f"[Iter {iteration}] Step 2: EXPLORE")

        # Inject skill strategies into exploration
        self._prepare_exploration(plan)

        dataset = self.explorer.explore(
            plan, self.evokg, self.environment, self.num_episodes,
        )
        metrics["num_trajectories"] = len(dataset)
        logger.info(f"  Collected {len(dataset)} trajectories")

        # Step 3: LEARN
        logger.info(f"[Iter {iteration}] Step 3: LEARN")
        train_metrics = self.learner.train(dataset, iteration)
        metrics["training"] = train_metrics
        logger.info(f"  Training: {train_metrics.get('status')}")

        # Step 4: EVALUATE (with failure collection)
        logger.info(f"[Iter {iteration}] Step 4: EVALUATE")
        # v15.6 — curriculum mode: build per-iter batch (n_new fresh + ALL
        # accumulated failures w/ demote schedule), evaluate it, then
        # update pools. Falls back to legacy fixed-set eval if curriculum
        # is disabled or returned no batch.
        cur_iter_metrics: dict[str, Any] = {}
        if self.curriculum_enabled and self._curriculum_state:
            batch, fresh_ids, revisit_ids = self._select_iter_batch(iteration)
            if batch:
                evaluation = self.critic.evaluate_instances(
                    self.learner, self.evokg, self.environment, batch, iteration,
                )
                cur_iter_metrics = self._update_curriculum_state(
                    evaluation.per_question_results, iteration, fresh_ids, revisit_ids,
                )
            else:
                evaluation = self.critic.evaluate(
                    self.learner, self.evokg, self.environment, iteration,
                )
        else:
            evaluation = self.critic.evaluate(
                self.learner, self.evokg, self.environment, iteration,
            )

        metrics["evaluation"] = evaluation.to_dict()
        metrics["overall_success_rate"] = evaluation.overall_success_rate
        if cur_iter_metrics:
            metrics["curriculum"] = cur_iter_metrics
        logger.info(f"  Overall success rate: {evaluation.overall_success_rate:.2%}")

        # ADDITIVE-WITH-DELTA-GUARD POLICY (v9.2):
        #
        # The KG is mostly additive — every principle, tool and split is
        # permanent in the node store. BUT we now compare the current
        # accuracy to the state BEFORE the previous iteration's EVOLVE, and
        # if EVOLVE caused a meaningful drop, we revert *only that iter's*
        # additions to the working strategy/principle slots. The node
        # history is preserved (we can inspect what was tried), but the
        # active configuration falls back to the pre-EVOLVE state.
        #
        # In addition, if the global accuracy ever falls more than 5% below
        # the best ever recorded, we restore the best snapshot wholesale —
        # this is the safety net for cumulative bad additions.
        current_accuracy = evaluation.overall_success_rate

        # Step 4a: per-iteration delta check vs pre-EVOLVE of LAST iter
        if hasattr(self, "_pre_evolve_acc") and self._pre_evolve_acc is not None:
            delta = current_accuracy - self._pre_evolve_acc
            if delta <= -0.03:
                logger.warning(
                    f"  Iter {iteration-1}'s EVOLVE caused {delta:.1%} drop "
                    f"({self._pre_evolve_acc:.2%} → {current_accuracy:.2%}) "
                    f"— reverting last EVOLVE's working slots"
                )
                if self._pre_evolve_snapshot is not None:
                    snap_nodes = json.loads(self._pre_evolve_snapshot)
                    # Only restore the principles/strategy/prompt_template
                    # for skills that existed BEFORE the bad iter. New nodes
                    # from that iter remain in the graph (history preserved)
                    # but their active influence is muted.
                    for nid, attrs in snap_nodes.items():
                        if nid in self.evokg.graph:
                            for k in ("principles", "strategy", "prompt_template"):
                                if k in attrs:
                                    self.evokg.graph.nodes[nid][k] = attrs[k]
                    if self._pre_evolve_router and hasattr(self.learner, '_router'):
                        self.learner._router._arms = json.loads(self._pre_evolve_router)
                    # Re-evaluate after the revert so subsequent EVOLVE acts
                    # on the actual restored accuracy, not the dropped one.
                    logger.info(f"  Re-evaluating after revert...")
                    evaluation = self.critic.evaluate(
                        self.learner, self.evokg, self.environment, iteration,
                    )
                    current_accuracy = evaluation.overall_success_rate
                    metrics["evaluation"] = evaluation.to_dict()
                    metrics["overall_success_rate"] = current_accuracy
                    logger.info(f"  After revert: {current_accuracy:.2%}")

        # Step 4b: catastrophic safety net — restore best snapshot if we're
        # 5% below best ever. This is the floor on regression.
        CATASTROPHIC_DROP = 0.05
        if (self._best_evokg_snapshot is not None
                and current_accuracy < self._best_accuracy - CATASTROPHIC_DROP):
            logger.warning(
                f"  CATASTROPHIC: {current_accuracy:.2%} is >{CATASTROPHIC_DROP:.0%} "
                f"below best {self._best_accuracy:.2%} — restoring best snapshot"
            )
            best_nodes = json.loads(self._best_evokg_snapshot)
            for nid, attrs in best_nodes.items():
                if nid in self.evokg.graph:
                    for k in ("principles", "strategy", "prompt_template"):
                        if k in attrs:
                            self.evokg.graph.nodes[nid][k] = attrs[k]
            if self._best_router_snapshot and hasattr(self.learner, '_router'):
                self.learner._router._arms = json.loads(self._best_router_snapshot)
            current_accuracy = self._best_accuracy

        if current_accuracy >= self._best_accuracy:
            logger.info(f"  New best accuracy: {current_accuracy:.2%}")
        else:
            gap = self._best_accuracy - current_accuracy
            logger.info(
                f"  Within tolerance ({current_accuracy:.2%}, "
                f"best {self._best_accuracy:.2%}, gap {gap:.2%}) "
                f"— EvoKG continues to grow"
            )

        # Save pre-EVOLVE snapshot so next iter can detect bad EVOLVE outcomes
        self._pre_evolve_acc = current_accuracy
        self._pre_evolve_snapshot = json.dumps({
            nid: dict(d) for nid, d in self.evokg.graph.nodes(data=True)
        })
        if hasattr(self.learner, '_router'):
            self._pre_evolve_router = json.dumps(self.learner._router._arms)

        self._collect_failures(evaluation, iteration)

        # v15: Search agent RL feedback + success-pattern storage
        self._update_search_agent(evaluation, iteration)

        # Step 5: UPDATE — curator updates mastery + experiences
        logger.info(f"[Iter {iteration}] Step 5: UPDATE")
        self.evokg = self.curator.update(self.evokg, evaluation, dataset, iteration)

        # Step 6: EVOLVE — guidance agent grows the KG
        logger.info(f"[Iter {iteration}] Step 6: EVOLVE")
        evolution_changes = self._evolve_graph(iteration)
        metrics["evolution_changes"] = evolution_changes

        # Update best-ever snapshot
        if current_accuracy >= self._best_accuracy:
            self._best_accuracy = current_accuracy
            self._best_evokg_snapshot = json.dumps({
                nid: dict(d) for nid, d in self.evokg.graph.nodes(data=True)
            })
            if hasattr(self.learner, '_router'):
                self._best_router_snapshot = json.dumps(self.learner._router._arms)
            logger.info(f"  Checkpoint saved (post-EVOLVE state)")

        metrics["evokg_stats"] = self.evokg.stats()
        logger.info(f"  EvoKG: {self.evokg}")

        # Step 7: SNAPSHOT
        snapshot_dir = self.output_dir / f"iter_{iteration}"
        self.evokg.save_snapshot(snapshot_dir / "evokg.json")

        eval_path = snapshot_dir / "evaluation.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        # Save bandit
        if (hasattr(self.explorer, '_bandit')
                and self.explorer._bandit is not None):
            self.explorer._bandit.save(self.output_dir / "bandit_state.json")

        return metrics

    def _prepare_exploration(self, plan: Any) -> None:
        """Inject skill strategies into exploration briefs.

        For each target skill, replace the generic exploration brief
        with the skill's prompt_template from the EvoKG.
        """
        for target in plan.targets:
            skill_data = self.evokg.get_node(target.skill_id)
            if skill_data and skill_data.get("prompt_template"):
                target.exploration_brief = skill_data["prompt_template"]

            # Also inject tool info into the brief
            if skill_data and skill_data.get("strategy", {}).get("type") == "tool":
                tool_name = skill_data["strategy"].get("tool", "")
                param_info = skill_data["strategy"].get("param_extraction", "")
                target.exploration_brief = (
                    f"{target.exploration_brief}\n\n"
                    f"IMPORTANT: Use the {tool_name} tool for computation. "
                    f"{param_info}"
                )

    def _update_search_agent(self, evaluation: Any, iteration: int) -> None:
        """v15: Update search agent bandit + store success patterns.

        v15.1 — Harvest success memories from the per-question results of
        the actual evaluation (200-question dev set). Previously we re-ran
        10 random questions and got <5 success memories across 20 iters;
        now every correct answer in the eval becomes a success memory
        (~80-100 per iter, dedup'd by question).

        v15.2 — Credit the bandit arm that was actually used on each
        question (not a hardcoded "hybrid"). The strategy string is
        captured in ``per_question_results[i]["search_strategy"]``.

        v15.3 — When a question has no strategy tag (early iters, or search
        was disabled), we fall back to per-task aggregate updates as before.
        """
        if not hasattr(self, '_search_agent') or self._search_agent is None:
            return

        per_q: list[dict] = list(
            getattr(evaluation, "per_question_results", []) or []
        )

        # ---- Bandit updates ----------------------------------------------
        # Prefer per-question updates (v15.2 credit assignment). For each
        # question where the search agent recorded which arm it picked,
        # update that arm with the binary correct/wrong reward. Questions
        # without a strategy tag (empty string) fall back to the old
        # per-task-aggregate update so legacy runs still get signal.
        strategy_updates = 0
        fallback_by_task: dict[str, list[float]] = {}
        for rec in per_q:
            task_type = rec.get("task_type", "") or ""
            reward = 1.0 if rec.get("correct") else 0.0
            strat = rec.get("search_strategy", "") or ""
            if strat:
                self._search_agent.update(
                    task_type=task_type, strategy=strat, reward=reward,
                )
                strategy_updates += 1
            else:
                fallback_by_task.setdefault(task_type, []).append(reward)

        # Fallback for questions without strategy tag: aggregate per task
        # and update the default "hybrid" arm (mirrors old behaviour).
        for task_type, rewards in fallback_by_task.items():
            if not rewards:
                continue
            avg = sum(rewards) / len(rewards)
            self._search_agent.update(
                task_type=task_type, strategy="hybrid", reward=avg,
            )

        # If per_q was empty (e.g. sequential env), keep legacy path
        if not per_q:
            for skill_eval in evaluation.evaluations:
                self._search_agent.update(
                    task_type=skill_eval.skill_id,
                    strategy="hybrid",
                    reward=skill_eval.success_rate,
                )

        # ---- Success-memory harvest --------------------------------------
        # v15.1: store every correct answer from the real eval. The EvoKG
        # dedupes by question_key so repeat-iteration successes are free.
        # v15.5: also pass the learner's CoT trace (raw_response) and any
        # chain decomposition so future similar questions see a proven
        # worked example, not just (Q, A).
        n_success_stored = 0
        n_success_candidates = 0
        n_with_reasoning = 0
        n_with_decomp = 0
        for rec in per_q:
            if not rec.get("correct"):
                continue
            n_success_candidates += 1
            reasoning_trace = rec.get("raw_response", "") or ""
            decomposition = rec.get("decomposition", []) or []
            if reasoning_trace:
                n_with_reasoning += 1
            if decomposition:
                n_with_decomp += 1
            added = self.evokg.add_success_memory(
                question=rec.get("question", ""),
                correct_answer=rec.get("predicted", ""),
                retrieved_context=(rec.get("context") or "")[:500],
                task_type=rec.get("task_type", ""),
                search_strategy=rec.get("search_strategy", ""),
                iteration=iteration,
                reasoning_trace=reasoning_trace,
                decomposition=decomposition,
            )
            if added:
                n_success_stored += 1

        logger.info(
            f"  Search agent: bandit updates={strategy_updates} "
            f"(per-strategy), fallback_tasks={len(fallback_by_task)}; "
            f"success memories stored={n_success_stored}/{n_success_candidates} "
            f"(w/ reasoning={n_with_reasoning}, w/ decomposition={n_with_decomp}; "
            f"rest were dedup-blocked)"
        )

        # Save search agent bandit state
        sa_path = self.output_dir / "search_agent_bandit.json"
        self._search_agent.save(sa_path)

    def _collect_failures(self, evaluation: Any, iteration: int) -> None:
        """Collect detailed failure information for the EVOLVE phase.

        Actually runs the Learner on sample questions to capture predictions,
        so Claude can see what went wrong.

        v15.5 — bug fix: previously this only collected failures for skills
        with success_rate < 0.8, which meant high-accuracy benchmarks like
        HotpotQA (87-97% per task) NEVER triggered the EVOLVE pipeline —
        Claude was never invited to analyze the remaining 5-15% failures,
        so the trajectory froze after iter 0. Now we collect failures
        whenever success_rate < 1.0 (i.e., any task with at least one
        failure). EVOLVE's own ``_select_evolve_targets`` (max_targets=3)
        bounds the per-iter Claude cost.
        """
        from mage.environments.base import ReasoningEnvironment

        for skill_eval in evaluation.evaluations:
            if skill_eval.success_rate < 1.0:
                instances = self.environment.get_evaluation_instances(
                    skill_id=skill_eval.skill_id, num_instances=10,
                )
                if not instances:
                    instances = self.environment.get_evaluation_instances(
                        skill_id=None, num_instances=10,
                    )

                # Get skill strategy for injection
                skill_data = self.evokg.get_node(skill_eval.skill_id)
                strategy_guidance = ""
                if skill_data:
                    if skill_data.get("prompt_template"):
                        strategy_guidance = f"\nStrategy: {skill_data['prompt_template']}"
                    strategy = skill_data.get("strategy", {})
                    if isinstance(strategy, dict) and strategy.get("type") == "tool":
                        tool_name = strategy.get("tool", "")
                        strategy_guidance += f"\nUse tool: {tool_name}"

                failures = []
                for inst in instances[:5]:
                    # Actually run the Learner to see what it predicts
                    predicted = "N/A"
                    inst_task_type = inst.metadata.get("task_type", "")
                    try:
                        if isinstance(self.environment, ReasoningEnvironment):
                            def agent_fn(prompt, instruction):
                                try:
                                    return self.learner.act(prompt + strategy_guidance, instruction, task_type=inst_task_type)
                                except TypeError:
                                    return self.learner.act(prompt + strategy_guidance, instruction)
                            result = self.environment.run_episode(
                                agent_fn=agent_fn,
                                instruction=inst.instruction,
                                context=inst.context,
                                max_steps=3,
                            )
                            predicted = result.metadata.get("final_answer", "N/A")
                        else:
                            predicted = self.learner.act(inst.instruction, inst.instruction)
                    except Exception:
                        pass

                    score = 0.0
                    if inst.expected_answer and predicted != "N/A":
                        score = self.environment.evaluate_answer(predicted, inst.expected_answer)

                    if score < 0.5:  # Only collect actual failures
                        failures.append({
                            "question": inst.instruction[:300],
                            "expected": inst.expected_answer or "N/A",
                            "predicted": predicted[:200],
                            "score": score,
                            "task_type": inst.metadata.get("task_type", ""),
                            "strategy_used": strategy_guidance[:100] if strategy_guidance else "none",
                            "failure_modes": [f.detail for f in skill_eval.failure_modes[:3]],
                        })

                if failures:
                    self._iteration_failures[skill_eval.skill_id] = failures
                    logger.info(
                        f"  Collected {len(failures)} failures for {skill_eval.skill_id} "
                        f"(success_rate={skill_eval.success_rate:.0%})"
                    )

    # Cap on EvoKG skills (after splits). Original DISCOVER produces ~10.
    SKILL_GROWTH_CAP = 30
    # Per-skill cap on principles. Generous so that additions from many
    # iterations can compose. Validation filters bad principles before they
    # are added; the bandit/embedding lookup picks the best at eval time.
    PRINCIPLES_PER_SKILL_CAP = 12

    def _accumulate_persistent_failures(self) -> None:
        """Merge this iteration's failures into the persistent failure memory.

        Persistent failures accumulate across iterations so Claude sees the
        full failure history when proposing principles/tools — not just the
        last iteration's snapshot.
        """
        if not hasattr(self, "_persistent_failures"):
            self._persistent_failures: dict[str, list[dict]] = {}

        for skill_id, failures in self._iteration_failures.items():
            if skill_id not in self._persistent_failures:
                self._persistent_failures[skill_id] = []
            # Dedupe by question text
            seen = {f.get("question", "") for f in self._persistent_failures[skill_id]}
            for f in failures:
                if f.get("question", "") not in seen:
                    self._persistent_failures[skill_id].append(f)
            # Keep at most 15 per skill (rolling window of failure patterns)
            self._persistent_failures[skill_id] = self._persistent_failures[skill_id][-15:]

    def _select_evolve_targets(self, max_targets: int = 3) -> list[tuple[str, list[dict]]]:
        """Select task_types to evolve this iteration with soft round-robin.

        v12.1a — anti-plateau:
            score = num_failures + 0.3 * iters_since_last_targeted

        STBench has 27 task types but the previous selector always picked
        the 3 with the most failures, starving the other 24. Soft round-
        robin keeps failure count as the dominant signal but adds a
        recency bonus so stale targets eventually get attention. Every
        task type gets targeted at least once per ~num_task_types/3 iters.
        """
        self._accumulate_persistent_failures()

        # Combine persistent + current with current taking priority
        merged: dict[str, list[dict]] = {}
        for sid, fs in self._persistent_failures.items():
            merged[sid] = list(fs)
        for sid, fs in self._iteration_failures.items():
            merged[sid] = fs  # current iter overrides

        if not merged:
            return []

        # Initialize history tracker on first call
        if not hasattr(self, "_task_targeting_history"):
            self._task_targeting_history: dict[str, int] = {}

        current_iter = self.evokg.iteration

        def score(tid: str) -> float:
            n_fails = len(merged[tid])
            last_iter = self._task_targeting_history.get(tid, -100)
            gap = current_iter - last_iter
            # Failures dominate, recency is a tiebreaker
            return float(n_fails) + 0.3 * gap

        ranked = sorted(merged.keys(), key=score, reverse=True)
        targets = ranked[:max_targets]

        # Mark these as targeted now
        for tid in targets:
            self._task_targeting_history[tid] = current_iter

        logger.info(
            f"  Evolve targets (round-robin): {targets} "
            f"[history covers {len(self._task_targeting_history)} task types]"
        )
        return [(tid, merged[tid]) for tid in targets]

    def _resolve_skill_id(self, skill_id: str) -> tuple[str | None, dict | None]:
        """Resolve a critic task_type to (evokg_skill_id, skill_data) or (None, None)."""
        skill_data = self.evokg.get_node(skill_id)
        if skill_data:
            return skill_id, skill_data
        mapped = self._map_task_to_skill(skill_id)
        if mapped:
            data = self.evokg.get_node(mapped)
            if data:
                return mapped, data
        return None, None

    def _evolve_graph(self, iteration: int) -> list[str]:
        """Generative EVOLVE with rotating action types.

        Each iteration takes a *different* kind of action so the system makes
        qualitatively different progress instead of plateauing on principles.

        Action rotation (mod 4):
            0: principles  — extract reusable rules from failures
            1: tools       — Claude writes Python tool code for 0% skills
            2: refine      — rewrite prompt_template for the worst skill
            3: split       — split a clustered failure skill into sub-skills

        Even when there are no current failures, the persistent failure
        history is used so EVOLVE always has work to do.
        """
        # Always run the rotating action even if current iter had no fails
        if not hasattr(self, "_persistent_failures"):
            self._persistent_failures = {}

        targets = self._select_evolve_targets(max_targets=3)
        if not targets:
            logger.info("  No failure history to evolve from yet")
            return []

        action = ["principles", "tools", "refine", "split"][iteration % 4]
        logger.info(f"  EVOLVE action [iter {iteration}]: {action}")

        all_changes: list[str] = []
        for raw_skill_id, failures in targets:
            skill_id, skill_data = self._resolve_skill_id(raw_skill_id)
            if skill_id is None:
                logger.debug(f"  Skipping unmapped skill: {raw_skill_id}")
                continue

            try:
                if action == "principles":
                    all_changes.extend(self._evolve_principles(
                        skill_id, skill_data, failures, iteration,
                    ))
                elif action == "tools":
                    all_changes.extend(self._evolve_tools(
                        skill_id, skill_data, failures, iteration,
                    ))
                elif action == "refine":
                    all_changes.extend(self._evolve_refine_strategy(
                        skill_id, skill_data, failures, iteration,
                    ))
                elif action == "split":
                    all_changes.extend(self._evolve_split_skill(
                        skill_id, skill_data, failures, iteration,
                    ))
            except Exception as e:
                logger.warning(f"  EVOLVE action '{action}' failed for {skill_id}: {e}")

        # ALWAYS write failure memories — independent of the rotating action.
        # v12.1c: Memory writing covers ALL failing task_types in this
        # iteration, not just the 3 picked for principle/tool/refine.
        # Memories are cheap (1 Claude call per task_type cluster) and the
        # more coverage we have, the more retrieval helps at eval time.
        all_failure_targets = list(self._iteration_failures.items())
        if not all_failure_targets:
            all_failure_targets = targets  # fall back to selected if empty
        memory_changes = self._evolve_failure_memories(iteration, all_failure_targets)
        all_changes.extend(memory_changes)

        # Persist any newly created dynamic tools
        self._save_dynamic_tools(iteration)

        # Invalidate embedding indexes after graph changes
        if hasattr(self.learner, '_embedding_index') and self.learner._embedding_index:
            self.learner._embedding_index.invalidate()
        # Rebuild memory index so newly-added memories are retrievable
        if hasattr(self.learner, '_rebuild_memory_index'):
            self.learner._rebuild_memory_index()

        if all_changes:
            logger.info(f"  Graph evolved [{action}]: {len(all_changes)} changes — {all_changes}")
        else:
            logger.info(f"  Graph evolved [{action}]: 0 changes (action found nothing to apply)")
        return all_changes

    def _evolve_failure_memories(
        self,
        iteration: int,
        targets: list[tuple[str, list[dict]]],
    ) -> list[str]:
        """Write concrete worked examples for this iteration's failures.

        Runs EVERY iteration (independent of rotating action). Each failure
        becomes a permanent retrievable memory. The memory bank only grows.
        """
        if not targets:
            return []

        added_total = 0
        for raw_skill_id, failures in targets:
            skill_id, _ = self._resolve_skill_id(raw_skill_id)
            if skill_id is None:
                skill_id = raw_skill_id  # store under raw task_type if no mapping
            try:
                memories = self.discovery.write_failure_memories(
                    failures[:5],  # cap per skill
                    skill_id=skill_id,
                )
            except Exception as e:
                logger.warning(f"  write_failure_memories failed for {skill_id}: {e}")
                continue

            for m in memories:
                added = self.evokg.add_failure_memory(
                    question=m["question"],
                    expected=m["expected"],
                    wrong_answer=m["wrong_answer"],
                    reasoning=m["reasoning"],
                    task_type=m.get("task_type", ""),
                    skill_id=skill_id,
                    iteration=iteration,
                    kind=m.get("kind", "specific"),
                )
                if added:
                    added_total += 1

        if added_total:
            total = self.evokg.count_failure_memories()
            logger.info(
                f"  Failure memory bank: +{added_total} new (total now {total})"
            )

        # v12.3 — also try to write a retrieval recipe per task_type whose
        # questions are factoid-style (no context provided in the eval).
        # Recipes are dedup'd inside add_retrieval_recipe.
        recipes_added = 0
        if not hasattr(self, "_recipe_attempted"):
            self._recipe_attempted: set[str] = set()

        for raw_skill_id, failures in targets:
            if not failures:
                continue
            tt = failures[0].get("task_type", "") or raw_skill_id
            if not tt or tt in self._recipe_attempted:
                continue
            # Only attempt recipes for task_types whose questions have NO
            # context (factoid lookups). Skip questions that already include
            # paragraphs/passages (HotpotQA-style).
            sample_q = failures[0].get("question", "")
            if len(sample_q) > 600:
                continue
            self._recipe_attempted.add(tt)

            try:
                recipe = self.discovery.write_retrieval_recipe(tt, failures)
            except Exception as e:
                logger.debug(f"  Recipe gen failed for {tt}: {e}")
                continue
            if not recipe:
                continue

            added = self.evokg.add_retrieval_recipe(
                task_type=tt,
                query_template=recipe["query_template"],
                extraction_pattern=recipe.get("extraction_pattern", ""),
                description=recipe.get("description", ""),
                source_tool="wikipedia",
                iteration=iteration,
            )
            if added:
                recipes_added += 1
                logger.info(
                    f"  Added retrieval recipe for {tt}: "
                    f"{recipe['query_template'][:60]}"
                )

        changes: list[str] = []
        if added_total:
            changes.append(f"memories(+{added_total})")
        if recipes_added:
            changes.append(f"recipes(+{recipes_added})")
        return changes

    # ------------------------------------------------------------------
    # EVOLVE actions (rotating, one per iteration)
    # ------------------------------------------------------------------

    def _evolve_principles(
        self,
        skill_id: str,
        skill_data: dict,
        failures: list[dict],
        iteration: int,
    ) -> list[str]:
        """Action: extract & validate new principles. Cap raised to 8/skill."""
        existing_principles = skill_data.get("principles", []) or []
        if len(existing_principles) >= self.PRINCIPLES_PER_SKILL_CAP:
            logger.info(
                f"  {skill_id}: at principles cap ({self.PRINCIPLES_PER_SKILL_CAP}) — skipping"
            )
            return []

        logger.info(f"  Extracting principles for: {skill_id}")
        # Pass per-environment domain context so Claude writes principles in
        # the right register (e.g., game tactics for Crafter, not roleplay).
        domain_context = None
        if hasattr(self.environment, "get_domain_context"):
            try:
                domain_context = self.environment.get_domain_context()
            except Exception:
                domain_context = None
        new_principles = self.discovery.extract_principles(
            skill_data, failures, existing_principles,
            domain_context=domain_context,
        )
        validated = self._validate_principles(
            skill_id, failures, existing_principles, new_principles,
        )
        if not validated:
            return []

        all_principles = (existing_principles + validated)[:self.PRINCIPLES_PER_SKILL_CAP]
        self.evokg.graph.nodes[skill_id]["principles"] = all_principles

        from mage.core.schemas import ExperienceNode, Edge
        for p_text in validated:
            p_id = f"principle_{skill_id}_{iteration}_{hash(p_text) % 10000}"
            p_node = ExperienceNode(
                node_id=p_id,
                action_pattern=p_text,
                outcome="principle",
                context_conditions=f"skill={skill_id}",
                confidence=1.0,
                domain_origin="principle",
            )
            self.evokg.add_experience(p_node)
            self.evokg.add_edge(Edge(p_id, skill_id, "principle_of_skill"))

        logger.info(f"  Validated {len(validated)}/{len(new_principles)} principles for {skill_id}")
        return [f"principles:{skill_id}(+{len(validated)})"]

    def _evolve_tools(
        self,
        skill_id: str,
        skill_data: dict,
        failures: list[dict],
        iteration: int,
    ) -> list[str]:
        """Action: Claude writes a Python tool for skills failing the most."""
        logger.info(f"  Asking Claude to create tool for: {skill_id}")
        analysis = self.discovery.analyze_failures(skill_data, failures)
        # Force tool creation regardless of Claude's recommendation — this
        # action's whole purpose is to introduce a tool.
        if analysis.get("recommendation") != "create_tool":
            logger.info(
                f"  Claude recommended {analysis.get('recommendation')} but we're "
                f"in tools mode; trying create_tool anyway"
            )
            analysis["recommendation"] = "create_tool"

        changes = self.discovery.apply_evolution(self.evokg, skill_id, analysis, iteration)
        if not changes:
            return []

        # Validate / fix the new tool against actual failures
        change_list = list(changes)
        self._validate_and_fix_tool(skill_id, failures, iteration, change_list)
        return change_list

    def _evolve_refine_strategy(
        self,
        skill_id: str,
        skill_data: dict,
        failures: list[dict],
        iteration: int,
    ) -> list[str]:
        """Action: Claude rewrites the skill's prompt_template / strategy."""
        logger.info(f"  Refining strategy for: {skill_id}")
        analysis = self.discovery.analyze_failures(skill_data, failures)
        if analysis.get("recommendation") not in ("refine_strategy", "create_tool"):
            analysis["recommendation"] = "refine_strategy"
        changes = self.discovery.apply_evolution(self.evokg, skill_id, analysis, iteration)
        return list(changes) if changes else []

    def _evolve_split_skill(
        self,
        skill_id: str,
        skill_data: dict,
        failures: list[dict],
        iteration: int,
    ) -> list[str]:
        """Action: Claude splits a skill into sub-skills with parent mastery inheritance."""
        from mage.core.schemas import NodeType
        cap_count = len(self.evokg.get_nodes_by_type(NodeType.CAPABILITY))
        if cap_count >= self.SKILL_GROWTH_CAP:
            logger.info(
                f"  Skill cap reached ({cap_count}/{self.SKILL_GROWTH_CAP}) — "
                f"falling back to refine_strategy"
            )
            return self._evolve_refine_strategy(skill_id, skill_data, failures, iteration)

        logger.info(f"  Asking Claude to split: {skill_id}")
        analysis = self.discovery.analyze_failures(skill_data, failures)
        if analysis.get("recommendation") != "split_skill":
            analysis["recommendation"] = "split_skill"
        changes = self.discovery.apply_evolution(self.evokg, skill_id, analysis, iteration)
        return list(changes) if changes else []

    def _validate_principles(
        self,
        skill_id: str,
        failures: list[dict],
        existing_principles: list[str],
        new_principles: list[str],
    ) -> list[str]:
        """Validate principles before applying — reject ones that hurt accuracy.

        2026-05-05 fix (was rejecting most principles due to noise):
        - temperature=0.0 (was 0.3 → non-deterministic; baseline and test
          scores were random samples)
        - 10 questions (was 5 → too small to dampen noise)
        - INCLUDE context (was empty string → for FinQA / context-rich
          tasks the model was being asked impossible questions and
          baseline was essentially random)
        - Margin tolerance: accept when test_score >= baseline_score - 1
          (small slack for ties; previously a single noise-induced flip
          would reject a good principle)
        """
        llm = getattr(self.learner, '_llm', None)
        if not new_principles or not llm:
            return []

        from mage.core.strategy_router import StrategyRouter

        n_test = 10  # widened from 5
        test_questions = failures[:n_test]

        def _score_pool(principles_in_use):
            score = 0
            for f in test_questions:
                ctx = f.get("context", "") or ""
                prompt = StrategyRouter().build_prompt_for_strategy(
                    f["question"], ctx, {"type": "direct"},
                    principles=principles_in_use,
                )
                response = llm.generate(prompt, temperature=0.0, max_tokens=256)
                predicted = self.learner._extract_answer(response) if hasattr(self.learner, '_extract_answer') else response.strip()
                score += self.environment.evaluate_answer(predicted, f.get("expected", ""))
            return score

        baseline_score = _score_pool(existing_principles)

        validated = []
        for principle in new_principles:
            test_score = _score_pool(existing_principles + [principle])
            # Accept with 1-question slack to dampen ties / borderline cases.
            if test_score >= baseline_score - 1:
                validated.append(principle)
                logger.info(f"  Principle ACCEPTED ({test_score:.1f} >= {baseline_score-1:.1f}): {principle[:60]}...")
            else:
                logger.info(f"  Principle REJECTED ({test_score:.1f} < {baseline_score-1:.1f}): {principle[:60]}...")

        return validated

    def _validate_and_fix_tool(
        self,
        skill_id: str,
        failures: list[dict],
        iteration: int,
        all_changes: list[str],
        max_attempts: int = 3,
    ) -> None:
        """Validate a newly created tool and fix extraction if it fails.

        The key: show Claude EXACTLY why extraction failed — the regex,
        the question text, what matched/didn't — so Claude can fix the
        EXTRACTOR, not just recreate the tool.
        """
        from mage.tools.tool_pipeline import run_tool_pipeline
        from mage.tools.extractors import TASK_EXTRACTORS
        import re

        skill_data = self.evokg.get_node(skill_id)
        if not skill_data:
            return

        strategy = skill_data.get("strategy", {})
        tool_name = strategy.get("tool", "")
        extractor_regex = strategy.get("extractor_regex", "")

        for attempt in range(max_attempts):
            successes = 0
            test_failures = []

            for f in failures[:5]:
                task_type = f.get("task_type", "")
                question = f["question"]

                # Diagnose: WHY did extraction fail?
                diagnosis = self._diagnose_extraction(question, task_type, skill_data)

                tool_answer = run_tool_pipeline(question, task_type, evokg=self.evokg)

                if tool_answer is not None:
                    score = self.environment.evaluate_answer(tool_answer, f.get("expected", ""))
                    if score > 0.5:
                        successes += 1
                    else:
                        test_failures.append({
                            "question": question[:300],
                            "expected": f.get("expected", "N/A"),
                            "tool_produced": tool_answer,
                            "score": score,
                            "diagnosis": f"Tool returned wrong answer. Tool output: {tool_answer}",
                        })
                else:
                    test_failures.append({
                        "question": question[:300],
                        "expected": f.get("expected", "N/A"),
                        "tool_produced": "EXTRACTION_FAILED",
                        "score": 0,
                        "diagnosis": diagnosis,
                    })

            test_total = min(len(failures), 5)
            accuracy = successes / max(test_total, 1)
            logger.info(
                f"  Tool validation for {skill_id} (attempt {attempt+1}): "
                f"{successes}/{test_total} = {accuracy:.0%}"
            )

            if accuracy >= 0.5 or not test_failures:
                all_changes.append(f"validated:{skill_id}({accuracy:.0%})")
                return

            if attempt >= max_attempts - 1:
                logger.warning(f"  Tool for {skill_id} failed validation after {max_attempts} attempts")
                return

            # Ask Claude to fix with DETAILED extraction diagnostics
            logger.info(f"  Asking Claude to fix extractor for {skill_id}...")
            fix_result = self.discovery.fix_extractor(
                skill_data, test_failures, tool_name, extractor_regex,
            )

            if fix_result:
                # Update the extractor in the EvoKG
                if fix_result.get("new_extractor_regex"):
                    self.evokg.graph.nodes[skill_id]["strategy"]["extractor_regex"] = fix_result["new_extractor_regex"]
                    all_changes.append(f"fixed_extractor:{skill_id}")
                    logger.info(f"  Updated extractor regex for {skill_id}")

                if fix_result.get("new_tool_code"):
                    self.discovery._register_dynamic_tool(
                        tool_name, fix_result["new_tool_code"],
                        strategy.get("tool_description", ""), "",
                    )
                    self._load_dynamic_tools()
                    all_changes.append(f"fixed_tool:{skill_id}")

    def _diagnose_extraction(self, question: str, task_type: str, skill_data: dict) -> str:
        """Diagnose WHY extraction failed for a question."""
        from mage.tools.extractors import TASK_EXTRACTORS
        import re

        parts = []

        # Check pre-built extractor
        extractor = TASK_EXTRACTORS.get(task_type)
        if extractor:
            try:
                params = extractor(question)
                if params:
                    parts.append(f"Pre-built extractor matched: {list(params.keys())}")
                else:
                    parts.append(f"Pre-built extractor for {task_type} returned None — regex didn't match")
            except Exception as e:
                parts.append(f"Pre-built extractor crashed: {e}")
        else:
            parts.append(f"No pre-built extractor for task_type={task_type}")

        # Check EvoKG dynamic extractor
        strategy = skill_data.get("strategy", {})
        dyn_regex = strategy.get("extractor_regex", "")
        if dyn_regex:
            try:
                import signal
                pattern = eval(dyn_regex)

                def _timeout(signum, frame):
                    raise TimeoutError()
                old = signal.signal(signal.SIGALRM, _timeout)
                signal.alarm(2)
                try:
                    match = re.search(pattern, question[:1000], re.DOTALL)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)

                if match:
                    parts.append(f"Dynamic regex matched: groups={match.groups()[:5]}")
                else:
                    parts.append(f"Dynamic regex '{dyn_regex}' did NOT match the question")
            except Exception as e:
                parts.append(f"Dynamic regex failed: {e}")
        else:
            parts.append("No dynamic extractor_regex in skill strategy")

        # Show question structure hints
        parts.append(f"Question starts with: '{question[:100]}'")
        parts.append(f"Question contains coords: {bool(re.search(r'[(-]?[0-9]+[.][0-9]+', question))}")

        return " | ".join(parts)

    def _map_task_to_skill(self, task_type: str) -> str | None:
        """Map task_type → EvoKG skill ID. Delegates to EvoKG.map_task_to_skill,
        with embedding-similarity as a final fallback (needs learner state).
        """
        if not task_type:
            return None

        # Primary: graph-based mapping
        mapped = self.evokg.map_task_to_skill(task_type)
        if mapped:
            return mapped

        # Fallback: embedding similarity (only available here, not in EvoKG)
        if hasattr(self.learner, '_embedding_index') and self.learner._embedding_index:
            matches = self.learner._embedding_index.find_skill(
                task_type.replace("_", " "), top_k=1,
            )
            if matches:
                return matches[0][0]

        return None

    def _save_dynamic_tools(self, iteration: int) -> None:
        """Persist dynamically created tools to disk."""
        from mage.tools.st_tools import TOOL_REGISTRY
        dynamic_tools = {
            name: {"code": info["code"], "description": info["description"]}
            for name, info in TOOL_REGISTRY.items()
            if info.get("source") == "claude_generated"
        }
        if dynamic_tools:
            path = self.output_dir / f"iter_{iteration}" / "dynamic_tools.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(dynamic_tools, f, indent=2)
            logger.info(f"  Saved {len(dynamic_tools)} dynamic tools")

    def _load_dynamic_tools(self) -> None:
        """Load previously saved dynamic tools from disk and per-iteration JSON.

        Two sources, in order:
        1. ``mage/tools/dynamic/<tool>.py`` — persistent across runs
        2. ``experiments/.../iter_*/dynamic_tools.json`` — per-iteration metadata
        """
        # 1. Load from persistent disk location (survives restarts)
        from mage.agents.discovery import SkillDiscovery
        SkillDiscovery.load_persisted_dynamic_tools()

        # 2. Load from per-iteration JSON (legacy, for current run)
        import glob
        tool_files = sorted(glob.glob(str(self.output_dir / "iter_*/dynamic_tools.json")))
        if tool_files:
            with open(tool_files[-1]) as f:
                tools = json.load(f)
            for name, info in tools.items():
                self.discovery._register_dynamic_tool(
                    name, info["code"], info["description"], "",
                )
            logger.info(f"Loaded {len(tools)} dynamic tools from {tool_files[-1]}")

    def _save_history(self) -> None:
        history_path = self.output_dir / "evolution_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Saved evolution history to {history_path}")
