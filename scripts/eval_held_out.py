"""Run a one-shot held-out evaluation on a completed MAGE experiment.

Given an experiment directory and an iteration index, this script:
  1. Loads the config the experiment was run with
  2. Rebuilds the environment with a held-out test pool (disjoint from
     the training pool the loop iterated over)
  3. Rebuilds the learner (RoutedLearner with the same config)
  4. Loads iter_N's EvoKG (which gives us all stored success + failure
     memories), strategy_router.json, and the experiment's
     search_agent_bandit.json
  5. Runs a single evaluation pass over the held-out pool using the
     same CoT prompt the critic uses in training
  6. Reports overall + per-task accuracy and writes the results to
     ``<experiment_dir>/held_out_eval_iter_<N>.json``

This is the "generalization number" for the paper: every held-out
question is something the learner has never seen during evolution, so
success memories keyed by the exact question cannot help. Any gain
comes from transferable skills (principles, decomposition patterns,
similar-question retrieval).

Usage:
    python scripts/eval_held_out.py \\
        --config configs/webqa_v15_5_large.yaml \\
        --experiment experiments/webqa_v15_5_large \\
        --iter 9 \\
        --num-test 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("eval_held_out")


def _latest_iter_dir(experiment_dir: Path) -> int:
    iters = []
    for p in experiment_dir.glob("iter_*"):
        try:
            iters.append(int(p.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not iters:
        raise FileNotFoundError(f"No iter_N directories found in {experiment_dir}")
    return max(iters)


def _build_env(env_config: dict[str, Any], held_out_size: int, seed: int):
    """Rebuild the environment WITH held-out pool populated."""
    env_type = env_config.get("type", "stbench")
    if env_type == "webqa":
        from mage.environments.webqa_env import WebQuestionsEnvironment
        return WebQuestionsEnvironment(
            split=env_config.get("split", "test"),
            max_instances=int(env_config.get("max_instances", 500)),
            held_out_size=held_out_size,
            seed=seed,
        )
    if env_type == "hotpotqa":
        from mage.environments.hotpotqa_env import HotpotQAEnvironment
        return HotpotQAEnvironment(
            split=env_config.get("split", "validation"),
            max_instances=int(env_config.get("max_instances", 500)),
            held_out_size=held_out_size,
            seed=seed,
        )
    if env_type == "stbench":
        from mage.environments.stbench_env import STBenchEnvironment
        return STBenchEnvironment(
            data_dir=env_config.get(
                "data_dir", "data/stbench/splits"
            ),
            split=env_config.get("split", "test"),
            max_instances=int(env_config.get("max_instances", 500)),
            held_out_size=held_out_size,
            seed=seed,
        )
    if env_type == "gsm8k":
        from mage.environments.gsm8k_env import GSM8KEnvironment
        return GSM8KEnvironment(
            split=env_config.get("split", "test"),
            max_instances=int(env_config.get("max_instances", 500)),
            held_out_size=held_out_size,
            seed=seed,
        )
    if env_type == "realmath":
        from mage.environments.realmath_env import RealMathEnvironment
        return RealMathEnvironment(
            split=env_config.get("split", "all"),
            max_instances=int(env_config.get("max_instances", 500)),
            held_out_size=held_out_size,
            seed=seed,
        )
    raise ValueError(f"Unknown environment type: {env_type}")


def _build_learner(config: dict[str, Any], environment, output_dir: Path, weak_llm):
    """Rebuild the RoutedLearner with same config the experiment used."""
    from mage.agents.learner import RoutedLearner

    learner_config = dict(config.get("learner", {}))
    learner_config["output_dir"] = str(output_dir)
    learner = RoutedLearner(config=learner_config, environment=environment)
    learner.set_llm(weak_llm)
    return learner


def main():
    parser = argparse.ArgumentParser(
        description="Held-out evaluation on a completed MAGE experiment"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the config YAML the experiment was run with",
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Path to experiments/<name>/ directory",
    )
    parser.add_argument(
        "--iter", type=int, default=-1,
        help="Which iter's state to load. -1 = latest available",
    )
    parser.add_argument(
        "--num-test", type=int, default=100,
        help="How many held-out questions to evaluate",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_dir():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = int(config.get("experiment", {}).get("seed", 42))
    random.seed(seed)

    iter_idx = args.iter if args.iter >= 0 else _latest_iter_dir(experiment_dir)
    iter_dir = experiment_dir / f"iter_{iter_idx}"
    evokg_path = iter_dir / "evokg.json"
    router_path = iter_dir / "strategy_router.json"
    sa_path = experiment_dir / "search_agent_bandit.json"

    logger.info(f"Loading state from iter {iter_idx}")
    logger.info(f"  evokg: {evokg_path}")
    logger.info(f"  router: {router_path}")
    logger.info(f"  search_agent: {sa_path}")

    if not evokg_path.exists():
        logger.error(f"EvoKG snapshot missing: {evokg_path}")
        sys.exit(1)

    # ---- Build strong (for judge) + weak (for learner) LLMs ------------
    from mage.utils.llm import create_backend
    strong_llm = create_backend(
        config.get("llm", {}).get("strong", {"type": "mock"})
    )
    weak_llm = create_backend(
        config.get("llm", {}).get("weak", {"type": "mock"})
    )

    # ---- Rebuild environment with held-out pool -----------------------
    environment = _build_env(
        env_config=config.get("environment", {}),
        held_out_size=args.num_test,
        seed=seed,
    )

    test_instances = environment.get_test_instances(num_instances=args.num_test)
    if not test_instances:
        logger.error(
            f"No held-out instances available — environment {environment.name} "
            f"may not support get_test_instances() or held_out_size=0."
        )
        sys.exit(1)
    logger.info(f"Held-out test set: {len(test_instances)} questions")

    # ---- Rebuild learner + load saved state ---------------------------
    from mage.core.evokg import EvoKG
    evokg = EvoKG.load_snapshot(str(evokg_path))
    logger.info(f"Loaded EvoKG: {evokg}")

    learner = _build_learner(config, environment, experiment_dir, weak_llm)

    # Plumb EvoKG → this triggers SkillEmbeddingIndex + memory index rebuild
    learner.set_evokg(evokg)

    # Load strategy router state (per-skill bandit arms) if present
    if router_path.exists():
        from mage.core.strategy_router import StrategyRouter
        learner._router = StrategyRouter.load(router_path)
        logger.info(f"Loaded strategy router from {router_path}")

    # Build search agent + load its bandit state (auto-backfills missing arms)
    from mage.tools.search_agent import SearchAgent
    sa_cfg = (config.get("search_agent", {}) or {})
    search_agent = SearchAgent(
        top_k=int(sa_cfg.get("top_k", 5)),
        min_pulls_per_arm=int(sa_cfg.get("min_pulls_per_arm", 20)),
    )
    search_agent.load(sa_path)
    learner.set_search_agent(search_agent)
    logger.info(f"Loaded search agent bandit with task_types: "
                f"{list(search_agent._bandit.keys())}")

    # ---- Run eval -----------------------------------------------------
    from mage.utils.llm_judge import LLMJudge
    judge = LLMJudge(llm_backend=strong_llm)

    total_success = 0
    total_attempts = 0
    by_task: dict[str, dict[str, int]] = {}
    per_question_results: list[dict[str, Any]] = []

    logger.info("Running held-out evaluation...")
    for i, inst in enumerate(test_instances):
        task_type = inst.metadata.get("task_type", "")
        by_task.setdefault(task_type, {"success": 0, "total": 0})
        by_task[task_type]["total"] += 1
        total_attempts += 1

        # CoT prompt — matches critic._evaluate_proper exactly
        if inst.context:
            prompt = (
                f"Answer the following question based on the provided context. "
                f"Think step by step, then give your final answer after 'Answer:'.\n\n"
                f"Context:\n{inst.context}\n\n"
                f"Question: {inst.instruction}\n\n"
                f"Let's think step by step:"
            )
        else:
            prompt = (
                f"Question: {inst.instruction}\n\n"
                f"Let's think step by step. "
                f"After your reasoning, give your final answer after 'Answer:'."
            )

        try:
            predicted = learner.act(prompt, inst.instruction, task_type=task_type)
        except TypeError:
            predicted = learner.act(prompt, inst.instruction)
        except Exception as e:
            logger.debug(f"Q{i} learner error: {e}")
            predicted = ""

        # Score via the judge (semantic equivalence) with F1/env fallback
        score = 0.0
        if inst.expected_answer and predicted:
            try:
                score = judge.evaluate(
                    predicted, inst.expected_answer,
                    question=inst.instruction, task_type=task_type,
                )
            except Exception:
                try:
                    score = environment.evaluate_answer(
                        predicted, inst.expected_answer, task_type=task_type,
                    )
                except TypeError:
                    score = environment.evaluate_answer(
                        predicted, inst.expected_answer,
                    )

        correct = score >= 0.5
        if correct:
            total_success += 1
            by_task[task_type]["success"] += 1

        per_question_results.append({
            "instance_id": inst.instance_id,
            "question": inst.instruction[:200],
            "expected": inst.expected_answer,
            "predicted": (predicted or "")[:300],
            "task_type": task_type,
            "score": float(score),
            "correct": bool(correct),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_instances):
            logger.info(
                f"  {i+1}/{len(test_instances)} done, "
                f"running accuracy = {total_success / max(total_attempts, 1):.1%}"
            )

    overall = total_success / max(total_attempts, 1)

    # ---- Persist + print ---------------------------------------------
    summary = {
        "experiment": str(experiment_dir),
        "iter_loaded": iter_idx,
        "num_test": len(test_instances),
        "overall_accuracy": overall,
        "total_success": total_success,
        "total_attempts": total_attempts,
        "by_task_accuracy": {
            t: {
                "success": v["success"],
                "total": v["total"],
                "accuracy": (v["success"] / v["total"]) if v["total"] else 0.0,
            }
            for t, v in by_task.items()
        },
        "per_question_results": per_question_results,
    }
    out_path = experiment_dir / f"held_out_eval_iter_{iter_idx}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    judge.flush_cache()

    print()
    print("=" * 60)
    print(f"HELD-OUT EVAL — {experiment_dir.name} @ iter {iter_idx}")
    print("=" * 60)
    print(f"Overall: {total_success}/{total_attempts} = {overall*100:.2f}%")
    print()
    print("Per task_type:")
    for t, v in sorted(by_task.items(), key=lambda x: -x[1]["total"]):
        acc = v["success"] / v["total"] if v["total"] else 0.0
        print(f"  {t:30s}  {v['success']:3d}/{v['total']:3d}  {acc*100:5.1f}%")
    print()
    print(f"Full results → {out_path}")


if __name__ == "__main__":
    main()
