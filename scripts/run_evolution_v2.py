"""Run the MAGE v2 evolution loop with dynamic skill discovery.

Usage:
    python scripts/run_evolution_v2.py --config configs/stbench_v2.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="MAGE v2 Evolution Loop")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--no-discover-cache",
        action="store_true",
        help="Force re-running Claude DISCOVER even if a cache exists at experiments/<env>/discovery/discovery_cache.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    random.seed(seed)

    num_iterations = args.iterations or config.get("experiment", {}).get("num_iterations", 5)
    output_dir = config.get("experiment", {}).get("output_dir", "experiments/default")

    # Build strong LLM (Claude) for Navigator/Critic/Curator
    from mage.utils.llm import create_backend
    strong_llm = create_backend(config.get("llm", {}).get("strong", {"type": "mock"}))

    # Build weak LLM (local) for Explorer/Learner
    weak_llm = create_backend(config.get("llm", {}).get("weak", {"type": "mock"}))

    # Build environment
    env_config = config.get("environment", {})
    env_type = env_config.get("type", "stbench")

    if env_type == "stbench":
        from mage.environments.stbench_env import STBenchEnvironment
        environment = STBenchEnvironment(
            data_dir=env_config.get("data_dir", "data/stbench/splits"),
            split=env_config.get("split", "test"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            seed=seed,
        )
    elif env_type == "webshop":
        from mage.environments.webshop_env import WebShopEnvironment
        environment = WebShopEnvironment(
            num_products=env_config.get("num_products", 100),
            max_steps=env_config.get("max_steps", 15),
            seed=seed,
        )
    elif env_type == "webqa":
        from mage.environments.webqa_env import WebQuestionsEnvironment
        environment = WebQuestionsEnvironment(
            held_out_size=int(env_config.get("held_out_size", 0)),
            split=env_config.get("split", "test"),
            max_instances=env_config.get("max_instances", 500),
            seed=seed,
        )
    elif env_type == "hotpotqa":
        from mage.environments.hotpotqa_env import HotpotQAEnvironment
        environment = HotpotQAEnvironment(
            split=env_config.get("split", "validation"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            seed=seed,
        )
    elif env_type == "gsm8k":
        from mage.environments.gsm8k_env import GSM8KEnvironment
        environment = GSM8KEnvironment(
            split=env_config.get("split", "test"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            seed=seed,
        )
    elif env_type == "realmath":
        from mage.environments.realmath_env import RealMathEnvironment
        environment = RealMathEnvironment(
            split=env_config.get("split", "all"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            seed=seed,
        )
    elif env_type == "finqa":
        from mage.environments.finqa_env import FinQAEnvironment
        environment = FinQAEnvironment(
            split=env_config.get("split", "test"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            max_context_chars=env_config.get("max_context_chars", 4000),
            seed=seed,
        )
    elif env_type == "medqa":
        from mage.environments.medqa_env import MedQAEnvironment
        environment = MedQAEnvironment(
            split=env_config.get("split", "test"),
            max_instances=env_config.get("max_instances", 500),
            held_out_size=int(env_config.get("held_out_size", 0)),
            seed=seed,
        )
    elif env_type == "crafter":
        from mage.environments.crafter_env import CrafterAdapter
        environment = CrafterAdapter(
            max_steps=env_config.get("max_steps", 100),
            seed=seed,
        )
    elif env_type == "alfworld":
        from mage.environments.alfworld_env import ALFWorldEnvironment
        environment = ALFWorldEnvironment(
            split=env_config.get("split", "eval_out_of_distribution"),
            max_steps=env_config.get("max_steps", 50),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment: {env_type}")

    # Build skill discovery agent
    from mage.agents.discovery import SkillDiscovery
    discovery = SkillDiscovery(strong_llm)

    # Build agents
    nav_config = config.get("navigator", {})
    exp_config = config.get("explorer", {})
    critic_config = config.get("critic", {})
    curator_config = config.get("curator", {})

    from mage.agents.navigator import HybridNavigator
    from mage.agents.explorer import LLMExplorer
    from mage.agents.critic import CrafterCritic
    from mage.agents.curator import EvoKGCurator

    navigator = HybridNavigator(llm=strong_llm, config=nav_config)
    explorer = LLMExplorer(llm=weak_llm, config=exp_config)

    # v15: LLM-as-judge for semantic answer evaluation
    from mage.utils.llm_judge import LLMJudge
    judge = LLMJudge(llm_backend=strong_llm)
    critic = CrafterCritic(config=critic_config, judge=judge)

    curator = EvoKGCurator(llm=strong_llm, config=curator_config)

    # Build learner
    learner_config = config.get("learner", {})
    learner_type = learner_config.get("type", "icl")

    if learner_type == "routed":
        from mage.agents.learner import RoutedLearner
        learner_config["output_dir"] = output_dir
        learner = RoutedLearner(config=learner_config, environment=environment)
        learner.set_llm(weak_llm)
        # v19.A: wire Claude as escalator if config requests it
        if learner_config.get("use_claude_escalation", False):
            learner.set_claude_escalator(strong_llm)
    elif learner_type == "icl":
        from mage.agents.learner import ICLLearner
        learner_config["output_dir"] = output_dir
        learner = ICLLearner(config=learner_config, environment=environment)
        learner.set_llm(weak_llm)
    elif learner_type == "grpo":
        from mage.agents.learner import GRPOLearner
        learner_config["output_dir"] = output_dir
        learner = GRPOLearner(config=learner_config, environment=environment)
    else:
        from mage.agents.learner import ICLLearner
        learner_config["output_dir"] = output_dir
        learner = ICLLearner(config=learner_config, environment=environment)
        learner.set_llm(weak_llm)

    # Build and run v2 evolution loop (evokg=None triggers DISCOVER phase)
    from mage.core.evolution_loop_v2 import EvolutionLoopV2
    from mage.environments.base import SequentialEnvironment

    # v15.6 fix: sequential envs (Crafter, WebShop) have deterministic
    # achievement IDs that the critic checks against `new_achievements` in
    # info. Claude's DISCOVER invents generic skill names that never match,
    # which gave Crafter 0% across all 22 achievements. Use the env's
    # built-in EvoKG for sequential envs instead.
    # 2026-05-01: also use env.build_evokg() when the reasoning env has a
    # hand-crafted skill graph AND opt-in flag set. This prevents the
    # Opus-DISCOVER-returns-empty bug observed on FinQA where 3/3 Opus seeds
    # had 0 capability nodes in their EvoKG (Sonnet seeds got 10 skills as
    # expected). The opt-in is per-env via a config flag.
    initial_evokg = None
    use_env_evokg = (
        isinstance(environment, SequentialEnvironment)
        or env_config.get("use_env_evokg", False)
    )
    if use_env_evokg and hasattr(environment, "build_evokg"):
        initial_evokg = environment.build_evokg()
        logging.info(
            f"Using env.build_evokg() "
            f"({initial_evokg.graph.number_of_nodes()} nodes) "
            f"instead of Claude DISCOVER."
        )

    loop = EvolutionLoopV2(
        evokg=initial_evokg,  # None for reasoning envs → DISCOVER; else preloaded
        navigator=navigator,
        explorer=explorer,
        learner=learner,
        critic=critic,
        curator=curator,
        environment=environment,
        discovery=discovery,
        config={
            "output_dir": output_dir,
            "num_episodes": config.get("experiment", {}).get("num_episodes", 50),
            "force_discover": args.no_discover_cache,
            # v15: forward the search-agent subconfig (min_pulls_per_arm, top_k)
            "search_agent": config.get("search_agent", {}),
            # v15.6: forward curriculum subconfig
            # (n_new, demote_after_n_fails, demote_revisit_every_k,
            #  final_exam_size). When absent the loop falls back to
            #  legacy fixed-set evaluation.
            "curriculum": config.get("curriculum", {}),
        },
    )

    history = loop.run(num_iterations=num_iterations)

    # Print summary
    print(f"\n{'='*60}")
    print("MAGE v2 EVOLUTION SUMMARY")
    print(f"{'='*60}")
    for entry in history:
        stats = entry.get("evokg_stats", {})
        changes = entry.get("evolution_changes", [])
        print(
            f"  Iter {entry['iteration']}: "
            f"success={entry.get('overall_success_rate', 0):.1%}, "
            f"skills={stats.get('capability_nodes', 0)}[mastered={stats.get('mastered_skills', 0)}], "
            f"exp={stats.get('experience_nodes', 0)}, "
            f"evolved={len(changes)} changes"
        )

    print(f"\nFinal: {history[-1].get('evokg_stats', {})}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
