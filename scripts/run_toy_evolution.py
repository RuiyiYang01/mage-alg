"""Run the MAGE evolution loop on the toy grid world environment."""

import logging
import random
import sys

from mage.agents.stub_agents import (
    StubCritic,
    StubCurator,
    StubExplorer,
    StubLearner,
    StubNavigator,
)
from mage.core.evolution_loop import EvolutionLoop
from mage.environments.toy_gridworld import ToyGridWorld, build_toy_evokg


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    random.seed(42)

    # Initialize
    evokg = build_toy_evokg()
    environment = ToyGridWorld()

    navigator = StubNavigator({"max_targets": 2, "mastery_threshold": 0.5})
    explorer = StubExplorer()
    learner = StubLearner()
    critic = StubCritic({"trials_per_skill": 10})
    curator = StubCurator()

    loop = EvolutionLoop(
        evokg=evokg,
        navigator=navigator,
        explorer=explorer,
        learner=learner,
        critic=critic,
        curator=curator,
        environment=environment,
        config={
            "output_dir": "experiments/toy_gridworld",
            "num_episodes": 20,
        },
    )

    # Run
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    history = loop.run(num_iterations=num_iterations)

    # Summary
    print("\n" + "=" * 60)
    print("EVOLUTION SUMMARY")
    print("=" * 60)
    for entry in history:
        stats = entry.get("evokg_stats", {})
        print(
            f"  Iter {entry['iteration']}: "
            f"success_rate={entry.get('overall_success_rate', 0):.2%}, "
            f"mastered={stats.get('mastered_skills', 0)}/{stats.get('capability_nodes', 0)}, "
            f"exp_nodes={stats.get('experience_nodes', 0)}, "
            f"frontier={stats.get('frontier_size', 0)}"
        )

    final_stats = history[-1].get("evokg_stats", {})
    print(f"\nFinal EvoKG: {final_stats}")


if __name__ == "__main__":
    main()
