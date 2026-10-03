"""Toy grid world environment for validating the MAGE evolution loop.

A simple environment with 6 skills in a prerequisite chain:
  explore → collect_wood → make_pickaxe → collect_stone → make_sword → defeat_zombie

The environment simulates skill execution with configurable success probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mage.core.evokg import EvoKG
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    RelationType,
    TaskNode,
)


# Skill definitions with prerequisite chains
TOY_SKILLS = [
    {"id": "explore", "name": "Explore", "prereqs": [], "base_difficulty": 0.1},
    {"id": "collect_wood", "name": "Collect Wood", "prereqs": ["explore"], "base_difficulty": 0.2},
    {"id": "make_pickaxe", "name": "Make Pickaxe", "prereqs": ["collect_wood"], "base_difficulty": 0.4},
    {"id": "collect_stone", "name": "Collect Stone", "prereqs": ["make_pickaxe"], "base_difficulty": 0.5},
    {"id": "make_sword", "name": "Make Sword", "prereqs": ["collect_stone"], "base_difficulty": 0.6},
    {"id": "defeat_zombie", "name": "Defeat Zombie", "prereqs": ["make_sword"], "base_difficulty": 0.8},
]


@dataclass
class ToyGridWorld:
    """Simple simulated environment for testing the evolution loop."""

    skill_defs: list[dict] = field(default_factory=lambda: list(TOY_SKILLS))

    def attempt_skill(self, skill_id: str, agent_mastery: float) -> tuple[bool, list[str], list[str]]:
        """Simulate attempting a skill.

        Returns (success, observations, actions).
        Success probability = agent_mastery * (1 - base_difficulty).
        """
        import random

        skill_def = next((s for s in self.skill_defs if s["id"] == skill_id), None)
        if skill_def is None:
            return False, ["Unknown skill"], ["noop"]

        success_prob = agent_mastery * (1.0 - skill_def["base_difficulty"] * 0.5)
        success = random.random() < success_prob

        if success:
            observations = [f"Approaching {skill_def['name']}", f"Executing {skill_def['name']}", "Success!"]
            actions = [f"move_to_{skill_id}", f"do_{skill_id}", "confirm"]
        else:
            observations = [f"Approaching {skill_def['name']}", f"Attempting {skill_def['name']}", "Failed."]
            actions = [f"move_to_{skill_id}", f"do_{skill_id}", "retry"]

        return success, observations, actions

    def get_skill_ids(self) -> list[str]:
        return [s["id"] for s in self.skill_defs]


def build_toy_evokg() -> EvoKG:
    """Build an EvoKG seeded with the toy grid world's skill tree."""
    evokg = EvoKG()

    # Add capability nodes
    for skill in TOY_SKILLS:
        evokg.add_node(CapabilityNode(
            node_id=skill["id"],
            name=skill["name"],
            mastery=0.0,
            domain="toy_gridworld",
            description=f"Skill: {skill['name']} (difficulty: {skill['base_difficulty']})",
        ))

    # Add prerequisite edges
    for skill in TOY_SKILLS:
        for prereq in skill["prereqs"]:
            evokg.add_edge(Edge(prereq, skill["id"], RelationType.PREREQUISITE_OF))

    # Add a few environment nodes
    evokg.add_node(EnvironmentNode("env_forest", "Forest", entity_type="location", domain="toy_gridworld"))
    evokg.add_node(EnvironmentNode("env_cave", "Cave", entity_type="location", domain="toy_gridworld"))

    # Add task nodes
    evokg.add_node(TaskNode("task_survive", "Survive and defeat zombie", difficulty=0.8, domain="toy_gridworld"))

    return evokg
