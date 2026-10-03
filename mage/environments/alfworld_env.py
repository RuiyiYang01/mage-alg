"""ALFWorld environment adapter for MAGE.

Text-based household tasks (put, clean, heat, cool, examine, pick&place).
The agent receives text observations of rooms/objects and must choose
actions from an admissible-commands list to achieve a goal (e.g., "put a
clean mug in the cabinet").

This demonstrates MAGE's generality beyond QA — the KG tracks household
skills (navigate, pick_up, clean, heat, etc.) and the memory bank
accumulates task-solving patterns across iterations.

Skills map to ReAct-style capabilities:
- navigation: move between rooms
- object_interaction: pick up, put down, open, close
- appliance_use: clean, heat, cool objects
- task_planning: decompose goal into sub-steps
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import yaml

from mage.core.evokg import EvoKG
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    RelationType,
    TaskNode,
)
from mage.environments.base import (
    EpisodeResult,
    EvaluationInstance,
    SequentialEnvironment,
    SkillDefinition,
)

logger = logging.getLogger(__name__)

ALFWORLD_SKILLS = {
    "navigation": {
        "name": "Room Navigation",
        "category": "spatial",
        "prereqs": [],
        "description": "Navigate between rooms by going to specific locations (go to shelf, go to countertop)",
    },
    "object_interaction": {
        "name": "Object Interaction",
        "category": "manipulation",
        "prereqs": ["navigation"],
        "description": "Pick up, put down, open, and close objects and receptacles",
    },
    "object_search": {
        "name": "Object Search",
        "category": "exploration",
        "prereqs": ["navigation"],
        "description": "Examine locations to find target objects (examine drawer, look at shelf)",
    },
    "appliance_use": {
        "name": "Appliance Use",
        "category": "tool_use",
        "prereqs": ["object_interaction"],
        "description": "Use appliances to transform objects: clean (sinkbasin), heat (microwave), cool (fridge)",
    },
    "task_planning": {
        "name": "Task Planning",
        "category": "reasoning",
        "prereqs": ["object_search", "appliance_use"],
        "description": "Decompose a household goal into a sequence of navigation, search, interaction, and appliance steps",
    },
}

# ALFWorld task type IDs → names
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


class ALFWorldEnvironment(SequentialEnvironment):
    """ALFWorld text-based household environment for MAGE.

    Uses the TextWorld backend (no vision). Each episode is a household
    task like "put a clean mug on the shelf". The agent gets text
    observations and an admissible-commands list.
    """

    def __init__(
        self,
        split: str = "eval_out_of_distribution",
        max_steps: int = 50,
        seed: int = 42,
    ):
        self._split = split
        self._max_steps = max_steps
        self._seed = seed
        self._env = None
        self._tw_env = None  # TextWorld sub-env
        self._current_obs: str = ""
        self._current_infos: dict = {}
        self._game_idx = 0
        self._loaded = False

    def _load_env(self) -> None:
        if self._loaded:
            return

        os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
        config_path = os.path.expanduser("~/.alfworld/base_config.yaml")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Set required config fields for non-RL usage
        config.setdefault("general", {})["training_method"] = "dqn"
        config.setdefault("rl", {}).setdefault("training", {})["max_nb_steps_per_episode"] = self._max_steps
        config["env"].setdefault("expert_type", "handcoded")

        from alfworld.agents.environment import get_environment
        AlfredTWEnv = get_environment("AlfredTWEnv")

        train_eval_map = {
            "train": "train",
            "eval_in_distribution": "eval_in_distribution",
            "eval_out_of_distribution": "eval_out_of_distribution",
        }
        self._alfred_env = AlfredTWEnv(config, train_eval=train_eval_map.get(self._split, self._split))
        # init_env returns a Gym env; reset/step are on THAT object
        self._tw_env = self._alfred_env.init_env(batch_size=1)
        self._loaded = True
        logger.info(f"Loaded ALFWorld ({self._split}, {len(self._alfred_env.game_files)} games)")

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def domain(self) -> str:
        return "household"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in ALFWORLD_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        """Admissible commands change per step; return generic action types."""
        return [
            "go to", "take", "put", "open", "close",
            "examine", "use", "clean", "heat", "cool",
            "toggle", "look", "inventory",
        ]

    def get_evaluation_instances(
        self, skill_id: str | None = None, num_instances: int = 10,
    ) -> list[EvaluationInstance]:
        self._load_env()

        instances = []
        n = min(num_instances, len(self._env.game_files))
        for i in range(n):
            game_file = self._env.game_files[i]
            # Extract task type from path
            task_type = "unknown"
            for tid, tname in TASK_TYPES.items():
                if tname in str(game_file):
                    task_type = tname
                    break

            instances.append(EvaluationInstance(
                instance_id=f"alfworld_{i}",
                instruction=f"Complete the household task (game {i})",
                target_skill="task_planning",
                seed=self._seed + i,
                metadata={"task_type": task_type, "game_idx": i},
            ))

        return instances

    # -- SequentialEnvironment interface --

    def reset(self, seed: int | None = None, **kwargs: Any) -> str:
        """Reset to a new game. Returns the initial text observation."""
        self._load_env()

        obs, infos = self._tw_env.reset()
        self._current_obs = obs[0] if isinstance(obs, list) else str(obs)
        self._current_infos = {
            k: (v[0] if isinstance(v, list) and v else v)
            for k, v in infos.items()
        }
        return self._current_obs

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute one action. Returns (observation, reward, done, info)."""
        result = self._tw_env.step([action])
        # TextWorld returns (obs, scores, dones, infos) but the exact
        # nesting varies by version. Handle both tuple and list forms.
        if len(result) == 4:
            obs_list, scores, dones, infos = result
        elif len(result) == 2:
            # Some versions return (obs, info_dict)
            obs_list, infos = result
            scores = [0.0]
            dones = [False]
        else:
            obs_list = result[0]
            scores = [0.0]
            dones = [False]
            infos = {}

        obs = obs_list[0] if isinstance(obs_list, (list, tuple)) else str(obs_list)
        # scores can be list, tuple of tuples, or scalar
        try:
            if isinstance(scores, (list, tuple)):
                s = scores[0]
                score = float(s[0]) if isinstance(s, (list, tuple)) else float(s)
            else:
                score = float(scores)
        except (TypeError, IndexError):
            score = 0.0
        try:
            if isinstance(dones, (list, tuple)):
                d = dones[0]
                done = bool(d[0]) if isinstance(d, (list, tuple)) else bool(d)
            else:
                done = bool(dones)
        except (TypeError, IndexError):
            done = False

        # Extract admissible commands for the next step
        admissible = []
        if "admissible_commands" in infos:
            cmds = infos["admissible_commands"]
            admissible = cmds[0] if isinstance(cmds, list) and cmds else []

        won = False
        if "won" in infos:
            w = infos["won"]
            won = bool(w[0]) if isinstance(w, list) else bool(w)

        info = {
            "admissible_commands": admissible,
            "won": won,
            "score": score,
        }

        self._current_obs = obs
        self._current_infos = info
        return obs, score, done, info

    def get_achievements(self) -> dict[str, bool]:
        """Return whether the current task was completed."""
        won = self._current_infos.get("won", False)
        return {"task_completed": won}

    def evaluate_answer(self, predicted: str, expected: str, **kwargs) -> float:
        """For ALFWorld, success is binary: did the agent complete the task?"""
        # This is called per-episode by the critic; `predicted` contains
        # the episode result. Check if the episode was won.
        if predicted.lower() in ("true", "1", "yes", "won", "success"):
            return 1.0
        return 0.0

    def build_evokg(self) -> EvoKG:
        return build_alfworld_evokg()


def build_alfworld_evokg() -> EvoKG:
    """Build an EvoKG for ALFWorld with household skill prerequisites."""
    evokg = EvoKG()

    for sid, sdef in ALFWORLD_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="household",
            description=sdef["description"],
            metadata={"category": sdef["category"]},
        ))

    for sid, sdef in ALFWORLD_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    for entity in ["room", "object", "receptacle", "appliance"]:
        evokg.add_node(EnvironmentNode(
            f"env_{entity}", entity.title(),
            entity_type="household_element", domain="household",
        ))

    evokg.add_node(TaskNode(
        "task_alfworld", "Complete household tasks (put, clean, heat, cool, examine)",
        difficulty=0.7, domain="household",
    ))

    logger.info(f"Built ALFWorld EvoKG: {evokg}")
    return evokg
