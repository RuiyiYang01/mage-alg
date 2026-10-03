"""Crafter environment adapter for MAGE.

Converts image-based observations to text format (following EXIF/BALROG approach)
and provides skill detection via the 22 built-in achievements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mage.core.evokg import EvoKG
from mage.environments.base import SequentialEnvironment
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    RelationType,
    TaskNode,
)

logger = logging.getLogger(__name__)

# All 22 Crafter achievements — these become capability nodes in EvoKG
CRAFTER_SKILLS = {
    # Harvest
    "collect_sapling": {"name": "Collect Sapling", "category": "harvest", "prereqs": [],
        "how": "Walk over grass tiles; saplings appear randomly when stepping on grass. Action: move toward grass, then `do` if a sapling is in front."},
    "place_plant": {"name": "Place Plant", "category": "harvest", "prereqs": ["collect_sapling"],
        "how": "Stand on grass with a sapling in inventory; face an empty grass tile. Action: `place_plant`."},
    "eat_plant": {"name": "Eat Plant", "category": "harvest", "prereqs": ["place_plant"],
        "how": "Wait for a placed plant to mature into a fruit, then face it and `do` to eat. Restores food."},
    # Status
    "wake_up": {"name": "Wake Up", "category": "status", "prereqs": [],
        "how": "Use `sleep` action when energy < 9 in a safe spot (no nearby enemies). Wakes up automatically when energy is full or daylight."},
    "eat_cow": {"name": "Eat Cow", "category": "status", "prereqs": [],
        "how": "Find a cow (visible in observation), face it, then `do` to attack and eat. Restores food. Cows roam grass."},
    "collect_drink": {"name": "Collect Drink", "category": "status", "prereqs": [],
        "how": "Find water tiles (lake/river), face the water tile, then `do` to drink. Restores drink stat."},
    # Wood
    "collect_wood": {"name": "Collect Wood", "category": "wood", "prereqs": [],
        "how": "Find a tree tile in the observation, move adjacent and face it, then `do` to chop. May need 1-3 `do` actions to fully harvest."},
    "place_table": {"name": "Place Table", "category": "wood", "prereqs": ["collect_wood"],
        "how": "Need at least 1 wood in inventory. Face an empty grass/dirt tile. Action: `place_table`. Required prereq for crafting."},
    "make_wood_pickaxe": {"name": "Make Wood Pickaxe", "category": "wood", "prereqs": ["collect_wood", "place_table"],
        "how": "Stand adjacent to a placed table with at least 1 wood in inventory. Action: `make_wood_pickaxe`. Needed to mine stone."},
    "make_wood_sword": {"name": "Make Wood Sword", "category": "wood", "prereqs": ["collect_wood", "place_table"],
        "how": "Stand adjacent to a placed table with at least 1 wood in inventory. Action: `make_wood_sword`. Helps defeat zombies."},
    # Stone
    "collect_stone": {"name": "Collect Stone", "category": "stone", "prereqs": ["make_wood_pickaxe"],
        "how": "Find a stone tile (gray rock), face it with a wood pickaxe in inventory, then `do` to mine."},
    "make_stone_pickaxe": {"name": "Make Stone Pickaxe", "category": "stone", "prereqs": ["collect_wood", "collect_stone", "place_table"],
        "how": "Stand adjacent to a table with wood + stone in inventory. Action: `make_stone_pickaxe`. Needed to mine iron and coal."},
    "make_stone_sword": {"name": "Make Stone Sword", "category": "stone", "prereqs": ["collect_wood", "collect_stone", "place_table"],
        "how": "Stand adjacent to a table with wood + stone in inventory. Action: `make_stone_sword`. Helps defeat skeletons."},
    "place_stone": {"name": "Place Stone", "category": "stone", "prereqs": ["collect_stone"],
        "how": "Need stone in inventory; face an empty tile. Action: `place_stone`. Useful for building defenses."},
    # Iron
    "collect_coal": {"name": "Collect Coal", "category": "iron", "prereqs": ["make_wood_pickaxe"],
        "how": "Find a coal tile (dark spots in stone areas), face it with a wood pickaxe, then `do` to mine."},
    "place_furnace": {"name": "Place Furnace", "category": "iron", "prereqs": ["collect_stone"],
        "how": "Need stone in inventory; face an empty tile. Action: `place_furnace`. Required to smelt iron."},
    "collect_iron": {"name": "Collect Iron", "category": "iron", "prereqs": ["make_stone_pickaxe"],
        "how": "Find an iron tile (orange specks in stone), face it with a stone pickaxe, then `do` to mine."},
    "make_iron_pickaxe": {"name": "Make Iron Pickaxe", "category": "iron", "prereqs": ["collect_wood", "collect_coal", "collect_iron", "place_table", "place_furnace"],
        "how": "Stand adjacent to BOTH a table and a furnace with wood + coal + iron in inventory. Action: `make_iron_pickaxe`. Required to mine diamond."},
    "make_iron_sword": {"name": "Make Iron Sword", "category": "iron", "prereqs": ["collect_wood", "collect_coal", "collect_iron", "place_table", "place_furnace"],
        "how": "Stand adjacent to BOTH a table and a furnace with wood + coal + iron in inventory. Action: `make_iron_sword`. Strongest weapon."},
    "collect_diamond": {"name": "Collect Diamond", "category": "iron", "prereqs": ["make_iron_pickaxe"],
        "how": "Find a diamond tile (light blue gem in stone areas), face it with an iron pickaxe in inventory, then `do` to mine."},
    # Hunt
    "defeat_skeleton": {"name": "Defeat Skeleton", "category": "hunt", "prereqs": [],
        "how": "Skeletons spawn at night and shoot arrows. Approach with any sword (stone/iron preferred), face the skeleton, then `do` to attack until defeated. Avoid arrows by moving."},
    "defeat_zombie": {"name": "Defeat Zombie", "category": "hunt", "prereqs": [],
        "how": "Zombies spawn at night and chase you. Equip a sword (wood works), face the zombie, then `do` to attack until defeated."},
}

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword",
]

# Semantic tile types from crafter's semantic map
SEMANTIC_NAMES = [
    "invalid", "water", "grass", "stone", "path", "sand",
    "tree", "lava", "coal", "iron", "diamond",
    "table", "furnace",
    "player", "cow", "zombie", "skeleton",
    "arrow", "plant",
]


@dataclass
class CrafterObservation:
    """Structured text observation from Crafter."""
    health: int = 9
    food: int = 9
    drink: int = 9
    energy: int = 9
    inventory: dict[str, int] = field(default_factory=dict)
    nearby_entities: list[str] = field(default_factory=list)
    facing: str = "grass"
    player_pos: tuple[int, int] = (0, 0)

    def to_text(self) -> str:
        lines = ["### Current Observation"]
        lines.append("Your status:")
        lines.append(f"- health: {self.health}/9")
        lines.append(f"- food: {self.food}/9")
        lines.append(f"- drink: {self.drink}/9")
        lines.append(f"- energy: {self.energy}/9")
        lines.append("")

        if self.inventory:
            lines.append("Your inventory:")
            for item, count in self.inventory.items():
                if count > 0:
                    lines.append(f"- {item}: {count}")
        else:
            lines.append("Your inventory: empty")
        lines.append("")

        if self.nearby_entities:
            lines.append("You see:")
            for entity in self.nearby_entities:
                lines.append(f"- {entity}")
        else:
            lines.append("You see: nothing nearby")
        lines.append("")

        lines.append(f"You are facing {self.facing}")
        return "\n".join(lines)


class CrafterAdapter(SequentialEnvironment):
    """Crafter environment implementing the MAGE SequentialEnvironment interface.

    Converts image observations to text format and tracks achievements.
    """

    def __init__(self, seed: int | None = None, max_steps: int = 100):
        import crafter
        self.env = crafter.Env()
        self.seed = seed
        self.max_steps = max_steps
        self._obs = None
        self._info: dict[str, Any] = {}
        self._step_count = 0
        self._cumulative_achievements: dict[str, int] = {}
        self._use_skeleton = False

    # -- BaseEnvironment interface --

    @property
    def name(self) -> str:
        return "crafter"

    @property
    def domain(self) -> str:
        return "crafter"

    def get_skill_definitions(self) -> list:
        from mage.environments.base import SkillDefinition
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=f"Crafter achievement: {sdef['name']}",
                metadata={"category": sdef["category"]},
            )
            for sid, sdef in CRAFTER_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        return list(ACTION_NAMES)

    def get_evaluation_instances(self, skill_id=None, num_instances=10):
        from mage.environments.base import EvaluationInstance
        skills = [skill_id] if skill_id else list(CRAFTER_SKILLS.keys())
        instances = []
        for sid in skills:
            sdef = CRAFTER_SKILLS.get(sid, {})
            for i in range(num_instances):
                instances.append(EvaluationInstance(
                    instance_id=f"{sid}_trial_{i}",
                    instruction=f"Perform skill: {sdef.get('name', sid)}",
                    target_skill=sid,
                    seed=42 + i,
                ))
        return instances

    # v15.6 — BALROG-style evaluation flag. When True, the critic runs
    # ONE long episode (max_steps steps) per trial and counts accumulated
    # achievements instead of running 22 separate per-skill trials.
    # This matches BALROG's methodology where Llama-3.1-8B scores 25%.
    USE_EPISODE_EVAL = True

    def evaluate_full_episode(
        self, agent_fn: Any, max_steps: int = 1000, seed: int = 42,
    ) -> dict[str, Any]:
        """BALROG-style Crafter evaluation: run ONE long episode, count
        all achievements unlocked.

        Returns dict with 'achievements' (which skills unlocked),
        'n_achievements' (count), 'score' (n/22), 'steps_taken'.
        """
        obs = self.reset(seed=seed)
        action_space = self.get_action_space()
        # v20.B — reset per-episode action-recipe collection
        self._episode_recipes: list[dict] = []

        # BALROG-style prompt: show action descriptions AND the 22 achievements
        # so the model knows what it's reaching for. Without the achievement
        # list the model has no idea Diamond/Iron Pickaxe are goals.
        action_descriptions = {
            "noop": "do nothing",
            "move_left": "move west on flat ground",
            "move_right": "move east on flat ground",
            "move_up": "move north on flat ground",
            "move_down": "move south on flat ground",
            "do": "multi-use action to collect material, drink, or hit creature in front",
            "sleep": "sleep when energy level is below maximum",
            "place_stone": "place a stone in front",
            "place_table": "place a crafting table",
            "place_furnace": "place a furnace",
            "place_plant": "place a sapling as a plant",
            "make_wood_pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
            "make_stone_pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone",
            "make_iron_pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron",
            "make_wood_sword": "craft a wood sword with a nearby table and wood in inventory",
            "make_stone_sword": "craft a stone sword with a nearby table, wood, and stone",
            "make_iron_sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron",
        }
        action_list_lines = "\n".join(
            f"- {a}: {action_descriptions.get(a, '')}" for a in action_space
        )

        achievements_prompt_list = (
            "1. Collect Wood\n2. Place Table\n3. Eat Cow\n4. Collect Sapling\n"
            "5. Collect Drink\n6. Make Wood Pickaxe\n7. Make Wood Sword\n"
            "8. Place Plant\n9. Defeat Zombie\n10. Collect Stone\n"
            "11. Place Stone\n12. Eat Plant\n13. Defeat Skeleton\n"
            "14. Make Stone Pickaxe\n15. Make Stone Sword\n16. Wake Up\n"
            "17. Place Furnace\n18. Collect Coal\n19. Collect Iron\n"
            "20. Make Iron Pickaxe\n21. Make Iron Sword\n22. Collect Diamond"
        )

        system_prompt = (
            "You are an agent playing Crafter, a 2D open-world survival game. "
            "The following are the only valid actions you can take, with a short description of each:\n\n"
            f"{action_list_lines}\n\n"
            "These are the game achievements you can get:\n"
            f"{achievements_prompt_list}\n\n"
            "Your goal is to unlock as many achievements as possible. "
            "Think briefly about the current situation and what achievement you can make progress on next, "
            "then output the single action to take. End your response with the exact action name on its own line."
        )

        from mage.agents.critic import CrafterCritic

        history: list[tuple[str, str, str]] = []  # (obs_summary, action, outcome)
        HISTORY_K = 16  # v16 A3: match BALROG's max_text_history=16 (was 8).
        # Adds ~1.2K chars to prompt; still comfortably in 16K context window.
        recent_actions: list[str] = []

        def _obs_summary(text: str, max_chars: int = 800) -> str:
            return text if len(text) <= max_chars else text[:max_chars] + "..."

        def _build_tech_tree_block() -> str:
            """Render the KG's prerequisite graph as a per-step status board.

            Reads the dependency edges from CRAFTER_SKILLS (== EvoKG's
            PREREQUISITE_OF edges) and tags every achievement as done /
            next-achievable / locked. Gives the agent an explicit curriculum
            signal at every turn — the same kind a human player keeps in
            their head when climbing the tech tree.
            """
            unlocked_now = {
                k for k, v in self._cumulative_achievements.items() if v
            }
            done_lines: list[str] = []
            next_lines: list[str] = []
            locked_lines: list[str] = []
            for skill_id, sdef in CRAFTER_SKILLS.items():
                name = sdef.get("name", skill_id)
                prereqs = sdef.get("prereqs", []) or []
                if skill_id in unlocked_now:
                    done_lines.append(f"  [DONE] {name}")
                    continue
                missing = [p for p in prereqs if p not in unlocked_now]
                if not missing:
                    next_lines.append(f"  [NEXT] {name}")
                else:
                    pretty_missing = ", ".join(
                        CRAFTER_SKILLS.get(p, {}).get("name", p) for p in missing
                    )
                    locked_lines.append(f"  [LOCKED] {name} (need: {pretty_missing})")
            block = ["TECH-TREE PROGRESS:"]
            if done_lines:
                block.append("Already unlocked:")
                block.extend(done_lines)
            if next_lines:
                block.append("Currently achievable (focus here next):")
                block.extend(next_lines)
            if locked_lines:
                # Limit to first 8 locked entries so the prompt stays compact
                block.append("Locked behind prerequisites (a few examples):")
                block.extend(locked_lines[:8])
            return "\n".join(block)

        for step in range(max_steps):
            # Build a prompt that includes (a) the system/goal info, (b) the
            # tech-tree status (what's done / what's next from the KG), (c) the
            # recent (obs,action,outcome) history, (d) the current observation.
            tech_tree_block = _build_tech_tree_block()
            history_block = ""
            if history:
                history_lines = []
                for i, (h_obs, h_act, h_outcome) in enumerate(history[-HISTORY_K:]):
                    history_lines.append(
                        f"[{max(0, step - HISTORY_K) + i}] obs: {h_obs}\n"
                        f"     action: {h_act}  → {h_outcome}"
                    )
                history_block = "Recent history (most recent last):\n" + "\n".join(history_lines) + "\n\n"

            instruction = (
                f"{system_prompt}\n\n"
                f"{tech_tree_block}\n\n"
                f"{history_block}"
                f"Current observation:\n{_obs_summary(obs)}\n\n"
                f"Pick ONE achievement under [NEXT] to focus on this turn, "
                f"then output the action that makes progress on it. "
                f"Your response must end with the action name on its own line."
            )

            response = agent_fn(obs, instruction)
            action = CrafterCritic._extract_action(response, action_space)

            # Anti-loop safety: with deterministic temperature=0.0, identical
            # observations produce identical actions, so true stuck loops can
            # last forever (agent dies of starvation/zombies). When ANY action
            # repeats 3+ times in a row, nudge with a different action — this
            # catches both stuck navigation (move_left at a wall) and stuck
            # invalid actions (do at empty air, make_wood_pickaxe without
            # ingredients). v5 attempted to exempt `do`/craft from override
            # but that let invalid-action loops kill the agent in <40 steps.
            # The 3-in-a-row threshold is permissive enough for the rare
            # legitimate `do`×3 chopping sequence.
            if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
                import random
                alt = [a for a in action_space if a != action and a != "noop"]
                if alt:
                    action = random.choice(alt)

            prev_cum = dict(self._cumulative_achievements)
            prev_obs_summary = _obs_summary(obs, max_chars=300)
            obs, reward, done, info = self.step(action)
            new_achs = [
                k for k, v in self._cumulative_achievements.items()
                if v and not prev_cum.get(k, 0)
            ]
            outcome = f"unlocked {', '.join(new_achs)}" if new_achs else "no change"
            recent_actions.append(action)
            history.append((prev_obs_summary, action, outcome))
            # v20.B — capture action recipe for each achievement unlock.
            # v20.B-fix: K=3 (down from 8). For Crafter, longer recipes
            # capture map-specific NAVIGATION steps (move_left → move_up …
            # to reach THIS map's tree) that don't transfer to new maps.
            # K=3 keeps the achievement-triggering action + 2 setup actions
            # → captures the CRAFTING / INTERACTION pattern without map noise.
            # Only store recipes for COMPOUND skills (skills with prereqs);
            # for atomic skills (do, place_X) the action name itself IS the
            # recipe and storing it adds nothing.
            if new_achs:
                if not hasattr(self, "_episode_recipes"):
                    self._episode_recipes = []
                K_RECIPE = 3
                recipe_actions = recent_actions[-K_RECIPE:]
                COMPOUND = {  # achievements that need a chain of preconditions
                    "place_table", "make_wood_pickaxe", "make_wood_sword",
                    "make_stone_pickaxe", "make_stone_sword",
                    "make_iron_pickaxe", "make_iron_sword",
                    "place_furnace", "collect_stone", "collect_coal",
                    "collect_iron", "collect_diamond", "eat_cow", "eat_plant",
                    "defeat_zombie", "defeat_skeleton",
                }
                for ach in new_achs:
                    if ach not in COMPOUND:
                        continue  # skip atomic skills
                    self._episode_recipes.append({
                        "skill_id": ach,
                        "actions": list(recipe_actions),
                        "ctx": prev_obs_summary[:300],
                    })
            if done:
                break

        # Count accumulated achievements
        unlocked = {k: v for k, v in self._cumulative_achievements.items() if v}
        return {
            "achievements": unlocked,
            "n_achievements": len(unlocked),
            "score": len(unlocked) / len(CRAFTER_SKILLS),
            "steps_taken": self._step_count,
            "total_skills": len(CRAFTER_SKILLS),
            "action_recipes": getattr(self, "_episode_recipes", []),
        }

    def get_domain_context(self) -> str:
        """Domain context fed to Claude during EVOLVE so it writes principles
        in the right register. Without this, Claude misreads skill names like
        'Place Plant' as roleplay/dialog tasks and writes useless principles
        like 'respond with a direct confirmation of action execution'.
        """
        return (
            "DOMAIN: Crafter — a 2D Minecraft-inspired survival game.\n"
            "The agent observes a text rendering of a small grid (player at "
            "center, surrounded by terrain, items, animals, monsters) and an "
            "inventory/status block. EACH TURN the agent picks ONE of these "
            "discrete actions:\n"
            "  noop, move_left, move_right, move_up, move_down,\n"
            "  do (interact with the tile in front: collect material, drink, "
            "attack creature),\n"
            "  sleep (restore energy),\n"
            "  place_stone, place_table, place_furnace, place_plant,\n"
            "  make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe,\n"
            "  make_wood_sword, make_stone_sword, make_iron_sword.\n\n"
            "Scoring: 22 achievements (collect_wood, place_table, eat_cow, "
            "make_wood_pickaxe, collect_stone, ..., collect_diamond) — the "
            "agent's score is the fraction unlocked per episode.\n\n"
            "Achievements have a tech-tree dependency: collect_wood → "
            "place_table → make_wood_pickaxe → collect_stone → "
            "make_stone_pickaxe → collect_iron + collect_coal + "
            "place_furnace → make_iron_pickaxe → collect_diamond. The agent "
            "must ALSO survive (eat cows/plants when hungry, drink from water, "
            "sleep when energy is low, kill or avoid zombies/skeletons).\n\n"
            "Principles must be CONCRETE GAME TACTICS — when to use which "
            "action, what to face before 'do', how to recover from low "
            "stats, what tech-tree step to chase next. Principles must NOT "
            "be roleplay/dialog advice ('respond with a confirmation', "
            "'enact the action') — those are useless because the agent "
            "outputs ONE discrete action per turn, not natural language."
        )

    def build_evokg(self) -> EvoKG:
        return build_crafter_evokg(use_skeleton=self._use_skeleton)

    # -- SequentialEnvironment interface --

    def reset(self, seed: int | None = None, **kwargs: Any) -> str:
        """Reset the environment and return initial text observation."""
        actual_seed = seed if seed is not None else self.seed
        if actual_seed is not None:
            self.env = __import__("crafter").Env(seed=actual_seed)

        self._obs = self.env.reset()
        # Take a noop to get info
        self._obs, _, _, self._info = self.env.step(0)
        self._step_count = 0
        self._cumulative_achievements = dict(self._info.get("achievements", {}))
        return self._build_observation().to_text()

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Take an action and return (observation_text, reward, done, info)."""
        action_idx = ACTION_NAMES.index(action) if action in ACTION_NAMES else 0

        self._obs, reward, done, self._info = self.env.step(action_idx)
        self._step_count += 1

        # Track new achievements
        new_achievements = {}
        current = self._info.get("achievements", {})
        for skill_id, achieved in current.items():
            if achieved and not self._cumulative_achievements.get(skill_id, 0):
                new_achievements[skill_id] = 1
        self._cumulative_achievements.update(current)

        if self._step_count >= self.max_steps:
            done = True

        info = {
            "achievements": dict(current),
            "new_achievements": new_achievements,
            "step": self._step_count,
            "inventory": dict(self._info.get("inventory", {})),
        }

        return self._build_observation().to_text(), float(reward), done, info

    def _build_observation(self) -> CrafterObservation:
        """Convert raw crafter info to structured text observation."""
        info = self._info
        inv = info.get("inventory", {})
        semantic = info.get("semantic", None)

        # Parse inventory
        inventory = {}
        for item, count in inv.items():
            if count > 0:
                inventory[item] = int(count)

        # Parse nearby entities from semantic map — v16 A2: widened to full view
        # (was limited to 5x5 → missed distant cows/stone/coal/iron) and now
        # uses the actual player_facing from info (was hardcoded "up" → caused
        # wrong facing reports). Phrasing matches BALROG: "wood 2 steps north
        # and 3 steps west" → easier for the LLM to parse than "SW".
        nearby = []
        if semantic is not None:
            semantic = np.array(semantic)
            center = semantic.shape[0] // 2, semantic.shape[1] // 2
            # Scan the FULL view (typically 9x7 in Crafter), not a 5x5 subset
            for dy in range(-center[0], semantic.shape[0] - center[0]):
                for dx in range(-center[1], semantic.shape[1] - center[1]):
                    if dy == 0 and dx == 0:
                        continue
                    y, x = center[0] + dy, center[1] + dx
                    if 0 <= y < semantic.shape[0] and 0 <= x < semantic.shape[1]:
                        tile_id = int(semantic[y, x])
                        if 0 < tile_id < len(SEMANTIC_NAMES):
                            tile_name = SEMANTIC_NAMES[tile_id]
                            if tile_name not in ("grass", "path", "sand", "invalid"):
                                # BALROG-style precise two-axis phrasing
                                parts: list[str] = []
                                if dy < 0:
                                    parts.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} north")
                                elif dy > 0:
                                    parts.append(f"{dy} step{'s' if dy > 1 else ''} south")
                                if dx < 0:
                                    parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} west")
                                elif dx > 0:
                                    parts.append(f"{dx} step{'s' if dx > 1 else ''} east")
                                loc = " and ".join(parts) if parts else "here"
                                nearby.append(f"{tile_name} {loc}")

            # Get facing tile via actual player_facing (info["player_facing"])
            # which is a (dy, dx) unit vector like (-1, 0) = north. Fallback to
            # "north" if info doesn't provide it.
            pf = info.get("player_facing", (-1, 0))
            try:
                pf_dy, pf_dx = int(pf[0]), int(pf[1])
            except (TypeError, ValueError, IndexError):
                pf_dy, pf_dx = -1, 0
            facing_y, facing_x = center[0] + pf_dy, center[1] + pf_dx
            facing = "grass"
            if 0 <= facing_y < semantic.shape[0] and 0 <= facing_x < semantic.shape[1]:
                fid = int(semantic[facing_y, facing_x])
                if 0 < fid < len(SEMANTIC_NAMES):
                    facing = SEMANTIC_NAMES[fid]
        else:
            facing = "unknown"

        # Sort by manhattan distance so the closest/most relevant tiles appear
        # first; keep the top 10 closest to cover useful objects without
        # overwhelming Llama-3.1-8B with noise (v16 first pass used top-20
        # and regressed by 3pp vs v6's implicit 5x5 scan).
        def _parse_dist(entry: str) -> int:
            import re as _re
            nums = _re.findall(r"\d+", entry)
            return sum(int(n) for n in nums[:2]) if nums else 99
        nearby = list(dict.fromkeys(nearby))
        nearby.sort(key=_parse_dist)
        nearby = nearby[:10]

        player_pos = tuple(info.get("player_pos", (0, 0)))

        return CrafterObservation(
            health=int(inv.get("health", 9)),
            food=int(inv.get("food", 9)),
            drink=int(inv.get("drink", 9)),
            energy=int(inv.get("energy", 9)),
            inventory={k: v for k, v in inventory.items() if k not in ("health", "food", "drink", "energy")},
            nearby_entities=nearby,
            facing=facing,
            player_pos=player_pos,
        )

    @staticmethod
    def _get_direction(dy: int, dx: int) -> str:
        if dy < 0 and dx == 0:
            return "north"
        elif dy > 0 and dx == 0:
            return "south"
        elif dy == 0 and dx < 0:
            return "west"
        elif dy == 0 and dx > 0:
            return "east"
        elif dy < 0 and dx < 0:
            return "north-west"
        elif dy < 0 and dx > 0:
            return "north-east"
        elif dy > 0 and dx < 0:
            return "south-west"
        else:
            return "south-east"

    def get_achievements(self) -> dict[str, int]:
        return dict(self._cumulative_achievements)

    def get_inference_hints(self, obs_text: str) -> dict[str, Any]:
        """v21 — Critical-priority hints derived from the CURRENT observation.

        Called by the learner's `_act_sequential` path to inject state-aware
        preemption BEFORE principles/recipes. Three categories:

        1. **Survival needs** (v21.D): if food/drink/energy/health below a
           threshold, return a high-priority "fix this now" instruction.
        2. **Prereq guardrails** (v21.B): parse inventory and return which
           craft actions are currently AVAILABLE vs BLOCKED.
        3. **Next-skill nudge** (v21.X): identify the next craftable skill
           given the inventory, so the agent doesn't waste steps trying
           skills whose prereqs aren't met.

        Returns a dict with keys: ``critical`` (str|None), ``available_crafts``
        (list[str]), ``blocked_crafts`` (dict[str, str]),
        ``next_craftable`` (str|None).
        """
        import re as _re

        # Parse health/food/drink/energy from obs text
        stats: dict[str, int] = {}
        for field in ("health", "food", "drink", "energy"):
            m = _re.search(rf"{field}\s*[:=]\s*(\d+)", obs_text, _re.IGNORECASE)
            if m:
                stats[field] = int(m.group(1))

        # Parse inventory — scan the chunk after "Inventory" keyword
        inventory: dict[str, int] = {}
        idx = obs_text.lower().find("inventory")
        if idx >= 0:
            # Read until next section marker (Facing/Nearby/Recent/etc.) or 600 chars
            chunk = obs_text[idx: idx + 600]
            cut = len(chunk)
            for marker in ("\nfacing", "\nnearby", "\nrecent", "\ncurrent",
                           "\ntech", "\n[next]", "\n["):
                p = chunk.lower().find(marker, len("inventory"))
                if 0 < p < cut:
                    cut = p
            chunk = chunk[:cut]
            for m in _re.finditer(r"(\w+)\s*[:=]\s*(\d+)", chunk):
                k = m.group(1).lower()
                if k in ("health", "food", "drink", "energy", "inventory"):
                    continue
                inventory[k] = int(m.group(2))

        # Crafting requirements (from Crafter source)
        CRAFT_REQS = {
            "place_table":        {"wood": 1},
            "place_furnace":      {"stone": 1},
            "place_plant":        {"sapling": 1},
            "place_stone":        {"stone": 1},
            "make_wood_pickaxe":  {"wood": 1},   # + near crafting_table
            "make_wood_sword":    {"wood": 1},
            "make_stone_pickaxe": {"wood": 1, "stone": 1},
            "make_stone_sword":   {"wood": 1, "stone": 1},
            "make_iron_pickaxe":  {"wood": 1, "coal": 1, "iron": 1},
            "make_iron_sword":    {"wood": 1, "coal": 1, "iron": 1},
        }
        available: list[str] = []
        blocked: dict[str, str] = {}
        for act, reqs in CRAFT_REQS.items():
            missing = [f"{c} {k}" for k, c in reqs.items() if inventory.get(k, 0) < c]
            if missing:
                blocked[act] = " and ".join(missing)
            else:
                available.append(act)

        # Next-craftable-uphill: most complex available craft
        tier_order = [
            "make_iron_sword", "make_iron_pickaxe",
            "make_stone_sword", "make_stone_pickaxe",
            "place_furnace",
            "make_wood_sword", "make_wood_pickaxe",
            "place_table",
        ]
        next_craftable: str | None = None
        for c in tier_order:
            if c in available:
                next_craftable = c
                break

        # Critical survival need — set if ANY stat is dangerously low
        critical: str | None = None
        if stats.get("food", 9) <= 3:
            critical = (
                "🚨 FOOD is low ({}). Immediate action: eat_cow (if cow nearby) "
                "or place_plant → wait → eat_plant. Food depletion → starvation."
            ).format(stats.get("food", "?"))
        elif stats.get("drink", 9) <= 3:
            critical = (
                "🚨 DRINK is low ({}). Immediate action: find water tile, "
                "face it, `do` to collect_drink. Thirst → death."
            ).format(stats.get("drink", "?"))
        elif stats.get("energy", 9) <= 3:
            critical = (
                "🚨 ENERGY is low ({}). Use `sleep` in a safe spot away from "
                "zombies. Ignoring energy → agent slows and then dies."
            ).format(stats.get("energy", "?"))
        elif stats.get("health", 9) <= 3:
            critical = (
                "🚨 HEALTH is low ({}). Move AWAY from any visible zombie/"
                "skeleton. Place_stone to block threats, regen by resting."
            ).format(stats.get("health", "?"))

        return {
            "critical": critical,
            "available_crafts": available,
            "blocked_crafts": blocked,
            "next_craftable": next_craftable,
            "stats": stats,
            "inventory": inventory,
        }

    @property
    def action_names(self) -> list[str]:
        return list(ACTION_NAMES)

    @property
    def skill_ids(self) -> list[str]:
        return list(CRAFTER_SKILLS.keys())


def build_crafter_evokg(use_skeleton: bool = False) -> EvoKG:
    """Build an EvoKG seeded with Crafter's 22 skills and prerequisite graph.

    Args:
        use_skeleton: If True, use domain-agnostic skeleton only (cold-start style).
                      If False, seed with full Crafter skill tree (default for main experiments).
    """
    if use_skeleton:
        return EvoKG.create_with_skeleton(domain="crafter")

    evokg = EvoKG()

    # Add all 22 skills as capability nodes
    for skill_id, skill_def in CRAFTER_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=skill_id,
            name=skill_def["name"],
            mastery=0.0,
            domain="crafter",
            description=(
                f"Crafter achievement '{skill_def['name']}' (category: {skill_def['category']}). "
                f"How to unlock: {skill_def.get('how', '')}"
            ),
            metadata={"category": skill_def["category"]},
        ))

    # Add prerequisite edges
    for skill_id, skill_def in CRAFTER_SKILLS.items():
        for prereq in skill_def["prereqs"]:
            evokg.add_edge(Edge(prereq, skill_id, RelationType.PREREQUISITE_OF))

    # Add environment nodes
    for tile in ["water", "grass", "stone", "tree", "coal", "iron", "diamond",
                 "table", "furnace", "cow", "zombie", "skeleton"]:
        evokg.add_node(EnvironmentNode(
            node_id=f"env_{tile}",
            name=tile.title(),
            entity_type="object" if tile in ("table", "furnace") else
                        "creature" if tile in ("cow", "zombie", "skeleton") else
                        "resource",
            domain="crafter",
        ))

    # Add top-level task
    evokg.add_node(TaskNode(
        "task_master_crafter", "Master all Crafter skills",
        difficulty=1.0, domain="crafter",
    ))

    logger.info(f"Built Crafter EvoKG: {evokg}")
    return evokg
