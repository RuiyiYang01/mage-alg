"""Explorer Agent: Collects trajectories by interacting with environments.

Guided by the evolution plan from Navigator and past experience from EvoKG.
Generates (instruction, trajectory) pairs for training the Learner.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from mage.agents.base import ExplorerAgent, SkillDataset, Trajectory
from mage.core.evokg import EvoKG
from mage.core.schemas import EvolutionPlan
from mage.environments.base import BaseEnvironment, SequentialEnvironment
from mage.utils.llm import LLMBackend

logger = logging.getLogger(__name__)

EXPLORER_SYSTEM_PROMPT = """\
You are an agent navigating and surviving in the Crafter game world. \
Your goal is to perform the given task by choosing the right actions.

### Valid Actions
- move_left: move one tile west
- move_right: move one tile east
- move_up: move one tile north
- move_down: move one tile south
- do: interact with the tile in front (collect material, drink water, attack creature, hunt cow)
- sleep: sleep to restore energy
- place_stone: place a stone in front
- place_table: place a crafting table in front
- place_furnace: place a stone furnace in front
- place_plant: place a plant in front
- make_wood_pickaxe: craft a wood pickaxe (requires nearby table + wood)
- make_stone_pickaxe: craft a stone pickaxe (requires nearby table + wood + stone)
- make_iron_pickaxe: craft an iron pickaxe (requires nearby table + furnace + wood + coal + iron)
- make_wood_sword: craft a wood sword (requires nearby table + wood)
- make_stone_sword: craft a stone sword (requires nearby table + wood + stone)
- make_iron_sword: craft an iron sword (requires nearby table + furnace + wood + coal + iron)
- noop: do nothing

### Instructions
- Plan progressively based on your inventory
- Identify and avoid repeating meaningless actions
- Stay alive: prioritize eating, drinking, sleeping when low
- Use the right tools for collecting resources

Respond with ONLY a valid action name from the list above."""

REASONING_SYSTEM_PROMPT = """\
You are a reasoning agent. Answer the question step by step using the provided context. \
Think carefully, then provide your final answer on a new line starting with 'Answer:'."""

RELABEL_PROMPT = """\
You are analyzing agent behavior in Crafter. Given the action sequence and final observation, \
infer the most likely task the agent was performing.

Action sequence: {actions}

Final observation:
{observation}

State the inferred goal in ~10 words. Output ONLY the goal, nothing else."""


class LLMExplorer(ExplorerAgent):
    """Explorer that uses an LLM to interact with Crafter.

    For each target skill in the plan, runs episodes collecting trajectories.
    Segments long trajectories into skill-relevant chunks.
    """

    def __init__(
        self,
        llm: LLMBackend,
        config: dict[str, Any] | None = None,
    ):
        super().__init__("LLMExplorer", config)
        self.llm = llm
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 50)
        self.segment_length = self.config.get("segment_length", 8)
        self.question_only = self.config.get("question_only", False)
        self.use_bandit = self.config.get("use_bandit", False)
        self._bandit: Any = None
        if self.use_bandit:
            from mage.core.experience_bandit import ExperienceBandit
            self._bandit = ExperienceBandit(k=self.config.get("bandit_k", 3))

    def set_bandit(self, bandit: Any) -> None:
        """Inject a pre-loaded bandit instance (e.g., from saved state)."""
        self._bandit = bandit
        self.use_bandit = True

    def explore(
        self,
        plan: EvolutionPlan,
        evokg: EvoKG,
        environment: Any,
        num_episodes: int = 50,
    ) -> SkillDataset:
        dataset = SkillDataset(iteration=plan.iteration)

        if not plan.targets:
            logger.warning("Empty plan — no exploration targets")
            return dataset

        episodes_per_target = max(1, num_episodes // len(plan.targets))

        for target in plan.targets:
            logger.info(f"  Exploring for skill: {target.skill_id} ({episodes_per_target} episodes)")

            # Select experience hints (bandit or fixed heuristic)
            if self.use_bandit and self._bandit is not None:
                experiences = evokg.get_experiences_for_skill(target.skill_id)
                selected = self._bandit.select(experiences)
                hints = self._bandit.format_hints(selected)
                selected_ids = [e["node_id"] for e in selected]
            else:
                hints = self._get_failure_hints(evokg, target.skill_id)
                selected_ids = []

            successes = 0
            for ep in range(episodes_per_target):
                trajectories = self._run_episode(
                    environment, target, hints, seed=plan.iteration * 1000 + ep,
                )
                for traj in trajectories:
                    dataset.add(traj)
                    if traj.skill_id == target.skill_id and traj.reward > 0:
                        successes += 1

            # Update bandit with reward signal
            if self.use_bandit and self._bandit is not None and selected_ids:
                reward = successes / max(episodes_per_target, 1)
                self._bandit.update(selected_ids, reward)
                logger.debug(f"  Bandit update for {target.skill_id}: reward={reward:.3f}, arms={selected_ids}")

        logger.info(f"  Total trajectories collected: {len(dataset)}")
        return dataset

    def _get_failure_hints(self, evokg: EvoKG, skill_id: str) -> str:
        """Retrieve past failure patterns for this skill from G_exp."""
        experiences = evokg.get_experiences_for_skill(skill_id)
        failures = [e for e in experiences if e.get("outcome") == "failure"]

        if not failures:
            return ""

        hints = "Avoid these known failure patterns:\n"
        for f in failures[:3]:
            hints += f"- {f.get('action_pattern', 'unknown pattern')}\n"
        return hints

    def _run_episode(
        self,
        environment: Any,
        target: Any,
        failure_hints: str,
        seed: int,
    ) -> list[Trajectory]:
        """Run one exploration episode and segment into trajectories."""
        from mage.environments.base import ReasoningEnvironment

        if isinstance(environment, ReasoningEnvironment):
            return self._run_reasoning_episode(environment, target, failure_hints, seed)

        if not isinstance(environment, SequentialEnvironment):
            return []

        obs_text = environment.reset(seed=seed)
        observations: list[str] = [obs_text]
        actions: list[str] = []

        task_prompt = target.exploration_brief
        if failure_hints:
            task_prompt += f"\n\n{failure_hints}"

        # Use the environment's run_episode or manual stepping
        action_space = environment.get_action_space()

        for step in range(self.max_steps_per_episode):
            action = self._get_action(obs_text, task_prompt, actions[-3:] if actions else [], action_space)
            obs_text, reward, done, info = environment.step(action)
            observations.append(obs_text)
            actions.append(action)

            if done:
                break

        # Detect new achievements from episode info
        new_achievements: set[str] = set()
        # For sequential envs that track achievements (like Crafter)
        if hasattr(environment, 'get_achievements'):
            all_achievements = environment.get_achievements()
            new_achievements = {k for k, v in all_achievements.items() if v}

        # Segment the trajectory
        trajectories = self._segment_trajectory(
            observations, actions, target, new_achievements,
        )

        return trajectories

    def _get_action(
        self, observation: str, task: str, recent_actions: list[str],
        action_space: list[str] | None = None,
    ) -> str:
        """Get next action from LLM given current observation."""
        recent = ""
        if recent_actions:
            recent = f"\nRecent actions: {', '.join(recent_actions)}"

        prompt = f"Task: {task}\n\n{observation}{recent}\n\nChoose your next action:"

        try:
            response = self.llm.generate(prompt, EXPLORER_SYSTEM_PROMPT, temperature=0.7, max_tokens=32)
            action = response.strip().lower().replace(" ", "_")
            # Validate action against action space
            valid_actions = action_space or ["noop"]
            if action in valid_actions:
                return action
            # Try partial match
            for a in valid_actions:
                if a in action or action in a:
                    return a
        except Exception as e:
            logger.debug(f"LLM action generation failed: {e}")

        # Fallback: random action from the action space
        fallback_actions = action_space or ["noop"]
        return random.choice(fallback_actions)

    def _segment_trajectory(
        self,
        observations: list[str],
        actions: list[str],
        target: Any,
        new_achievements: set[str],
    ) -> list[Trajectory]:
        """Segment a long trajectory into skill-relevant chunks."""
        trajectories = []

        if not actions:
            return trajectories

        # If achievements were gained, create a trajectory for each
        if new_achievements:
            for skill_id in new_achievements:
                # Use the last N steps as the trajectory for this achievement
                seg_len = min(self.segment_length, len(actions))
                traj = Trajectory(
                    instruction=f"Perform skill: {skill_id}",
                    observations=observations[-seg_len - 1:],
                    actions=actions[-seg_len:],
                    reward=1.0,
                    skill_id=skill_id,
                    metadata={"achievement": skill_id, "source": "exploration"},
                )
                trajectories.append(traj)

        # Always create a trajectory for the target skill
        seg_len = min(self.segment_length, len(actions))
        traj = Trajectory(
            instruction=target.exploration_brief,
            observations=observations[-seg_len - 1:],
            actions=actions[-seg_len:],
            reward=1.0 if target.skill_id in new_achievements else 0.0,
            skill_id=target.skill_id,
            metadata={"target": target.skill_id, "source": "exploration"},
        )
        trajectories.append(traj)

        return trajectories

    def _run_reasoning_episode(
        self,
        environment: Any,
        target: Any,
        hints: str,
        seed: int,
    ) -> list[Trajectory]:
        """Run one QA/reasoning episode. Returns a single trajectory."""
        # Get a batch of instances; fall back to unfiltered if skill has none
        instances = environment.get_evaluation_instances(
            skill_id=target.skill_id, num_instances=100,
        )
        if not instances:
            instances = environment.get_evaluation_instances(
                skill_id=None, num_instances=100,
            )
        if not instances:
            return []

        inst = instances[seed % len(instances)]

        # GRPO mode: return question metadata only, no LLM generation
        if self.question_only:
            traj = Trajectory(
                instruction=inst.instruction,
                observations=[],
                actions=[],
                reward=0.0,
                skill_id=target.skill_id,
                metadata={
                    "expected": inst.expected_answer,
                    "context": inst.context,
                    "instance_id": inst.instance_id,
                    "task_type": inst.metadata.get("task_type", ""),
                    "source": "question_selection",
                },
            )
            return [traj]

        task_prompt = f"{inst.instruction}"

        # Inject skill strategy from target's exploration brief
        if target.exploration_brief and target.exploration_brief != f"Practice skill: {target.skill_id}. Current mastery: 0%.":
            task_prompt += f"\n\nStrategy: {target.exploration_brief}"

        if hints:
            task_prompt += f"\n\n{hints}"

        def agent_fn(prompt: str, instruction: str) -> str:
            return self.llm.generate(prompt, REASONING_SYSTEM_PROMPT, max_tokens=512)

        result = environment.run_episode(
            agent_fn=agent_fn,
            instruction=task_prompt,
            seed=seed,
            context=inst.context,
        )

        # Score the answer
        final_answer = result.metadata.get("final_answer", "")
        if inst.expected_answer and final_answer:
            score = environment.evaluate_answer(final_answer, inst.expected_answer)
        else:
            score = 0.0

        traj = Trajectory(
            instruction=inst.instruction,
            observations=result.observations,
            actions=result.actions,
            reward=score,
            skill_id=target.skill_id,
            metadata={
                "expected": inst.expected_answer,
                "predicted": final_answer,
                "score": score,
                "source": "exploration",
            },
        )
        return [traj]


class RandomExplorer(ExplorerAgent):
    """Random exploration baseline — no LLM, random actions."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("RandomExplorer", config)
        self.max_steps = self.config.get("max_steps_per_episode", 50)

    def explore(
        self,
        plan: EvolutionPlan,
        evokg: EvoKG,
        environment: Any,
        num_episodes: int = 50,
    ) -> SkillDataset:
        dataset = SkillDataset(iteration=plan.iteration)

        if not isinstance(environment, BaseEnvironment):
            return dataset

        action_space = environment.get_action_space()

        for ep in range(num_episodes):
            if isinstance(environment, SequentialEnvironment):
                obs_text = environment.reset(seed=plan.iteration * 1000 + ep)
            else:
                obs_text = ""
            observations = [obs_text]
            actions = []

            for _ in range(self.max_steps):
                action = random.choice(action_space)
                if isinstance(environment, SequentialEnvironment):
                    obs_text, _, done, _ = environment.step(action)
                    observations.append(obs_text)
                    actions.append(action)
                    if done:
                        break
                else:
                    actions.append(action)
                    break

            new_achievements: set[str] = set()
            if hasattr(environment, 'get_achievements'):
                new_achievements = {
                    k for k, v in environment.get_achievements().items() if v
                }

            for skill_id in new_achievements:
                dataset.add(Trajectory(
                    instruction=f"Perform: {skill_id}",
                    observations=observations[-6:],
                    actions=actions[-5:],
                    reward=1.0,
                    skill_id=skill_id,
                ))

            # Add the full episode as a generic trajectory
            skill_id = plan.targets[0].skill_id if plan.targets else "unknown"
            dataset.add(Trajectory(
                instruction=f"Explore for: {skill_id}",
                observations=observations[-6:],
                actions=actions[-5:],
                reward=1.0 if new_achievements else 0.0,
                skill_id=skill_id,
            ))

        return dataset
