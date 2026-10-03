"""Critic Agent: Evaluates the Learner and produces structured feedback.

Outputs structured evaluations with per-skill success rates and failure mode
analysis — directly writable to EvoKG. This is a key differentiator from EXIF's
NL-only feedback.
"""

from __future__ import annotations

import logging
from typing import Any

from mage.agents.base import CriticAgent, LearnerAgent
from mage.core.evokg import EvoKG
from mage.core.schemas import (
    FailureMode,
    NodeType,
    SkillEvaluation,
    StructuredEvaluation,
)
from mage.environments.base import BaseEnvironment, ReasoningEnvironment, SequentialEnvironment

logger = logging.getLogger(__name__)


class CrafterCritic(CriticAgent):
    """Evaluates the Learner on Crafter skills via test episodes.

    For each skill in EvoKG's G_cap:
    - Runs N trials with the Learner
    - Tracks success/failure + failure modes
    - Produces structured evaluation directly writable to EvoKG

    This replaces EXIF's NL feedback with structured, KG-compatible output.
    """

    def __init__(self, config: dict[str, Any] | None = None, judge: Any = None):
        super().__init__("CrafterCritic", config)
        self.trials_per_skill = self.config.get("trials_per_skill", 10)
        self.max_steps_per_trial = self.config.get("max_steps_per_trial", 30)
        # v15: LLM-as-judge for semantic answer evaluation
        self._judge = judge

    def evaluate(
        self,
        learner: LearnerAgent,
        evokg: EvoKG,
        environment: Any,
        iteration: int,
    ) -> StructuredEvaluation:
        if not isinstance(environment, BaseEnvironment):
            return StructuredEvaluation(iteration=iteration)

        # Use PROPER EVAL: fixed set of questions evaluated through the
        # Learner's full pipeline (including task_type routing and tools)
        if isinstance(environment, ReasoningEnvironment):
            return self._evaluate_proper(learner, evokg, environment, iteration)

        # v15.6 — BALROG-style episode evaluation for Crafter-like envs.
        # Instead of per-skill trials (which start from scratch for each
        # of 22 skills), run N full episodes of `max_steps` and count
        # accumulated achievements. This matches BALROG where Llama-3.1-8B
        # scores 25%.
        if getattr(environment, "USE_EPISODE_EVAL", False) and hasattr(environment, "evaluate_full_episode"):
            logger.info(f"  Using BALROG-style episode evaluation")
            n_episodes = self.trials_per_skill  # reuse config knob
            max_ep_steps = self.max_steps_per_trial
            total_achievements = 0
            per_skill_hits: dict[str, int] = {}

            # v20.B — collect action recipes across episodes
            all_recipes: list[dict] = []
            for ep in range(n_episodes):
                def agent_fn(obs, instruction):
                    try:
                        return learner.act(obs, instruction)
                    except Exception:
                        return "noop"
                result = environment.evaluate_full_episode(
                    agent_fn, max_steps=max_ep_steps, seed=42 + ep,
                )
                total_achievements += result["n_achievements"]
                for sk in result["achievements"]:
                    per_skill_hits[sk] = per_skill_hits.get(sk, 0) + 1
                # v20.B — gather any recipes this episode produced
                for r in result.get("action_recipes", []):
                    all_recipes.append(r)
                logger.info(
                    f"  Episode {ep}: {result['n_achievements']}/{result['total_skills']} "
                    f"achievements in {result['steps_taken']} steps "
                    f"({list(result['achievements'].keys())})"
                )

            # v20.B — persist the recipes to the EvoKG so the learner can
            # retrieve them in the next iteration's PLAN/EXPLORE/EVALUATE
            if all_recipes and evokg is not None and hasattr(evokg, "add_action_recipe"):
                added = 0
                for r in all_recipes:
                    nid = evokg.add_action_recipe(
                        skill_id=r.get("skill_id", ""),
                        actions=r.get("actions", []),
                        ctx_summary=r.get("ctx", ""),
                        iteration=iteration,
                        source="episode",
                    )
                    if nid:
                        added += 1
                logger.info(f"  v20.B — wrote {added} action recipes to EvoKG (out of {len(all_recipes)} captured)")

            # v15.6: env-agnostic — use the env's own skill list rather than
            # hardcoding Crafter's 22. Required for WebShop (5 skills), etc.
            env_skills = [s.skill_id for s in environment.get_skill_definitions()]
            total_skill_count = max(len(env_skills), 1)
            overall = total_achievements / (n_episodes * total_skill_count) if n_episodes > 0 else 0.0
            evaluations = []
            for sid in env_skills:
                hits = per_skill_hits.get(sid, 0)
                evaluations.append(SkillEvaluation(
                    skill_id=sid,
                    success_rate=hits / max(n_episodes, 1),
                    attempts=n_episodes,
                    failure_modes=[],
                    recommended_focus=[],
                ))
            return StructuredEvaluation(
                iteration=iteration,
                evaluations=evaluations,
                overall_success_rate=overall,
                summary=f"BALROG-style: {total_achievements} achievements across {n_episodes} episodes = {overall:.1%}",
            )

        # Sequential environments: per-skill evaluation (original path)
        evaluations = []
        total_success = 0
        total_attempts = 0

        skills = evokg.get_nodes_by_type(NodeType.CAPABILITY)
        for skill_node in skills:
            skill_id = skill_node["node_id"]
            if skill_id.startswith("meta_"):
                continue
            skill_eval = self._evaluate_skill(
                learner, environment, evokg, skill_id, iteration,
            )
            evaluations.append(skill_eval)
            total_success += int(skill_eval.success_rate * skill_eval.attempts)
            total_attempts += skill_eval.attempts

        overall = total_success / total_attempts if total_attempts > 0 else 0.0

        return StructuredEvaluation(
            iteration=iteration,
            evaluations=evaluations,
            overall_success_rate=overall,
            summary=self._generate_summary(evaluations, overall, iteration),
        )

    def _evaluate_proper(
        self,
        learner: LearnerAgent,
        evokg: EvoKG,
        environment: Any,
        iteration: int,
    ) -> StructuredEvaluation:
        """Legacy v15.5 path: pull a fixed N from the env and score it.

        Same questions every iteration. The new v15.6 curriculum path uses
        ``evaluate_instances`` with caller-supplied lists instead.
        """
        instances = environment.get_evaluation_instances(
            skill_id=None, num_instances=self.trials_per_skill * 10,
        )
        if not instances:
            return StructuredEvaluation(iteration=iteration)
        return self.evaluate_instances(learner, evokg, environment, instances, iteration)

    def evaluate_instances(
        self,
        learner: LearnerAgent,
        evokg: EvoKG,
        environment: Any,
        instances: list[Any],
        iteration: int,
    ) -> StructuredEvaluation:
        """v15.6 — Score a caller-supplied list of EvaluationInstances.

        Used by the curriculum loop to evaluate each iter's mixed batch
        (n_new fresh + ALL accumulated failures) and again for the
        final-exam pool. Per-question results carry the same fields as
        the legacy ``_evaluate_proper`` path so the success-memory harvest
        loop in ``evolution_loop_v2._update_search_agent`` works unchanged.
        """
        if not instances:
            return StructuredEvaluation(iteration=iteration)

        total_success = 0
        total_attempts = 0
        by_task: dict[str, dict] = {}
        per_question_results: list[dict[str, Any]] = []

        for inst in instances:
            task_type = inst.metadata.get("task_type", "")
            if task_type not in by_task:
                by_task[task_type] = {"success": 0, "total": 0}
            by_task[task_type]["total"] += 1
            total_attempts += 1

            # CoT prompt — same as v15.5 _evaluate_proper.
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

            score = 0.0
            if inst.expected_answer and predicted:
                if hasattr(self, '_judge') and self._judge is not None:
                    score = self._judge.evaluate(
                        predicted, inst.expected_answer,
                        question=inst.instruction, task_type=task_type,
                    )
                else:
                    try:
                        score = environment.evaluate_answer(
                            predicted, inst.expected_answer, task_type=task_type,
                        )
                    except TypeError:
                        score = environment.evaluate_answer(
                            predicted, inst.expected_answer,
                        )
                if score >= 0.5:
                    total_success += 1
                    by_task[task_type]["success"] += 1

            search_strategy = ""
            raw_response = ""
            decomposition: list[dict[str, Any]] = []
            last_meta = getattr(learner, "_last_call_metadata", None)
            if isinstance(last_meta, dict):
                search_strategy = last_meta.get("search_strategy", "") or ""
                raw_response = last_meta.get("raw_response", "") or ""
                decomposition = list(last_meta.get("decomposition", []) or [])

            per_question_results.append({
                "instance_id": getattr(inst, "instance_id", ""),
                "question": inst.instruction,
                "context": inst.context or "",
                "predicted": predicted or "",
                "expected": inst.expected_answer or "",
                "task_type": task_type,
                "score": float(score),
                "correct": bool(score >= 0.5),
                "search_strategy": search_strategy,
                "raw_response": raw_response,
                "decomposition": decomposition,
                # v15.6: tag whether this question was a "fresh" probe or
                # a re-test of a prior failure. Curriculum loop fills this
                # in via ``_curriculum_tag`` on the EvaluationInstance metadata.
                "curriculum_tag": inst.metadata.get("_curriculum_tag", ""),
            })

        overall = total_success / max(total_attempts, 1)
        evaluations = []
        for task_type, stats in by_task.items():
            evaluations.append(SkillEvaluation(
                skill_id=task_type,
                success_rate=stats["success"] / max(stats["total"], 1),
                attempts=stats["total"],
                failure_modes=[],
                recommended_focus=[],
            ))

        return StructuredEvaluation(
            iteration=iteration,
            evaluations=evaluations,
            overall_success_rate=overall,
            summary=f"Eval: {total_success}/{total_attempts} = {overall:.1%}",
            per_question_results=per_question_results,
        )

    def _evaluate_skill(
        self,
        learner: LearnerAgent,
        environment: BaseEnvironment,
        evokg: EvoKG,
        skill_id: str,
        iteration: int,
    ) -> SkillEvaluation:
        """Evaluate a single skill over multiple trials."""
        successes = 0
        failure_reasons: dict[str, int] = {}

        for trial in range(self.trials_per_skill):
            # FIXED seed — same questions every iteration (dev set)
            # This ensures accuracy changes reflect real learning, not different questions
            seed = hash(skill_id) % 10000 + trial
            success, failure_reason = self._run_trial(
                learner, environment, evokg, skill_id, seed,
            )
            if success:
                successes += 1
            elif failure_reason:
                failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1

        success_rate = successes / self.trials_per_skill

        # Build failure modes
        failure_modes = []
        total_failures = self.trials_per_skill - successes
        if total_failures > 0:
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                failure_modes.append(FailureMode(
                    pattern=reason.split(":")[0] if ":" in reason else reason,
                    detail=reason,
                    frequency=count / total_failures,
                ))

        # Determine recommended focus
        recommended = []
        if success_rate < 0.5:
            # Check if prerequisites are the issue
            prereqs = evokg.get_prerequisites(skill_id)
            unmastered = [p for p in prereqs if evokg.get_mastery(p) < 0.5]
            if unmastered:
                recommended.extend(unmastered)
            else:
                recommended.append(skill_id)

        return SkillEvaluation(
            skill_id=skill_id,
            success_rate=success_rate,
            attempts=self.trials_per_skill,
            failure_modes=failure_modes,
            recommended_focus=recommended,
        )

    def _run_trial(
        self,
        learner: LearnerAgent,
        environment: BaseEnvironment,
        evokg: EvoKG,
        skill_id: str,
        seed: int,
    ) -> tuple[bool, str]:
        """Run one evaluation trial for a skill.

        Returns (success, failure_reason).
        """
        # Get skill name from environment definitions
        skill_defs = {s.skill_id: s for s in environment.get_skill_definitions()}
        skill_name = skill_defs[skill_id].name if skill_id in skill_defs else skill_id
        instruction = f"Perform skill: {skill_name}"
        action_space = environment.get_action_space()

        if isinstance(environment, ReasoningEnvironment):
            # For QA environments: run episode, score the answer
            instances = environment.get_evaluation_instances(
                skill_id=skill_id, num_instances=100,
            )
            if not instances:
                instances = environment.get_evaluation_instances(
                    skill_id=None, num_instances=100,
                )
            if not instances:
                return False, "no_evaluation_instances"
            inst = instances[seed % len(instances)]

            # Inject skill strategy into the agent's prompt
            skill_data = evokg.get_node(skill_id)
            strategy_guidance = ""
            if skill_data:
                if skill_data.get("prompt_template"):
                    strategy_guidance += f"\nStrategy: {skill_data['prompt_template']}"
                strategy = skill_data.get("strategy", {})
                if isinstance(strategy, dict) and strategy.get("type") == "tool":
                    tool_name = strategy.get("tool", "")
                    param_info = strategy.get("param_extraction", "")
                    strategy_guidance += (
                        f"\nIMPORTANT: Use the {tool_name} tool for computation. "
                        f"{param_info}"
                    )

            # Get task_type from the evaluation instance
            eval_task_type = inst.metadata.get("task_type", "")

            def agent_fn(prompt: str, instruction: str) -> str:
                augmented = prompt + strategy_guidance if strategy_guidance else prompt
                # Pass task_type so the tool pipeline can fire
                try:
                    return learner.act(augmented, instruction, task_type=eval_task_type)
                except TypeError:
                    return learner.act(augmented, instruction)

            result = environment.run_episode(
                agent_fn=agent_fn, instruction=inst.instruction,
                seed=seed, max_steps=self.max_steps_per_trial,
                context=inst.context,
            )
            final_answer = result.metadata.get("final_answer", "")
            if inst.expected_answer and final_answer:
                try:
                    score = environment.evaluate_answer(
                        final_answer, inst.expected_answer, task_type=eval_task_type,
                    )
                except TypeError:
                    score = environment.evaluate_answer(final_answer, inst.expected_answer)
                if score >= 0.5:
                    return True, ""
                return False, f"insufficient_skill:wrong_answer(f1={score:.2f})"
            return False, "insufficient_skill:no_answer"

        if not isinstance(environment, SequentialEnvironment):
            return False, "unsupported_environment"

        # Sequential environment: step-by-step evaluation
        obs_text = environment.reset(seed=seed)
        prev_obs = ""
        repeated_actions = 0

        # v15.6 fix — sequential envs need the LLM to know which actions
        # are valid. Previously the critic called learner.act(obs, instruction)
        # without exposing the action space, so the LLM produced free-form
        # text and we always fell through to action_space[0]=noop. Now we
        # build a per-step prompt with the action list and parse the LLM's
        # response by scanning for the first valid action token.
        action_list_str = ", ".join(action_space)
        instruction_with_actions = (
            f"Goal: {instruction}\n\n"
            f"Valid actions: {action_list_str}\n"
            f"Choose ONE action from the list above. "
            f"Output only the action name (e.g. 'do' or 'move_left'). "
            f"No explanation."
        )

        # v15.6 — envs with parameterised actions (WebShop's
        # ``search[query]`` / ``click[id]``) implement ``parse_action`` so
        # the critic doesn't try to validate against a static enum.
        env_parser = getattr(environment, "parse_action", None)

        for step in range(self.max_steps_per_trial):
            response = learner.act(obs_text, instruction_with_actions)
            if callable(env_parser):
                action = env_parser(response, obs_text)
            else:
                action = self._extract_action(response, action_space)

            obs_text, reward, done, info = environment.step(action)

            # Check if skill was achieved
            new_achievements = (info or {}).get("new_achievements", {})
            if skill_id in new_achievements:
                return True, ""

            # Also check cumulative achievements
            if hasattr(environment, 'get_achievements'):
                current = environment.get_achievements()
                if current.get(skill_id, 0):
                    return True, ""

            # Detect repeated actions (stuck)
            if obs_text == prev_obs:
                repeated_actions += 1
            else:
                repeated_actions = 0
            prev_obs = obs_text

            if repeated_actions > 5:
                return False, "stuck:repeated_actions_no_effect"

            if done:
                break

        # Determine failure reason
        prereqs = evokg.get_prerequisites(skill_id)
        unmastered_prereqs = [p for p in prereqs if evokg.get_mastery(p) < 0.5]
        if unmastered_prereqs:
            return False, f"missing_prerequisite:{','.join(unmastered_prereqs)}"

        return False, "insufficient_skill:timeout"

    @staticmethod
    def _extract_action(response: str, action_space: list[str]) -> str:
        """Pull the first valid action name from a free-form LLM response.

        v15.6 — Qwen3-8B's text observations make it produce things like
        ``"<think>I should move left to find wood</think>\\nmove_left"``,
        ``"Action: do"``, or natural-language ``"Make wood pickaxe"``. We
        normalise underscores↔spaces↔hyphens so ``"make wood pickaxe"``
        matches ``"make_wood_pickaxe"``. Scan for the EARLIEST occurrence
        of any action token. Falls back to ``noop`` if nothing matches.
        """
        if not response:
            return "noop" if "noop" in action_space else action_space[0]
        import re

        def norm(s: str) -> str:
            # Lowercase, strip <think>...</think> (closed) AND truncated
            # <think> blocks that never closed (Qwen3-8B's thinking cut
            # off by max_tokens → false positive action matches inside
            # reasoning text). Strip from <think> to end-of-string.
            s = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL)
            if "<think>" in s and "</think>" not in s:
                s = s.split("<think>")[0]
            # v16 A5: prefer the content inside BALROG-style <|ACTION|>...<|END|>
            # tags when present; models fine-tuned for agentic benchmarks
            # emit them and the action sits inside the tags.
            tag_m = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", s, flags=re.DOTALL | re.IGNORECASE)
            if tag_m:
                s = tag_m.group(1)
            s = s.lower()
            # Strip surrounding brackets/quotes/backticks/parens/hashes so
            # outputs like ``[do]``, ``"do"``, ``` `do` ``` or ``Action: do.``
            # all parse to ``do``. Do this before normalising space runs.
            s = re.sub(r"[\[\]`'\"()#*]", " ", s)
            s = re.sub(r"[_\-]+", " ", s)
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        normalised_text = norm(response)
        # Sort longer action names first so e.g. "make_iron_pickaxe" matches
        # before "make_iron" (which is not in the space, but defensive).
        sorted_actions = sorted(action_space, key=len, reverse=True)
        best_pos = None
        best_action = None
        for act in sorted_actions:
            act_norm = norm(act)
            # Whole-word match against normalised text
            for m in re.finditer(r"\b" + re.escape(act_norm) + r"\b", normalised_text):
                if best_pos is None or m.start() < best_pos:
                    best_pos = m.start()
                    best_action = act
                break
        if best_action is not None:
            return best_action
        return "noop" if "noop" in action_space else action_space[0]

    def _generate_summary(
        self,
        evaluations: list[SkillEvaluation],
        overall: float,
        iteration: int,
    ) -> str:
        """Generate a text summary of the evaluation."""
        mastered = [e for e in evaluations if e.success_rate >= 0.5]
        struggling = [e for e in evaluations if 0 < e.success_rate < 0.5]
        unlearned = [e for e in evaluations if e.success_rate == 0.0]

        lines = [f"Iteration {iteration} evaluation: {overall:.1%} overall success rate"]
        if mastered:
            lines.append(f"  Mastered ({len(mastered)}): {[e.skill_id for e in mastered]}")
        if struggling:
            lines.append(f"  Struggling ({len(struggling)}): {[e.skill_id for e in struggling]}")
        if unlearned:
            lines.append(f"  Not learned ({len(unlearned)}): {[e.skill_id for e in unlearned]}")

        return "\n".join(lines)
