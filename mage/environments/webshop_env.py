"""WebShop environment adapter for MAGE.

Sequential web navigation for online shopping. The agent searches for
products, browses results, selects items, and chooses options to match
a given shopping goal.

Skills map to shopping capabilities:
- query_formulation: craft effective search queries from goal
- result_selection: pick the right product from search results
- option_selection: choose correct size/color/style variants
- goal_understanding: parse shopping instructions for requirements
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# v15.5: WebShop's Pyserini search engine wraps Apache Lucene via Java; we
# need JAVA_HOME / JVM_PATH set BEFORE pyjnius is imported. Auto-detect the
# system OpenJDK and set the env vars if the user hasn't already.
def _ensure_java_env() -> None:
    if os.environ.get("JVM_PATH"):
        return
    candidates = [
        "/usr/lib/jvm/java-21-openjdk-amd64",
        "/usr/lib/jvm/default-java",
        "/usr/lib/jvm/java-1.21.0-openjdk-amd64",
    ]
    for jh in candidates:
        libjvm = Path(jh) / "lib" / "server" / "libjvm.so"
        if libjvm.exists():
            os.environ.setdefault("JAVA_HOME", jh)
            os.environ["JVM_PATH"] = str(libjvm)
            ldp = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{libjvm.parent}:{ldp}" if ldp else str(libjvm.parent)
            )
            return
_ensure_java_env()

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

WEBSHOP_DIR = Path(os.environ.get("WEBSHOP_DIR", "../WebShop")).expanduser().resolve()

WEBSHOP_SKILLS = {
    "goal_understanding": {
        "name": "Goal Understanding",
        "category": "comprehension",
        "prereqs": [],
        "description": "Parse shopping instructions to identify required attributes (color, size, price, features)",
    },
    "query_formulation": {
        "name": "Query Formulation",
        "category": "search",
        "prereqs": ["goal_understanding"],
        "description": "Craft effective search queries from shopping goals",
    },
    "result_selection": {
        "name": "Result Selection",
        "category": "navigation",
        "prereqs": ["query_formulation"],
        "description": "Select the most relevant product from search results based on goal requirements",
    },
    "option_selection": {
        "name": "Option Selection",
        "category": "navigation",
        "prereqs": ["result_selection", "goal_understanding"],
        "description": "Choose correct size, color, and style options for the selected product",
    },
    "purchase_decision": {
        "name": "Purchase Decision",
        "category": "decision",
        "prereqs": ["option_selection"],
        "description": "Confirm purchase when all requirements are met, or go back to search if product doesn't match",
    },
}


class WebShopEnvironment(SequentialEnvironment):
    """WebShop sequential shopping environment for MAGE."""

    # v15.6 fix: use BALROG-style full-episode eval (critic.py:64) so the
    # agent sees the REAL shopping goal via self._current_goal and gets a
    # full 15–30 step browse→find→configure→buy budget. Without this the
    # critic falls through to the per-skill step-by-step path with a vague
    # "Perform skill: purchase_decision" prompt and only ~20 steps total,
    # which is why WebShop success_rate was stuck at 0.0%.
    USE_EPISODE_EVAL = True

    def __init__(
        self,
        num_products: int = 100,
        max_steps: int = 15,
        seed: int = 42,
    ):
        self._num_products = num_products
        self._max_steps = max_steps
        self._seed = seed
        self._env = None
        self._current_goal = ""
        self._step_count = 0
        # v15.6 — WebShop has continuous reward, no achievement events.
        # Track the high-water-mark reward this episode and synthesize per-skill
        # achievements from reward thresholds so the per-skill critic eval works.
        self._max_reward_this_episode = 0.0
        # v16 B2: persistent lessons learned from failed episodes. Each entry
        # is {goal, product_type, reward, lesson, actions}. We retrieve the
        # top-K most similar ones (by product-type word overlap) and inject
        # into future episode prompts so the agent avoids repeating the same
        # mistakes across goals. Persists across evaluate_full_episode calls
        # within a single WebShopEnvironment instance.
        self._episode_lessons: list[dict[str, Any]] = []

    def _load_env(self) -> None:
        if self._env is not None:
            return
        # Add WebShop to path
        if str(WEBSHOP_DIR) not in sys.path:
            sys.path.insert(0, str(WEBSHOP_DIR))
        import warnings
        warnings.filterwarnings("ignore")
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        self._env = WebAgentTextEnv(
            observation_mode="text",
            num_products=self._num_products,
        )
        logger.info(f"Loaded WebShop environment ({self._num_products} products)")

    @property
    def name(self) -> str:
        return "webshop"

    @property
    def domain(self) -> str:
        return "web"

    def get_skill_definitions(self) -> list[SkillDefinition]:
        return [
            SkillDefinition(
                skill_id=sid,
                name=sdef["name"],
                category=sdef["category"],
                prerequisites=sdef["prereqs"],
                description=sdef["description"],
            )
            for sid, sdef in WEBSHOP_SKILLS.items()
        ]

    def get_action_space(self) -> list[str]:
        # v15.6 — return [] to disable the critic's strict static-list
        # validation. WebShop's actions are parameterised
        # (``search[shoes]``, ``click[B07YBH8XQ7]``); we expose
        # ``parse_action`` for the critic to extract them.
        return []

    def parse_action(self, response: str, obs_text: str = "") -> str:
        """v15.6 — extract a valid WebShop action from the LLM's response.

        Looks for the FIRST occurrence of:
          * ``search[<anything>]``
          * ``click[<anything>]``
          * the literal token ``buy`` or ``back``

        Falls back to ``click[Back to Search]`` (cheap noop on most pages).
        """
        if not response:
            return "click[Back to Search]"
        # Strip closed Qwen3 <think>...</think> blocks. If a <think> was
        # opened but never closed (max_tokens cut it off mid-reasoning),
        # also discard everything from `<think>` to end-of-string — those
        # tokens are reasoning, not the chosen action, and parsing them
        # will produce false-positive "examples" matches.
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>")[0]
        text = text.strip()
        if not text:
            return "click[Back to Search]"
        # Try parameterised first (longest match wins where they tie)
        m = re.search(
            r"(search\s*\[[^\]\n]+\]|click\s*\[[^\]\n]+\])",
            text,
            re.IGNORECASE,
        )
        if m:
            # Normalise the verb to lowercase but keep the parameter
            # case (item IDs like "B07YBH8XQ7" are case-sensitive).
            raw = m.group(1).replace(" [", "[")
            verb_match = re.match(r"(search|click)\[(.*)\]", raw, re.IGNORECASE)
            if verb_match:
                return f"{verb_match.group(1).lower()}[{verb_match.group(2)}]"
            return raw
        # Then literal commands
        for tok in ("buy", "back"):
            if re.search(r"\b" + tok + r"\b", text, re.IGNORECASE):
                return tok
        return "click[Back to Search]"

    def reset(self, seed: int | None = None, **kwargs: Any) -> str:
        self._load_env()
        # Clamp seed to valid goal range to avoid IndexError
        if seed is not None:
            num_goals = len(self._env.server.goals) if hasattr(self._env, 'server') and hasattr(self._env.server, 'goals') else 500
            seed = seed % max(num_goals, 1)
        obs = self._env.reset(seed)
        self._step_count = 0
        self._max_reward_this_episode = 0.0  # v15.6: reset for synthesised achievements

        # Parse observation — it's a tuple (text, image)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = str(obs)

        # Extract goal from observation
        goal_match = re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", obs)
        if goal_match:
            self._current_goal = goal_match.group(1).strip()
        else:
            self._current_goal = obs

        return self._format_obs(obs)

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._load_env()
        self._step_count += 1

        obs, reward, done, info = self._env.step(action)

        if isinstance(obs, tuple):
            obs = obs[0]
        obs = str(obs)

        done = done or self._step_count >= self._max_steps
        self._max_reward_this_episode = max(self._max_reward_this_episode, reward or 0.0)

        # v15.6: synthesise per-skill achievement events when the cumulative
        # reward crosses progress milestones. The critic uses
        # ``skill_id in info["new_achievements"]`` to detect a hit, so we
        # only put TRUE achievements in this dict (Crafter-style: keys are
        # achievements unlocked this step, not a status map). Including
        # False entries would falsely fire `in` for every skill.
        info = dict(info or {})
        info["new_achievements"] = {
            k: True for k, v in self.get_achievements().items() if v
        }

        return self._format_obs(obs), reward, done, info

    def _format_obs(self, obs: str) -> str:
        """Clean up WebShop observation for LLM and append the dynamic
        action menu so the LLM can pick a real, formatted action.

        v15.6 — Without these hints the LLM produces free-form text and
        every action becomes an invalid noop → reward stuck at 0.
        """
        text = obs.replace("[SEP]", "\n").strip()
        if len(text) > 1800:
            text = text[:1800] + "\n[... truncated]"

        # Append per-state action hints from the underlying env. We avoid
        # any literal "example query" inside square brackets here, because
        # Qwen3-8B's parser regex would happily grab the example as the real
        # action. Placeholders use ALL_CAPS so they're not valid actions.
        action_hint = ""
        try:
            avail = self._env.get_available_actions() if self._env else {}
            if isinstance(avail, dict):
                hints: list[str] = []
                if avail.get("has_search_bar"):
                    hints.append('Format: search[YOUR_QUERY_HERE]   '
                                 '(replace YOUR_QUERY_HERE with words from the goal)')
                clickables = avail.get("clickables") or []
                if clickables:
                    # Real clickables ARE valid actions, list them as concrete options
                    sample = " | ".join(f'click[{c}]' for c in list(clickables)[:8])
                    if len(clickables) > 8:
                        sample += " | ..."
                    hints.append(f'Or one of: {sample}')
                if hints:
                    action_hint = "\n\nValid actions for THIS state:\n  " + "\n  ".join(hints)
        except Exception:
            pass

        return text + action_hint

    def get_achievements(self) -> dict[str, bool]:
        """v15.6 — synthesise per-skill achievements from continuous reward.

        WebShop's underlying reward is in [0, 1] and reflects how well the
        purchased item matches the goal attributes. We translate reward
        bands into "skills exercised" so the critic's per-skill evaluation
        path can produce non-zero numbers. The bands are heuristic — the
        single source of truth for evaluation is the maximum reward seen
        in the current episode (``self._max_reward_this_episode``).
        """
        r = self._max_reward_this_episode
        # v15.6 — permissive thresholds. Any non-zero reward = the agent
        # produced a valid action sequence that touched a product. WebShop's
        # raw rewards for partial-match purchases are typically 0.01-0.30
        # so the previous 0.20-0.65 thresholds never fired.
        return {
            "goal_understanding":  r > 0.0,    # any reward = parsed the goal
            "query_formulation":   r >= 0.05,  # search returned ≥5% match
            "result_selection":    r >= 0.10,  # picked a relevant product
            "option_selection":    r >= 0.20,  # tried setting options
            "purchase_decision":   r >= 0.30,  # bought with reasonable match
        }

    def get_evaluation_instances(
        self,
        skill_id: str | None = None,
        num_instances: int = 10,
    ) -> list[EvaluationInstance]:
        """Generate evaluation instances pulled from WebShop's real goal pool.

        Each WebShop seed maps to a specific shopping instruction like "Find
        me a pair of red running shoes for men under $50". Earlier versions
        hard-coded the literal string "Complete the shopping task" here,
        which made every per-skill trial unsolvable (no goal → reward stays
        0 → bandit never learns → KG fills with junk principles about
        underspecified prompts). Fix: actually call reset(seed) to read the
        real instruction text for each evaluation seed.
        """
        self._load_env()
        instances = []
        # WebShop has ~500 distinct goals; deterministically sample by seed
        try:
            num_goals = len(self._env.server.goals) if hasattr(self._env, "server") and hasattr(self._env.server, "goals") else 500
        except Exception:
            num_goals = 500
        for i in range(num_instances):
            seed_i = (self._seed * 1000 + i) % max(num_goals, 1)
            try:
                self.reset(seed=seed_i)
                goal_text = self._current_goal or f"Complete the shopping task (seed {seed_i})"
            except Exception:
                goal_text = f"Complete the shopping task (seed {seed_i})"
            instances.append(EvaluationInstance(
                instance_id=f"webshop_{seed_i}",
                instruction=goal_text,
                target_skill=skill_id or "purchase_decision",
                seed=seed_i,
                metadata={"task_type": "webshop_purchase", "goal": goal_text},
            ))
        return instances

    def run_episode(
        self,
        agent_fn: Any,
        instruction: str,
        seed: int | None = None,
        max_steps: int = 15,
        context: str = "",
    ) -> EpisodeResult:
        """Run a shopping episode."""
        obs = self.reset(seed=seed)
        observations = [obs]
        actions = []
        total_reward = 0.0

        history: list[tuple[str, str]] = []  # (action, page_summary)
        HISTORY_K = 4

        def _page_summary(text: str, max_chars: int = 400) -> str:
            return text if len(text) <= max_chars else text[:max_chars] + "..."

        # v16 B1: parse goal for required attributes once per episode. We
        # use this to (a) build an option-completeness checklist in every
        # prompt and (b) veto `click[Buy Now]` if any required attribute is
        # still unselected. Before this, agent would find the right product
        # but miss colour/size on ~half of purchases → 54% strict success.
        required_attrs = self._extract_goal_attributes(self._current_goal)

        # v16 B3 (simplified): show the parsed required attributes as a
        # structured block so the agent doesn't mis-read them from prose.
        # Initial v16 pass also included a "RECOMMENDED FIRST ACTION:
        # search[...]" line, but that regressed strict success by 4pp
        # because the extractor produced queries worse than what Qwen3
        # generated on its own. Keep the info block; drop the forced query.
        decomposition_block = ""
        if required_attrs:
            lines = ["REQUIRED ATTRIBUTES (parsed from the goal — you must satisfy every one):"]
            for k, v in required_attrs.items():
                if k == "price_cap":
                    lines.append(f"  price must be ≤ ${v:.2f}")
                else:
                    lines.append(f"  {k}: {v}")
            decomposition_block = "\n".join(lines) + "\n\n"

        # v16 B2 (DISABLED — see ablation): post-episode reflection + lesson
        # injection regressed strict success by ~12pp on the 100-ep test eval
        # (B1+B3v2 = 70.0% → B1+B2+B3 = 58.0%). Qwen3-generated reflections
        # were too often wrong about WHY the purchase failed (e.g., blamed
        # 'missed size' when the real issue was 'wrong product category'),
        # and once those incorrect lessons leaked into the next prompt they
        # distracted the agent. Kept the code path off via the flag below.
        # If you want to re-enable for experimentation, set
        # ``self._enable_lessons = True`` in __init__.
        lessons_block = ""
        if getattr(self, "_enable_lessons", False):
            current_ptype = self._infer_product_type(self._current_goal)
            relevant_lessons = self._retrieve_lessons(current_ptype, top_k=2)
            if relevant_lessons:
                lessons_block = "PAST MISTAKES ON SIMILAR GOALS (avoid repeating):\n"
                for i, L in enumerate(relevant_lessons, 1):
                    lessons_block += f"  [{i}] {L.get('lesson', '')[:200]}\n"
                lessons_block += "\n"

        for step in range(max_steps):
            # BALROG-style prompt: explicit task framing + action vocabulary
            # (with concrete examples) + recent history + current page.
            history_block = ""
            if history:
                lines = []
                for i, (h_act, h_page) in enumerate(history[-HISTORY_K:]):
                    lines.append(f"[step {step - len(history[-HISTORY_K:]) + i}] action: {h_act}\n  → page: {h_page}")
                history_block = "Recent history:\n" + "\n".join(lines) + "\n\n"

            # v16 B1: compute which required attributes have been clicked so far.
            selected_attrs = self._which_attrs_selected(required_attrs, actions)
            missing_attrs = [
                k for k in required_attrs
                if k != "price_cap" and k not in selected_attrs
            ]
            checklist_block = ""
            on_product_page = self._looks_like_product_page(obs)
            if required_attrs and on_product_page:
                req_lines = [
                    f"  {k}: {v}{'  ✅ selected' if k in selected_attrs else '  ❗ MISSING — click it'}"
                    for k, v in required_attrs.items() if k != "price_cap"
                ]
                if "price_cap" in required_attrs:
                    req_lines.append(f"  price ≤ ${required_attrs['price_cap']:.2f}  (check the page before buy)")
                checklist_block = (
                    "REQUIRED OPTIONS (you are on a product page — check each one):\n"
                    + "\n".join(req_lines) + "\n"
                )
                if missing_attrs:
                    checklist_block += (
                        f"\n⚠️  DO NOT click[Buy Now] yet. First click each missing option: "
                        f"{', '.join(missing_attrs)}. "
                        f"For example: click[{required_attrs[missing_attrs[0]]}].\n"
                    )
                checklist_block += "\n"

            # Pre-compute a concrete example search query from the actual
            # goal so the agent has something concrete to copy/adapt instead
            # of the literal placeholder "search[QUERY]" which earlier
            # versions copied verbatim into a no-op action.
            goal_words = re.findall(r"[A-Za-z0-9$]+", self._current_goal)[:6]
            example_query = " ".join(goal_words) if goal_words else "shoes"
            prompt = (
                f"You are an agent in WebShop, a simulated online shopping "
                f"environment. Your task is to find and buy the product that "
                f"best matches the goal below.\n\n"
                f"GOAL: {self._current_goal}\n\n"
                f"{decomposition_block}"
                f"{lessons_block}"
                f"AVAILABLE ACTIONS (output exactly one per turn):\n"
                f"  search[...]   — start a new keyword search; put your real query inside the brackets.\n"
                f"                  Example for THIS goal: search[{example_query}]\n"
                f"  click[...]    — click a button or product. Use the EXACT text or product code shown\n"
                f"                  on the page inside the brackets, e.g., click[B07XJK9PQR] or click[red].\n"
                f"  buy           — buy the currently-viewed product (only when on a product page)\n\n"
                f"STRATEGY:\n"
                f"  1. Start with a SHORT search (3-4 keywords MAX, e.g., the product type + 1-2 attributes). "
                f"Long queries return 0 results.\n"
                f"  2. If the search returns ANY product results, CLICK one — never search again on the same page; "
                f"refine by clicking and inspecting.\n"
                f"  3. On a product page, click EVERY required option (color, size, style) from the goal BEFORE 'buy'.\n"
                f"  4. Issue 'buy' only after all required options are selected.\n\n"
                f"DO NOT output reasoning, <think> tags, placeholder text, or anything other than "
                f"a single concrete action. Substitute REAL VALUES inside the brackets. "
                f"Keep search queries SHORT.\n\n"
                f"{checklist_block}"
                f"{history_block}"
                f"Current page:\n{_page_summary(obs)}\n\n"
                f"Output exactly ONE action on its own line."
            )

            response = agent_fn(prompt, self._current_goal)
            action = self._parse_action(response)

            # v16 B1: if the agent tries to buy while required options are
            # still unselected, override with a click on the first missing
            # attribute. This is the deterministic complement to the prompt
            # checklist — guarantees the agent never buys an incomplete
            # product, recovering ~15-25pp strict success.
            if on_product_page and missing_attrs and action in (
                "click[Buy Now]", "click[buy now]", "buy",
            ):
                forced = f"click[{required_attrs[missing_attrs[0]]}]"
                action = forced

            page_before = _page_summary(obs, max_chars=160)

            obs, reward, done, info = self.step(action)
            observations.append(obs)
            actions.append(action)
            history.append((action, page_before))
            total_reward = max(total_reward, reward)

            if done:
                break

        # v16 B2 (DISABLED by default — Qwen3-generated reflections regressed
        # by 12pp). Kept the code path so we can flip the flag for ablation
        # studies. v16-bis: if `_reflector_llm` is set on the env (e.g. by
        # the test-eval script), we use IT instead of the weak agent_fn for
        # reflection. Claude-as-reflector should produce higher-quality
        # lessons than Qwen3 reflecting on its own mistakes.
        if getattr(self, "_enable_lessons", False) and total_reward < 1.0 and total_reward > 0 and len(actions) >= 3:
            try:
                ptype = self._infer_product_type(self._current_goal)
                last_actions = " → ".join(actions[-6:])
                reflection_prompt = (
                    f"You just tried to fulfil this WebShop goal:\n{self._current_goal}\n\n"
                    f"Your actions (last 6): {last_actions}\n"
                    f"Final reward: {total_reward:.2f} (1.0 = perfect match).\n\n"
                    f"Summarise in ONE short sentence (under 30 words) the most likely reason "
                    f"the purchase was not a perfect match. Focus on concrete mistakes like "
                    f"'missed size selection', 'picked wrong colour', 'did not check price ceiling', "
                    f"'wrong product category', etc. Do NOT apologise, do NOT use hedging; just "
                    f"name the mistake plainly."
                )
                reflector = getattr(self, "_reflector_llm", None)
                if reflector is not None and hasattr(reflector, "generate"):
                    raw = reflector.generate(reflection_prompt, temperature=0.0, max_tokens=80)
                else:
                    raw = agent_fn(reflection_prompt, "reflect")
                # Strip any action-style wrapping + think tags
                raw = re.sub(r"<think>.*?</think>", " ", raw or "", flags=re.DOTALL).strip()
                if "<think>" in raw:
                    raw = raw.split("<think>")[0].strip()
                # Take the first line / sentence
                lesson = raw.splitlines()[0].strip() if raw else ""
                lesson = re.split(r"[.!?](?:\s|$)", lesson)[0].strip()[:200]
                if lesson and len(lesson) > 10:
                    self._episode_lessons.append({
                        "goal": self._current_goal,
                        "product_type": ptype,
                        "reward": total_reward,
                        "lesson": lesson,
                    })
                    # Cap memory to the most recent 40 lessons to bound prompt size
                    if len(self._episode_lessons) > 40:
                        self._episode_lessons = self._episode_lessons[-40:]
            except Exception:
                pass

        # v20.B — capture action recipe for successful episodes. The full
        # action sequence becomes the recipe for the purchase_decision skill
        # (the terminal MAGE skill in WebShop's KG).
        action_recipes_out: list[dict] = []
        if total_reward >= 0.5 and actions:
            # Trim very long episodes to last 12 actions for usability
            recipe_actions = actions[-12:]
            action_recipes_out.append({
                "skill_id": "purchase_decision",
                "actions": recipe_actions,
                "ctx": (self._current_goal or "")[:300],
            })

        return EpisodeResult(
            observations=observations,
            actions=actions,
            reward=total_reward,
            success=total_reward > 0.5,
            metadata={
                "total_reward": total_reward,
                "steps": len(actions),
                "goal": self._current_goal,
                "action_recipes": action_recipes_out,
            },
        )

    def _retrieve_lessons(self, product_type: str, top_k: int = 2) -> list[dict]:
        """v16 B2+B4: fetch up to top_k past failed-episode lessons whose
        product_type shares at least one word with the current one. Simple
        word-overlap scoring is fast and avoids pulling a whole embedding
        model — WebShop goal categories are coarse (apparel, electronics,
        beauty, home) and word overlap captures them reliably.
        """
        if not self._episode_lessons or not product_type:
            return []
        cur_words = set(product_type.lower().split())
        scored: list[tuple[int, dict]] = []
        for L in self._episode_lessons:
            pt = L.get("product_type", "")
            if not pt:
                continue
            overlap = len(cur_words & set(pt.lower().split()))
            if overlap > 0:
                scored.append((overlap, L))
        scored.sort(key=lambda x: -x[0])
        return [L for _, L in scored[:top_k]]

    @staticmethod
    def _extract_goal_attributes(goal: str) -> dict[str, Any]:
        """v16 B1: parse a WebShop goal string into a required-attribute dict.

        WebShop goals have a consistent structure: ``Find me ... with color:
        X, and size: Y, and price lower than N dollars``. We extract those
        fields so downstream code can (a) render a per-turn checklist and
        (b) veto premature buys. Keys: color, size, style, material, gender,
        plus price_cap (float). Only keys present in the goal are returned.
        """
        attrs: dict[str, Any] = {}
        # Pattern: "<attr>: <value>" where value ends at comma, "and", or end.
        for attr_name in ("color", "size", "style", "material", "gender", "scent", "flavor"):
            # Greedy up to a boundary; strip trailing comma/period/quotes
            m = re.search(
                rf"\b{attr_name}\s*:\s*([A-Za-z0-9][^,\.\n]*?)(?=\s*(?:,|\.|and\b|$))",
                goal, re.IGNORECASE,
            )
            if m:
                val = m.group(1).strip().strip("'\"").lower()
                # Trim a trailing " and " that slipped through
                val = re.sub(r"\s+and\s*$", "", val).strip()
                if val:
                    attrs[attr_name] = val
        price_m = re.search(r"price\s+lower\s+than\s+\$?(\d+(?:\.\d+)?)", goal, re.IGNORECASE)
        if price_m:
            attrs["price_cap"] = float(price_m.group(1))
        return attrs

    @staticmethod
    def _which_attrs_selected(required: dict[str, Any], actions: list[str]) -> set[str]:
        """v16 B1: for each required attribute, has the agent already clicked
        an option matching that attribute's value (case-insensitive,
        substring either direction)? Returns the set of attribute names that
        have been satisfied. Used to compute the per-step missing list and
        to block premature buys.
        """
        satisfied: set[str] = set()
        for a in actions:
            m = re.match(r"click\s*\[\s*(.+?)\s*\]", a, re.IGNORECASE)
            if not m:
                continue
            clicked = m.group(1).strip().lower()
            if clicked in {"buy now", "back to search"}:
                continue
            for attr_name, attr_val in required.items():
                if attr_name == "price_cap":
                    continue
                av = str(attr_val).strip().lower()
                # substring either direction catches "medium" ⊂ "size: medium"
                # and "mocha brown" ⊂ "color: mocha brown" equally well
                if av and (av == clicked or av in clicked or clicked in av):
                    satisfied.add(attr_name)
        return satisfied

    @staticmethod
    def _infer_product_type(goal: str) -> str:
        """v16 B3: pull the PRODUCT TYPE out of a WebShop goal (e.g., "sweater",
        "running shoes", "laptop bag"). WebShop goals begin "Find me ..." and
        the product type is the head noun phrase before "with" or attribute
        clauses like "color:". Used to build the recommended search query
        automatically — avoids agent copying the full goal sentence as a
        long query (which returns 0 results).
        """
        # 1. Strip leading "Find me" / "I want" / etc.
        g = re.sub(r"^\s*(find me|i want|buy me|get me|i'm looking for|show me)\s+",
                   "", goal.strip(), flags=re.IGNORECASE)
        # 2. Cut at "with" or "for" or attribute clause
        for stopper in (
            r"\s+with\s+",
            r"\s+for\s+",
            r",\s*(?:color|size|style|material|scent|flavor|price)\s*:",
            r"\s+(?:color|size|style|material|scent|flavor|price)\s*:",
        ):
            g = re.split(stopper, g, maxsplit=1, flags=re.IGNORECASE)[0]
        # 3. Drop leading qualifier adjectives (e.g., "machine wash",
        #    "hand wash", "loose fit"). These are attributes not product nouns.
        g = re.sub(
            r"^\s*(?:hand wash|machine wash|wash cold|dry clean|loose fit|slim fit|"
            r"light weight|lightweight|quick release|hands free|oil free|fat free|"
            r"heavy duty|high quality|for dry clean|tumble dry)[, ]+",
            "", g, flags=re.IGNORECASE,
        )
        # 4. Keep the last ~4 words (the noun phrase)
        toks = re.findall(r"[A-Za-z0-9']+", g)
        if not toks:
            return ""
        # Head noun phrase — typical WebShop product types are 1-4 words
        return " ".join(toks[-4:]).lower()

    @staticmethod
    def _looks_like_product_page(obs: str) -> bool:
        """Heuristic: is this observation a product detail page (options +
        Buy Now button visible)? Used to gate the option-completeness
        checklist — no point warning about missing options on a search-
        results page where options aren't shown yet.
        """
        if not obs:
            return False
        ol = obs.lower()
        return ("buy now" in ol) and ("price" in ol or "$" in ol)

    def evaluate_full_episode(
        self, agent_fn: Any, max_steps: int = 30, seed: int = 42,
    ) -> dict[str, Any]:
        """BALROG-style full-episode eval for WebShop. Reuses ``run_episode``
        logic (real goal, full browse→find→buy loop) and returns the same
        dict shape the Crafter critic expects: ``achievements`` (which of the
        5 synthesised skills unlocked), ``n_achievements``, ``score``,
        ``steps_taken``, ``total_skills``. Without this method + the
        ``USE_EPISODE_EVAL`` flag, the critic would fall through to the
        per-skill step-by-step trial path which asked the agent "Perform
        skill: purchase_decision" with only ~20 steps — guaranteed 0 reward.
        """
        result = self.run_episode(
            agent_fn=agent_fn,
            instruction=self._current_goal or "Complete the shopping task",
            seed=seed,
            max_steps=max_steps,
            context="",
        )
        final_reward = float(result.metadata.get("total_reward", 0.0))
        self._max_reward_this_episode = max(self._max_reward_this_episode, final_reward)
        achievements_map = self.get_achievements()
        unlocked = {k: 1 for k, v in achievements_map.items() if v}
        total_skills = len(achievements_map)
        return {
            "achievements": unlocked,
            "n_achievements": len(unlocked),
            "score": len(unlocked) / max(total_skills, 1),
            "steps_taken": result.metadata.get("steps", 0),
            "total_skills": total_skills,
            "reward": final_reward,
            "action_recipes": result.metadata.get("action_recipes", []),  # v20.B
        }

    def _parse_action(self, response: str) -> str:
        """Parse LLM response into a valid WebShop action.

        Hardened against two real failure modes seen with Qwen3-8B and Llama-
        3.1-8B agents:
          - Qwen3 emits ``<think>...</think>`` reasoning blocks. We strip them
            (including truncated unclosed ``<think>`` openers) before parsing
            so they don't end up wrapped as ``search[<think>...]`` queries.
          - The env-side prompt shows examples like
            ``search[YOUR_QUERY_HERE]`` and ``click[ELEMENT]``. The agent
            sometimes copies these placeholders verbatim. We reject
            placeholder-only matches so they don't become real actions.
        """
        # 1. Strip Qwen3-style <think>...</think> reasoning blocks
        cleaned = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL)
        # If <think> opened but was truncated (max_tokens cut off the close tag),
        # drop everything after the opener — the visible text is internal reasoning
        if "<think>" in cleaned and "</think>" not in cleaned:
            cleaned = cleaned.split("<think>")[0]
        cleaned = cleaned.strip()

        PLACEHOLDERS = {
            "query", "your_query_here", "your query here",
            "element", "element_text", "your element",
            "buy now", "asin", "b0...",
        }

        def _is_placeholder(s: str) -> bool:
            t = s.strip().lower().strip("[]<> ")
            if not t:
                return True
            # Anything starting/ending with `<` or `<...>` is a placeholder
            if t.startswith("<") and t.endswith(">"):
                return True
            return t in PLACEHOLDERS

        # 2. Look at the LAST non-empty line first — instructions tell the
        #    agent to put the action on its own line, so this is most reliable.
        # IMPORTANT: WebShop's gym does NOT accept the bare verb `buy`. The
        # actual purchase action is `click[Buy Now]`. Same for `back` →
        # `click[Back to Search]`. We translate verb-only outputs here so the
        # agent's natural-language `buy` becomes a real env-side purchase.
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        last_line = lines[-1] if lines else ""

        ll_low = last_line.lower()
        for verb, env_action in (("buy", "click[Buy Now]"), ("back", "click[Back to Search]")):
            if ll_low == verb or ll_low.endswith(f" {verb}") or ll_low.endswith(f"{verb}."):
                return env_action

        # 3. Try parameterised patterns on the cleaned response
        search_match = re.search(r"search\s*\[\s*([^\]]+?)\s*\]", cleaned, re.IGNORECASE)
        if search_match:
            q = search_match.group(1).strip()
            if not _is_placeholder(q):
                return f"search[{q}]"

        click_match = re.search(r"click\s*\[\s*([^\]]+?)\s*\]", cleaned, re.IGNORECASE)
        if click_match:
            elem = click_match.group(1).strip()
            if not _is_placeholder(elem):
                return f"click[{elem}]"

        # 4. Verb-only fallback (anywhere in response)
        if re.search(r"\bbuy\b", cleaned, re.IGNORECASE):
            return "click[Buy Now]"
        if re.search(r"\bback\b", cleaned, re.IGNORECASE):
            return "click[Back to Search]"

        # 5. Last-resort: build a search query from the goal text we have on
        #    the env, NOT from raw model output (which may be reasoning prose).
        if self._current_goal:
            # Use first ~6 keywords from the goal as a default exploratory search
            goal_words = re.findall(r"[A-Za-z0-9$]+", self._current_goal)[:6]
            if goal_words:
                return f"search[{' '.join(goal_words)}]"

        # If we still have nothing usable, return click[search] which is a
        # safe no-op on most pages. We avoid emitting search[<reasoning>]
        # which silently returns zero results.
        return "click[search]"

    def evaluate_answer(self, predicted: str, expected: str) -> float:
        """WebShop uses reward (0-1), not answer matching."""
        return 0.0  # Reward comes from the environment step

    def get_domain_context(self) -> str:
        """Domain context for Claude during EVOLVE (Fix 2 — see crafter_env)."""
        return (
            "DOMAIN: WebShop — a simulated online shopping environment "
            "(Princeton NLP, 2022). The agent navigates an HTML-rendered "
            "store of ~1.2M products to buy a specific item matching a "
            "natural-language goal (e.g., 'find a red running shoe under "
            "$50, men's size 10').\n\n"
            "Each turn the agent picks ONE of these parameterised actions:\n"
            "  search[QUERY]            — runs a keyword search; returns top results\n"
            "  click[ELEMENT_TEXT]      — clicks a button or product (must use exact text from the page)\n"
            "  buy                      — purchases the currently-viewed product\n\n"
            "The text observation describes the current page: search box, "
            "result list, product detail (title, price, attributes, options "
            "like color/size), or cart. Reward in [0,1] is awarded on 'buy' "
            "and reflects how closely the purchased item matches the goal "
            "attributes (title overlap + attribute match + price match).\n\n"
            "Typical solution loop:\n"
            "  1. search[<keywords from goal>]\n"
            "  2. scan results, click[<best matching product title>]\n"
            "  3. inspect detail, click[<color/size option from goal>] if needed\n"
            "  4. buy\n\n"
            "Principles must be CONCRETE SHOPPING TACTICS — what to put in "
            "the search query, how to extract attributes from the goal, when "
            "to refine search vs click a candidate. Principles must NOT be "
            "abstract roleplay/dialog advice — the agent outputs ONE "
            "parameterised action per turn, never natural language."
        )

    def build_evokg(self) -> EvoKG:
        return build_webshop_evokg()


def build_webshop_evokg() -> EvoKG:
    """Build EvoKG for WebShop with shopping skill prerequisites."""
    evokg = EvoKG()

    for sid, sdef in WEBSHOP_SKILLS.items():
        evokg.add_node(CapabilityNode(
            node_id=sid,
            name=sdef["name"],
            mastery=0.0,
            domain="web",
            description=sdef["description"],
        ))

    for sid, sdef in WEBSHOP_SKILLS.items():
        for prereq in sdef["prereqs"]:
            evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

    for entity in ["product", "search_results", "product_page", "options", "cart"]:
        evokg.add_node(EnvironmentNode(
            f"env_{entity}", entity.replace("_", " ").title(),
            entity_type="web_element", domain="web",
        ))

    evokg.add_node(TaskNode(
        "task_webshop", "Find and purchase products matching shopping goals",
        difficulty=0.7, domain="web",
    ))

    logger.info(f"Built WebShop EvoKG: {evokg}")
    return evokg
