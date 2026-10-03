"""Strategy Router: Bandit-based strategy selection for EvoKG skills.

Instead of updating LLM weights (LoRA/GRPO), this learns WHICH strategy
to apply for each skill type. The LLM stays frozen.

Each skill can have multiple candidate strategies:
  - direct: pass question to LLM as-is
  - tool:<name>: extract params, call tool, inject result into prompt
  - cot:<template>: chain-of-thought with specific template
  - decompose: break into sub-questions

The router uses Thompson Sampling (per-skill, per-strategy) to learn
which strategy works best, updated by answer accuracy.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StrategyRouter:
    """Thompson Sampling bandit that selects the best strategy per skill.

    State: skill_id
    Action: which strategy to apply
    Reward: answer accuracy (0 or 1)

    Each (skill, strategy) pair has Beta(alpha, beta) parameters.
    """

    def __init__(self):
        # {skill_id: {strategy_key: {"alpha": float, "beta": float}}}
        self._arms: dict[str, dict[str, dict[str, float]]] = {}

    def register_strategies(self, skill_id: str, strategies: list[dict[str, Any]]) -> None:
        """Register candidate strategies for a skill."""
        if skill_id not in self._arms:
            self._arms[skill_id] = {}

        for strategy in strategies:
            key = self._strategy_key(strategy)
            if key not in self._arms[skill_id]:
                self._arms[skill_id][key] = {"alpha": 1.0, "beta": 1.0, "strategy": strategy}

    def select_strategy(self, skill_id: str, rng: random.Random | None = None, deterministic: bool = False) -> dict[str, Any]:
        """Select strategy for a skill.

        Args:
            deterministic: If True, pick strategy with highest expected value
                          (for evaluation — reproducible). If False, use
                          Thompson Sampling (for training — exploration).
        """
        if skill_id not in self._arms or not self._arms[skill_id]:
            return {"type": "direct"}

        # Prefer tools ONLY if they have proven success
        for key, arm in self._arms[skill_id].items():
            if "tool:" in key:
                successes = arm["alpha"] - 1
                if successes >= 1:
                    return arm.get("strategy", {"type": "direct"})

        if deterministic:
            # Pick strategy with highest expected value: alpha / (alpha + beta)
            best_key = None
            best_ev = -1.0
            for key, arm in self._arms[skill_id].items():
                ev = arm["alpha"] / (arm["alpha"] + arm["beta"])
                if ev > best_ev:
                    best_ev = ev
                    best_key = key
            return self._arms[skill_id][best_key].get("strategy", {"type": "direct"})
        else:
            # Thompson Sampling (exploration)
            rng = rng or random.Random()
            best_key = None
            best_sample = -1.0
            for key, arm in self._arms[skill_id].items():
                sample = rng.betavariate(arm["alpha"], arm["beta"])
                if sample > best_sample:
                    best_sample = sample
                    best_key = key
            return self._arms[skill_id][best_key].get("strategy", {"type": "direct"})

    def update(self, skill_id: str, strategy: dict[str, Any], reward: float) -> None:
        """Update strategy arm with observed reward."""
        key = self._strategy_key(strategy)
        if skill_id in self._arms and key in self._arms[skill_id]:
            self._arms[skill_id][key]["alpha"] += reward
            self._arms[skill_id][key]["beta"] += 1.0 - reward

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        stats = {}
        for skill_id, arms in self._arms.items():
            skill_stats = {}
            for key, arm in arms.items():
                total = arm["alpha"] + arm["beta"] - 2  # subtract prior
                win_rate = (arm["alpha"] - 1) / max(total, 1)
                skill_stats[key] = {
                    "win_rate": round(win_rate, 3),
                    "trials": int(total),
                    "alpha": round(arm["alpha"], 2),
                    "beta": round(arm["beta"], 2),
                }
            stats[skill_id] = skill_stats
        return stats

    def _strategy_key(self, strategy: dict[str, Any]) -> str:
        """Create a hashable key for a strategy."""
        stype = strategy.get("type", "direct")
        if stype == "tool":
            return f"tool:{strategy.get('tool', 'unknown')}"
        elif stype == "cot":
            return f"cot:{hash(strategy.get('prompt', ''))}"
        elif stype == "compose":
            return f"compose:{'+'.join(strategy.get('skills', []))}"
        return stype

    # -- Serialization --

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._arms, f, indent=2, default=str)
        logger.info(f"Saved strategy router to {path}")

    @classmethod
    def load(cls, path: str | Path) -> StrategyRouter:
        router = cls()
        path = Path(path)
        if path.exists():
            with open(path) as f:
                router._arms = json.load(f)
            logger.info(f"Loaded strategy router from {path}")
        return router

    def build_prompt_for_strategy(
        self,
        question: str,
        context: str,
        strategy: dict[str, Any],
        principles: list[str] | None = None,
        tool_result: str | None = None,
    ) -> str:
        """Build the prompt based on selected strategy + principles + tool result."""
        parts = []

        # 1. Inject principles (abstract rules from Claude)
        if principles:
            parts.append("IMPORTANT RULES for this type of question:")
            for i, p in enumerate(principles, 1):
                parts.append(f"  {i}. {p}")
            parts.append("")

        # 2. Context + question
        if context:
            parts.append(context)

        # 3. Strategy-specific augmentation
        stype = strategy.get("type", "direct")

        if stype == "tool" and tool_result is not None:
            parts.append(f"\nCOMPUTED ANSWER (verified): {tool_result}")
            parts.append("Match this result to the correct option below.\n")

        # 4. Always append the question
        parts.append(question)

        return "\n".join(parts)

    def execute_hybrid_verification(
        self,
        question: str,
        strategy: dict[str, Any],
        llm: Any,
    ) -> str | None:
        """For tool-type strategies, extract params and call the tool.

        Three extraction methods tried in order:
        1. Regex-based extraction (fast, deterministic, no LLM needed)
        2. LLM extraction with focused prompt
        3. Give up → return None (fallback to direct LLM answer)
        """
        if strategy.get("type") != "tool":
            return None

        tool_name = strategy.get("tool", "")
        if not tool_name:
            return None

        from mage.tools.st_tools import TOOL_REGISTRY, call_tool
        if tool_name not in TOOL_REGISTRY:
            return None

        # Method 1: Regex-based extraction (no LLM needed)
        params = self._regex_extract_params(question, tool_name)
        if params:
            result = call_tool(tool_name, __import__("json").dumps(params))
            if not result.startswith("Error"):
                return result

        # Method 2: LLM extraction with examples
        param_info = strategy.get("param_extraction", "")
        example = strategy.get("example_params", self._get_tool_example(tool_name))

        extraction_prompt = (
            f"Extract parameters from the question below for the function '{tool_name}'.\n"
            f"Instructions: {param_info}\n\n"
            f"Question: {question[:500]}\n\n"
            f"Output ONLY a valid JSON object. Example format:\n{example}"
        )

        try:
            import json as _json
            import re
            response = llm.generate(extraction_prompt, temperature=0.0, max_tokens=256)
            # Find JSON in response (may be nested)
            for match in re.finditer(r'\{[^{}]*\}', response):
                try:
                    params = _json.loads(match.group(0))
                    result = call_tool(tool_name, _json.dumps(params))
                    if not result.startswith("Error"):
                        return result
                except (_json.JSONDecodeError, Exception):
                    continue
        except Exception:
            pass

        return None

    def _regex_extract_params(self, question: str, tool_name: str) -> dict | None:
        """Try to extract parameters using regex patterns (no LLM needed).

        This handles common STBench/STARK question formats deterministically.
        """
        import re

        if tool_name == "compute_direction":
            # Pattern: "A has a longitude of X and a latitude of Y, while B has a longitude of X2 and a latitude of Y2"
            m = re.search(
                r'[Aa]\s+has\s+a\s+longitude\s+of\s+(-?[\d]+\.[\d]+)\s+and\s+a\s+latitude\s+of\s+(-?[\d]+\.[\d]+).*?'
                r'[Bb]\s+has\s+a\s+longitude\s+of\s+(-?[\d]+\.[\d]+)\s+and\s+a\s+latitude\s+of\s+(-?[\d]+\.[\d]+)',
                question, re.DOTALL
            )
            if m:
                return {"lon_a": float(m.group(1)), "lat_a": float(m.group(2)),
                        "lon_b": float(m.group(3)), "lat_b": float(m.group(4))}

        elif tool_name in ("check_temporal_relation", "check_allen_relation"):
            # Pattern: "time interval (X, Y) has the temporal relationship 'Z' with (A, B)"
            m = re.search(
                r'interval\s*\(?([\d.\-]+)[,\s]+([\d.\-]+)\)?\s+.*?'
                r'(?:relationship|relation)\s+["\']?(\w+)["\']?\s+.*?'
                r'\(?([\d.\-]+)[,\s]+([\d.\-]+)\)?',
                question, re.DOTALL | re.IGNORECASE
            )
            if m:
                return {"interval_a": [float(m.group(1)), float(m.group(2))],
                        "relation": m.group(3).lower(),
                        "interval_b": [float(m.group(4)), float(m.group(5))]}

        elif tool_name in ("check_spatial_relation", "check_spatial_relationship"):
            # Pattern: "spatial relationship ** contains ** between..."
            m = re.search(r'relationship\s+\*{0,2}\s*(\w+)\s*\*{0,2}', question, re.IGNORECASE)
            if m:
                relation = m.group(1).lower()
                # Extract coordinate lists
                coords = re.findall(r'\[([\d.\-]+),\s*([\d.\-]+)\]', question)
                if len(coords) >= 2:
                    # Split into two geometries (heuristic: first half / second half)
                    mid = len(coords) // 2
                    geom1 = [[float(x), float(y)] for x, y in coords[:mid]]
                    geom2 = [[float(x), float(y)] for x, y in coords[mid:]]
                    return {"relation": relation, "geom1": geom1, "geom2": geom2}

        return None

    def _map_tool_result_to_option(self, question: str, tool_result: str) -> str | None:
        """Try to map tool output directly to an answer option in the question.

        Returns the option number/letter if mapping succeeds, None otherwise.
        """
        import json as _json
        try:
            result = _json.loads(tool_result) if isinstance(tool_result, str) else tool_result
        except _json.JSONDecodeError:
            return None

        if not isinstance(result, dict):
            return None

        # Get the main value from tool result
        value = None
        for key in ["direction", "holds", "inside", "result", "distance_km"]:
            if key in result:
                value = result[key]
                break
        if value is None:
            return None

        # Boolean results → map to 1.0/0.0 or True/False or Yes/No
        if isinstance(value, bool):
            # Check question for [1.0]/[0.0] format (STARK style)
            return "1.0" if value else "0.0"

        # String results (like "Southwest") → find matching option number
        if isinstance(value, str):
            import re
            value_lower = value.lower().strip()

            # Direction abbreviation mapping
            dir_aliases = {
                "north": ["north", "n"], "northeast": ["northeast", "ne", "north-east"],
                "east": ["east", "e"], "southeast": ["southeast", "se", "south-east"],
                "south": ["south", "s"], "southwest": ["southwest", "sw", "south-west"],
                "west": ["west", "w"], "northwest": ["northwest", "nw", "north-west"],
            }

            # Find numbered options: (1) North, (2) Northeast, etc.
            options = re.findall(r'\((\d+)\)\s*([^,()\n]+)', question)
            if not options:
                options = re.findall(r'(\d+)[):]\s*([^,\n]+)', question)

            # First pass: exact match
            for num, text in options:
                text_lower = text.lower().strip()
                if value_lower == text_lower:
                    return num

            # Second pass: alias match (full word match, not substring)
            for num, text in options:
                text_lower = text.lower().strip()
                for full, aliases in dir_aliases.items():
                    if value_lower == full or value_lower in aliases:
                        if text_lower == full or text_lower in aliases:
                            return num

        return None

    def _get_tool_example(self, tool_name: str) -> str:
        """Get an example parameter JSON for a tool."""
        examples = {
            "compute_direction": '{"lon_a": 114.28, "lat_a": 36.22, "lon_b": 117.05, "lat_b": 34.76}',
            "compute_distance": '{"lat1": 40.7, "lon1": -74.0, "lat2": 34.0, "lon2": -118.2}',
            "check_point_in_region": '{"point": [39.9, -75.0], "polygon": [[39.8,-75.1],[40.0,-75.1],[40.0,-74.9],[39.8,-74.9]]}',
            "check_temporal_relation": '{"relation": "precedes", "interval_a": [1.0, 3.0], "interval_b": [4.0, 6.0]}',
            "check_spatial_relation": '{"relation": "contains", "geom1": [[0,0],[10,10]], "geom2": [[5,5]]}',
        }
        return examples.get(tool_name, '{}')
