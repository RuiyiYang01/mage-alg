"""Thompson Sampling bandit for adaptive experience retrieval.

Learns which experience patterns (success and failure) are most useful
for guiding the Explorer, replacing the fixed "first 3 failures" heuristic.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperienceBandit:
    """Thompson Sampling bandit over experience nodes.

    Each arm corresponds to an experience node_id. The bandit maintains
    Beta(alpha, beta) parameters per arm. New experiences are lazily
    registered with a Beta(1, 1) uniform prior on first selection.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self._arms: dict[str, dict[str, float]] = {}

    def select(
        self,
        experiences: list[dict[str, Any]],
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        """Select up to K experiences using Thompson Sampling.

        Args:
            experiences: Experience node dicts from evokg.get_experiences_for_skill().
            rng: Optional Random instance for reproducibility.

        Returns:
            Up to K experience dicts, ranked by Thompson sample.
        """
        if not experiences:
            return []

        rng = rng or random.Random()

        # Lazy-register new arms
        for exp in experiences:
            nid = exp["node_id"]
            if nid not in self._arms:
                self._arms[nid] = {"alpha": 1.0, "beta": 1.0}

        # Sample from Beta distribution for each candidate
        scored = []
        for exp in experiences:
            arm = self._arms[exp["node_id"]]
            theta = rng.betavariate(arm["alpha"], arm["beta"])
            scored.append((theta, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[: self.k]]

    def update(self, node_ids: list[str], reward: float) -> None:
        """Update selected arms with observed reward.

        Args:
            node_ids: IDs of the experiences that were selected.
            reward: Fraction of episodes that succeeded (in [0, 1]).
        """
        for nid in node_ids:
            if nid in self._arms:
                self._arms[nid]["alpha"] += reward
                self._arms[nid]["beta"] += 1.0 - reward

    def format_hints(self, selected: list[dict[str, Any]]) -> str:
        """Format selected experiences into a prompt hint string.

        Success patterns → "Follow this pattern: ..."
        Failure patterns → "Avoid: ..."
        """
        if not selected:
            return ""

        lines = ["Experience-based guidance for this skill:"]
        for exp in selected:
            pattern = exp.get("action_pattern", "unknown")
            if exp.get("outcome") == "success":
                lines.append(f"- Follow this pattern: {pattern}")
            else:
                lines.append(f"- Avoid: {pattern}")
        return "\n".join(lines)

    # -- Serialization -------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"k": self.k, "arms": dict(self._arms)}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> ExperienceBandit:
        bandit = cls(k=data.get("k", 3))
        bandit._arms = {k: dict(v) for k, v in data.get("arms", {}).items()}
        return bandit

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)
        logger.info(f"Saved bandit state to {path} ({len(self._arms)} arms)")

    @classmethod
    def load(cls, path: str | Path) -> ExperienceBandit:
        path = Path(path)
        if not path.exists():
            logger.info(f"No bandit state at {path}, starting fresh")
            return cls()
        with open(path) as f:
            data = json.load(f)
        bandit = cls.from_state_dict(data)
        logger.info(f"Loaded bandit state from {path} ({len(bandit._arms)} arms)")
        return bandit
