"""Tests for the Thompson Sampling experience bandit."""

import random
import tempfile
from pathlib import Path

import pytest

from mage.core.experience_bandit import ExperienceBandit


def _make_exp(node_id: str, outcome: str = "failure", pattern: str = "test") -> dict:
    return {"node_id": node_id, "outcome": outcome, "action_pattern": pattern}


class TestExperienceBanditSelect:
    def test_select_empty(self):
        bandit = ExperienceBandit(k=3)
        assert bandit.select([]) == []

    def test_select_returns_up_to_k(self):
        bandit = ExperienceBandit(k=3)
        exps = [_make_exp(f"e{i}") for i in range(5)]
        selected = bandit.select(exps)
        assert len(selected) == 3

    def test_select_returns_all_when_fewer_than_k(self):
        bandit = ExperienceBandit(k=3)
        exps = [_make_exp("e1"), _make_exp("e2")]
        selected = bandit.select(exps)
        assert len(selected) == 2

    def test_select_lazy_registers_arms(self):
        bandit = ExperienceBandit(k=2)
        exps = [_make_exp("new1"), _make_exp("new2")]
        bandit.select(exps)
        assert "new1" in bandit._arms
        assert "new2" in bandit._arms
        assert bandit._arms["new1"] == {"alpha": 1.0, "beta": 1.0}

    def test_select_deterministic_with_seed(self):
        bandit = ExperienceBandit(k=2)
        exps = [_make_exp(f"e{i}") for i in range(5)]
        r1 = bandit.select(exps, rng=random.Random(42))
        r2 = bandit.select(exps, rng=random.Random(42))
        assert [e["node_id"] for e in r1] == [e["node_id"] for e in r2]

    def test_select_favors_high_reward_arms(self):
        bandit = ExperienceBandit(k=1)
        exps = [_make_exp("good"), _make_exp("bad")]
        # Give "good" a strong track record
        bandit._arms["good"] = {"alpha": 50.0, "beta": 1.0}
        bandit._arms["bad"] = {"alpha": 1.0, "beta": 50.0}
        # Over many samples, "good" should almost always win
        wins = sum(1 for _ in range(100) if bandit.select(exps)[0]["node_id"] == "good")
        assert wins > 90


class TestExperienceBanditUpdate:
    def test_update_increases_alpha_on_reward(self):
        bandit = ExperienceBandit()
        bandit._arms["e1"] = {"alpha": 1.0, "beta": 1.0}
        bandit.update(["e1"], reward=1.0)
        assert bandit._arms["e1"]["alpha"] == 2.0
        assert bandit._arms["e1"]["beta"] == 1.0

    def test_update_increases_beta_on_no_reward(self):
        bandit = ExperienceBandit()
        bandit._arms["e1"] = {"alpha": 1.0, "beta": 1.0}
        bandit.update(["e1"], reward=0.0)
        assert bandit._arms["e1"]["alpha"] == 1.0
        assert bandit._arms["e1"]["beta"] == 2.0

    def test_update_fractional_reward(self):
        bandit = ExperienceBandit()
        bandit._arms["e1"] = {"alpha": 1.0, "beta": 1.0}
        bandit.update(["e1"], reward=0.3)
        assert bandit._arms["e1"]["alpha"] == pytest.approx(1.3)
        assert bandit._arms["e1"]["beta"] == pytest.approx(1.7)

    def test_update_unknown_arm_ignored(self):
        bandit = ExperienceBandit()
        bandit.update(["nonexistent"], reward=1.0)  # should not raise


class TestExperienceBanditFormat:
    def test_format_empty(self):
        bandit = ExperienceBandit()
        assert bandit.format_hints([]) == ""

    def test_format_success_pattern(self):
        bandit = ExperienceBandit()
        result = bandit.format_hints([_make_exp("e1", "success", "gather then craft")])
        assert "Follow this pattern: gather then craft" in result

    def test_format_failure_pattern(self):
        bandit = ExperienceBandit()
        result = bandit.format_hints([_make_exp("e1", "failure", "stuck in loop")])
        assert "Avoid: stuck in loop" in result

    def test_format_mixed(self):
        bandit = ExperienceBandit()
        selected = [
            _make_exp("e1", "success", "good approach"),
            _make_exp("e2", "failure", "bad approach"),
        ]
        result = bandit.format_hints(selected)
        assert "Follow this pattern: good approach" in result
        assert "Avoid: bad approach" in result


class TestExperienceBanditPersistence:
    def test_state_dict_roundtrip(self):
        bandit = ExperienceBandit(k=5)
        bandit._arms = {"e1": {"alpha": 3.0, "beta": 2.0}}
        restored = ExperienceBandit.from_state_dict(bandit.state_dict())
        assert restored.k == 5
        assert restored._arms == {"e1": {"alpha": 3.0, "beta": 2.0}}

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bandit.json"
            bandit = ExperienceBandit(k=4)
            bandit._arms = {"a": {"alpha": 5.0, "beta": 1.0}}
            bandit.save(path)

            loaded = ExperienceBandit.load(path)
            assert loaded.k == 4
            assert loaded._arms == {"a": {"alpha": 5.0, "beta": 1.0}}

    def test_load_missing_file_returns_fresh(self):
        bandit = ExperienceBandit.load("/nonexistent/path/bandit.json")
        assert bandit._arms == {}
        assert bandit.k == 3
