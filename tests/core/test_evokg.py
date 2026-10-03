"""Tests for EvoKG core data structure."""

import json
import tempfile
from pathlib import Path

import pytest

from mage.core.evokg import EvoKG
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    ExperienceNode,
    NodeType,
    RelationType,
    Subgraph,
    TaskNode,
)


class TestEvoKGBasicOperations:
    def test_create_empty(self):
        kg = EvoKG()
        assert kg.graph.number_of_nodes() == 0
        assert kg.graph.number_of_edges() == 0
        assert kg.iteration == 0

    def test_add_capability_node(self):
        kg = EvoKG()
        node = CapabilityNode("collect_wood", "Collect Wood", mastery=0.0, domain="crafter")
        kg.add_node(node)
        assert kg.has_node("collect_wood")
        data = kg.get_node("collect_wood")
        assert data["name"] == "Collect Wood"
        assert data["mastery"] == 0.0
        assert data["node_type"] == "capability"
        assert data["subgraph"] == "G_cap"

    def test_add_task_node(self):
        kg = EvoKG()
        node = TaskNode("task_survive", "Survive 100 steps", difficulty=0.8, domain="crafter")
        kg.add_node(node)
        assert kg.has_node("task_survive")
        data = kg.get_node("task_survive")
        assert data["node_type"] == "task"
        assert data["difficulty"] == 0.8

    def test_add_experience_node(self):
        kg = EvoKG()
        node = ExperienceNode(
            "exp_1", "collect_wood → place_table → craft_pickaxe",
            outcome="success", domain_origin="crafter",
        )
        kg.add_node(node)
        assert kg.has_node("exp_1")

    def test_add_environment_node(self):
        kg = EvoKG()
        node = EnvironmentNode("env_tree", "tree", entity_type="object", domain="crafter")
        kg.add_node(node)
        assert kg.has_node("env_tree")

    def test_remove_node(self):
        kg = EvoKG()
        kg.add_node(CapabilityNode("skill_a", "Skill A"))
        assert kg.has_node("skill_a")
        kg.remove_node("skill_a")
        assert not kg.has_node("skill_a")

    def test_get_nonexistent_node(self):
        kg = EvoKG()
        assert kg.get_node("nonexistent") is None

    def test_add_edge(self):
        kg = EvoKG()
        kg.add_node(CapabilityNode("a", "A"))
        kg.add_node(CapabilityNode("b", "B"))
        kg.add_edge(Edge("a", "b", RelationType.PREREQUISITE_OF))
        edges = kg.get_edges_from("a")
        assert len(edges) == 1
        assert edges[0]["dst"] == "b"
        assert edges[0]["relation"] == RelationType.PREREQUISITE_OF.value

    def test_get_edges_to(self):
        kg = EvoKG()
        kg.add_node(CapabilityNode("a", "A"))
        kg.add_node(CapabilityNode("b", "B"))
        kg.add_edge(Edge("a", "b", RelationType.PREREQUISITE_OF))
        edges = kg.get_edges_to("b")
        assert len(edges) == 1
        assert edges[0]["src"] == "a"

    def test_get_nodes_by_type(self):
        kg = EvoKG()
        kg.add_node(CapabilityNode("s1", "Skill 1"))
        kg.add_node(CapabilityNode("s2", "Skill 2"))
        kg.add_node(TaskNode("t1", "Task 1"))
        caps = kg.get_nodes_by_type(NodeType.CAPABILITY)
        assert len(caps) == 2
        tasks = kg.get_nodes_by_type(NodeType.TASK)
        assert len(tasks) == 1


class TestSubgraphViews:
    def _build_mixed_kg(self) -> EvoKG:
        kg = EvoKG()
        kg.add_node(CapabilityNode("s1", "Skill 1"))
        kg.add_node(CapabilityNode("s2", "Skill 2"))
        kg.add_node(TaskNode("t1", "Task 1"))
        kg.add_node(ExperienceNode("e1", "pattern", "success"))
        kg.add_node(EnvironmentNode("env1", "tree"))
        return kg

    def test_g_cap_view(self):
        kg = self._build_mixed_kg()
        assert len(kg.G_cap) == 2

    def test_g_task_view(self):
        kg = self._build_mixed_kg()
        assert len(kg.G_task) == 1

    def test_g_exp_view(self):
        kg = self._build_mixed_kg()
        assert len(kg.G_exp) == 1

    def test_g_env_view(self):
        kg = self._build_mixed_kg()
        assert len(kg.G_env) == 1


class TestMasteryAndPrerequisites:
    def _build_skill_chain(self) -> EvoKG:
        """Build: collect_wood → make_pickaxe → collect_stone → make_sword"""
        kg = EvoKG()
        kg.add_node(CapabilityNode("collect_wood", "Collect Wood", mastery=0.0))
        kg.add_node(CapabilityNode("make_pickaxe", "Make Pickaxe", mastery=0.0))
        kg.add_node(CapabilityNode("collect_stone", "Collect Stone", mastery=0.0))
        kg.add_node(CapabilityNode("make_sword", "Make Sword", mastery=0.0))
        kg.add_edge(Edge("collect_wood", "make_pickaxe", RelationType.PREREQUISITE_OF))
        kg.add_edge(Edge("make_pickaxe", "collect_stone", RelationType.PREREQUISITE_OF))
        kg.add_edge(Edge("collect_stone", "make_sword", RelationType.PREREQUISITE_OF))
        return kg

    def test_update_mastery(self):
        kg = self._build_skill_chain()
        kg.update_mastery("collect_wood", 0.8, "iter_1")
        assert kg.get_mastery("collect_wood") == 0.8
        data = kg.get_node("collect_wood")
        assert data["last_evaluated"] == "iter_1"

    def test_mastery_clamped(self):
        kg = self._build_skill_chain()
        kg.update_mastery("collect_wood", 1.5)
        assert kg.get_mastery("collect_wood") == 1.0
        kg.update_mastery("collect_wood", -0.5)
        assert kg.get_mastery("collect_wood") == 0.0

    def test_update_mastery_wrong_type(self):
        kg = EvoKG()
        kg.add_node(TaskNode("t1", "Task"))
        with pytest.raises(TypeError):
            kg.update_mastery("t1", 0.5)

    def test_get_prerequisites(self):
        kg = self._build_skill_chain()
        prereqs = kg.get_prerequisites("make_pickaxe")
        assert prereqs == ["collect_wood"]

    def test_get_prerequisites_none(self):
        kg = self._build_skill_chain()
        prereqs = kg.get_prerequisites("collect_wood")
        assert prereqs == []

    def test_learnable_frontier_initial(self):
        """Initially only collect_wood (no prereqs) should be in frontier."""
        kg = self._build_skill_chain()
        frontier = kg.get_learnable_frontier(mastery_threshold=0.5)
        frontier_ids = [n["node_id"] for n in frontier]
        assert "collect_wood" in frontier_ids
        assert "make_pickaxe" not in frontier_ids  # prereq not mastered

    def test_learnable_frontier_after_mastery(self):
        """After mastering collect_wood, make_pickaxe enters frontier."""
        kg = self._build_skill_chain()
        kg.update_mastery("collect_wood", 0.8)
        frontier = kg.get_learnable_frontier(mastery_threshold=0.5)
        frontier_ids = [n["node_id"] for n in frontier]
        assert "collect_wood" not in frontier_ids  # already mastered
        assert "make_pickaxe" in frontier_ids
        assert "collect_stone" not in frontier_ids  # prereq chain not satisfied

    def test_learnable_frontier_progressive(self):
        """Master skills progressively, frontier advances."""
        kg = self._build_skill_chain()
        kg.update_mastery("collect_wood", 0.9)
        kg.update_mastery("make_pickaxe", 0.7)
        frontier = kg.get_learnable_frontier(mastery_threshold=0.5)
        frontier_ids = [n["node_id"] for n in frontier]
        assert frontier_ids == ["collect_stone"]

    def test_all_mastered_empty_frontier(self):
        kg = self._build_skill_chain()
        for skill in ["collect_wood", "make_pickaxe", "collect_stone", "make_sword"]:
            kg.update_mastery(skill, 1.0)
        frontier = kg.get_learnable_frontier(mastery_threshold=0.5)
        assert len(frontier) == 0


class TestExperience:
    def test_add_and_retrieve_experience(self):
        kg = EvoKG()
        kg.add_node(CapabilityNode("skill_a", "Skill A"))
        exp = ExperienceNode("exp_1", "do X then Y", "success")
        kg.add_experience(exp)
        kg.add_edge(Edge("exp_1", "skill_a", RelationType.EXPERIENCE_OF_SKILL))
        exps = kg.get_experiences_for_skill("skill_a")
        assert len(exps) == 1
        assert exps[0]["action_pattern"] == "do X then Y"

    def test_prune_low_confidence(self):
        kg = EvoKG()
        kg.add_experience(ExperienceNode("e1", "p1", "success", confidence=0.1))
        kg.add_experience(ExperienceNode("e2", "p2", "success", confidence=0.8))
        kg.add_experience(ExperienceNode("e3", "p3", "failure", confidence=0.2))
        removed = kg.prune_low_confidence(threshold=0.3)
        assert removed == 2
        assert not kg.has_node("e1")
        assert kg.has_node("e2")
        assert not kg.has_node("e3")


class TestSerialization:
    def test_save_and_load(self, tmp_path):
        kg = EvoKG()
        kg.iteration = 3
        kg.add_node(CapabilityNode("s1", "Skill 1", mastery=0.7, domain="test"))
        kg.add_node(CapabilityNode("s2", "Skill 2", mastery=0.3, domain="test"))
        kg.add_edge(Edge("s1", "s2", RelationType.PREREQUISITE_OF, confidence=0.9))

        path = tmp_path / "evokg.json"
        kg.save_snapshot(path)

        # Verify JSON is valid and readable
        with open(path) as f:
            data = json.load(f)
        assert data["metadata"]["iteration"] == 3
        assert "s1" in data["nodes"]
        assert len(data["edges"]) == 1

        # Load and verify
        kg2 = EvoKG.load_snapshot(path)
        assert kg2.iteration == 3
        assert kg2.has_node("s1")
        assert kg2.has_node("s2")
        assert kg2.get_mastery("s1") == 0.7
        edges = kg2.get_edges_from("s1")
        assert len(edges) == 1


class TestSkeleton:
    def test_create_with_skeleton(self):
        kg = EvoKG.create_with_skeleton(domain="crafter")
        assert kg.has_node("meta_explore")
        assert kg.has_node("meta_collect")
        assert kg.has_node("meta_compose")
        assert kg.has_node("meta_use")
        assert kg.has_node("meta_defend")
        assert len(kg.G_cap) == 5

        # Check prerequisite chain: explore → collect → compose → use
        prereqs = kg.get_prerequisites("meta_collect")
        assert "meta_explore" in prereqs

    def test_skeleton_frontier(self):
        kg = EvoKG.create_with_skeleton()
        frontier = kg.get_learnable_frontier()
        frontier_ids = [n["node_id"] for n in frontier]
        # Only explore and defend should be in frontier (no prereqs)
        assert "meta_explore" in frontier_ids
        assert "meta_defend" in frontier_ids
        assert "meta_collect" not in frontier_ids  # needs explore


class TestStats:
    def test_stats(self):
        kg = EvoKG.create_with_skeleton()
        kg.iteration = 2
        kg.update_mastery("meta_explore", 0.9)
        s = kg.stats()
        assert s["iteration"] == 2
        assert s["capability_nodes"] == 5
        assert s["mastered_skills"] == 1
        assert s["total_edges"] == 3

    def test_repr(self):
        kg = EvoKG.create_with_skeleton()
        r = repr(kg)
        assert "EvoKG(" in r
        assert "nodes=5" in r
