"""Node and edge type definitions for EvoKG."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Subgraph identifiers
# ---------------------------------------------------------------------------

class Subgraph(str, Enum):
    CAPABILITY = "G_cap"
    TASK = "G_task"
    EXPERIENCE = "G_exp"
    ENVIRONMENT = "G_env"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    CAPABILITY = "capability"
    TASK = "task"
    EXPERIENCE = "experience"
    ENVIRONMENT = "environment"


@dataclass
class CapabilityNode:
    """A skill or capability the agent has or could have.

    Extended for MAGE v2: each skill has a strategy (how to solve),
    prompt_template (what to tell the Learner), and composability.
    """
    node_id: str
    name: str
    mastery: float = 0.0          # [0, 1]
    domain: str = "general"
    last_evaluated: str | None = None   # iteration tag, e.g. "iter_3"
    description: str = ""
    # v2: strategy and teaching
    strategy: dict[str, Any] = field(default_factory=dict)
    # e.g. {"type": "tool", "tool": "compute_direction",
    #        "param_extraction": "Extract lon_a, lat_a, lon_b, lat_b"}
    # e.g. {"type": "direct"} — just reason
    # e.g. {"type": "compose", "skills": ["coord_extract", "distance_calc"]}
    prompt_template: str = ""     # specific instruction for the Learner
    composable_with: list[str] = field(default_factory=list)  # skill A+B
    principles: list[str] = field(default_factory=list)  # abstract rules from Claude
    discovered_at: str = ""       # which iteration discovered this skill
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> NodeType:
        return NodeType.CAPABILITY

    @property
    def subgraph(self) -> Subgraph:
        return Subgraph.CAPABILITY

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["subgraph"] = self.subgraph.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilityNode:
        d = {k: v for k, v in d.items() if k not in ("node_type", "subgraph")}
        return cls(**d)


@dataclass
class TaskNode:
    """A task or goal the agent needs to accomplish."""
    node_id: str
    name: str
    difficulty: float = 0.5       # [0, 1]
    domain: str = "general"
    success_rate: float = 0.0     # [0, 1]
    attempts: int = 0
    completed: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> NodeType:
        return NodeType.TASK

    @property
    def subgraph(self) -> Subgraph:
        return Subgraph.TASK

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["subgraph"] = self.subgraph.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskNode:
        d = {k: v for k, v in d.items() if k not in ("node_type", "subgraph")}
        return cls(**d)


@dataclass
class ExperienceNode:
    """An abstracted experience pattern (not raw trajectory)."""
    node_id: str
    action_pattern: str           # e.g. "gather_wood → place_table → craft_pickaxe"
    outcome: str                  # "success" | "failure"
    context_conditions: str = ""  # when this pattern applies
    confidence: float = 1.0       # [0, 1]
    domain_origin: str = "general"
    frequency: int = 1            # how many times observed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> NodeType:
        return NodeType.EXPERIENCE

    @property
    def subgraph(self) -> Subgraph:
        return Subgraph.EXPERIENCE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["subgraph"] = self.subgraph.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperienceNode:
        d = {k: v for k, v in d.items() if k not in ("node_type", "subgraph")}
        return cls(**d)


@dataclass
class EnvironmentNode:
    """An entity, state, or object in the environment."""
    node_id: str
    name: str
    entity_type: str = "object"   # "object" | "location" | "state" | "action"
    domain: str = "general"
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> NodeType:
        return NodeType.ENVIRONMENT

    @property
    def subgraph(self) -> Subgraph:
        return Subgraph.ENVIRONMENT

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        d["subgraph"] = self.subgraph.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentNode:
        d = {k: v for k, v in d.items() if k not in ("node_type", "subgraph")}
        return cls(**d)


# Factory for deserialization
NODE_CLASSES: dict[str, type] = {
    NodeType.CAPABILITY.value: CapabilityNode,
    NodeType.TASK.value: TaskNode,
    NodeType.EXPERIENCE.value: ExperienceNode,
    NodeType.ENVIRONMENT.value: EnvironmentNode,
}


def node_from_dict(d: dict[str, Any]) -> CapabilityNode | TaskNode | ExperienceNode | EnvironmentNode:
    """Deserialize a node from a dictionary."""
    node_type = d.get("node_type")
    if node_type not in NODE_CLASSES:
        raise ValueError(f"Unknown node_type: {node_type}")
    return NODE_CLASSES[node_type].from_dict(d)


# ---------------------------------------------------------------------------
# Edge / relation types
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    # G_cap edges
    PREREQUISITE_OF = "prerequisite_of"
    COMPOSES_INTO = "composes_into"

    # G_task edges
    DECOMPOSES_TO = "decomposes_to"
    DEPENDS_ON = "depends_on"

    # G_exp edges
    LED_TO_SUCCESS = "led_to_success"
    CAUSED_FAILURE = "caused_failure"
    SIMILAR_TO = "similar_to"
    GENERALIZATION_OF = "generalization_of"

    # G_env edges
    SPATIAL_RELATION = "spatial_relation"
    CAUSAL_RELATION = "causal_relation"
    TEMPORAL_RELATION = "temporal_relation"

    # Cross-subgraph edges
    SKILL_FOR_TASK = "skill_for_task"        # G_cap → G_task
    EXPERIENCE_OF_SKILL = "experience_of_skill"  # G_exp → G_cap
    PRINCIPLE_OF_SKILL = "principle_of_skill"      # G_exp → G_cap (abstract rules)
    CROSS_DOMAIN_ANALOGY = "cross_domain_analogy"  # G_cap ↔ G_cap across domains


@dataclass
class Edge:
    """An edge in the EvoKG."""
    src: str
    dst: str
    relation: RelationType | str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["relation"] = self.relation.value if isinstance(self.relation, RelationType) else self.relation
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Edge:
        rel = d.get("relation", "")
        try:
            d["relation"] = RelationType(rel)
        except ValueError:
            d["relation"] = rel
        return cls(**d)


# ---------------------------------------------------------------------------
# Structured evaluation output (from Critic)
# ---------------------------------------------------------------------------

@dataclass
class FailureMode:
    pattern: str          # e.g. "missing_prerequisite"
    detail: str           # e.g. "no stone_pickaxe"
    frequency: float      # [0, 1] — proportion of failures with this pattern

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillEvaluation:
    """Structured evaluation of a single skill."""
    skill_id: str
    success_rate: float
    attempts: int
    failure_modes: list[FailureMode] = field(default_factory=list)
    recommended_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_modes"] = [fm.to_dict() for fm in self.failure_modes]
        return d


@dataclass
class StructuredEvaluation:
    """Full structured evaluation from Critic.

    v15.1: ``per_question_results`` stores per-question outcomes so the
    evolution loop can harvest every correct answer as a success memory
    (previously only 10 re-run questions were checked per iter, yielding
    <5 success memories across a 20-iter run). Each entry contains:

        {
            "question": str,
            "context": str,          # QA context (HotpotQA) or "" (WebQA/STBench/GSM8K)
            "predicted": str,        # model output
            "expected": str,         # ground truth
            "task_type": str,
            "score": float,          # 0..1 from judge/F1
            "correct": bool,         # score >= 0.5
            "search_strategy": str,  # v15.2: which arm was used (if any)
        }
    """
    iteration: int
    evaluations: list[SkillEvaluation] = field(default_factory=list)
    overall_success_rate: float = 0.0
    summary: str = ""
    per_question_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "overall_success_rate": self.overall_success_rate,
            "summary": self.summary,
            # Keep per-question results out of the on-disk dict by default
            # (they can be very large). The evolution loop reads them directly
            # from the in-memory StructuredEvaluation before serialization.
        }


# ---------------------------------------------------------------------------
# Evolution plan (from Navigator)
# ---------------------------------------------------------------------------

@dataclass
class EvolutionTarget:
    """A single skill to target in the next evolution round.

    v12.2 — multi-hop chains: ``skill_chain`` is the topologically-ordered
    list of prerequisite skills leading to ``skill_id``. Empty if the
    Navigator did not compute one (e.g., no prerequisites). Downstream
    consumers (Explorer/Critic) can use the chain to evaluate per-step
    progress instead of treating the target as a single atomic skill.
    """
    skill_id: str
    priority: float               # higher = more important
    reason: str = ""              # why this skill was chosen
    exploration_brief: str = ""   # guidance for Explorer
    skill_chain: list[str] = field(default_factory=list)  # multi-hop prereq chain

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionPlan:
    """Plan produced by Navigator for one evolution iteration."""
    iteration: int
    targets: list[EvolutionTarget] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "targets": [t.to_dict() for t in self.targets],
            "rationale": self.rationale,
        }
