"""EvoKG: The co-evolutionary knowledge graph at the heart of MAGE."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from mage.core.schemas import (
    CapabilityNode,
    Edge,
    EnvironmentNode,
    ExperienceNode,
    NodeType,
    RelationType,
    Subgraph,
    TaskNode,
    node_from_dict,
)

logger = logging.getLogger(__name__)


class EvoKG:
    """Co-evolutionary Knowledge Graph with four subgraphs.

    Subgraphs
    ---------
    G_cap : Capability graph — skills and their prerequisites / mastery.
    G_task : Task dependency graph — tasks, decomposition, ordering.
    G_exp : Experience graph — abstracted success/failure patterns.
    G_env : Environment model graph — entities, spatial/causal relations.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._iteration = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def iteration(self) -> int:
        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self._iteration = value

    # ------------------------------------------------------------------
    # Subgraph views
    # ------------------------------------------------------------------

    def _subgraph_nodes(self, subgraph: Subgraph) -> list[str]:
        return [
            n for n, d in self._graph.nodes(data=True)
            if d.get("subgraph") == subgraph.value
        ]

    @property
    def G_cap(self) -> nx.DiGraph:
        return self._graph.subgraph(self._subgraph_nodes(Subgraph.CAPABILITY))

    @property
    def G_task(self) -> nx.DiGraph:
        return self._graph.subgraph(self._subgraph_nodes(Subgraph.TASK))

    @property
    def G_exp(self) -> nx.DiGraph:
        return self._graph.subgraph(self._subgraph_nodes(Subgraph.EXPERIENCE))

    @property
    def G_env(self) -> nx.DiGraph:
        return self._graph.subgraph(self._subgraph_nodes(Subgraph.ENVIRONMENT))

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node: CapabilityNode | TaskNode | ExperienceNode | EnvironmentNode,
    ) -> None:
        """Add a typed node to the graph."""
        attrs = node.to_dict()
        node_id = attrs.pop("node_id")
        self._graph.add_node(node_id, **attrs)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        if node_id not in self._graph:
            return None
        d = dict(self._graph.nodes[node_id])
        d["node_id"] = node_id
        return d

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def remove_node(self, node_id: str) -> None:
        if node_id in self._graph:
            self._graph.remove_node(node_id)

    def map_task_to_skill(self, task_type: str) -> str | None:
        """Map a task_type (from critic eval metadata) to an EvoKG skill ID.

        The critic evaluates by task_type (data-defined names like
        ``STBENCH_DIRECTION_DETERMINATION`` or short ones like ``bridge``,
        ``comparison``, ``answer_retrieval``), but EvoKG stores skills with
        Claude-discovered names like ``coordinate_parsing`` or
        ``multi_hop_entity_resolution``. This bridges the two ID spaces so
        curator + evolve can find the right skill node.

        Tries in order (short task_types use a more lenient threshold):
        1. Exact match
        2. STBench TASK_TO_SKILL static mapping
        3. Keyword overlap (lenient: ≥1 for short task_types, ≥2 for long)
        4. Substring containment
        5. Semantic-keyword fallback for QA task_types
        """
        if not task_type:
            return None

        # 1. Exact match
        if self.has_node(task_type):
            return task_type

        # 2. STBench static mapping
        try:
            from mage.environments.stbench_env import TASK_TO_SKILL
            mapped = TASK_TO_SKILL.get(task_type)
            if mapped and self.has_node(mapped):
                return mapped
        except ImportError:
            pass

        # 3. Keyword overlap with lenient threshold for short task_types
        task_words = set(task_type.lower().replace("_", " ").split())
        # Single-word task_types ("bridge", "comparison") need ≥1 overlap
        # to have any chance of matching multi-word skill names.
        min_overlap = 1 if len(task_words) <= 2 else 2

        skills = self.get_nodes_by_type(NodeType.CAPABILITY)
        best_match = None
        best_overlap = 0
        for skill in skills:
            sid = skill["node_id"]
            skill_words = set(
                sid.lower().replace("_", " ").split()
                + skill.get("name", "").lower().split()
                + skill.get("description", "").lower().split()
            )
            overlap = len(task_words & skill_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = sid
        if best_match and best_overlap >= min_overlap:
            return best_match

        # 4. Substring containment
        for skill in skills:
            sid = skill["node_id"]
            if sid.lower() in task_type.lower() or task_type.lower() in sid.lower():
                return sid

        # 5. Semantic fallback for common QA task_types
        SEMANTIC_HINTS = {
            "bridge": ["multi_hop", "bridge", "chain", "two_hop", "multi-hop", "entity_resolution", "fact_chain"],
            "comparison": ["compar", "contrast", "select", "decide", "differ"],
            "answer_retrieval": ["lookup", "retriev", "fact", "entity", "find"],
            "complex_question_parsing": ["complex", "constraint", "parse", "decompos", "multi_part"],
            "multi_answer_handling": ["multi", "list", "enumeration", "set", "value"],
        }
        hints = SEMANTIC_HINTS.get(task_type.lower(), [])
        for hint in hints:
            for skill in skills:
                sid_l = skill["node_id"].lower()
                desc_l = skill.get("description", "").lower()
                if hint in sid_l or hint in desc_l:
                    return skill["node_id"]

        return None

    def get_nodes_by_type(self, node_type: NodeType) -> list[dict[str, Any]]:
        results = []
        for n, d in self._graph.nodes(data=True):
            if d.get("node_type") == node_type.value:
                nd = dict(d)
                nd["node_id"] = n
                results.append(nd)
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """Add a typed edge to the graph."""
        attrs = edge.to_dict()
        src = attrs.pop("src")
        dst = attrs.pop("dst")
        self._graph.add_edge(src, dst, **attrs)

    def get_edges_from(self, node_id: str) -> list[dict[str, Any]]:
        if node_id not in self._graph:
            return []
        return [
            {"src": node_id, "dst": dst, **data}
            for dst, data in self._graph[node_id].items()
        ]

    def get_edges_to(self, node_id: str) -> list[dict[str, Any]]:
        if node_id not in self._graph:
            return []
        return [
            {"src": src, "dst": node_id, **data}
            for src, data in self._graph.pred[node_id].items()
        ]

    # ------------------------------------------------------------------
    # Capability-specific operations
    # ------------------------------------------------------------------

    def update_mastery(self, skill_id: str, mastery: float, iteration_tag: str | None = None) -> None:
        """Update the mastery level of a capability node."""
        if skill_id not in self._graph:
            raise KeyError(f"Node {skill_id!r} not in graph")
        node_data = self._graph.nodes[skill_id]
        if node_data.get("node_type") != NodeType.CAPABILITY.value:
            raise TypeError(f"Node {skill_id!r} is not a capability node")
        node_data["mastery"] = max(0.0, min(1.0, mastery))
        if iteration_tag is not None:
            node_data["last_evaluated"] = iteration_tag

    def get_mastery(self, skill_id: str) -> float:
        if skill_id not in self._graph:
            raise KeyError(f"Node {skill_id!r} not in graph")
        return self._graph.nodes[skill_id].get("mastery", 0.0)

    # ------------------------------------------------------------------
    # v12.1d: Per-task-type mastery (parallel to per-skill mastery)
    # ------------------------------------------------------------------
    # The Critic produces evaluations keyed by task_type (e.g.
    # ``STBENCH_DIRECTION_DETERMINATION``), but the EvoKG stores skills
    # under Claude-discovered names like ``coordinate_parsing``. The
    # task_type → skill mapping is fragile, so we ALSO track mastery
    # directly per task_type. This signal is what the Navigator's
    # round-robin selector and curriculum can read without depending on
    # the mapping.
    # ------------------------------------------------------------------

    def update_task_mastery(
        self,
        task_type: str,
        mastery: float,
        iteration_tag: str | None = None,
    ) -> None:
        """Update mastery for a critic task_type (graph-level dict)."""
        if not task_type:
            return
        store = self._graph.graph.setdefault("task_mastery", {})
        store[task_type] = max(0.0, min(1.0, float(mastery)))
        if iteration_tag is not None:
            ts = self._graph.graph.setdefault("task_mastery_ts", {})
            ts[task_type] = iteration_tag

    def get_task_mastery(self, task_type: str) -> float:
        store = self._graph.graph.get("task_mastery", {}) or {}
        return float(store.get(task_type, 0.0))

    def all_task_mastery(self) -> dict[str, float]:
        return dict(self._graph.graph.get("task_mastery", {}) or {})

    def retrieve_via_pagerank(
        self,
        seed_weights: dict[str, float],
        top_k: int = 5,
        alpha: float = 0.85,
        node_filter: Any = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        treat_as_undirected: bool = True,
    ) -> list[tuple[str, float, dict]]:
        """v17 Phase 1 — HippoRAG-style retrieval over the EvoKG.

        Personalized PageRank from caller-supplied seeds (typically the
        top-K embedding-similar nodes) propagates relevance through KG
        edges. Memory nodes attached (via EXPERIENCE_OF_SKILL) to a skill
        whose siblings/prerequisites are seeded get boosted, even if their
        own embedding match was weaker.

        Args:
          seed_weights: ``{node_id: similarity_score}`` from a dense
            retrieval pass. Scores are renormalised to sum to 1 for the
            personalisation vector.
          top_k: how many nodes to return.
          alpha: PageRank damping. 0.85 is the canonical value; lower
            (e.g. 0.5) makes the walk stick closer to seeds.
          node_filter: optional ``(node_id, node_dict) -> bool``; only
            nodes passing the filter are included in results. Use this to
            select e.g. only success_memory nodes for the few-shot block.
          max_iter: PageRank iteration cap.
          treat_as_undirected: if True (default), run PR on the undirected
            view so that memory→skill edges (EXPERIENCE_OF_SKILL points
            from memory to skill) propagate BOTH ways. Without this, a
            skill seed wouldn't reach its own memories under directed PR.

        Returns: ``[(node_id, pagerank_score, node_dict), ...]`` sorted by
        score desc, length ≤ top_k.
        """
        if not seed_weights:
            return []
        seeds = {nid: float(w) for nid, w in seed_weights.items() if nid in self._graph}
        if not seeds:
            return []
        s = sum(seeds.values())
        if s <= 0:
            return []
        personalization = {nid: w / s for nid, w in seeds.items()}
        # Initialize PR vector at seeds for faster convergence.
        nstart = {nid: personalization.get(nid, 0.0) for nid in self._graph.nodes()}
        try:
            graph = self._graph.to_undirected() if treat_as_undirected else self._graph
            try:
                pr = nx.pagerank(
                    graph,
                    alpha=alpha,
                    personalization=personalization,
                    nstart=nstart,
                    max_iter=max_iter,
                    tol=tol,
                )
            except nx.PowerIterationFailedConvergence:
                # Loosen tolerance and retry — small KGs sometimes need it
                pr = nx.pagerank(
                    graph,
                    alpha=alpha,
                    personalization=personalization,
                    nstart=nstart,
                    max_iter=max_iter * 2,
                    tol=tol * 100,
                )
        except Exception:
            # Fall back: just return the seeds themselves
            ranked = sorted(seeds.items(), key=lambda x: -x[1])[:top_k]
            return [(nid, w, self.get_node(nid)) for nid, w in ranked if self.get_node(nid)]

        candidates: list[tuple[str, float, dict]] = []
        for nid, score in pr.items():
            nd = self.get_node(nid)
            if nd is None:
                continue
            if node_filter is not None and not node_filter(nid, nd):
                continue
            candidates.append((nid, float(score), nd))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def get_prerequisites(self, skill_id: str) -> list[str]:
        """Get prerequisite skill IDs for a given skill.

        Prerequisites are nodes with a PREREQUISITE_OF edge pointing to skill_id,
        i.e. (prereq) --prerequisite_of--> (skill_id).
        """
        prereqs = []
        for src, data in self._graph.pred.get(skill_id, {}).items():
            if data.get("relation") == RelationType.PREREQUISITE_OF.value:
                prereqs.append(src)
        return prereqs

    def get_skill_chain(self, target_skill_id: str) -> list[str]:
        """v12.2 — DFS over PREREQUISITE_OF edges; returns the multi-hop
        prerequisite chain ending at ``target_skill_id`` in topological order.

        Example: if A -> B -> C are prerequisite edges (A is required for B,
        B is required for C), then ``get_skill_chain('C')`` returns
        ``['A', 'B', 'C']`` — the order in which the skills must be acquired.

        Cycles are guarded by a ``visited`` set so the worst case is O(V+E)
        even on a malformed graph.
        """
        chain: list[str] = []
        visited: set[str] = set()

        def dfs(sid: str) -> None:
            if sid in visited or not self.has_node(sid):
                return
            visited.add(sid)
            for prereq in self.get_prerequisites(sid):
                dfs(prereq)
            chain.append(sid)

        dfs(target_skill_id)
        return chain

    def get_learnable_frontier(self, mastery_threshold: float = 0.5) -> list[dict[str, Any]]:
        """Get skills that are ready to learn: all prereqs mastered but own mastery below threshold.

        Returns capability nodes sorted by mastery (ascending — least mastered first).
        """
        frontier = []
        for node_id in self._subgraph_nodes(Subgraph.CAPABILITY):
            node_data = self._graph.nodes[node_id]
            mastery = node_data.get("mastery", 0.0)
            if mastery >= mastery_threshold:
                continue

            prereqs = self.get_prerequisites(node_id)
            all_prereqs_mastered = all(
                self._graph.nodes.get(p, {}).get("mastery", 0.0) >= mastery_threshold
                for p in prereqs
            )
            if all_prereqs_mastered:
                nd = dict(node_data)
                nd["node_id"] = node_id
                frontier.append(nd)

        frontier.sort(key=lambda x: x.get("mastery", 0.0))
        return frontier

    # ------------------------------------------------------------------
    # Experience-specific operations
    # ------------------------------------------------------------------

    def add_experience(self, exp: ExperienceNode) -> None:
        """Add an experience pattern. If a similar pattern exists, increment frequency."""
        self.add_node(exp)

    def get_experiences_for_skill(self, skill_id: str) -> list[dict[str, Any]]:
        """Get experience patterns linked to a skill via EXPERIENCE_OF_SKILL edges."""
        results = []
        for src, data in self._graph.pred.get(skill_id, {}).items():
            if data.get("relation") == RelationType.EXPERIENCE_OF_SKILL.value:
                nd = dict(self._graph.nodes.get(src, {}))
                nd["node_id"] = src
                results.append(nd)
        return results

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """Remove experience nodes with confidence below threshold.

        Failure-memory experiences (outcome='failure_memory') are NEVER pruned —
        they are concrete worked examples that grow monotonically and serve as
        the agent's persistent retrieval memory.
        """
        to_remove = [
            n for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == NodeType.EXPERIENCE.value
            and d.get("outcome") != "failure_memory"
            and d.get("outcome") != "principle"
            and d.get("confidence", 1.0) < threshold
        ]
        for n in to_remove:
            self._graph.remove_node(n)
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} low-confidence experience nodes")
        return len(to_remove)

    # ------------------------------------------------------------------
    # Memory bank — both failure memories AND success patterns
    # ------------------------------------------------------------------
    # v15: Success memories store (question, retrieved_passage, correct_answer)
    # for questions where the search agent's retrieval LED to a correct answer.
    # These are retrievable by the search agent's history search, so future
    # similar questions can reuse the proven retrieval without re-searching
    # Wikipedia. Over iterations, the memory bank increasingly REPLACES
    # external search.
    # ------------------------------------------------------------------

    def get_skill_lattice(
        self,
        target_skill_id: str,
        max_depth: int = 4,
        max_total: int = 10,
    ) -> list[dict]:
        """v20.C — Build a topologically-sorted plan from PREREQUISITE_OF
        edges leading TO the target skill.

        Returns an ordered list (root prereqs first → target last) of dicts:
          {skill_id, name, mastery, principles, recipes}

        Critical for sequential envs: the agent doesn't just need "what
        to do for skill X" — it needs to know "you can't do X until Y, and
        Y needs Z first" so it plans the action sequence accordingly.

        Differs from `get_skill_chain` (which only returns IDs) in that it
        bundles principles + recipes per node, making it directly
        injectable into the agent's prompt as a structured plan.
        """
        if not self.has_node(target_skill_id):
            return []
        # DFS gather + topological order
        order: list[str] = []
        seen: set[str] = set()

        def dfs(sid: str, depth: int):
            if sid in seen or depth > max_depth or not self.has_node(sid):
                return
            seen.add(sid)
            try:
                prereqs = self.get_prerequisites(sid) or []
            except Exception:
                prereqs = []
            for pr in prereqs:
                dfs(pr, depth + 1)
            order.append(sid)

        dfs(target_skill_id, 0)

        # Cap at max_total (keeping target + nearest prereqs)
        if len(order) > max_total:
            # Keep last max_total (closer to target)
            order = order[-max_total:]

        out: list[dict] = []
        for sid in order:
            n = self.get_node(sid) or {}
            name = n.get("name", sid)
            mastery = n.get("mastery", 0.0)
            ps = (n.get("principles") or [])[:2]
            ps_text: list[str] = []
            for p in ps:
                if isinstance(p, dict):
                    p = p.get("text", "")
                if p:
                    ps_text.append(str(p)[:200])
            recipes = []
            try:
                if hasattr(self, "get_action_recipes"):
                    recipes = [r.get("actions", []) for r in self.get_action_recipes(sid, top_k=1)]
            except Exception:
                pass
            out.append({
                "skill_id": sid,
                "name": name,
                "mastery": mastery,
                "principles": ps_text,
                "recipes": recipes,
            })
        return out

    def add_action_recipe(
        self,
        skill_id: str,
        actions: list[str],
        ctx_summary: str = "",
        iteration: int = 0,
        source: str = "episode",
    ) -> str | None:
        """v20.B — store an action sequence that successfully unlocked a
        skill in a sequential env (Crafter, WebShop). The K most-recent
        actions before the achievement become the "recipe" for that skill.

        Stored as ExperienceNode with outcome="action_recipe" so it lives in
        the same subgraph as principles/memories and can be retrieved via
        the standard EvoKG queries.

        Dedup: if a recipe with same (skill_id, actions tuple) already
        exists, return None.
        """
        from mage.core.schemas import ExperienceNode

        actions = [str(a) for a in actions if a]
        if not skill_id or not actions:
            return None
        actions_key = "|".join(actions)
        # Dedup
        for n, d in self._graph.nodes(data=True):
            if (d.get("outcome") == "action_recipe"
                    and d.get("metadata", {}).get("skill_id") == skill_id
                    and d.get("metadata", {}).get("actions_key") == actions_key):
                return None
        node_id = f"recipe_{iteration}_{skill_id}_{abs(hash(actions_key)) % 10000}"
        exp = ExperienceNode(
            node_id=node_id,
            action_pattern=" → ".join(actions),
            outcome="action_recipe",
            context_conditions=ctx_summary[:300],
            confidence=1.0,
            domain_origin=skill_id,
            metadata={
                "skill_id": skill_id,
                "actions": actions,
                "actions_key": actions_key,
                "ctx_summary": ctx_summary[:300],
                "iteration_added": iteration,
                "source": source,
            },
        )
        self.add_node(exp)
        return node_id

    def get_action_recipes(
        self, skill_id: str | None = None, top_k: int = 5,
    ) -> list[dict]:
        """v20.B — retrieve action recipes, optionally filtered by skill.
        Returns list of dicts with keys: skill_id, actions (list[str]),
        ctx_summary, iteration_added.
        """
        recipes: list[dict] = []
        for n, d in self._graph.nodes(data=True):
            if d.get("outcome") != "action_recipe":
                continue
            md = d.get("metadata", {})
            if skill_id and md.get("skill_id") != skill_id:
                continue
            recipes.append({
                "node_id": n,
                "skill_id": md.get("skill_id", ""),
                "actions": md.get("actions", []),
                "ctx_summary": md.get("ctx_summary", ""),
                "iteration_added": md.get("iteration_added", 0),
                "source": md.get("source", ""),
            })
        # Most recent first
        recipes.sort(key=lambda r: -r.get("iteration_added", 0))
        return recipes[:top_k]

    def add_success_memory(
        self,
        question: str,
        correct_answer: str,
        retrieved_context: str,
        task_type: str = "",
        skill_id: str = "",
        search_strategy: str = "",
        iteration: int = 0,
        reasoning_trace: str = "",
        decomposition: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Store a successful (question, retrieval, reasoning, answer) pattern.

        v15.5 — now stores the learner's own CoT reasoning trace and, if
        the answer came from a multi-step skill chain, the per-step
        decomposition. This turns success memories from (Q, A) pairs into
        full demonstrations that future similar questions can imitate.

        Args:
            question: original question text.
            correct_answer: learner's final answer (which was correct).
            retrieved_context: passages the learner saw (may be empty).
            task_type, skill_id: indexing metadata.
            search_strategy: which bandit arm produced the retrieval.
            iteration: current iteration index.
            reasoning_trace: the raw LLM response (full CoT), truncated to
                1500 chars. If empty, the memory falls back to the old
                (Q, A)-only format on retrieval.
            decomposition: list of ``{"step": skill_name, "output": str}``
                entries from ``_run_skill_chain``. Empty if single-shot.

        Returns the new node id, or None if a near-duplicate already
        exists (dedup by question_key).
        """
        q_key = question.strip()[:300]
        # Dedup by question text
        for n, d in self._graph.nodes(data=True):
            if (d.get("node_type") == NodeType.EXPERIENCE.value
                    and d.get("outcome") == "success_memory"
                    and d.get("metadata", {}).get("question_key") == q_key):
                return None

        node_id = f"succ_{iteration}_{abs(hash(q_key + str(iteration))) % 1000000}"
        # Truncate decomposition step outputs to keep memory compact
        decomp_clean: list[dict[str, Any]] = []
        for step in decomposition or []:
            if not isinstance(step, dict):
                continue
            decomp_clean.append({
                "step": str(step.get("step", ""))[:80],
                "output": str(step.get("output", ""))[:300],
            })

        exp = ExperienceNode(
            node_id=node_id,
            action_pattern=reasoning_trace[:1000] if reasoning_trace
                else f"Q: {question[:200]} → A: {correct_answer[:100]}",
            outcome="success_memory",
            context_conditions=f"task_type={task_type},strategy={search_strategy}",
            confidence=1.0,
            domain_origin=task_type or "general",
            metadata={
                "question": question[:1000],
                "question_key": q_key,
                "correct_answer": correct_answer[:300],
                "retrieved_context": retrieved_context[:500],
                "task_type": task_type,
                "skill_id": skill_id,
                "search_strategy": search_strategy,
                "iteration_added": iteration,
                # v15.5 — structural provenance
                "reasoning_trace": (reasoning_trace or "")[:1500],
                "decomposition": decomp_clean,
            },
        )
        self.add_experience(exp)
        return node_id

    def get_success_memories(
        self,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all success-memory nodes, optionally filtered.

        v15.5 — surface reasoning_trace and decomposition so downstream
        indices/formatters can present them as rich few-shot demonstrations.
        """
        results = []
        for n, d in self._graph.nodes(data=True):
            if d.get("node_type") != NodeType.EXPERIENCE.value:
                continue
            if d.get("outcome") != "success_memory":
                continue
            md = d.get("metadata", {}) or {}
            if task_type is not None and md.get("task_type") != task_type:
                continue
            results.append({
                "node_id": n,
                "question": md.get("question", ""),
                "correct_answer": md.get("correct_answer", ""),
                "retrieved_context": md.get("retrieved_context", ""),
                "task_type": md.get("task_type", ""),
                "skill_id": md.get("skill_id", ""),
                "search_strategy": md.get("search_strategy", ""),
                "iteration_added": md.get("iteration_added", 0),
                "reasoning_trace": md.get("reasoning_trace", ""),
                "decomposition": md.get("decomposition", []) or [],
            })
        return results

    # v12.1b: gap (in iters) before a duplicate memory for the same question
    # is allowed. Different EVOLVE iterations may produce different worked
    # examples for the same failed question — both can be useful.
    MEMORY_REFRESH_GAP = 5

    def add_failure_memory(
        self,
        question: str,
        expected: str,
        wrong_answer: str,
        reasoning: str,
        task_type: str = "",
        skill_id: str = "",
        iteration: int = 0,
        kind: str = "specific",
    ) -> str | None:
        """Add a concrete worked example from a failed question.

        Args:
            kind: 'specific' (question-grounded reasoning) or 'type_strategy'
                (general strategy for a question type). Type strategies
                generalize across entities; specifics target exact recurrences.

        v12.1b — loosened dedup: allow a duplicate if the existing memory
        for the same question was added more than ``MEMORY_REFRESH_GAP``
        iterations ago. This stops the memory bank from freezing when the
        same questions keep failing.

        Returns the new node id, or None if an existing memory blocks it.
        """
        q_key = question.strip()[:300]
        for n, d in self._graph.nodes(data=True):
            if (d.get("node_type") == NodeType.EXPERIENCE.value
                    and d.get("outcome") == "failure_memory"
                    and d.get("metadata", {}).get("question_key") == q_key
                    and d.get("metadata", {}).get("kind", "specific") == kind):
                existing_iter = d.get("metadata", {}).get("iteration_added", 0)
                if iteration - existing_iter < self.MEMORY_REFRESH_GAP:
                    return None
                # else: enough time has passed; allow a refreshed memory

        prefix = "type" if kind == "type_strategy" else "mem"
        # Include iteration in the hash so refreshed memories don't collide
        node_id = f"{prefix}_{iteration}_{abs(hash(q_key + str(iteration))) % 1000000}"
        exp = ExperienceNode(
            node_id=node_id,
            action_pattern=reasoning[:1000],
            outcome="failure_memory",
            context_conditions=f"task_type={task_type},skill={skill_id},kind={kind}",
            confidence=1.0,
            domain_origin=task_type or "general",
            metadata={
                "question": question[:1000],
                "question_key": q_key,
                "expected": str(expected)[:300],
                "wrong_answer": str(wrong_answer)[:300],
                "task_type": task_type,
                "skill_id": skill_id,
                "iteration_added": iteration,
                "kind": kind,
            },
        )
        self.add_experience(exp)
        if skill_id and self.has_node(skill_id):
            try:
                self.add_edge(Edge(node_id, skill_id, RelationType.EXPERIENCE_OF_SKILL))
            except Exception:
                pass
        return node_id

    def get_failure_memories(
        self,
        task_type: str | None = None,
        skill_id: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all failure-memory nodes, optionally filtered.

        Each returned dict has: node_id, question, expected, wrong_answer,
        reasoning, task_type, skill_id, iteration_added, kind.
        """
        results = []
        for n, d in self._graph.nodes(data=True):
            if d.get("node_type") != NodeType.EXPERIENCE.value:
                continue
            if d.get("outcome") != "failure_memory":
                continue
            md = d.get("metadata", {}) or {}
            if task_type is not None and md.get("task_type") != task_type:
                continue
            if skill_id is not None and md.get("skill_id") != skill_id:
                continue
            if kind is not None and md.get("kind", "specific") != kind:
                continue
            results.append({
                "node_id": n,
                "question": md.get("question", ""),
                "expected": md.get("expected", ""),
                "wrong_answer": md.get("wrong_answer", ""),
                "reasoning": d.get("action_pattern", ""),
                "task_type": md.get("task_type", ""),
                "skill_id": md.get("skill_id", ""),
                "iteration_added": md.get("iteration_added", 0),
                "kind": md.get("kind", "specific"),
            })
        return results

    def count_failure_memories(self) -> int:
        return sum(
            1 for _, d in self._graph.nodes(data=True)
            if d.get("node_type") == NodeType.EXPERIENCE.value
            and d.get("outcome") == "failure_memory"
        )

    # ------------------------------------------------------------------
    # v12.3 — Retrieval recipes (stored as ExperienceNode metadata)
    # ------------------------------------------------------------------
    # A retrieval recipe is a learned mapping from "task_type X needs to
    # retrieve from external source Y using template Z". They live in G_exp
    # because they're a kind of experience pattern: "this query template
    # has worked for this kind of question before". No schema migration —
    # outcome="retrieval_recipe" + structured metadata.

    def add_retrieval_recipe(
        self,
        task_type: str,
        query_template: str,
        extraction_pattern: str,
        description: str = "",
        source_tool: str = "wikipedia",
        iteration: int = 0,
    ) -> str | None:
        """Store a learned retrieval recipe.

        Args:
            task_type: The task_type this recipe applies to (e.g.
                "answer_retrieval", "factoid_lookup").
            query_template: A format-string template; the learner formats it
                with the question text. Example:
                ``"{question} site:en.wikipedia.org"``.
            extraction_pattern: An optional regex / hint for extracting the
                answer from the retrieved text.
            source_tool: Name of the retrieval tool (default "wikipedia").

        Returns the new node id, or None if a similar recipe already exists.
        """
        if not task_type or not query_template:
            return None
        # Dedup: same task_type + same query_template
        for n, d in self._graph.nodes(data=True):
            if (d.get("node_type") == NodeType.EXPERIENCE.value
                    and d.get("outcome") == "retrieval_recipe"
                    and d.get("metadata", {}).get("task_type") == task_type
                    and d.get("metadata", {}).get("query_template") == query_template):
                return None

        node_id = f"recipe_{task_type}_{abs(hash(query_template)) % 1000000}"
        exp = ExperienceNode(
            node_id=node_id,
            action_pattern=(
                f"[{source_tool}] query='{query_template[:200]}' "
                f"extract='{extraction_pattern[:200]}'"
            ),
            outcome="retrieval_recipe",
            context_conditions=f"task_type={task_type},source={source_tool}",
            confidence=1.0,
            domain_origin=task_type,
            metadata={
                "task_type": task_type,
                "query_template": query_template,
                "extraction_pattern": extraction_pattern,
                "description": description,
                "source_tool": source_tool,
                "iteration_added": iteration,
            },
        )
        self.add_experience(exp)
        return node_id

    def get_retrieval_recipes(
        self,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all retrieval recipes, optionally filtered by task_type."""
        results = []
        for n, d in self._graph.nodes(data=True):
            if d.get("node_type") != NodeType.EXPERIENCE.value:
                continue
            if d.get("outcome") != "retrieval_recipe":
                continue
            md = d.get("metadata", {}) or {}
            if task_type is not None and md.get("task_type") != task_type:
                continue
            results.append({
                "node_id": n,
                "task_type": md.get("task_type", ""),
                "query_template": md.get("query_template", ""),
                "extraction_pattern": md.get("extraction_pattern", ""),
                "description": md.get("description", ""),
                "source_tool": md.get("source_tool", "wikipedia"),
                "iteration_added": md.get("iteration_added", 0),
                "action_pattern": d.get("action_pattern", ""),
            })
        return results

    def count_retrieval_recipes(self) -> int:
        return sum(
            1 for _, d in self._graph.nodes(data=True)
            if d.get("node_type") == NodeType.EXPERIENCE.value
            and d.get("outcome") == "retrieval_recipe"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics of the EvoKG."""
        cap_nodes = self._subgraph_nodes(Subgraph.CAPABILITY)
        masteries = [self._graph.nodes[n].get("mastery", 0.0) for n in cap_nodes]
        return {
            "iteration": self._iteration,
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "capability_nodes": len(cap_nodes),
            "task_nodes": len(self._subgraph_nodes(Subgraph.TASK)),
            "experience_nodes": len(self._subgraph_nodes(Subgraph.EXPERIENCE)),
            "environment_nodes": len(self._subgraph_nodes(Subgraph.ENVIRONMENT)),
            "avg_mastery": sum(masteries) / len(masteries) if masteries else 0.0,
            "mastered_skills": sum(1 for m in masteries if m >= 0.5),
            "frontier_size": len(self.get_learnable_frontier()),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_snapshot(self, path: str | Path) -> None:
        """Save EvoKG to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = {}
        for n, d in self._graph.nodes(data=True):
            nodes[n] = dict(d)

        edges = []
        for src, dst, d in self._graph.edges(data=True):
            edge_data = dict(d)
            edge_data["src"] = src
            edge_data["dst"] = dst
            edges.append(edge_data)

        snapshot = {
            "metadata": {
                "iteration": self._iteration,
                "stats": self.stats(),
            },
            "nodes": nodes,
            "edges": edges,
        }

        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.info(f"Saved EvoKG snapshot to {path} ({self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges)")

    @classmethod
    def load_snapshot(cls, path: str | Path) -> EvoKG:
        """Load EvoKG from JSON file."""
        path = Path(path)
        with open(path) as f:
            snapshot = json.load(f)

        evokg = cls()
        evokg._iteration = snapshot.get("metadata", {}).get("iteration", 0)

        for node_id, attrs in snapshot.get("nodes", {}).items():
            evokg._graph.add_node(node_id, **attrs)

        for edge_data in snapshot.get("edges", []):
            src = edge_data.pop("src")
            dst = edge_data.pop("dst")
            evokg._graph.add_edge(src, dst, **edge_data)

        logger.info(f"Loaded EvoKG from {path} ({evokg._graph.number_of_nodes()} nodes, {evokg._graph.number_of_edges()} edges)")
        return evokg

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_with_skeleton(cls, domain: str = "general") -> EvoKG:
        """Create an EvoKG with a domain-agnostic skeleton.

        The skeleton provides abstract meta-capabilities that are universal
        across domains: explore, collect, compose, use, defend.
        """
        evokg = cls()

        skeleton_skills = [
            CapabilityNode("meta_explore", "explore", mastery=0.0, domain=domain,
                           description="Navigate and discover the environment"),
            CapabilityNode("meta_collect", "collect", mastery=0.0, domain=domain,
                           description="Gather resources or information"),
            CapabilityNode("meta_compose", "compose", mastery=0.0, domain=domain,
                           description="Combine elements to create new items or solutions"),
            CapabilityNode("meta_use", "use", mastery=0.0, domain=domain,
                           description="Apply tools or skills to achieve goals"),
            CapabilityNode("meta_defend", "defend", mastery=0.0, domain=domain,
                           description="Protect against threats or recover from failures"),
        ]
        for skill in skeleton_skills:
            evokg.add_node(skill)

        # Basic prerequisite chain: explore → collect → compose → use
        evokg.add_edge(Edge("meta_explore", "meta_collect", RelationType.PREREQUISITE_OF))
        evokg.add_edge(Edge("meta_collect", "meta_compose", RelationType.PREREQUISITE_OF))
        evokg.add_edge(Edge("meta_compose", "meta_use", RelationType.PREREQUISITE_OF))

        logger.info(f"Created EvoKG with domain-agnostic skeleton ({len(skeleton_skills)} nodes)")
        return evokg

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"EvoKG(iter={s['iteration']}, nodes={s['total_nodes']}, edges={s['total_edges']}, "
            f"skills={s['capability_nodes']}[mastered={s['mastered_skills']}], "
            f"exp={s['experience_nodes']}, tasks={s['task_nodes']})"
        )
