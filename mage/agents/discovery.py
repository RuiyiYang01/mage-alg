"""Skill Discovery and Graph Evolution agents for MAGE v2.

Uses a strong LLM (Claude Sonnet) to:
1. Discover skills from a batch of questions (cold start)
2. Analyze failures and propose strategy fixes
3. Evolve the graph structure (split/merge/add skills)

These agents operate on the EvoKG structure itself, not just mastery values.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mage.core.evokg import EvoKG
from mage.core.schemas import (
    CapabilityNode,
    Edge,
    RelationType,
    StructuredEvaluation,
)
from mage.utils.llm import LLMBackend

logger = logging.getLogger(__name__)

DISCOVER_SKILLS_PROMPT = """\
You are analyzing a set of questions from a new domain to discover the skills needed to answer them.

Here are {n} sample questions:

{questions}

Analyze these questions and identify 5-10 distinct reasoning skills needed to answer them.
For each skill, provide:
- A short skill_id (snake_case)
- A human-readable name
- A description of what this skill involves
- The strategy type: "direct" (pure reasoning), "tool" (needs computation), or "compose" (combination of other skills)
- If tool: which tool function would help (e.g., compute_distance, check_temporal_relation)
- If compose: which other skills it combines
- Prerequisites: which other skills must be mastered first
- A prompt_template: specific instruction for an LLM to follow when solving this type of question

Output ONLY valid JSON:
```json
{{
  "skills": [
    {{
      "skill_id": "direction_computation",
      "name": "Direction Computation",
      "description": "Determine compass direction between two GPS coordinates",
      "strategy": {{"type": "tool", "tool": "compute_direction", "param_extraction": "Extract lon_a, lat_a, lon_b, lat_b from the question"}},
      "prerequisites": ["coordinate_parsing"],
      "prompt_template": "Extract the two coordinates. Use compute_direction tool to get the compass direction. Map to the answer option."
    }},
    ...
  ]
}}
```

Be specific about strategies. Do NOT just say "reason about X" — say exactly what computation or steps are needed."""

ANALYZE_FAILURES_PROMPT = """\
You are analyzing why an AI agent failed on certain questions. The agent was trying to use skill "{skill_name}".

Skill strategy: {strategy}
Skill prompt: {prompt_template}

Here are {n} failure cases:

{failures}

For each failure, analyze:
1. WHY did the agent fail? (wrong computation? wrong tool? misunderstood question? missing prerequisite?)
2. What SPECIFIC fix would help?

Then provide ONE of these recommendations:
- "refine_strategy": The skill exists but the strategy/prompt needs updating. Provide the new strategy and prompt.
- "split_skill": This skill is too broad — split into sub-skills. Provide the new skill definitions.
- "create_tool": The agent needs a COMPUTATIONAL tool (Python function) to solve this deterministically. YOU MUST WRITE the complete Python function.
- "add_prerequisite": The agent needs another skill first. Name it.

Output ONLY valid JSON:
```json
{{
  "analysis": "Brief summary of failure patterns",
  "recommendation": "refine_strategy|split_skill|create_tool|add_prerequisite",
  "details": {{
    // For refine_strategy:
    "new_strategy": {{}},
    "new_prompt_template": "..."
    // For split_skill:
    "new_skills": [...]
    // For create_tool:
    "tool_name": "compute_something",
    "tool_description": "What this tool computes",
    "tool_code": "def compute_something(param1: float, param2: float) -> dict:\\n    result = param1 + param2\\n    return {{\\"answer\\": result}}",
    "param_extraction": "Extract param1 and param2 as floats from the question",
    "example_params": "{{\\"param1\\": 1.0, \\"param2\\": 2.0}}",
    "extractor_regex": "r'param1\\s*=\\s*([\\d.]+).*?param2\\s*=\\s*([\\d.]+)'",
    "answer_mapping": "Return result['answer'] directly as the option number"
    // For add_prerequisite:
    "prerequisite_skill_id": "...", "prerequisite_description": "..."
  }}
}}
```

IMPORTANT for create_tool:
- The function MUST be pure Python (use only math, re, json — no external libraries)
- It MUST return a dict with the computed answer
- Include ALL needed computation logic in the function body
- The param_extraction tells the LLM how to extract parameters from questions
- The example_params shows the expected JSON format
- The extractor_regex is a Python regex to extract params WITHOUT LLM (deterministic)
- The answer_mapping explains how to convert tool output to the expected answer format"""

EXTRACT_PRINCIPLES_PROMPT = """\
You are analyzing failures for skill "{skill_name}" to extract reusable PRINCIPLES.

Current strategy: {strategy}
Current principles: {current_principles}

Here are {n} failure cases (agent's wrong answers):

{failures}

Extract 2-3 ABSTRACT PRINCIPLES (reusable rules) that would help answer these correctly.

Good principle: "For direction questions, extract (lon_a, lat_a, lon_b, lat_b), compute bearing = atan2(dx, dy), map to 8-way compass: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW"
Bad principle: "The answer to question 5 is East" (too specific)

Output ONLY valid JSON:
```json
{{
  "principles": [
    "principle text here",
    ...
  ]
}}
```"""


class SkillDiscovery:
    """Discovers skills from question batches using a strong LLM."""

    def __init__(self, strong_llm: LLMBackend):
        self.llm = strong_llm

    def discover_skills(
        self,
        questions: list[dict[str, Any]],
        existing_skills: list[str] | None = None,
        available_tools: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze a batch of questions and propose skills.

        Args:
            questions: List of {question, answer, task_type} dicts
            existing_skills: Skills already in the EvoKG (to avoid duplicates)
            available_tools: Tool names the agent can use

        Returns:
            List of skill definitions ready to add to EvoKG
        """
        # Sample diverse questions (max 20 for prompt length)
        sample = questions[:20]
        q_text = "\n\n".join(
            f"Q{i+1} [type={q.get('task_type', 'unknown')}]: {q['question'][:300]}\n"
            f"Answer: {q.get('answer', 'N/A')}"
            for i, q in enumerate(sample)
        )

        context = ""
        if available_tools:
            context += f"\n\nAvailable computational tools: {', '.join(available_tools)}"
        if existing_skills:
            context += f"\n\nAlready discovered skills (avoid duplicates): {', '.join(existing_skills)}"

        prompt = DISCOVER_SKILLS_PROMPT.format(
            n=len(sample),
            questions=q_text,
        ) + context

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt="You are an expert at analyzing reasoning tasks and designing skill taxonomies.",
            )
            skills = result.get("skills", [])
            logger.info(f"Discovered {len(skills)} skills from {len(sample)} questions")
            return skills
        except Exception as e:
            logger.error(f"Skill discovery failed: {e}")
            return []

    def analyze_failures(
        self,
        skill_node: dict[str, Any],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze why the agent failed on a skill and propose fixes.

        Args:
            skill_node: The skill's EvoKG node data
            failures: List of {question, expected, predicted, score} dicts

        Returns:
            Analysis with recommendation (refine_strategy/split_skill/add_tool/add_prerequisite)
        """
        if not failures:
            return {"recommendation": "none", "analysis": "No failures to analyze"}

        failure_text = "\n\n".join(
            f"Question: {f['question'][:200]}\n"
            f"Expected: {f.get('expected', 'N/A')}\n"
            f"Agent answered: {f.get('predicted', 'N/A')}\n"
            f"Score: {f.get('score', 0)}"
            for f in failures[:10]
        )

        strategy_str = json.dumps(skill_node.get("strategy", {}))
        prompt_template = skill_node.get("prompt_template", "None")

        prompt = ANALYZE_FAILURES_PROMPT.format(
            skill_name=skill_node.get("name", skill_node.get("node_id", "unknown")),
            strategy=strategy_str,
            prompt_template=prompt_template,
            n=len(failures),
            failures=failure_text,
        )

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt="You are an expert at debugging AI agent failures and improving reasoning strategies.",
            )
            logger.info(f"Failure analysis for {skill_node.get('node_id')}: {result.get('recommendation')}")
            return result
        except Exception as e:
            logger.error(f"Failure analysis failed: {e}")
            return {"recommendation": "none", "analysis": str(e)}

    def extract_principles(
        self,
        skill_node: dict[str, Any],
        failures: list[dict[str, Any]],
        existing_principles: list[str] | None = None,
        domain_context: str | None = None,
    ) -> list[str]:
        """Extract abstract principles from failure cases using Claude.

        Args:
            skill_node: capability node dict from EvoKG
            failures: list of failure dicts ({question, expected, predicted})
            existing_principles: principles already attached to this skill
            domain_context: an optional preamble describing the task domain
                (e.g., "Crafter is a 2D survival game; agents pick one of 17
                discrete actions per turn; principles must be concrete game
                tactics, not roleplay advice"). Without this, Claude often
                writes generic dialog/roleplay principles for skill names that
                look like commands (e.g., "Place Plant").

        Returns list of principle strings (reusable rules, not raw examples).
        """
        if not failures:
            return []

        failure_text = "\n\n".join(
            f"Q: {f['question'][:200]}\n"
            f"Expected: {f.get('expected', 'N/A')}\n"
            f"Agent said: {f.get('predicted', 'N/A')}"
            for f in failures[:8]
        )

        prompt = EXTRACT_PRINCIPLES_PROMPT.format(
            skill_name=skill_node.get("name", skill_node.get("node_id", "?")),
            strategy=json.dumps(skill_node.get("strategy", {})),
            current_principles=json.dumps(existing_principles or []),
            n=len(failures),
            failures=failure_text,
        )
        if domain_context:
            prompt = (
                f"DOMAIN CONTEXT (read carefully — principles MUST fit this domain):\n"
                f"{domain_context.strip()}\n\n"
                f"---\n\n" + prompt
            )

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt="You are an expert at extracting reusable reasoning patterns.",
            )
            principles = result.get("principles", [])
            # Filter: must be abstract (>20 chars) and not too long (<500)
            principles = [p for p in principles if 20 < len(p) < 500]
            logger.info(f"Extracted {len(principles)} principles for {skill_node.get('node_id')}")
            return principles[:3]
        except Exception as e:
            logger.error(f"Principle extraction failed: {e}")
            return []

    def write_retrieval_recipe(
        self,
        task_type: str,
        failures: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        """Ask Claude to write a retrieval recipe for a task_type.

        v12.3 — Given recurring failures on a given task_type (e.g.
        WebQA factoid questions about birthplaces), Claude proposes:
            * a query template (with `{question}` placeholder) that
              should be sent to a retrieval tool (Wikipedia)
            * an extraction pattern hint for parsing the answer

        Returns ``{"query_template": ..., "extraction_pattern": ..., "description": ...}``
        or ``None`` on failure / non-applicability.
        """
        if not failures:
            return None

        sample = "\n\n".join(
            f"Q: {f.get('question', '')[:200]}\n"
            f"Expected: {f.get('expected', '')}\n"
            f"Wrong answer: {f.get('predicted', '')}"
            for f in failures[:5]
        )

        prompt = (
            f"You are designing a Wikipedia retrieval recipe for a class of "
            f"factoid questions the agent keeps getting wrong.\n\n"
            f"Task type: {task_type}\n\n"
            f"Recent failures on this task type:\n{sample}\n\n"
            f"Write a SINGLE retrieval recipe that would help the agent answer "
            f"future questions of this type. The recipe has two parts:\n"
            f"  1. A query template — a string with {{question}} placeholder "
            f"that will be sent to a Wikipedia search tool.\n"
            f"  2. An extraction pattern — a Python regex (or simple hint) "
            f"that pulls the answer from the retrieved text.\n\n"
            f"Output JSON of the form:\n"
            f'{{\n'
            f'  "applicable": true,\n'
            f'  "query_template": "{{question}} site:en.wikipedia.org",\n'
            f'  "extraction_pattern": "born in (?P<answer>[A-Z][a-z]+(?: [A-Z][a-z]+)*)",\n'
            f'  "description": "Person birthplace lookups"\n'
            f'}}\n\n'
            f"If a Wikipedia retrieval recipe is NOT a good fit for this task "
            f'type (e.g., the questions are computational, not factual), '
            f'set "applicable" to false and leave the other fields empty.'
        )

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt=(
                    "You are an expert at designing retrieval queries and "
                    "extraction patterns for factoid QA."
                ),
            )
        except Exception as e:
            logger.error(f"Recipe generation failed for {task_type}: {e}")
            return None

        if not result.get("applicable", False):
            return None
        if not result.get("query_template"):
            return None

        return {
            "query_template": result["query_template"][:500],
            "extraction_pattern": result.get("extraction_pattern", "")[:500],
            "description": result.get("description", "")[:300],
        }

    def write_failure_memories(
        self,
        failures: list[dict[str, Any]],
        skill_id: str = "",
        max_failures: int = 8,
    ) -> list[dict[str, Any]]:
        """Convert failed (question, expected, predicted) tuples into a
        HIERARCHY of worked examples that the learner can retrieve at eval time.

        v11: Hierarchical memory. Each Claude call now produces:
          1. **Specific memories** — one per failure, with question-grounded
             reasoning. Used when a near-identical question recurs.
          2. **Type-level strategy** — a single shared strategy that abstracts
             the *kind* of question (e.g., "person_birthplace" → "Output the
             birth city; check 'born in X' patterns"). Used when similar but
             not identical questions recur.

        Returns a list of memory dicts with keys:
            kind: 'specific' | 'type_strategy'
            question / question_pattern
            expected
            wrong_answer
            reasoning
            task_type
            skill_id
        """
        if not failures:
            return []

        failure_block = "\n\n---\n\n".join(
            f"FAILURE {i+1}:\n"
            f"Question: {f['question'][:600]}\n"
            f"Correct answer: {f.get('expected', 'N/A')}\n"
            f"Agent's wrong answer: {f.get('predicted', 'N/A')}"
            for i, f in enumerate(failures[:max_failures])
        )

        prompt = (
            f"You are teaching a frozen LLM agent how to solve a class of "
            f"questions it has been getting wrong. Below are several failures "
            f"on questions of the same general kind.\n\n"
            f"Produce TWO things:\n\n"
            f"  (A) A list of SPECIFIC worked examples — one per failure — "
            f"each showing the exact reasoning that produces the correct "
            f"answer for that question. Keep it concrete and short (≤4 "
            f"sentences). End each with 'Answer: <correct answer>'.\n\n"
            f"  (B) A SINGLE type-level strategy that captures what KIND of "
            f"question these are (e.g. 'person_birthplace', 'comparison_taller', "
            f"'directional_bearing') and a SHORT instruction (≤2 sentences) "
            f"describing how to recognize and solve this question type. The "
            f"strategy must be GENERAL enough to apply to similar questions "
            f"about different entities, not just the failures shown.\n\n"
            f"FAILURES:\n{failure_block}\n\n"
            f"Output JSON of the form:\n"
            f'{{\n'
            f'  "type_label": "snake_case_question_type",\n'
            f'  "type_strategy": "Recognize: ... How to solve: ...",\n'
            f'  "type_pattern_example": "where was <person> born?",\n'
            f'  "memories": [\n'
            f'    {{"reasoning": "Step 1: ... Answer: X."}},\n'
            f'    ... one per failure in order ...\n'
            f'  ]\n'
            f'}}'
        )

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt=(
                    "You are an expert at extracting both concrete examples "
                    "and reusable type-level strategies from failure cases."
                ),
            )
        except Exception as e:
            logger.error(f"Failure memory writing failed: {e}")
            return []

        memories: list[dict[str, Any]] = []

        # 1. Specific memories
        memories_raw = result.get("memories", [])
        for f, m in zip(failures[:max_failures], memories_raw):
            reasoning = m.get("reasoning", "") if isinstance(m, dict) else str(m)
            if not reasoning or len(reasoning) < 10:
                continue
            memories.append({
                "kind": "specific",
                "question": f.get("question", ""),
                "expected": str(f.get("expected", "")),
                "wrong_answer": str(f.get("predicted", "")),
                "reasoning": reasoning[:800],
                "task_type": f.get("task_type", ""),
                "skill_id": skill_id,
            })

        # 2. Type-level strategy (one shared "memory" that generalizes)
        type_label = (result.get("type_label", "") or "").strip()
        type_strategy = (result.get("type_strategy", "") or "").strip()
        type_pattern = (result.get("type_pattern_example", "") or "").strip()

        if type_label and type_strategy and len(type_strategy) > 15:
            # The "question" for a type strategy is the pattern example so
            # retrieval finds it on similar wording. The "reasoning" is the
            # general instruction.
            memories.append({
                "kind": "type_strategy",
                "question": type_pattern or f"Question type: {type_label}",
                "expected": "(varies by entity)",
                "wrong_answer": "",
                "reasoning": f"[Question type: {type_label}] {type_strategy}",
                "task_type": failures[0].get("task_type", ""),
                "skill_id": skill_id,
            })

        n_specific = sum(1 for m in memories if m["kind"] == "specific")
        n_type = sum(1 for m in memories if m["kind"] == "type_strategy")
        logger.info(
            f"Wrote {n_specific} specific + {n_type} type-strategy memories "
            f"from {len(failures)} failures"
        )
        return memories

    def apply_evolution(
        self,
        evokg: EvoKG,
        skill_id: str,
        analysis: dict[str, Any],
        iteration: int,
    ) -> list[str]:
        """Apply the recommended evolution to the EvoKG.

        Returns list of new/modified skill IDs.
        """
        recommendation = analysis.get("recommendation", "none")
        details = analysis.get("details", {})
        modified = []

        if recommendation == "refine_strategy":
            # Update existing skill's strategy and prompt.
            # IMPORTANT: never overwrite a working tool strategy with a 'direct'
            # strategy. If the existing strategy is type=tool, only update the
            # prompt_template; preserve the tool wiring.
            if evokg.has_node(skill_id):
                node_data = evokg.graph.nodes[skill_id]
                existing_strategy = node_data.get("strategy", {}) or {}
                existing_is_tool = (
                    isinstance(existing_strategy, dict)
                    and existing_strategy.get("type") == "tool"
                )
                new_strategy = details.get("new_strategy")
                if new_strategy:
                    if existing_is_tool and (
                        not isinstance(new_strategy, dict)
                        or new_strategy.get("type") != "tool"
                    ):
                        logger.info(
                            f"  Refused to overwrite tool strategy for {skill_id} "
                            f"with non-tool — only updating prompt_template"
                        )
                    else:
                        node_data["strategy"] = new_strategy
                if details.get("new_prompt_template"):
                    node_data["prompt_template"] = details["new_prompt_template"]
                logger.info(f"Refined strategy for {skill_id}")
                modified.append(skill_id)

        elif recommendation == "split_skill":
            # Add new sub-skills with prerequisite edges.
            # CRITICAL: child skills inherit parent mastery (not 0.0) so the
            # navigator's curriculum doesn't get reset every time we split.
            # Also inherit the parent's principles so accumulated knowledge
            # transfers to the children.
            parent_data = evokg.graph.nodes.get(skill_id, {})
            parent_mastery = parent_data.get("mastery", 0.0)
            parent_principles = parent_data.get("principles", []) or []
            parent_domain = parent_data.get("domain", "general")

            new_skills = details.get("new_skills", [])
            for skill_def in new_skills:
                new_id = skill_def.get("skill_id", f"{skill_id}_sub_{len(modified)}")
                if not evokg.has_node(new_id):
                    node = CapabilityNode(
                        node_id=new_id,
                        name=skill_def.get("name", new_id),
                        mastery=parent_mastery,  # INHERIT parent's mastery
                        domain=parent_domain,
                        description=skill_def.get("description", ""),
                        strategy=skill_def.get("strategy", parent_data.get("strategy", {})),
                        prompt_template=skill_def.get("prompt_template", ""),
                        discovered_at=f"iter_{iteration}",
                    )
                    evokg.add_node(node)
                    # Carry over parent's principles so children start informed
                    if parent_principles:
                        evokg.graph.nodes[new_id]["principles"] = list(parent_principles)
                    # New sub-skill is prerequisite of the original
                    evokg.add_edge(Edge(new_id, skill_id, RelationType.PREREQUISITE_OF))
                    modified.append(new_id)
                    logger.info(
                        f"Split {skill_id} → added sub-skill {new_id} "
                        f"(inherited mastery={parent_mastery:.2f}, "
                        f"{len(parent_principles)} principles)"
                    )

        elif recommendation == "create_tool":
            # Claude wrote a tool — register it dynamically
            tool_name = details.get("tool_name", "")
            tool_code = details.get("tool_code", "")
            tool_desc = details.get("tool_description", "")
            param_extraction = details.get("param_extraction", "")
            example_params = details.get("example_params", "{}")

            if tool_name and tool_code:
                success = self._register_dynamic_tool(
                    tool_name, tool_code, tool_desc, param_extraction,
                )
                if success:
                    # Update skill strategy with tool + extractor + mapping
                    if evokg.has_node(skill_id):
                        extractor_regex = details.get("extractor_regex", "")
                        answer_mapping = details.get("answer_mapping", "")
                        evokg.graph.nodes[skill_id]["strategy"] = {
                            "type": "tool",
                            "tool": tool_name,
                            "param_extraction": param_extraction,
                            "example_params": example_params,
                            "extractor_regex": extractor_regex,
                            "answer_mapping": answer_mapping,
                        }
                        evokg.graph.nodes[skill_id]["prompt_template"] = (
                            f"Use the {tool_name} tool to compute the answer. "
                            f"{param_extraction}"
                        )
                    modified.append(f"tool:{tool_name}")
                    logger.info(f"Created tool '{tool_name}' for {skill_id}")
                else:
                    logger.warning(f"Failed to create tool '{tool_name}'")

        elif recommendation == "add_prerequisite":
            prereq_id = details.get("prerequisite_skill_id", "")
            if prereq_id and not evokg.has_node(prereq_id):
                node = CapabilityNode(
                    node_id=prereq_id,
                    name=details.get("prerequisite_description", prereq_id),
                    mastery=0.0,
                    domain=evokg.graph.nodes.get(skill_id, {}).get("domain", "general"),
                    description=details.get("prerequisite_description", ""),
                    discovered_at=f"iter_{iteration}",
                )
                evokg.add_node(node)
                logger.info(f"Added prerequisite {prereq_id} for {skill_id}")
            if prereq_id and evokg.has_node(prereq_id):
                evokg.add_edge(Edge(prereq_id, skill_id, RelationType.PREREQUISITE_OF))
                modified.append(prereq_id)

        return modified

    def fix_extractor(
        self,
        skill_node: dict[str, Any],
        failures: list[dict[str, Any]],
        tool_name: str,
        current_regex: str,
    ) -> dict[str, Any] | None:
        """Ask Claude to fix the extraction regex based on actual failure diagnostics.

        Claude sees: the question text, what regex was tried, why it failed,
        and what parameters the tool needs. Claude writes a FIXED regex.
        """
        failure_details = "\n\n".join(
            f"Question: {f['question'][:300]}\n"
            f"Expected answer: {f.get('expected', 'N/A')}\n"
            f"Extraction diagnosis: {f.get('diagnosis', 'unknown')}"
            for f in failures[:5]
        )

        # Show Claude the EXACT text that needs matching
        sample_question = failures[0]["question"][:500] if failures else ""

        prompt = (
            f"The tool '{tool_name}' exists but parameter extraction keeps failing.\n\n"
            f"Current extractor regex: {current_regex or 'NONE'}\n\n"
            f"EXACT QUESTION TEXT TO MATCH (first 500 chars):\n"
            f"'''{sample_question}'''\n\n"
            f"Extraction failures:\n{failure_details}\n\n"
            f"Write a Python regex that:\n"
            f"1. Matches this EXACT question format\n"
            f"2. Extracts the numeric parameters the tool needs\n"
            f"3. Works with: match = re.search(pattern, question, re.DOTALL)\n\n"
            f"IMPORTANT: Test your regex mentally against the exact question text above.\n"
            f"The regex MUST match that specific text.\n\n"
            f"Output ONLY valid JSON:\n"
            f'{{"new_extractor_regex": "r\'...pattern...\'", '
            f'"explanation": "what the regex matches and what groups extract"}}'
        )

        try:
            result = self.llm.generate_json(
                prompt,
                system_prompt="You are an expert at writing Python regex patterns for text extraction.",
            )
            if result.get("new_extractor_regex"):
                # Validate the regex compiles
                import re
                try:
                    re.compile(eval(result["new_extractor_regex"]))
                    logger.info(f"Fixed extractor for {tool_name}: {result.get('explanation', '')[:100]}")
                    return result
                except Exception:
                    logger.warning(f"Claude's regex doesn't compile: {result['new_extractor_regex']}")
                    return None
        except Exception as e:
            logger.error(f"Extractor fix failed: {e}")

        return None

    def _register_dynamic_tool(
        self,
        tool_name: str,
        tool_code: str,
        description: str,
        param_extraction: str,
    ) -> bool:
        """Dynamically register a tool written by Claude.

        Compiles the function, validates it, registers it in TOOL_REGISTRY,
        AND persists the source code to disk so the next run can load it.

        The persistent location is ``mage/tools/dynamic/<tool_name>.py``.
        Returns True if successful.
        """
        import math
        from pathlib import Path
        from mage.tools.st_tools import TOOL_REGISTRY

        if tool_name in TOOL_REGISTRY:
            logger.info(f"Tool '{tool_name}' already exists, skipping")
            return True

        # Safe namespace for compilation
        safe_globals = {
            "__builtins__": {
                "abs": abs, "round": round, "max": max, "min": min,
                "sum": sum, "len": len, "int": int, "float": float,
                "str": str, "bool": bool, "list": list, "dict": dict,
                "tuple": tuple, "range": range, "enumerate": enumerate,
                "zip": zip, "sorted": sorted, "isinstance": isinstance,
                "ValueError": ValueError, "TypeError": TypeError,
                "True": True, "False": False, "None": None,
            },
            "math": math,
            "re": __import__("re"),
        }

        try:
            # Compile the function
            exec(tool_code, safe_globals)

            # Find the function in the namespace
            func = None
            for name, obj in safe_globals.items():
                if callable(obj) and name == tool_name:
                    func = obj
                    break

            if func is None:
                # Try to find any new callable
                for name, obj in safe_globals.items():
                    if callable(obj) and not name.startswith("_") and name not in ("math", "re"):
                        func = obj
                        break

            if func is None:
                logger.error(f"Could not find function in tool code for '{tool_name}'")
                return False

            # Register in TOOL_REGISTRY
            TOOL_REGISTRY[tool_name] = {
                "function": func,
                "description": description,
                "prereqs": [],
                "name": tool_name,
                "source": "claude_generated",
                "code": tool_code,
            }

            # PERSIST to disk so the tool survives process restarts
            try:
                dyn_dir = Path(__file__).parent.parent / "tools" / "dynamic"
                dyn_dir.mkdir(parents=True, exist_ok=True)
                tool_file = dyn_dir / f"{tool_name}.py"
                header = (
                    f'"""Dynamically generated tool: {tool_name}\n\n'
                    f'{description}\n\n'
                    f'Generated by Claude during MAGE EVOLVE phase.\n'
                    f'"""\n\n'
                    f'import math\nimport re\n\n'
                )
                tool_file.write_text(header + tool_code)
                logger.info(f"Persisted dynamic tool to {tool_file}")
            except Exception as persist_err:
                logger.warning(f"Could not persist dynamic tool to disk: {persist_err}")

            logger.info(f"Dynamically registered tool: {tool_name} — {description}")
            return True

        except Exception as e:
            logger.error(f"Failed to compile tool '{tool_name}': {e}")
            return False

    @staticmethod
    def load_persisted_dynamic_tools() -> int:
        """Load all dynamic tools from disk into TOOL_REGISTRY.

        Called at the start of each iteration so tools created in earlier
        iterations (or earlier runs) are available. Returns count loaded.
        """
        import math
        from pathlib import Path
        from mage.tools.st_tools import TOOL_REGISTRY

        dyn_dir = Path(__file__).parent.parent / "tools" / "dynamic"
        if not dyn_dir.exists():
            return 0

        loaded = 0
        for tool_file in sorted(dyn_dir.glob("*.py")):
            tool_name = tool_file.stem
            if tool_name in TOOL_REGISTRY:
                continue
            try:
                code = tool_file.read_text()
                safe_globals = {
                    "__builtins__": {
                        "abs": abs, "round": round, "max": max, "min": min,
                        "sum": sum, "len": len, "int": int, "float": float,
                        "str": str, "bool": bool, "list": list, "dict": dict,
                        "tuple": tuple, "range": range, "enumerate": enumerate,
                        "zip": zip, "sorted": sorted, "isinstance": isinstance,
                        "ValueError": ValueError, "TypeError": TypeError,
                        "True": True, "False": False, "None": None,
                    },
                    "math": math,
                    "re": __import__("re"),
                }
                exec(code, safe_globals)
                func = safe_globals.get(tool_name)
                if func is None:
                    # Find any non-builtin callable
                    for name, obj in safe_globals.items():
                        if callable(obj) and not name.startswith("_") and name not in ("math", "re"):
                            func = obj
                            tool_name = name
                            break
                if func is None:
                    continue
                TOOL_REGISTRY[tool_name] = {
                    "function": func,
                    "description": f"Loaded from {tool_file.name}",
                    "prereqs": [],
                    "name": tool_name,
                    "source": "claude_generated",
                    "code": code,
                }
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load persisted tool {tool_file.name}: {e}")
        if loaded:
            logger.info(f"Loaded {loaded} persisted dynamic tools")
        return loaded

    def build_initial_evokg(
        self,
        questions: list[dict[str, Any]],
        domain: str,
        available_tools: list[str] | None = None,
        cache_path: "Path | None" = None,
        force_discover: bool = False,
    ) -> EvoKG:
        """Build an EvoKG from scratch by discovering skills.

        v12.0 — DISCOVER caching:
            If `cache_path` is provided and exists (and `force_discover` is
            False), load the discovered skills from disk instead of calling
            Claude. This makes runs reproducible across versions: the same
            set of Claude-discovered skills is reused, removing variance
            from skill-discovery randomness.

            On a cache miss (file doesn't exist), Claude is called as
            before and the result is saved to `cache_path` for next time.
        """
        from pathlib import Path
        skills: list[dict[str, Any]] | None = None

        # 1. Try cache
        if cache_path is not None and not force_discover:
            cache_path = Path(cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        cached = json.load(f)
                    skills = cached.get("skills", [])
                    logger.info(
                        f"Loaded {len(skills)} discovered skills from cache: {cache_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load discovery cache {cache_path}: {e}")
                    skills = None

        # 2. Cache miss → call Claude
        if skills is None:
            logger.info("Calling Claude for skill discovery (no cache or forced)")
            skills = self.discover_skills(questions, available_tools=available_tools)
            # Save to cache for next time
            if cache_path is not None and skills:
                try:
                    cache_path = Path(cache_path)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump({"skills": skills, "n_questions": len(questions)}, f, indent=2)
                    logger.info(f"Saved discovered skills to cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save discovery cache: {e}")

        evokg = EvoKG()

        # Add discovered skills
        for skill_def in skills:
            sid = skill_def["skill_id"]
            node = CapabilityNode(
                node_id=sid,
                name=skill_def.get("name", sid),
                mastery=0.0,
                domain=domain,
                description=skill_def.get("description", ""),
                strategy=skill_def.get("strategy", {}),
                prompt_template=skill_def.get("prompt_template", ""),
                discovered_at="iter_0",
            )
            evokg.add_node(node)

        # Add prerequisite edges
        skill_ids = {s["skill_id"] for s in skills}
        for skill_def in skills:
            sid = skill_def["skill_id"]
            for prereq in skill_def.get("prerequisites", []):
                if prereq in skill_ids:
                    evokg.add_edge(Edge(prereq, sid, RelationType.PREREQUISITE_OF))

        logger.info(f"Built EvoKG from discovery: {evokg}")
        return evokg
