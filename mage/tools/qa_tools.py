"""QA reasoning tools for MAGE.

Tools for multi-hop question answering:
- Search context paragraphs for specific entities/facts
- Extract and compare entity attributes
- Decompose complex questions into sub-questions
"""

from __future__ import annotations

import re
from typing import Any

from mage.tools.st_tools import register_tool


@register_tool(
    "search_paragraphs",
    "Search context paragraphs for mentions of a keyword/entity. Args: context (full text), query (search term)",
    prereqs=["fact_retrieval"],
)
def search_paragraphs(context: str, query: str) -> dict:
    """Find paragraphs containing the query term."""
    paragraphs = re.split(r"\n\n|\[.*?\]\n", context)
    matches = []
    query_lower = query.lower()
    for i, para in enumerate(paragraphs):
        if query_lower in para.lower():
            matches.append({"index": i, "text": para.strip()[:200]})
    return {"matches": matches, "count": len(matches)}


@register_tool(
    "extract_entities",
    "Extract named entities (people, places, dates, numbers) from text. Args: text",
    prereqs=["fact_retrieval"],
)
def extract_entities(text: str) -> dict:
    """Simple regex-based entity extraction."""
    # Dates
    dates = re.findall(r"\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b", text)
    # Numbers
    numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
    # Capitalized phrases (likely named entities)
    names = re.findall(r"(?<!\. )\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    # Filter common words
    stop = {"The", "This", "That", "These", "Those", "What", "When", "Where", "How", "Who", "Which", "Answer", "Question"}
    names = [n for n in names if n not in stop]

    return {"names": list(set(names))[:10], "dates": dates[:5], "numbers": numbers[:10]}


@register_tool(
    "compare_values",
    "Compare two values (numeric or string). Args: value_a, value_b, comparison (greater/less/equal/contains)",
    prereqs=[],
)
def compare_values(value_a: str, value_b: str, comparison: str = "equal") -> dict:
    """Compare two values."""
    try:
        a_num = float(value_a.replace(",", ""))
        b_num = float(value_b.replace(",", ""))
        if comparison == "greater":
            return {"result": a_num > b_num, "a": a_num, "b": b_num}
        elif comparison == "less":
            return {"result": a_num < b_num, "a": a_num, "b": b_num}
        elif comparison == "equal":
            return {"result": abs(a_num - b_num) < 0.01, "a": a_num, "b": b_num}
    except (ValueError, TypeError):
        pass

    a_str = str(value_a).lower().strip()
    b_str = str(value_b).lower().strip()
    if comparison == "contains":
        return {"result": b_str in a_str, "a": value_a, "b": value_b}
    return {"result": a_str == b_str, "a": value_a, "b": value_b}
