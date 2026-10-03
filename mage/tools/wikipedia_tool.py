"""Wikipedia retrieval tool for factoid QA.

For benchmarks like WebQA where the dataset includes a Freebase URL but no
context, this tool fetches the Wikipedia summary for the linked entity and
returns the first ~500 chars as retrieval context.

Cache is persisted to mage/tools/wikipedia_cache/<entity>.txt so the same
entity is only fetched once across all runs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "wikipedia_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def freebase_url_to_entity(url: str) -> str | None:
    """Convert a Freebase URL to a candidate Wikipedia entity title.

    Examples:
        http://www.freebase.com/view/en/james_k_polk → "james k polk"
        http://www.freebase.com/view/en/oregon_ducks → "oregon ducks"
    """
    if not url:
        return None
    m = re.search(r"/en/([^/?#]+)", url)
    if not m:
        return None
    slug = m.group(1)
    # Replace underscores with spaces, strip trailing _N (disambiguation)
    return slug.replace("_", " ").strip()


def _cache_path(entity: str) -> Path:
    safe = re.sub(r"[^a-z0-9_]", "_", entity.lower())[:120]
    return CACHE_DIR / f"{safe}.txt"


def _load_cache(entity: str) -> str | None:
    p = _cache_path(entity)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _save_cache(entity: str, content: str) -> None:
    try:
        _cache_path(entity).write_text(content, encoding="utf-8")
    except Exception as e:
        logger.debug(f"Wikipedia cache write failed for {entity}: {e}")


def _candidate_titles(entity: str) -> list[str]:
    """Generate candidate Wikipedia titles from an entity slug.

    'james k polk' →
        ['James_K_Polk', 'James_K._Polk', 'James K Polk',
         'James_k_polk', 'james k polk']
    """
    words = entity.split()
    if not words:
        return []
    cap_words = [w.capitalize() for w in words]
    candidates = []
    # Title case (most common)
    candidates.append("_".join(cap_words))
    # Title case with periods on single-letter middle initials
    cap_with_periods = []
    for w in cap_words:
        if len(w) == 1 and w.isalpha():
            cap_with_periods.append(f"{w}.")
        else:
            cap_with_periods.append(w)
    candidates.append("_".join(cap_with_periods))
    # Original (in case it's already a proper title)
    candidates.append("_".join(words))
    # Lowercase fallback
    candidates.append("_".join(w.lower() for w in words))
    # Dedupe while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_one_title(title: str) -> dict | None:
    """Single REST API call. Returns dict with extract/title or None."""
    try:
        import requests
    except ImportError:
        return None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        resp = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "MAGE-Research/1.0 (research)"},
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def fetch_wikipedia_summary(entity: str, max_chars: int = 800) -> str:
    """Fetch the Wikipedia summary (lead section) for an entity.

    Tries multiple candidate titles (capitalized, with periods, etc.) and
    falls back to free-text search if none of the direct lookups work.
    Returns empty string if nothing is found. Cached on disk.
    """
    if not entity:
        return ""
    cached = _load_cache(entity)
    if cached is not None and cached.strip():
        return cached[:max_chars]

    data = None
    for candidate in _candidate_titles(entity):
        data = _fetch_one_title(candidate)
        if data and data.get("extract"):
            break
        data = None

    # Fallback: free-text search → top result → summary
    if data is None:
        try:
            titles = search_wikipedia(entity, max_results=1)
            if titles:
                data = _fetch_one_title(titles[0].replace(" ", "_"))
        except Exception:
            pass

    if data is None or not data.get("extract"):
        _save_cache(entity, "")
        return ""

    extract = data.get("extract", "") or ""
    description = data.get("description", "") or ""
    title = data.get("title", entity)
    content = f"{title}\n{description}\n{extract}".strip()
    _save_cache(entity, content)
    return content[:max_chars]


def fetch_for_freebase_url(url: str, max_chars: int = 800) -> str:
    """Convenience: extract entity from Freebase URL and fetch Wikipedia."""
    entity = freebase_url_to_entity(url)
    if not entity:
        return ""
    return fetch_wikipedia_summary(entity, max_chars=max_chars)


def search_wikipedia(query: str, max_results: int = 3) -> list[str]:
    """Free-text Wikipedia search — used when no Freebase URL is available.

    Returns the top result titles. The agent can then call fetch_wikipedia_summary
    on the chosen title.
    """
    if not query:
        return []
    cached_q = _cache_path(f"search_{query}")
    if cached_q.exists():
        try:
            return json.loads(cached_q.read_text())[:max_results]
        except Exception:
            pass

    try:
        import requests
    except ImportError:
        return []

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            },
            timeout=5,
            headers={"User-Agent": "MAGE-Research/1.0 (research)"},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        titles = [r["title"] for r in data.get("query", {}).get("search", [])]
        try:
            cached_q.write_text(json.dumps(titles))
        except Exception:
            pass
        return titles[:max_results]
    except Exception:
        return []
