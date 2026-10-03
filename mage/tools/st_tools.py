"""Spatial-temporal computation tools for MAGE.

Lightweight tool layer for STBench tasks. The agent calls these via
ReAct-style interaction: think → tool_call(name, args) → observe result → answer.

Tools are registered as skills in the EvoKG so the agent can learn
when and how to use them.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

# ── Tool registry ──────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str, description: str, prereqs: list[str] | None = None):
    """Decorator to register a tool function."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "function": fn,
            "description": description,
            "prereqs": prereqs or [],
            "name": name,
        }
        return fn
    return decorator


def call_tool(name: str, args_str: str) -> str:
    """Call a registered tool by name with JSON args string. Returns result as string."""
    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'. Available: {list(TOOL_REGISTRY.keys())}"
    try:
        args = json.loads(args_str) if args_str.strip() else {}
        result = TOOL_REGISTRY[name]["function"](**args)
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as e:
        return f"Error calling {name}: {e}"


def get_tool_descriptions() -> str:
    """Format all tools for inclusion in LLM prompts."""
    lines = ["Available tools:"]
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"  - {name}: {info['description']}")
    lines.append("")
    lines.append('To use a tool, write: TOOL_CALL: {"tool": "name", "args": {...}}')
    lines.append("The result will be shown. Then continue reasoning.")
    return "\n".join(lines)


# ── Spatial tools ──────────────────────────────────────────────────

@register_tool(
    "compute_distance",
    "Compute haversine distance (km) between two GPS coordinates. Args: lat1, lon1, lat2, lon2",
)
def compute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return {"distance_km": round(R * c, 4)}


@register_tool(
    "compute_direction",
    "Compute compass direction from point A to point B. Args: lon_a, lat_a, lon_b, lat_b",
)
def compute_direction(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> dict:
    dx = lon_b - lon_a
    dy = lat_b - lat_a
    angle = math.degrees(math.atan2(dx, dy)) % 360
    directions = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
    idx = int((angle + 22.5) / 45) % 8
    return {"direction": directions[idx], "bearing_degrees": round(angle, 2)}


@register_tool(
    "check_point_in_region",
    "Check if a point (lat, lon) is inside a polygon region. Args: point [lat, lon], polygon [[lat,lon], ...]",
)
def check_point_in_region(point: list[float], polygon: list[list[float]]) -> dict:
    """Ray casting algorithm for point-in-polygon."""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return {"inside": inside}


@register_tool(
    "check_spatial_relation",
    "Check spatial relation between two geometries. Args: relation (contains/intersects/within), geom1 [[x,y],...], geom2 [[x,y],...]",
)
def check_spatial_relation(relation: str, geom1: list, geom2: list) -> dict:
    try:
        from shapely.geometry import LineString, Point, Polygon

        def make_geom(coords):
            if len(coords) == 1:
                return Point(coords[0])
            elif len(coords) >= 3 and coords[0] == coords[-1]:
                return Polygon(coords)
            else:
                return LineString(coords)

        g1 = make_geom(geom1)
        g2 = make_geom(geom2)

        checks = {
            "contains": g1.contains(g2),
            "within": g1.within(g2),
            "intersects": g1.intersects(g2),
            "crosses": g1.crosses(g2),
            "touches": g1.touches(g2),
            "overlaps": g1.overlaps(g2),
        }
        result = checks.get(relation.lower(), False)
        return {"relation": relation, "holds": result}
    except ImportError:
        return {"error": "shapely not installed"}


# ── Temporal tools ─────────────────────────────────────────────────

@register_tool(
    "check_temporal_relation",
    "Check Allen's temporal relation between two intervals. Args: relation, interval_a [start, end], interval_b [start, end]",
)
def check_temporal_relation(relation: str, interval_a: list[float], interval_b: list[float]) -> dict:
    a_s, a_e = interval_a
    b_s, b_e = interval_b
    tol = 1e-6

    checks = {
        "precedes": a_e < b_s - tol,
        "meets": abs(a_e - b_s) < tol,
        "overlaps": a_s < b_s and a_e > b_s and a_e < b_e,
        "contains": a_s <= b_s and a_e >= b_e and not (abs(a_s-b_s) < tol and abs(a_e-b_e) < tol),
        "during": b_s <= a_s and b_e >= a_e and not (abs(a_s-b_s) < tol and abs(a_e-b_e) < tol),
        "starts": abs(a_s - b_s) < tol and a_e < b_e,
        "finishes": abs(a_e - b_e) < tol and a_s > b_s,
        "equals": abs(a_s - b_s) < tol and abs(a_e - b_e) < tol,
    }
    result = checks.get(relation.lower(), False)
    return {"relation": relation, "holds": result}


@register_tool(
    "parse_time",
    "Convert time string (HH:MM) to minutes since midnight. Args: time_str",
)
def parse_time(time_str: str) -> dict:
    match = re.match(r"(\d{1,2}):(\d{2})", time_str.strip())
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        return {"minutes": h * 60 + m}
    return {"error": f"Cannot parse time: {time_str}"}


# ── Statistics tools ───────────────────────────────────────────────

@register_tool(
    "compute_stats",
    "Compute statistics for a numeric series. Args: series [float, ...]",
)
def compute_stats(series: list[float]) -> dict:
    if not series:
        return {"error": "empty series"}
    n = len(series)
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series) / n
    std = variance ** 0.5
    # Simple trend via linear regression
    if n >= 2:
        x_mean = (n - 1) / 2
        slope = sum((i - x_mean) * (series[i] - mean) for i in range(n)) / max(sum((i - x_mean)**2 for i in range(n)), 1e-8)
        trend = "increasing" if slope > 0.01 else ("decreasing" if slope < -0.01 else "stable")
    else:
        slope = 0.0
        trend = "stable"
    return {
        "mean": round(mean, 4), "std": round(std, 4),
        "min": round(min(series), 4), "max": round(max(series), 4),
        "trend": trend, "slope": round(slope, 4), "n": n,
    }


@register_tool(
    "compute_correlation",
    "Compute Pearson correlation between two series. Args: series_a, series_b",
)
def compute_correlation(series_a: list[float], series_b: list[float]) -> dict:
    n = min(len(series_a), len(series_b))
    if n < 2:
        return {"correlation": 0.0}
    a, b = series_a[:n], series_b[:n]
    mean_a, mean_b = sum(a)/n, sum(b)/n
    cov = sum((a[i]-mean_a)*(b[i]-mean_b) for i in range(n)) / n
    std_a = (sum((x-mean_a)**2 for x in a)/n) ** 0.5
    std_b = (sum((x-mean_b)**2 for x in b)/n) ** 0.5
    if std_a < 1e-8 or std_b < 1e-8:
        return {"correlation": 0.0}
    return {"correlation": round(cov / (std_a * std_b), 4)}


@register_tool(
    "python_compute",
    "Execute a simple Python expression for custom computation. Args: expression (string)",
)
def python_compute(expression: str) -> dict:
    """Safe eval for simple math expressions."""
    allowed = set("0123456789+-*/().,%[] ")
    if not all(c in allowed or c.isalpha() for c in expression):
        return {"error": "expression contains unsafe characters"}
    try:
        # Only allow math operations
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min, "sum": sum, "len": len, "math": math})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
