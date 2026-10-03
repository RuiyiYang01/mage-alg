"""Regex-based parameter extractors for STBench/STARK questions.

Ported from ST-Reasoner's agent parameter parsers. These are deterministic —
no LLM needed. Each extractor matches a specific question format and returns
structured parameters ready for tool invocation.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── Direction ──────────────────────────────────────────────────────

_DIRECTION_PATTERN = re.compile(
    r'[Aa]\s+has\s+a\s+longitude\s+of\s+(-?[\d]+\.[\d]+)\s+and\s+a\s+latitude\s+of\s+(-?[\d]+\.[\d]+).*?'
    r'[Bb]\s+has\s+a\s+longitude\s+of\s+(-?[\d]+\.[\d]+)\s+and\s+a\s+latitude\s+of\s+(-?[\d]+\.[\d]+)',
    re.DOTALL,
)


def extract_direction(question: str) -> dict | None:
    m = _DIRECTION_PATTERN.search(question)
    if m:
        return {
            "lon_a": float(m.group(1)), "lat_a": float(m.group(2)),
            "lon_b": float(m.group(3)), "lat_b": float(m.group(4)),
        }
    return None


# ── Temporal relations ─────────────────────────────────────────────

_TEMPORAL_INTERVAL = re.compile(r'\((-?[\d.]+),\s*(-?[\d.]+)\)')
_TEMPORAL_RELATION = re.compile(
    r'(?:temporal\s+relationship|relation)\s+\*{0,2}["\']?\s*([\w\s]+?)\s*["\']?\*{0,2}\s+with',
    re.IGNORECASE,
)


def extract_temporal_relation(question: str) -> dict | None:
    intervals = _TEMPORAL_INTERVAL.findall(question)
    rel_m = _TEMPORAL_RELATION.search(question)
    if len(intervals) >= 2 and rel_m:
        raw_rel = rel_m.group(1).strip().lower()
        # Normalize Allen's relation names (handle "is X by" variants)
        rel_map = {
            "meets": "meets", "is met by": "met_by", "met by": "met_by",
            "overlaps": "overlaps", "is overlapped by": "overlapped_by", "overlapped by": "overlapped_by",
            "contains": "contains", "is contained by": "during", "during": "during",
            "starts": "starts", "is started by": "started_by", "started by": "started_by",
            "finishes": "finishes", "is finished by": "finished_by", "finished by": "finished_by",
            "precedes": "precedes", "is preceded by": "preceded_by", "preceded by": "preceded_by",
            "equals": "equals",
        }
        relation = rel_map.get(raw_rel, raw_rel.replace(" ", "_"))
        return {
            "interval_a": [float(intervals[0][0]), float(intervals[0][1])],
            "interval_b": [float(intervals[1][0]), float(intervals[1][1])],
            "relation": relation,
        }
    return None


# ── Spatial relations ──────────────────────────────────────────────

_SPATIAL_RELATION = re.compile(
    r'(?:spatial\s+)?relationship\s+\*{0,2}\s*(\w+)\s*\*{0,2}',
    re.IGNORECASE,
)
_COORD_PAIR = re.compile(r'\((-?[\d.]+),\s*(-?[\d.]+)\)')
_GEOM_TYPE = re.compile(r'(Point|LineString|Polygon)', re.IGNORECASE)


def extract_spatial_relation(question: str) -> dict | None:
    rel_m = _SPATIAL_RELATION.search(question)
    if not rel_m:
        return None

    relation = rel_m.group(1).lower()

    # Find geometry sections
    geom_types = _GEOM_TYPE.findall(question)
    coords = _COORD_PAIR.findall(question)

    if len(coords) < 2:
        return None

    # Heuristic: split coordinates by geometry type markers
    # Look for "Geometry A:" and "Geometry B:" or similar markers
    geom_sections = re.split(r'(?:Geometry\s+[AB12]|Target|Query)\s*:', question, flags=re.IGNORECASE)

    if len(geom_sections) >= 3:
        coords_a = _COORD_PAIR.findall(geom_sections[1])
        coords_b = _COORD_PAIR.findall(geom_sections[2])
    else:
        # Fallback: split coordinates in half
        mid = len(coords) // 2
        coords_a = coords[:mid] if mid > 0 else coords[:1]
        coords_b = coords[mid:] if mid > 0 else coords[1:]

    geom1 = [[float(x), float(y)] for x, y in coords_a]
    geom2 = [[float(x), float(y)] for x, y in coords_b]

    type1 = geom_types[0].lower() if len(geom_types) > 0 else ("point" if len(geom1) == 1 else "linestring")
    type2 = geom_types[1].lower() if len(geom_types) > 1 else ("point" if len(geom2) == 1 else "linestring")

    return {
        "relation": relation,
        "geom1": geom1, "geom1_type": type1,
        "geom2": geom2, "geom2_type": type2,
    }


# ── Sensor localization ────────────────────────────────────────────

_SENSOR_COORD = re.compile(r'Sensor\s+([A-Z]):\s*\[(-?[\d.]+),\s*(-?[\d.]+)\]')
_SENSOR_READINGS = re.compile(r'(?:readings?\s+(?:are|is)\s*)([\d.,\s\-nan]+)', re.IGNORECASE)
_WAVE_SPEED = re.compile(r'(?:wave\s*speed|speed)\s*(?:is|=|:)\s*([\d.]+)', re.IGNORECASE)
_BEARING_ANGLES = re.compile(r'(?:bearing|angle)\s+(?:is|=|:)?\s*([\d.,\s]+)(?:deg|°)?', re.IGNORECASE)


def extract_localization(question: str) -> dict | None:
    sensors = _SENSOR_COORD.findall(question)
    if len(sensors) < 2:
        return None

    sensor_positions = [[float(x), float(y)] for _, x, y in sensors]

    # Try to find readings
    readings_m = _SENSOR_READINGS.findall(question)
    wave_m = _WAVE_SPEED.search(question)
    bearing_m = _BEARING_ANGLES.findall(question)

    result = {"sensor_positions": sensor_positions}

    if readings_m:
        readings = []
        for r_str in readings_m[0].split(","):
            r_str = r_str.strip()
            if r_str.lower() == "nan" or not r_str:
                readings.append(float("nan"))
            else:
                readings.append(float(r_str))
        result["readings"] = readings

    if wave_m:
        result["wave_speed"] = float(wave_m.group(1))

    if bearing_m:
        angles = [float(a.strip()) for a in bearing_m[0].split(",") if a.strip()]
        result["bearing_angles"] = angles

    return result


# ── Node time series (ST-Bench-new) ───────────────────────────────

_NODE_SERIES = re.compile(
    r'Node\s+(\d+)\s+time\s+series\s+with\s+length\s+of\s+(\d+)\s*:\s*\[([\d.,\s\-]+)\]'
)
_GRAPH_EDGE = re.compile(r'Node\s+(\d+)\s*->\s*Node\s+(\d+)')
_TIME_WINDOW = re.compile(r'time\s+steps?\s+(\d+)\s*[-–to]+\s*(\d+)', re.IGNORECASE)
_FORECAST_STEPS = re.compile(r'(?:next|predict|forecast)\s+(\d+)\s+(?:steps?|values?)', re.IGNORECASE)


def extract_timeseries(question: str) -> dict | None:
    series_matches = _NODE_SERIES.findall(question)
    if not series_matches:
        return None

    node_series = {}
    for node_id, length, values_str in series_matches:
        values = []
        for v in values_str.split(","):
            v = v.strip()
            if v and v != "..." and v != "…":
                try:
                    values.append(float(v))
                except ValueError:
                    continue
        node_series[int(node_id)] = values

    edges = [(int(s), int(d)) for s, d in _GRAPH_EDGE.findall(question)]

    result = {"node_series": node_series, "edges": edges}

    window_m = _TIME_WINDOW.search(question)
    if window_m:
        result["window"] = [int(window_m.group(1)), int(window_m.group(2))]

    forecast_m = _FORECAST_STEPS.search(question)
    if forecast_m:
        result["forecast_steps"] = int(forecast_m.group(1))

    return result


# ── Point in region ────────────────────────────────────────────────

_ADMIN_COORD = re.compile(r'"latitude"\s*:\s*(-?[\d.]+).*?"longitude"\s*:\s*(-?[\d.]+)', re.DOTALL)
_ADMIN_OPTIONS = re.compile(r'\((\d+)\)\s*:\s*([^()\n]+)')


def extract_admin_region(question: str) -> dict | None:
    coord_m = _ADMIN_COORD.search(question)
    options = _ADMIN_OPTIONS.findall(question)
    if coord_m and options:
        return {
            "latitude": float(coord_m.group(1)),
            "longitude": float(coord_m.group(2)),
            "options": {num: name.strip() for num, name in options},
        }
    return None


# ── Trajectory parsing ────────────────────────────────────────────

_TRAJECTORY_POINT = re.compile(r'\((-?[\d.]+),\s*(-?[\d.]+)(?:,\s*(-?[\d.]+))?\)')


def extract_trajectory(question: str) -> dict | None:
    points = _TRAJECTORY_POINT.findall(question)
    if len(points) < 2:
        return None
    trajectory = []
    for lon, lat, ts in points:
        pt = [float(lon), float(lat)]
        if ts:
            pt.append(float(ts))
        trajectory.append(pt)
    return {"trajectory": trajectory}


# ── Master dispatch ────────────────────────────────────────────────

TASK_EXTRACTORS: dict[str, Any] = {
    "STBENCH_DIRECTION_DETERMINATION": extract_direction,
    "STBENCH_ADMIN_REGION": extract_admin_region,
    "STBENCH_POINT_REGION": extract_admin_region,
    "STARK_TEMPORAL_RELATIONSHIP": extract_temporal_relation,
    "STARK_SPATIAL_RELATIONSHIP": extract_spatial_relation,
    "STARK_SPATIAL_LOCALIZATION": extract_localization,
    "STARK_TEMPORAL_LOCALIZATION": extract_localization,
    "STARK_SPATIOTEMPORAL_RELATIONSHIP": extract_spatial_relation,
    "ST_BENCH_NEW_CORRELATION": extract_timeseries,
    "ST_BENCH_NEW_ENTITY": extract_timeseries,
    "ST_BENCH_NEW_ETIOLOGICAL": extract_timeseries,
    "ST_BENCH_NEW_FORECASTING": extract_timeseries,
    "STBENCH_TRAJECTORY_TRAJECTORY": extract_trajectory,
    "STBENCH_TRAJECTORY_REGION": extract_trajectory,
    "STBENCH_TRAJECTORY_ANOMALY": extract_trajectory,
    "STBENCH_TRAJECTORY_PREDICTION": extract_trajectory,
    "STBENCH_POINT_TRAJECTORY": extract_trajectory,
    "STBENCH_FLOW_PREDICTION": extract_timeseries,
}
