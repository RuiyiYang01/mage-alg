"""End-to-end tool pipeline: extract → compute → map answer.

This is the deterministic path that achieves high accuracy.
For each task_type, it:
1. Extracts parameters from the question using regex
2. Calls the appropriate ST-Reasoner tool
3. Maps the tool result to the answer option

No LLM needed for supported task types.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

# Import ST-Reasoner tools
import os

if os.environ.get("ST_REASONER_DIR"):
    sys.path.insert(0, os.environ["ST_REASONER_DIR"])

import logging

logger = logging.getLogger(__name__)


def run_tool_pipeline(question: str, task_type: str, expected_answer: str = "", evokg: Any = None) -> str | None:
    """Run the full deterministic pipeline for a question.

    Three extraction methods tried in order:
    1. Pre-built regex extractor (from extractors.py)
    2. EvoKG-stored dynamic extractor (from Claude)
    3. Give up → return None → LLM handles it
    """
    from mage.tools.extractors import TASK_EXTRACTORS

    # Method 1: Pre-built extractor
    extractor = TASK_EXTRACTORS.get(task_type)
    if extractor:
        params = extractor(question)
        if params:
            try:
                result = _compute(task_type, params, question)
                if result is not None:
                    answer = _map_to_answer(question, task_type, result)
                    if answer is not None:
                        return answer
            except Exception as e:
                logger.debug(f"Pre-built pipeline failed for {task_type}: {e}")

    # Method 2: EvoKG dynamic extractor (from Claude)
    if evokg:
        answer = _try_dynamic_extractor(question, task_type, evokg)
        if answer is not None:
            return answer

    return None


def _try_dynamic_extractor(question: str, task_type: str, evokg: Any) -> str | None:
    """Try extractors stored in EvoKG by Claude during EVOLVE."""
    import re as _re
    from mage.core.schemas import NodeType

    skills = evokg.get_nodes_by_type(NodeType.CAPABILITY)
    for skill in skills:
        strategy = skill.get("strategy", {})
        if not isinstance(strategy, dict) or strategy.get("type") != "tool":
            continue

        extractor_regex = strategy.get("extractor_regex", "")
        tool_name = strategy.get("tool", "")

        if not extractor_regex or not tool_name:
            continue

        # Try the dynamic regex with timeout protection
        # Claude-written regex can cause catastrophic backtracking
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Regex timed out")

            pattern = eval(extractor_regex)

            # Set 2-second timeout for regex execution
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(2)
            try:
                match = _re.search(pattern, question[:1000], _re.DOTALL)  # cap input length too
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            if match:
                # Build params from regex groups
                groups = match.groups()
                params = {f"param_{i}": float(g) if g.replace(".", "").replace("-", "").isdigit() else g
                         for i, g in enumerate(groups)}

                from mage.tools.st_tools import TOOL_REGISTRY, call_tool
                if tool_name in TOOL_REGISTRY:
                    import json
                    result_str = call_tool(tool_name, json.dumps(params))
                    if not result_str.startswith("Error"):
                        result = json.loads(result_str)
                        answer = _map_to_answer(question, task_type, result)
                        if answer is not None:
                            return answer
        except Exception:
            continue

    return None


def _compute(task_type: str, params: dict, question: str) -> Any:
    """Call the appropriate tool with extracted parameters."""

    # ── Direction ──
    if task_type == "STBENCH_DIRECTION_DETERMINATION":
        from tools.spatial_ops import compute_compass_direction
        # Returns 1-8 matching STBench option numbering directly
        option_num = compute_compass_direction(
            params["lon_a"], params["lat_a"],
            params["lon_b"], params["lat_b"],
        )
        return {"option_number": option_num}

    # ── Temporal relations ──
    if task_type in ("STARK_TEMPORAL_RELATIONSHIP",):
        from tools.temporal_ops import (
            check_allen_meets, check_allen_overlaps, check_allen_contains,
            check_allen_during, check_allen_starts, check_allen_finishes,
            check_allen_precedes, check_allen_equals,
            check_allen_met_by, check_allen_overlapped_by,
            check_allen_started_by, check_allen_finished_by,
            check_allen_preceded_by,
        )
        a = tuple(params["interval_a"])
        b = tuple(params["interval_b"])
        rel = params["relation"]

        checkers = {
            "meets": check_allen_meets,
            "met_by": check_allen_met_by,
            "overlaps": check_allen_overlaps,
            "overlapped_by": check_allen_overlapped_by,
            "contains": check_allen_contains,
            "during": check_allen_during,
            "starts": check_allen_starts,
            "started_by": check_allen_started_by,
            "finishes": check_allen_finishes,
            "finished_by": check_allen_finished_by,
            "precedes": check_allen_precedes,
            "preceded_by": check_allen_preceded_by,
            "equals": check_allen_equals,
        }
        checker = checkers.get(rel)
        if checker:
            holds = checker(a, b)
            return {"holds": holds, "relation": rel}
        return None

    # ── Spatial relations ──
    if task_type in ("STARK_SPATIAL_RELATIONSHIP", "STARK_SPATIOTEMPORAL_RELATIONSHIP"):
        from tools.spatial_ops import (
            check_contains, check_crosses, check_intersects,
            check_within, check_touches, check_overlaps, check_equals,
        )
        raw_rel = params["relation"]
        g1 = params["geom1"]
        g2 = params["geom2"]
        t1 = params.get("geom1_type", "linestring")
        t2 = params.get("geom2_type", "point")
        is_poly = t1 == "polygon"

        # Normalize relation names (STARK uses various forms)
        rel_map = {
            "contains": "contains", "in": "within", "within": "within",
            "crosses": "crosses", "intersects": "intersects",
            "touches": "touches", "overlaps": "overlaps",
            "equals": "equals", "equal": "equals",
            "disjoint": "disjoint",
        }
        rel = rel_map.get(raw_rel, raw_rel)

        checkers = {
            "contains": lambda: check_contains(g1, g2, is_poly),
            "within": lambda: check_within(g1, g2, t1, t2),
            "crosses": lambda: check_crosses(g1, g2, t1, t2),
            "intersects": lambda: check_intersects(g1, g2, t1, t2),
            "touches": lambda: check_touches(g1, g2, t1, t2),
            "overlaps": lambda: check_overlaps(g1, g2, t1, t2),
            "equals": lambda: check_equals(g1, g2),
            "disjoint": lambda: not check_intersects(g1, g2, t1, t2),
        }
        checker = checkers.get(rel)
        if checker:
            try:
                holds = checker()
                return {"holds": holds, "relation": rel}
            except Exception:
                return None
        return None

    # ── Localization ──
    if task_type in ("STARK_SPATIAL_LOCALIZATION", "STARK_TEMPORAL_LOCALIZATION"):
        if "readings" in params and "wave_speed" in params:
            from tools.spatial_ops import localize_event_location, localize_event_time
            sp = params["sensor_positions"]
            rd = params["readings"]
            ws = params["wave_speed"]
            if "time" in task_type.lower() or "temporal" in task_type.lower():
                t = localize_event_time(sp, rd, ws)
                return {"event_time": t}
            else:
                loc = localize_event_location(sp, rd, ws)
                return {"location": loc}
        if "bearing_angles" in params:
            from tools.spatial_ops import triangulate_from_bearings
            loc = triangulate_from_bearings(
                params["sensor_positions"], params["bearing_angles"],
            )
            return {"location": loc}
        return None

    # ── Time series / graph (ST-Bench-new) ──
    if task_type in ("ST_BENCH_NEW_CORRELATION", "ST_BENCH_NEW_ENTITY",
                     "ST_BENCH_NEW_ETIOLOGICAL", "ST_BENCH_NEW_FORECASTING"):
        return _compute_graph_analysis(params, question, task_type)

    # ── Flow prediction ──
    if task_type == "STBENCH_FLOW_PREDICTION":
        return _compute_flow_forecast(params, question)

    # ── Trajectory anomaly ──
    if task_type == "STBENCH_TRAJECTORY_ANOMALY":
        if "trajectory" in params and len(params["trajectory"]) >= 3:
            from tools.spatial_ops import detect_trajectory_anomaly
            traj = params["trajectory"]
            coords = [(p[0], p[1]) for p in traj]
            timestamps = [p[2] for p in traj if len(p) > 2]
            if timestamps and len(timestamps) == len(coords):
                try:
                    result = detect_trajectory_anomaly(coords, timestamps)
                    return result
                except Exception:
                    pass
        return None

    # ── Trajectory region ──
    if task_type == "STBENCH_TRAJECTORY_REGION":
        # Complex: need both trajectory + region polygons from question
        # Defer to LLM for now (extraction too complex for regex)
        return None

    # ── Trajectory prediction ──
    if task_type == "STBENCH_TRAJECTORY_PREDICTION":
        if "trajectory" in params and len(params["trajectory"]) >= 2:
            # Simple linear extrapolation from last 2 points
            traj = params["trajectory"]
            p1 = traj[-2]
            p2 = traj[-1]
            predicted = [round(2*p2[i] - p1[i], 5) for i in range(min(len(p1), 2))]
            return {"predicted_point": predicted}
        return None

    return None


def _compute_graph_analysis(params: dict, question: str, task_type: str) -> dict | None:
    """Graph/time-series analysis for ST-Bench-new questions."""
    from tools.graph_ops import (
        build_flow_graph, analyze_topology, compute_pairwise_causality,
        compute_graph_feature_vector, verify_option_against_features,
    )
    from tools.stats_ops import (
        compute_pairwise_correlations, compute_window_statistics,
        simple_linear_forecast,
    )

    node_series = params.get("node_series", {})
    edges = params.get("edges", [])

    if not node_series:
        return None

    # Build graph
    G = build_flow_graph(edges) if edges else None

    if task_type == "ST_BENCH_NEW_FORECASTING" and "forecast_steps" in params:
        # Simple forecast
        target = max(node_series.keys())
        series = node_series[target]
        n_steps = params["forecast_steps"]
        forecast = simple_linear_forecast(series, n_steps)
        return {"forecast": [round(v, 2) for v in forecast]}

    if G and node_series:
        # Compute features for MCQ
        features = compute_graph_feature_vector(G, node_series, target_node=0)
        return {"features": features, "type": "graph_analysis"}

    return None


def _compute_flow_forecast(params: dict, question: str) -> dict | None:
    """Flow/traffic prediction."""
    from tools.stats_ops import simple_linear_forecast, moving_average_forecast

    node_series = params.get("node_series", {})
    if not node_series:
        return None

    # Find the target series and forecast
    for node_id, series in node_series.items():
        if len(series) >= 6:
            forecast = moving_average_forecast(series, 6, window=min(5, len(series)))
            return {"forecast": [round(v, 1) for v in forecast]}

    return None


def _map_to_answer(question: str, task_type: str, result: dict) -> str | None:
    """Map tool result to the answer format expected by the question."""

    # Direct option number (from tools that return the answer directly)
    if "option_number" in result:
        return str(result["option_number"])

    # Boolean results → [1.0] / [0.0]
    if "holds" in result:
        return "1.0" if result["holds"] else "0.0"

    # Direction → option number
    if "direction" in result:
        direction = result["direction"].lower()
        dir_aliases = {
            "north": ["north", "n"], "northeast": ["northeast", "ne"],
            "east": ["east", "e"], "southeast": ["southeast", "se"],
            "south": ["south", "s"], "southwest": ["southwest", "sw"],
            "west": ["west", "w"], "northwest": ["northwest", "nw"],
        }
        # Find numbered options in question
        options = re.findall(r'\((\d+)\)\s*:?\s*([^,()\n]+)', question)
        if not options:
            options = re.findall(r'(\d+)\)\s*([^,\n]+)', question)

        for num, text in options:
            text_lower = text.lower().strip()
            for full_name, aliases in dir_aliases.items():
                if direction == full_name or direction in aliases:
                    if text_lower == full_name or text_lower in aliases:
                        return num
        return None

    # Predicted point → match to options
    if "predicted_point" in result:
        pred = result["predicted_point"]
        # Find closest option point
        import re as _re
        options = _re.findall(r'\((\d+)\)\s+Point\s+\d+\s+\((-?[\d.]+),\s*(-?[\d.]+)\)', question)
        if options:
            best_opt = None
            best_dist = float('inf')
            for num, x, y in options:
                dist = ((pred[0]-float(x))**2 + (pred[1]-float(y))**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_opt = num
            if best_opt and best_dist < 0.1:
                return best_opt

    # Anomaly detection → "anomalous" / "normal"
    if "is_anomalous" in result:
        return "anomalous" if result["is_anomalous"] else "normal"

    # Location → formatted coordinates
    if "location" in result:
        loc = result["location"]
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            return f"{loc[0]:.4f}, {loc[1]:.4f}"

    # Forecast → formatted list
    if "forecast" in result:
        vals = result["forecast"]
        return str(vals)

    # Event time
    if "event_time" in result:
        return str(round(result["event_time"], 4))

    return None
