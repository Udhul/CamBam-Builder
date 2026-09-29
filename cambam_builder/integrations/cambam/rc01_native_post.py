"""Independent, bounded stock replay for an RC01 native Pocket Default post.

The posted straight moves, rather than native MOP settings or planned RC01 paths,
are the removal evidence. Polygon results retain an explicit numerical caveat.
"""

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .rc01_adapter import normalize
from .rc01_post import read_default_post
from ...native.reader import read_cambam_bytes


LEVELS = (-1.0, -2.0, -3.0)
SETUP = (-10.0, -10.0, 5.0)
QUAD_SEGS = 256
ARC_SAGITTA_MM = 0.0001


def _path_for_move(item, level=None):
    """Polyline and XY deviation bound for a posted line or relative-I/J arc."""
    a, b = item["start"], item["end"]
    if level is None:
        first = 0.0
    elif a[2] <= level and b[2] <= level:
        first = 0.0
    elif b[2] <= level and b[2] < a[2]:
        first = (level - a[2]) / (b[2] - a[2])
    else:
        return None
    if item["g"] in (0, 1):
        if level is not None and b[2] > a[2]:
            return None
        start = tuple(a[i] + first * (b[i] - a[i]) for i in (0, 1))
        return ((start, tuple(b[:2])), 0.0)
    if item["g"] not in (2, 3):
        raise ValueError(f"line {item['line']}: unknown cutting interpolation")
    cx, cy = item["center"]
    radius0 = math.hypot(a[0] - cx, a[1] - cy)
    radius1 = math.hypot(b[0] - cx, b[1] - cy)
    angle0 = math.atan2(a[1] - cy, a[0] - cx)
    angle1 = math.atan2(b[1] - cy, b[0] - cx)
    if item["g"] == 3:
        sweep = (angle1 - angle0) % (2 * math.pi)
    else:
        sweep = -((angle0 - angle1) % (2 * math.pi))
    max_radius = max(radius0, radius1)
    step = 2 * math.acos(1 - ARC_SAGITTA_MM / max_radius)
    count = max(1, math.ceil(abs(sweep) * (1 - first) / step))
    points = []
    for index in range(count + 1):
        t = first + (1 - first) * index / count
        angle = angle0 + sweep * t
        radius = radius0 + (radius1 - radius0) * t
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    if first == 0:
        points[0] = tuple(a[:2])
    points[-1] = tuple(b[:2])
    # A controller's handling of a rounded arc endpoint is not encoded here;
    # bracket both centerline interpolation and chord flattening conservatively.
    deviation = abs(radius0 - radius1) + 2 * ARC_SAGITTA_MM + 1e-9
    return (tuple(points), deviation)


def _target(job):
    from shapely.geometry import box
    outer = job.outer
    island = job.island
    return box(float(outer.xmin), float(outer.ymin), float(outer.xmax),
               float(outer.ymax)).difference(box(
                   float(island.xmin), float(island.ymin),
                   float(island.xmax), float(island.ymax)))


def _footprints(moves, level):
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    unique = {}
    for item in moves:
        if item["g"] not in (1, 2, 3) or item["end"][2] > item["start"][2]:
            continue
        path = _path_for_move(item, level)
        if path is not None:
            points, error = path
            unique[(points, item["tool"])] = (
                points, 3.0 if item["tool"] == "T1" else 1.0, error)
    inflation = 1 / math.cos(math.pi / (4 * QUAD_SEGS))
    inner = unary_union([
        LineString(points).buffer(radius - error, quad_segs=QUAD_SEGS)
        for points, radius, error in unique.values()])
    outer = unary_union([
        LineString(points).buffer((radius + error) * inflation,
                                  quad_segs=QUAD_SEGS)
        for points, radius, error in unique.values()])
    return inner, outer, len(unique)


def _area_by_depth(moves, job, ideal_radius):
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union

    target = _target(job)
    outer = job.outer
    corners = ((outer.xmin, outer.ymin, 1, 1),
               (outer.xmax, outer.ymin, -1, 1),
               (outer.xmax, outer.ymax, -1, -1),
               (outer.xmin, outer.ymax, 1, -1))
    ideal_parts = []
    for x, y, sx, sy in corners:
        x, y, radius = float(x), float(y), float(ideal_radius)
        square = box(min(x, x + sx * radius), min(y, y + sy * radius),
                     max(x, x + sx * radius), max(y, y + sy * radius))
        disk = LineString(((x + sx * radius, y + sy * radius),) * 2).buffer(
            radius, quad_segs=QUAD_SEGS)
        ideal_parts.append(square.difference(disk))
    allowed_rest = unary_union(ideal_parts).buffer(0.05).union(
        target.boundary.buffer(0.05))
    result = []
    for level in LEVELS:
        inner, outer_bound, count = _footprints(moves, level)
        rest_upper = target.difference(inner)
        result.append({
            "bottom_z": level,
            "rest_area_mm2": [max(0.0, target.difference(outer_bound).area),
                              rest_upper.area],
            "residual_outside_ideal_or_boundary_0_05mm_mm2":
                rest_upper.difference(allowed_rest).area,
            "cut_segments": count,
        })
    return result


def _t2_vertical_access(rough_moves, combined_items):
    """Check T2 vertical columns against the posted T1 full-depth footprint."""
    from shapely.geometry import Point

    def exact_witness(xy):
        px, py = (Fraction(str(value)) for value in xy)
        for move in rough_moves:
            if (move["g"] != 1 or move["start"][2] != -3 or
                    move["end"][2] != -3):
                continue
            ax, ay = (Fraction(str(value)) for value in move["start"][:2])
            bx, by = (Fraction(str(value)) for value in move["end"][:2])
            if ay == by:
                dx = max(min(ax, bx) - px, px - max(ax, bx), 0)
                dy = py - ay
            elif ax == bx:
                dx = px - ax
                dy = max(min(ay, by) - py, py - max(ay, by), 0)
            else:
                continue
            if dx * dx + dy * dy <= 4:  # (T1 radius 3 - T2 radius 1)^2
                return move["line"]
        return None

    rough_inner, _, _ = _footprints(rough_moves, -3.0)
    columns = sorted({tuple(item["start"][:2]) for item in combined_items
                      if item["type"] == "move" and item["tool"] == "T2" and
                      item["start"][:2] == item["end"][:2] and
                      min(item["start"][2], item["end"][2]) < 0})
    outer_radius = 1 / math.cos(math.pi / (4 * QUAD_SEGS))
    return [{"xy": list(xy),
             "exact_single_t1_cut_witness_line": exact_witness(xy),
             "uncovered_area_mm2": Point(*xy).buffer(
                 outer_radius, quad_segs=QUAD_SEGS).difference(rough_inner).area}
            for xy in columns]


def _motion_findings(items, warnings, job):
    """Check emitted events and bounded motion/target conditions."""
    from shapely.geometry import LineString, box

    issues = list(warnings)
    moves = [i for i in items if i["type"] == "move"]
    tool_changes = [i for i in items if i["type"] == "event" and
                    i["kind"] == "tool_change"]
    if [i["tool"] for i in tool_changes] not in (["T1"], ["T1", "T2"]):
        issues.append("tool sequence is not T1 followed by optional T2")
    for event in (i for i in items if i["type"] == "event"):
        if event["kind"] in ("tool_change", "spindle_stop") and tuple(
                event["position"]) != SETUP:
            issues.append(f"line {event['line']}: {event['kind']} away from setup")
        if event["kind"] == "spindle_start" and event["rpm"] != 12000:
            issues.append(f"line {event['line']}: spindle speed is not 12000 rpm")
    for move in moves:
        a, b = move["start"], move["end"]
        line, radius = move["line"], 3 if move["tool"] == "T1" else 1
        if (any(not (-15 <= p[0] <= 55 and -15 <= p[1] <= 45 and
                     -3 <= p[2] <= 10) for p in (a, b))):
            issues.append(f"line {line}: travel or floor bound exceeded")
            continue
        if move["g"] == 0 and (a[2] < 5 or b[2] < 5):
            issues.append(f"line {line}: rapid below clearance +5")
        if min(a[2], b[2]) < 0:
            points, error = _path_for_move(move)
            margin = min(min(p[0] - radius, 40 - radius - p[0],
                             p[1] - radius, 30 - radius - p[1]) for p in points)
            if margin < -error:
                issues.append(f"line {line}: cutter crosses outer protected wall")
            elif error and margin <= error:
                issues.append(f"line {line}: outer-wall clearance unresolved")
            island = job.island
            island_box = box(float(island.xmin), float(island.ymin),
                             float(island.xmax), float(island.ymax))
            distance = LineString(points).distance(island_box)
            if distance < radius - error - 1e-9:
                issues.append(f"line {line}: cutter crosses protected island")
            elif abs(distance - radius) <= error + 1e-9:
                # Exact tangent is allowed; GEOS cannot certify the sign at
                # this boundary from floating coordinates alone.
                issues.append(f"line {line}: island tangency numerically unresolved")
        if move["g"] in (1, 2, 3):
            xy = a[:2] != b[:2]
            if xy and a[2] == b[2] and a[2] < 0 and move["feed"] != 300:
                issues.append(f"line {line}: low-level XY feed lacks RC01 motion role")
            if not xy and b[2] < a[2] and b[2] < 0 and move["feed"] != 60:
                issues.append(f"line {line}: plunge feed is not 60 mm/min")
            if xy and a[2] != b[2] and min(a[2], b[2]) < 0:
                issues.append(f"line {line}: ramped XY engagement lacks RC01 process proof")
    return issues


def _budget(rough, final):
    rough_ideal = (4 - math.pi) * 9
    final_ideal = 4 - math.pi
    # Radius inflation is an outward motion bound, so its lower rest bound may
    # lie below the analytic ideal even for a safe path. Protected overcut is
    # a separate continuous-motion check and cannot be inferred from this area.
    rough_ok = all(row["rest_area_mm2"][1] <= rough_ideal + 0.5 and
                   row["residual_outside_ideal_or_boundary_0_05mm_mm2"] <= 1e-7
                   for row in rough)
    final_ok = all(row["rest_area_mm2"][1] <= final_ideal + 0.5 and
                   row["residual_outside_ideal_or_boundary_0_05mm_mm2"] <= 1e-7 and
                   a["rest_area_mm2"][0] - row["rest_area_mm2"][1] >=
                   (4 - math.pi) * 8 - 0.5
                   for a, row in zip(rough, final))
    return rough_ok, final_ok


def audit_native_posts(manifest_path, rough_post, combined_post):
    """Hash-guard two native candidates and replay their actual Default posts."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "rc01-native-v1":
        raise ValueError("not an RC01 native comparison manifest")
    source_path = manifest_path.parent / manifest["source_file"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("RC01 source changed after manifest generation")
    setup = json.loads((manifest_path.parent / "setup.json").read_text(encoding="utf-8"))
    parsed = {}
    for key, post_path in (("rough", rough_post), ("combined", combined_post)):
        entry = manifest["variants"][key]
        candidate = manifest_path.parent / entry["file"]
        data = candidate.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"{key} candidate changed after manifest generation")
        project = read_cambam_bytes(data, source_name=str(candidate))
        job = normalize(project, setup, allow_attachments=True)
        if job.fingerprint != manifest["job_fingerprint"]:
            raise ValueError("RC01 job fingerprint changed")
        post = Path(post_path).read_text(encoding="utf-8-sig")
        if "( Post processor: Default )" not in post:
            raise ValueError(f"{key} post is not labeled Default")
        if not any(line.startswith(f"( {candidate.stem} ")
                   for line in post.splitlines()[:6]):
            raise ValueError(f"{key} post header does not name its candidate")
        items, warnings = read_default_post(post, allow_arcs=True)
        parsed[key] = (items, warnings)
    rough_items, rough_warnings = parsed["rough"]
    combined_items, combined_warnings = parsed["combined"]
    rough_moves = [i for i in rough_items if i["type"] == "move" and
                   i["tool"] == "T1"]
    combined_rough = [i for i in combined_items if i["type"] == "move" and
                      i["tool"] == "T1"]
    def signature(moves):
        return [(i["g"], i["start"], i["end"], i["feed"], i.get("center"))
                for i in moves]
    same_prefix = signature(rough_moves) == signature(combined_rough)
    rough_rest = _area_by_depth(rough_moves, job, 3)
    final_rest = _area_by_depth([i for i in combined_items if i["type"] == "move"],
                                job, 1)
    t2_columns = _t2_vertical_access(rough_moves, combined_items)
    required_columns = _t2_vertical_access(rough_moves, [
        {"type": "move", "tool": "T2", "g": 0,
         "start": [x, y, 5], "end": [x, y, -3]}
        for x, y in ((5, 5), (35, 5), (35, 25), (5, 25))])
    rough_budget, final_budget = _budget(rough_rest, final_rest)
    issues = {
        "rough": _motion_findings(rough_items, rough_warnings, job),
        "combined": _motion_findings(combined_items, combined_warnings, job),
    }
    if not same_prefix:
        issues["combined"].append("T1 posted prefix differs from rough-only post")
    if not rough_budget:
        issues["rough"].append("T1 rest area or location exceeds RC01 budget")
    if not final_budget:
        issues["combined"].append("final rest or cleanup benefit exceeds RC01 budget")
    if not t2_columns or any(row["exact_single_t1_cut_witness_line"] is None
                             for row in t2_columns):
        issues["combined"].append("T2 vertical access lacks T1 full-depth clearance")
    if any(row["exact_single_t1_cut_witness_line"] is None
           for row in required_columns):
        issues["rough"].append("required RC01 corner column lacks T1 full-depth clearance")
    counts = {key: len(value) for key, value in issues.items()}
    issue_kinds = {key: dict(Counter(
        message.split(": ", 1)[-1] for message in value))
        for key, value in issues.items()}
    motion_summary = {}
    for key, items in (("rough", rough_items), ("combined", combined_items)):
        moves = [item for item in items if item["type"] == "move"]
        motion_summary[key] = {
            "moves": len(moves),
            "arcs": sum(item["g"] in (2, 3) for item in moves),
            "ramped_xy": sum(item["start"][:2] != item["end"][:2] and
                              item["start"][2] != item["end"][2]
                              for item in moves),
            "minimum_tip_z": min(min(item["start"][2], item["end"][2])
                                 for item in moves),
            "tool_changes": [{"tool": item["tool"], "position": item["position"]}
                             for item in items if item["type"] == "event" and
                             item["kind"] == "tool_change"],
        }
    return {
        "format": "rc01-native-post-audit-v1",
        "candidate_sha256": {key: manifest["variants"][key]["sha256"]
                             for key in ("rough", "combined")},
        "post_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for key, path in (("rough", rough_post),
                                          ("combined", combined_post))},
        "rough_prefix_identical": same_prefix,
        "rough_rest_by_depth": rough_rest,
        "final_rest_by_depth": final_rest,
        "rough_rest_budget_met": rough_budget,
        "final_rest_budget_met": final_budget,
        "t2_vertical_columns_from_t1": t2_columns,
        "required_corner_columns_from_t1": required_columns,
        "motion_summary": motion_summary,
        "issue_counts": counts,
        "issue_kinds": issue_kinds,
        "issues": {key: value[:24] for key, value in issues.items()},
        "status": "fails_RC01" if any(counts.values()) else
                  "bounded_checks_passed_access_unverified",
        "numeric_limit": "GEOS floating topology is not an interval proof; "
                         "radial polygon enclosure excludes that error",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replay RC01 native Default posts")
    parser.add_argument("manifest")
    parser.add_argument("rough_post")
    parser.add_argument("combined_post")
    args = parser.parse_args()
    print(json.dumps(audit_native_posts(args.manifest, args.rough_post,
                                        args.combined_post), indent=2))
