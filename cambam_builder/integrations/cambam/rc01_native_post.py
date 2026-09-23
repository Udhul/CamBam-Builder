"""Independent, bounded stock replay for an RC01 native Pocket Default post.

The posted straight moves, rather than native MOP settings or planned RC01 paths,
are the removal evidence. Polygon results retain an explicit numerical caveat.
"""

import hashlib
import json
import math
from pathlib import Path

from .rc01_adapter import normalize
from .rc01_post import read_default_post
from ...cambam_reader import read_cambam_bytes


LEVELS = (-1.0, -2.0, -3.0)
SETUP = (-10.0, -10.0, 5.0)
QUAD_SEGS = 256


def _at_or_below(start, end, level):
    """Conservative full-depth XY part of a descending or level straight cut."""
    z0, z1 = start[2], end[2]
    if z0 <= level and z1 <= level:
        return (start[:2], end[:2])
    if z1 > level or z1 > z0:
        return None
    ratio = (level - z0) / (z1 - z0)
    point = tuple(start[i] + ratio * (end[i] - start[i]) for i in (0, 1))
    return (point, end[:2])


def _target(job):
    from shapely.geometry import box
    outer = job.outer
    island = job.island
    return box(float(outer.xmin), float(outer.ymin), float(outer.xmax),
               float(outer.ymax)).difference(box(
                   float(island.xmin), float(island.ymin),
                   float(island.xmax), float(island.ymax)))


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
    inflation = 1 / math.cos(math.pi / (4 * QUAD_SEGS))
    for level in LEVELS:
        unique = {}
        for item in moves:
            if item["g"] != 1 or item["end"][2] > item["start"][2]:
                continue
            segment = _at_or_below(item["start"], item["end"], level)
            if segment is not None:
                unique[(tuple(segment[0]), tuple(segment[1]), item["tool"])] = (
                    segment, 3.0 if item["tool"] == "T1" else 1.0)
        inner = unary_union([LineString(segment).buffer(radius, quad_segs=QUAD_SEGS)
                             for segment, radius in unique.values()])
        outer_bound = unary_union([
            LineString(segment).buffer(radius * inflation, quad_segs=QUAD_SEGS)
            for segment, radius in unique.values()])
        rest_upper = target.difference(inner)
        result.append({
            "bottom_z": level,
            "rest_area_mm2": [max(0.0, target.difference(outer_bound).area),
                              rest_upper.area],
            "residual_outside_ideal_or_boundary_0_05mm_mm2":
                rest_upper.difference(allowed_rest).area,
            "cut_segments": len(unique),
        })
    return result


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
            if (not all(radius <= p[0] <= 40 - radius and
                        radius <= p[1] <= 30 - radius for p in (a, b))):
                issues.append(f"line {line}: cutter crosses outer protected wall")
            island = job.island
            island_box = box(float(island.xmin), float(island.ymin),
                             float(island.xmax), float(island.ymax))
            distance = LineString((a[:2], b[:2])).distance(island_box)
            if distance < radius - 1e-9:
                issues.append(f"line {line}: cutter crosses protected island")
            elif abs(distance - radius) <= 1e-9:
                # Exact tangent is allowed; GEOS cannot certify the sign at
                # this boundary from floating coordinates alone.
                issues.append(f"line {line}: island tangency numerically unresolved")
        if move["g"] == 1:
            xy = a[:2] != b[:2]
            if xy and a[2] == b[2] and a[2] < 0 and move["feed"] != 300:
                issues.append(f"line {line}: cutting feed is not 300 mm/min")
            if not xy and b[2] < a[2] and b[2] < 0 and move["feed"] != 60:
                issues.append(f"line {line}: plunge feed is not 60 mm/min")
            if xy and a[2] != b[2] and min(a[2], b[2]) < 0:
                issues.append(f"line {line}: ramped XY engagement lacks RC01 process proof")
    return issues


def _budget(rough, final):
    rough_ideal = (4 - math.pi) * 9
    final_ideal = 4 - math.pi
    rough_ok = all(rough_ideal - 1e-6 <= row["rest_area_mm2"][0] and
                   row["rest_area_mm2"][1] <= rough_ideal + 0.5 and
                   row["residual_outside_ideal_or_boundary_0_05mm_mm2"] <= 1e-7
                   for row in rough)
    final_ok = all(final_ideal - 1e-6 <= row["rest_area_mm2"][0] and
                   row["rest_area_mm2"][1] <= final_ideal + 0.5 and
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
        items, warnings = read_default_post(post)
        parsed[key] = (items, warnings)
    rough_items, rough_warnings = parsed["rough"]
    combined_items, combined_warnings = parsed["combined"]
    rough_moves = [i for i in rough_items if i["type"] == "move" and
                   i["tool"] == "T1"]
    combined_rough = [i for i in combined_items if i["type"] == "move" and
                      i["tool"] == "T1"]
    def signature(moves):
        return [(i["g"], i["start"], i["end"], i["feed"]) for i in moves]
    same_prefix = signature(rough_moves) == signature(combined_rough)
    rough_rest = _area_by_depth(rough_moves, job, 3)
    final_rest = _area_by_depth([i for i in combined_items if i["type"] == "move"],
                                job, 1)
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
    counts = {key: len(value) for key, value in issues.items()}
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
        "issue_counts": counts,
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
