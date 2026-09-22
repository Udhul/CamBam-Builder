"""Development-only planar backend probe; install Shapely in an isolated environment.

Run from the repository root: python tools/evaluate_shapely.py --output <report.json>.
Consumes the backend-neutral corpus; does not import or modify the runtime package.
Curved references are direct trigonometric constructions, not GEOS buffers. Reported
Hausdorff observations use GEOS's discrete vertex metric. Separately, both boundary
directions are sampled at bounded spacing, adding half that spacing using the
1-Lipschitz distance bound and adding the analytic reference chord sagitta. This
conservative bound is charged against the boundary budget. This is synthetic evaluation,
not a conservative swept-stock guarantee or machining/path acceptance.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


QUAD_SEGS = 256
REFERENCE_QUAD_SEGS = 512
BOUNDARY_SAMPLE_SPACING = 0.001


def arc(cx, cy, radius, start, stop):
    count = max(1, math.ceil(abs(stop - start) * REFERENCE_QUAD_SEGS * 2 / math.pi))
    return [(cx + radius * math.cos(start + (stop - start) * i / count),
             cy + radius * math.sin(start + (stop - start) * i / count))
            for i in range(count + 1)]


def circle(radius, center=(0, 0)):
    return Polygon(arc(*center, radius, 0, 2 * math.pi))


def rounded_rectangle(bounds, radius):
    """Exact tangent locations plus sampled circular arcs, constructed without buffer."""
    x0, y0, x1, y1 = bounds
    points = []
    for cx, cy, start in ((x1, y1, 0), (x0, y1, math.pi / 2),
                          (x0, y0, math.pi), (x1, y0, 3 * math.pi / 2)):
        points.extend(arc(cx, cy, radius, start, start + math.pi / 2))
    return Polygon(points)


def topology(geometry):
    if geometry.is_empty:
        return {"components": 0, "holes": 0}
    if geometry.geom_type == "Polygon":
        return {"components": 1, "holes": len(geometry.interiors)}
    if geometry.geom_type == "MultiPolygon":
        return {"components": len(geometry.geoms),
                "holes": sum(len(part.interiors) for part in geometry.geoms)}
    return {"components": None, "holes": None}


def directed_boundary_bound(source, target):
    """Distance to a closed set is 1-Lipschitz; every segment point is <= step/2
    from a sampled endpoint. Hence sampled maximum + step/2 bounds the supremum.
    The numeric floating-point geometry calculations themselves are not intervals.
    """
    sampled = shapely.segmentize(source, BOUNDARY_SAMPLE_SPACING)
    points = shapely.points(shapely.get_coordinates(sampled))
    observed = float(shapely.distance(points, target).max())
    return observed, observed + BOUNDARY_SAMPLE_SPACING / 2


class Evaluation:
    def __init__(self, limits):
        self.limits = limits
        self.checks = []

    def scalar(self, name, actual, expected, limit):
        error = actual - expected
        passed = math.isfinite(actual) and abs(error) <= limit
        self.checks.append({"name": name, "actual": actual, "expected": expected,
                            "signed_error": error, "limit": limit, "passed": passed})

    def condition(self, name, actual, expected=True):
        self.checks.append({"name": name, "actual": actual, "expected": expected,
                            "passed": actual == expected})

    def region(self, name, actual, area=None, reference=None, components=1, holes=0,
               reference_radius=0):
        self.condition(name + ".valid", actual.is_valid)
        self.checks.append({"name": name + ".finite_area_mm2", "actual": actual.area,
                            "expected": "finite", "passed": math.isfinite(actual.area)})
        self.condition(name + ".topology", topology(actual),
                       {"components": components, "holes": holes})
        if area is not None:
            self.scalar(name + ".area_mm2", actual.area, area,
                        self.limits["area_error_mm2"])
        if reference is not None:
            self.scalar(name + ".symmetric_difference_mm2",
                        actual.symmetric_difference(reference).area, 0,
                        self.limits["area_error_mm2"])
            # Hausdorff distance is symmetric: both boundary directions participate.
            distance = actual.boundary.hausdorff_distance(reference.boundary)
            sagitta = reference_radius * (1 - math.cos(math.pi / (4 * REFERENCE_QUAD_SEGS)))
            forward, forward_bound = directed_boundary_bound(actual.boundary, reference.boundary)
            reverse, reverse_bound = directed_boundary_bound(reference.boundary, actual.boundary)
            self.scalar(name + ".bidirectional_boundary_bound_mm",
                        max(forward_bound, reverse_bound) + sagitta, 0,
                        self.limits["boundary_distance_mm"])
            self.checks[-1].update(polygon_boundary_hausdorff_mm=distance,
                                   actual_to_reference_sample_max_mm=forward,
                                   reference_to_actual_sample_max_mm=reverse,
                                   sample_spacing_mm=BOUNDARY_SAMPLE_SPACING,
                                   reference_chord_sagitta_mm=sagitta)


def evaluate_case(case, limits):
    key = case["id"].split("_")[0]
    data, expected = case["input"], case["expected"]
    check = Evaluation(limits)
    notes = []
    evidence = "planar_geometry"
    status = "pass"
    if key in ("C01", "C02"):
        return {"id": case["id"], "status": "out_of_planar_scope",
                "notes": ["Cutter-profile inversion and joins belong to the tool-profile layer."],
                "checks": []}
    if key == "G01":
        a, b = box(*data["a"]), box(*data["b"])
        for name, result, reference in (
                ("union", a.union(b), box(0, 0, 15, 10)),
                ("intersection", a.intersection(b), box(5, 0, 10, 10)),
                ("difference", a.difference(b), box(0, 0, 5, 10))):
            check.region(name, result, expected[name + "_area"], reference)
    elif key == "R01":
        side = data["side"]
        opening = box(0, 0, side, side)
        rests = []
        for radius, area in zip(data["tool_radii"], expected["rest_areas"]):
            centers = opening.buffer(-radius)
            reachable = centers.buffer(radius, quad_segs=QUAD_SEGS)
            rest = opening.difference(reachable)
            reference = opening.difference(rounded_rectangle(
                (radius, radius, side - radius, side - radius), radius))
            check.region("rest_r" + str(radius), rest, area, reference,
                         expected["rest_components"], reference_radius=radius)
            rests.append(rest.area)
        check.scalar("second_tool_gain_mm2", rests[0] - rests[1],
                     expected["second_tool_gain"], limits["area_error_mm2"])
        evidence = "ideal_reachability_not_generated_path_coverage"
    elif key == "R02":
        side, radius, allowance = data["side"], data["radius"], data["allowance"]
        opening = box(0, 0, side, side)
        coordinates = data["profile_centers"]
        path = LineString(coordinates + [coordinates[0]])
        sweep = path.buffer(radius, quad_segs=QUAD_SEGS)
        low, high = radius + allowance, side - radius - allowance
        rounded = rounded_rectangle((low, low, high, high), radius)
        center = box(low + radius, low + radius, high - radius, high - radius)
        reference = Polygon(rounded.exterior.coords, [center.exterior.coords])
        pocket = opening.buffer(-low).buffer(radius, quad_segs=QUAD_SEGS)
        check.region("profile_sweep", sweep, expected["profile_sweep_area"],
                     reference, holes=1, reference_radius=radius)
        check.region("profile_rest", opening.difference(sweep), expected["profile_rest_area"],
                     opening.difference(reference), components=2, holes=1,
                     reference_radius=radius)
        check.region("ideal_pocket", pocket, expected["ideal_pocket_sweep_area"],
                     rounded, reference_radius=radius)
        check.region("ideal_pocket_rest", opening.difference(pocket),
                     expected["ideal_pocket_rest_area"], opening.difference(rounded),
                     holes=1, reference_radius=radius)
        check.scalar("unswept_center_side_mm", math.sqrt(Polygon(sweep.interiors[0]).area),
                     expected["profile_unswept_center_side"], limits["boundary_distance_mm"])
        check.scalar("intentional_allowance_area_mm2", opening.area -
                     opening.buffer(-allowance).area, expected["intentional_allowance_area"],
                     limits["area_error_mm2"])
        evidence = "explicit_profile_trajectory_and_separate_ideal_pocket"
    elif key == "G02":
        outer = Point(0, 0).buffer(data["outer_radius"], quad_segs=QUAD_SEGS)
        inner = Point(0, 0).buffer(data["inner_radius"], quad_segs=QUAD_SEGS)
        result = outer.difference(inner).buffer(-data["inset"], quad_segs=QUAD_SEGS)
        reference = Polygon(circle(expected["outer_radius"]).exterior.coords,
                            [circle(expected["inner_radius"]).exterior.coords])
        check.region("inset", result, expected["area"], reference, holes=1,
                     reference_radius=expected["outer_radius"])
    elif key in ("V01", "V02"):
        tangent = math.tan(math.radians(data["included_angle_degrees"] / 2))
        opening = (box(0, 0, data["side"], data["side"]) if key == "V01" else
                   Point(0, 0).buffer(data["radius"], quad_segs=QUAD_SEGS))
        areas = []
        for index, depth in enumerate(data["section_depths"]):
            inset = depth * tangent
            section = opening.buffer(-inset)
            if key == "V01":
                reference = box(inset, inset, data["side"] - inset, data["side"] - inset)
                area = expected["section_areas"][index]
                radius = 0
            else:
                radius = expected["section_radii"][index]
                reference = circle(radius)
                area = math.pi * radius ** 2
            check.region("section_depth_" + str(depth), section, area, reference,
                         reference_radius=radius)
            areas.append(section.area)
        depths = data["section_depths"]
        check.condition("simpson_nodes", depths, [0, data["cap_depth"] / 2, data["cap_depth"]])
        volume = data["cap_depth"] * (areas[0] + 4 * areas[1] + areas[2]) / 6
        check.scalar("simpson_volume_mm3", volume, expected["volume"], limits["volume_error_mm3"])
        check.scalar("flat_floor_area_mm2", areas[-1], expected["flat_floor_area"],
                     limits["area_error_mm2"])
        notes.append("Simpson integration is exact for these analytic quadratic section areas; polygon approximation remains measured.")
        evidence = "desired_target_sections_not_cut_occupancy"
    elif key in ("S01", "S02"):
        start, end = data["start"], data["end"]
        if start[1] != end[1] or end[0] <= start[0]:
            raise ValueError("Sweep references require the corpus's horizontal positive-X segment")
        if key == "S01":
            radius = data["radius"]
            result = LineString([start, end]).buffer(radius, quad_segs=QUAD_SEGS)
            reference = Polygon(arc(*end, radius, -math.pi / 2, math.pi / 2) +
                                arc(*start, radius, math.pi / 2, 3 * math.pi / 2))
        else:
            r0, radius = data["start_radius"], data["end_radius"]
            if not 0 <= radius - r0 < end[0] - start[0]:
                raise ValueError("Affine reference requires increasing, non-containing endpoint disks")
            result = unary_union([Point(start).buffer(r0, quad_segs=QUAD_SEGS),
                                  Point(end).buffer(radius, quad_segs=QUAD_SEGS)]).convex_hull
            # Horizontal fixture: common tangent outward normal has cos(theta)=-dr/d.
            theta = math.acos(-(radius - r0) / (end[0] - start[0]))
            reference = Polygon(arc(*end, radius, -theta, theta) +
                                arc(*start, r0, theta, 2 * math.pi - theta))
            notes.append("Independent boundary uses analytic common tangent endpoints and exposed circle arcs; no candidate convex hull in reference.")
        check.region("sweep", result, expected["area"], reference, reference_radius=radius)
        for i, (actual, wanted) in enumerate(zip(result.bounds, expected["bounds"])):
            check.scalar("bound_" + str(i), actual, wanted, limits["boundary_distance_mm"])
        midpoint = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        check.condition("contains_midpoint", result.contains(midpoint), expected["contains_midpoint"])
        evidence = "continuous_XY_sweep_at_one_depth_not_XYZ_acceptance"
    elif key == "T01":
        results = [box(*data["rectangle"]).buffer(-data["tool_radius"]),
                   Point(data["circle_center"]).buffer(data["circle_radius"],
                                                       quad_segs=QUAD_SEGS).buffer(-data["tool_radius"])]
        notes.append("Raw polygon erosion cannot represent the nominal exact-fit segment and point: it may be empty or a tiny numerical polygon. Requires explicit lower-dimensional feasible-set adapter; this is not a passed acceptance case.")
        for name, result in zip(("rectangle", "circle"), results):
            check.condition(name + ".raw_geometry_valid_and_finite",
                            result.is_valid and math.isfinite(result.area))
            check.checks[-1].update(geometry_type=result.geom_type, area_mm2=result.area,
                                   empty=result.is_empty,
                                   bounds=None if result.is_empty else list(result.bounds),
                                   centroid=None if result.is_empty else list(result.centroid.coords)[0],
                                   expected_nominal_dimension=1 if name == "rectangle" else 0,
                                   nominal_dimension_match=False)
        references = [LineString(expected["rectangle_center_segment"]), Point(expected["circle_center_point"])]
        notes.append("Expected dimensions/locations: " + "; ".join(g.wkt for g in references))
        status = "expected_limitation"
    elif key == "T02":
        side = data["side"]
        for gap, area, components in zip(data["horizontal_gaps"], expected["union_areas"],
                                         expected["interior_components"]):
            left = box(0, 0, side, side)
            right = box(side + gap, 0, 2 * side + gap, side)
            reference = box(0, 0, 2 * side + gap, side) if gap <= 0 else shapely.geometry.MultiPolygon([left, right])
            check.region("gap_" + str(gap), left.union(right), area, reference, components)
    elif key == "T03":
        opening = unary_union([box(*data[name]) for name in ("left_box", "right_box", "bridge")])
        check.region("opening", opening, expected["initial_area"], components=expected["initial_components"])
        inset = opening.buffer(-data["inset"], quad_segs=QUAD_SEGS)
        check.region("inset", inset, components=expected["inset_components"], holes=expected["inset_holes"])
        notes.append("Corpus specifies input area and inset topology only; no independent inset boundary/area oracle is claimed.")
    elif key == "A01":
        opening = Polygon(data["outer"], data["holes"])
        reference = Polygon(list(reversed(data["outer"])),
                            [list(reversed(ring)) for ring in data["holes"]])
        check.region("opening", opening, expected["opening_area"], reference,
                     expected["components"], expected["holes"])
        notes.append(expected["path_acceptance"])
        evidence = "input_composition_only_not_combined_tool_path_acceptance"
    else:
        raise ValueError("Unimplemented corpus case: " + key)
    if not all(item["passed"] for item in check.checks):
        status = "fail"
    return {"id": case["id"], "status": status, "evidence_class": evidence,
            "checks": check.checks, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/rest_vcarve_acceptance.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if corpus["version"] != 1:
        raise ValueError("Only acceptance corpus version 1 is supported")
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "shapely": shapely.__version__, "geos": shapely.geos_version_string,
              "corpus": str(args.corpus), "corpus_version": corpus["version"],
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": corpus["backend_limits"], "quad_segs": QUAD_SEGS,
              "reference_quad_segs": REFERENCE_QUAD_SEGS,
              "boundary_metric": "both directed sampled maxima + half sample spacing (1-Lipschitz bound) + reference chord sagitta; floating-point calculations, not interval arithmetic",
              "cases": [evaluate_case(case, corpus["backend_limits"]) for case in corpus["cases"]]}
    report["unexpected_failures"] = sum(case["status"] == "fail" for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "unexpected_failures": report["unexpected_failures"],
                      "statuses": {case["id"]: case["status"] for case in report["cases"]}}, indent=2))
    return int(report["unexpected_failures"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
