"""Development-only adversarial planar acceptance; no runtime imports or dependency.

Run in the isolated Shapely environments documented in DEVELOPMENT.md. All units
are synthetic mm. The finite/valid input and exact primitive helpers below are
executable contract prototypes, not a public geometry implementation. No repair,
precision grid, arbitrary contour offset oracle or conservative stock claim.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box

from evaluate_shapely import Evaluation, topology


LIMITS = {"boundary_distance_mm": 0.001, "area_error_mm2": 0.01}
CURVE_ERROR = 0.000004


def directed_boundary_bound(source, target):
    """Same Lipschitz bound as the initial probe, indexed for dense contours."""
    sampled = shapely.points(shapely.get_coordinates(shapely.segmentize(source, 0.001)))
    segments = []
    for line in shapely.get_parts(target):
        coordinates = shapely.get_coordinates(line)
        segments.extend(shapely.linestrings(np.stack([coordinates[:-1], coordinates[1:]], axis=1)))
    tree = shapely.STRtree(segments)
    _, distances = tree.query_nearest(sampled, return_distance=True, all_matches=False)
    observed = float(distances.max())
    return observed, observed + 0.0005


def strict_polygon(shell, holes=(), factory=Polygon):
    """Reject malformed/nonfinite/underresolved coordinates before calling GEOS.

    Ring topology and ownership are checked before constructing the filled region.
    Four coordinate ULPs reserve a measurable input-resolution allowance only;
    this is not a proof bounding every GEOS intermediate computation.
    """
    rings = [shell] + list(holes)
    for ring in rings:
        if any(len(p) != 2 for p in ring):
            raise ValueError("xy_required")
        if any(not math.isfinite(v) for p in ring for v in p):
            raise ValueError("nonfinite_coordinate")
        if any(4 * math.ulp(float(v)) > LIMITS["boundary_distance_mm"] / 10
               for p in ring for v in p):
            raise ValueError("coordinate_resolution")
        if len(set(map(tuple, ring))) < 3:
            raise ValueError("degenerate_ring")
        if any(tuple(a) == tuple(b) for a, b in zip(ring, ring[1:])):
            raise ValueError("zero_length_edge")
    polygons = []
    for ring in rings:
        polygon = factory(ring)
        if not polygon.exterior.is_simple:
            raise ValueError("self_intersection")
        if polygon.area == 0:
            raise ValueError("degenerate_ring")
        if not polygon.is_valid:
            raise ValueError("invalid_ring")
        polygons.append(polygon)
    outer = polygons[0]
    for hole in polygons[1:]:
        if not outer.contains(hole) or outer.boundary.intersects(hole.boundary):
            raise ValueError("hole_not_strictly_inside")
    for i, hole in enumerate(polygons[1:]):
        if any(hole.intersects(other) for other in polygons[1:i + 1]):
            raise ValueError("holes_not_disjoint")
    result = factory(shell, holes)
    if not result.is_valid:
        raise ValueError("invalid_region")
    return result


def primitive_centers(kind, dimensions, radius):
    """Exact predicates on supplied binary64 values, local primitive coordinates.

    Rational coordinates retain sub-ULP dimensional distinctions in this probe.
    General contour collapse is explicitly unsupported; no polygon buffer fallback.
    """
    if kind not in ("rectangle", "circle"):
        raise ValueError("unsupported_general_collapse")
    if (len(dimensions) != (2 if kind == "rectangle" else 1) or
            not all(math.isfinite(v) and v > 0 for v in dimensions) or
            not math.isfinite(radius) or radius < 0):
        raise ValueError("invalid_primitive")
    dims = [Fraction(v) for v in dimensions]
    r = Fraction(radius)
    remaining = [v - (2 * r if kind == "rectangle" else r) for v in dims]
    dimension = (-1 if min(remaining) < 0 else
                 sum(v > 0 for v in remaining) if kind == "rectangle" else
                 2 if remaining[0] > 0 else 0)
    return {"dimension": dimension,
            "remaining_exact": [str(v) for v in remaining],
            "center_exact": [str(v / 2) for v in dims] if kind == "rectangle" else ["0", "0"],
            "rectangle_intervals_exact": [[str(r), str(v - r)] for v in dims]
            if kind == "rectangle" and dimension >= 0 else None}


def invalid_inputs(check):
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(2, 2), (4, 2), (4, 4), (2, 4)]
    fixtures = [
        ("nan", [(0, 0), (1, 0), (0, float("nan"))], [], "nonfinite_coordinate"),
        ("infinity", [(0, 0), (1, 0), (0, float("inf"))], [], "nonfinite_coordinate"),
        ("negative_infinity", [(0, 0), (1, 0), (0, -float("inf"))], [], "nonfinite_coordinate"),
        ("z", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], "xy_required"),
        ("self_crossing", [(0, 0), (4, 4), (0, 4), (4, 0)], [], "self_intersection"),
        ("too_few", [(0, 0), (1, 0), (0, 0)], [], "degenerate_ring"),
        ("duplicate_vertex", [(0, 0), (1, 0), (1, 0), (0, 1)], [], "zero_length_edge"),
        ("collinear", [(0, 0), (1, 0), (2, 0)], [], "self_intersection"),
        ("outside_hole", shell, [[(11, 1), (12, 1), (12, 2), (11, 2)]], "hole_not_strictly_inside"),
        ("point_touch_shell", shell, [[(0, 5), (2, 4), (2, 6)]], "hole_not_strictly_inside"),
        ("edge_touch_shell", shell, [[(0, 2), (2, 2), (2, 4), (0, 4)]], "hole_not_strictly_inside"),
        ("overlap_holes", shell, [inner, [(3, 3), (5, 3), (5, 5), (3, 5)]], "holes_not_disjoint"),
        ("nested_holes", shell, [inner, [(2.5, 2.5), (3, 2.5), (3, 3)]], "holes_not_disjoint"),
        ("point_touch_holes", shell, [inner, [(4, 4), (5, 4), (5, 5)]], "holes_not_disjoint"),
        ("underresolved_world", [(1e15 + x, 1e15 + y) for x, y in shell], [], "coordinate_resolution"),
    ]
    for name, outer, holes, expected in fixtures:
        calls = []
        def tracked(*args):
            calls.append(1)
            return Polygon(*args)
        try:
            strict_polygon(outer, holes, tracked)
            reason = "accepted"
        except ValueError as exc:
            reason = str(exc)
        check.condition("invalid." + name, reason, expected)
        if expected in ("nonfinite_coordinate", "xy_required", "coordinate_resolution"):
            check.condition("invalid." + name + ".before_geos", len(calls), 0)
    check.condition("valid.strict_hole", strict_polygon(shell, [inner]).area, 96.0)
    check.condition("valid.optional_ring_closure", strict_polygon(shell + [shell[0]]).area, 100.)


def radial_ring(radius, amplitude=0, waves=0, epsilon=CURVE_ERROR, center=(0, 0)):
    # ||p''|| <= |r''| + 2|r'| + |r|. Linear interpolation error <= M*h^2/8.
    second_bound = radius + abs(amplitude) * (1 + 2 * waves + waves ** 2)
    count = math.ceil(2 * math.pi * math.sqrt(second_bound / (8 * epsilon)))
    points = []
    for i in range(count):
        theta = 2 * math.pi * i / count
        r = radius + amplitude * math.sin(waves * theta)
        points.append((center[0] + r * math.cos(theta), center[1] + r * math.sin(theta)))
    error = second_bound * (2 * math.pi / count) ** 2 / 8
    length_bound = 2 * math.pi * math.hypot(radius + abs(amplitude), abs(amplitude * waves))
    return points, error, length_bound


def curve_comparison(check, name, actual, reference, reference_error, analytic_area,
                     approximation_area_bound, components=1, holes=0):
    check.condition(name + ".valid", actual.is_valid)
    check.condition(name + ".topology", topology(actual), {"components": components, "holes": holes})
    check.scalar(name + ".analytic_area_mm2", actual.area, analytic_area, LIMITS["area_error_mm2"])
    check.scalar(name + ".derived_area_budget_mm2", approximation_area_bound, 0, LIMITS["area_error_mm2"])
    forward, fbound = directed_boundary_bound(actual.boundary, reference.boundary)
    reverse, rbound = directed_boundary_bound(reference.boundary, actual.boundary)
    check.scalar(name + ".boundary_bound_mm", max(fbound, rbound) + reference_error, 0,
                 LIMITS["boundary_distance_mm"])
    check.checks[-1].update(reference_interpolation_bound_mm=reference_error,
                           actual_to_reference_sample_max_mm=forward,
                           reference_to_actual_sample_max_mm=reverse)
    check.scalar(name + ".symmetric_difference_mm2", actual.symmetric_difference(reference).area,
                 0, LIMITS["area_error_mm2"])


def curves(check):
    points, error, length = radial_ring(20, 2, 17)
    reference, ref_error, _ = radial_ring(20, 2, 17, CURVE_ERROR / 4)
    # Integral (r^2)/2 dtheta = pi*(R^2 + amplitude^2/2).
    curve_comparison(check, "organic", Polygon(points), Polygon(reference), ref_error,
                     math.pi * 402, 2 * length * error + math.pi * error ** 2)
    check.condition("organic.vertex_count", len(points) > 10000)
    # Six circular holes; exact inset increases each hole radius and decreases shell radius.
    centers = [(10 * math.cos(i * math.pi / 3), 10 * math.sin(i * math.pi / 3)) for i in range(6)]
    shell, _, _ = radial_ring(20)
    holes = [radial_ring(2, center=c)[0] for c in centers]
    original = strict_polygon(shell, holes)
    for inset in (0, 0.5):
        actual = original if inset == 0 else original.buffer(-inset, quad_segs=1024)
        outer, outer_error, outer_length = radial_ring(20 - inset, epsilon=CURVE_ERROR / 4)
        inner = [radial_ring(2 + inset, epsilon=CURVE_ERROR / 4, center=c) for c in centers]
        ref = Polygon(outer, [v[0] for v in inner])
        ref_error = max([outer_error] + [v[1] for v in inner])
        perimeter = outer_length + sum(v[2] for v in inner)
        # Convex regular shell offset amplifies its sagitta by at most 1/cos(pi/n).
        # Hole expansion adds its arc chord sagitta. Bound each by <= 2*CURVE_ERROR.
        curve_comparison(check, "arc_holes.inset_" + str(inset), actual, ref, ref_error,
                         math.pi * ((20 - inset) ** 2 - 6 * (2 + inset) ** 2),
                         2 * perimeter * (2 * CURVE_ERROR) + 7 * math.pi * (2 * CURVE_ERROR) ** 2,
                         holes=6)


def invariants(check):
    shell = [(0, 0), (30, 0), (30, 20), (19, 20), (19, 15), (11, 15), (11, 20), (0, 20)]
    holes = [[(3, 3), (7, 3), (7, 7), (3, 7)], [(23, 3), (27, 3), (27, 7), (23, 7)]]
    a = strict_polygon(shell, holes)
    b = box(15, -2, 34, 12)
    operations = {"union": lambda x, y, r: x.union(y),
                  "intersection": lambda x, y, r: x.intersection(y),
                  "difference": lambda x, y, r: x.difference(y),
                  "inset": lambda x, y, r: x.buffer(-r, quad_segs=256)}
    angle = math.radians(37)
    transforms = [("rotation", 1, angle, 1, 0, 0), ("reflection", 1, 0, -1, 0, 0),
                  ("scale", 3.7, angle, 1, 0, 0), ("world_1e9", 1, 0, 1, 1e9, -1e9),
                  ("mm_to_inch", 1 / 25.4, 0, 1, 0, 0)]
    for name, operation in operations.items():
        baseline = operation(a, b, 0.5)
        reordered = strict_polygon(list(reversed(shell[2:] + shell[:2])),
                                   [list(reversed(h)) for h in reversed(holes)])
        candidate = operation(reordered, b, 0.5)
        check.region(name + ".ring_order", candidate, baseline.area, baseline,
                     **topology(baseline))
        if name in ("union", "intersection"):
            check.region(name + ".operand_order", operation(b, a, 0.5), baseline.area,
                         baseline, **topology(baseline))
        for label, scale, theta, reflection, tx, ty in transforms:
            c, s = math.cos(theta), math.sin(theta)
            aa, bb, dd, ee = scale * c * reflection, -scale * s, scale * s * reflection, scale * c
            matrix = [aa, bb, dd, ee, tx, ty]
            determinant = aa * ee - bb * dd
            inverse = [ee / determinant, -bb / determinant, -dd / determinant, aa / determinant,
                       (bb * ty - ee * tx) / determinant, (dd * tx - aa * ty) / determinant]
            result = operation(affine_transform(a, matrix), affine_transform(b, matrix), scale * 0.5)
            restored = affine_transform(result, inverse)
            check.region(name + "." + label, restored, baseline.area, baseline,
                         **topology(baseline))
    check.condition("valid.world_1e9", strict_polygon([(x + 1e9, y - 1e9) for x, y in shell]).is_valid)


def exact_fit(check):
    cases = [("rectangle_segment", "rectangle", [10., 20.], 5., 1),
             ("rectangle_point", "rectangle", [10., 10.], 5., 0),
             ("circle_point", "circle", [5.], 5., 0)]
    for label, kind, dims, radius, dimension in cases:
        for suffix, r, expected in (("exact", radius, dimension),
                                    ("below_1ulp", math.nextafter(radius, 0), 2),
                                    ("above_1ulp", math.nextafter(radius, math.inf), -1),
                                    ("below_1e-6", radius - 1e-6, 2),
                                    ("above_1e-6", radius + 1e-6, -1)):
            result = primitive_centers(kind, dims, r)
            check.condition("centers." + label + "." + suffix, result["dimension"], expected)
            check.checks[-1]["exact_result"] = result
    check.condition("centers.rectangle_segment.location",
                    primitive_centers("rectangle", [10., 20.], 5.)["rectangle_intervals_exact"],
                    [["5", "5"], ["5", "15"]])
    check.condition("centers.rectangle_point.location",
                    primitive_centers("rectangle", [10., 10.], 5.)["center_exact"], ["5", "5"])
    check.condition("centers.circle_point.location",
                    primitive_centers("circle", [5.], 5.)["center_exact"], ["0", "0"])
    for kind, dims in (("rectangle", [10., 20.]), ("circle", [5.])):
        check.condition("centers.zero_radius." + kind, primitive_centers(kind, dims, 0.)["dimension"], 2)
    result = primitive_centers("rectangle", [10., 20.], 5.)
    x, y = [list(map(Fraction, interval)) for interval in result["rectangle_intervals_exact"]]
    endpoints = [(x[0], y[0]), (x[1], y[1])]
    # Reflection in Y followed by 90-degree rotation and translation; preserve dimension metadata.
    placed = [(Fraction(100) - py, Fraction(200) - px) for px, py in endpoints]
    check.condition("centers.rigid_reflected_segment.location", [[str(v) for v in p] for p in placed],
                    [["95", "195"], ["85", "195"]])
    check.condition("centers.rigid_reflected_segment.dimension", result["dimension"], 1)
    for kind, dims in (("rectangle", [10., 10.]), ("circle", [5.])):
        result = primitive_centers(kind, dims, 5.)
        px, py = map(Fraction, result["center_exact"])
        check.condition("centers.rigid_reflected_point." + kind,
                        [str(Fraction(100) - py), str(Fraction(200) - px), result["dimension"]],
                        ["95", "195", 0] if kind == "rectangle" else ["100", "200", 0])
    # Unit conversion happens on authoritative rationals, before float output.
    ratio = Fraction(5, 127)
    for kind, dims, radius in (("rectangle", [10., 20.], 5.), ("circle", [5.], 5.)):
        expected = primitive_centers(kind, dims, radius)["dimension"]
        actual = primitive_centers(kind, [Fraction(v) * ratio for v in dims],
                                   Fraction(radius) * ratio)["dimension"]
        check.condition("centers.exact_mm_to_inch." + kind, actual, expected)
    for name, kind, dims, r, reason in [
            ("general", "polygon", [10.], 5., "unsupported_general_collapse"),
            ("negative_radius", "circle", [5.], -1., "invalid_primitive"),
            ("nan_radius", "circle", [5.], math.nan, "invalid_primitive"),
            ("zero_size", "rectangle", [0., 5.], 1., "invalid_primitive"),
            ("infinite_size", "circle", [math.inf], 1., "invalid_primitive")]:
        try:
            primitive_centers(kind, dims, r)
            actual = "accepted"
        except ValueError as exc:
            actual = str(exc)
        check.condition("centers.reject." + name, actual, reason)


def output_semantics(check):
    outer = box(0, 0, 10, 10)
    touching_hole = outer.difference(Polygon([(0, 5), (2, 4), (2, 6)]))
    check.condition("output.point_contact.geos_valid", touching_hole.is_valid)
    try:
        strict_polygon(list(touching_hole.exterior.coords),
                       [list(r.coords) for r in touching_hole.interiors])
        status = "accepted"
    except ValueError:
        status = "unsupported_output_topology"
    check.condition("output.point_contact.owned_contract", status, "unsupported_output_topology")
    boundary_only = outer.intersection(box(10, 0, 20, 10))
    check.condition("output.boundary_intersection.raw_dimension", int(shapely.get_dimensions(boundary_only)), 1)
    # The filled-area contract takes the closure of the interior: lines contribute no area.
    area_parts = [p for p in shapely.get_parts(boundary_only) if p.geom_type == "Polygon"]
    check.condition("output.boundary_intersection.regularized_empty", not area_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check = Evaluation(LIMITS)
    for gate in (invalid_inputs, curves, invariants, exact_fit, output_semantics):
        try:
            gate(check)
        except Exception as exc:
            check.condition(gate.__name__ + ".unexpected_exception", type(exc).__name__ + ": " + str(exc), "none")
    paths = [Path(__file__), Path(__file__).with_name("evaluate_shapely.py")]
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "shapely": shapely.__version__, "geos": shapely.geos_version_string,
              "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "limits": LIMITS, "curve_interpolation_budget_mm": CURVE_ERROR,
              "boundary_metric": "bidirectional sampled maxima + 0.0005 mm sampling bound + reference interpolation bound; floating point, not intervals",
              "scope": "strict finite simple XY polygons with disjoint strictly interior holes; measured overlay/inset invariants; independently referenced radial contour and six circular holes; analytic primitive dimensions only",
              "limitations": ["No arbitrary organic inset accuracy oracle", "No general collapsed-center solver", "No source Arc/Pline-bulge detachment implemented; circular fixtures test analytic-to-chord backend geometry only", "No conservative containment, machining, motion, stock or physical clearance acceptance", "Four-ULP coordinate admission is an input-resolution gate, not an end-to-end numerical proof", "Exact binary64 primitive dimensions do not certify uncertain measurements or rounded transformed dimensions"],
              "checks": check.checks,
              "unexpected_failures": sum(not item["passed"] for item in check.checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "checks": len(check.checks),
                      "unexpected_failures": report["unexpected_failures"],
                      "failures": [item for item in check.checks if not item["passed"]]}, indent=2))
    return int(report["unexpected_failures"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
