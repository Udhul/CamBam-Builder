"""Bounded circular-arc Region approximation for cylindrical rest planning.

Input rings contain (x, y, bulge) records. The native owner validates their
analytic topology before this module is called. Chords are nominal geometry;
an inward offset is the conservative cutter-access target.
"""

from dataclasses import dataclass
import math

from shapely.geometry import Polygon

from . import polygon_rest, replay


@dataclass(frozen=True)
class Approximation:
    nominal: Polygon
    safe: Polygon
    outer: Polygon
    sagitta_mm: float
    area_error_mm2: float
    analytic_area_mm2: float
    segment_count: int

    def target(self, name, depth):
        if self.safe.geom_type != "Polygon" or self.safe.is_empty:
            raise ValueError("curved safe center region disconnected or empty")
        shell = tuple(tuple(p) for p in self.safe.exterior.coords[:-1])
        holes = tuple(tuple(tuple(p) for p in ring.coords[:-1])
                      for ring in self.safe.interiors)
        return replay.Target(name, self.safe.bounds, depth,
                             region_shell=shell, region_holes=holes)


def _ring(records, sagitta):
    if len(records) < 2:
        raise ValueError("curved ring needs at least two records")
    points = []
    area = error = 0.0
    count = 0
    for index, (x, y, bulge) in enumerate(records):
        xx, yy, _ = records[(index + 1) % len(records)]
        if not all(math.isfinite(v) for v in (x, y, xx, yy, bulge)):
            raise ValueError("nonfinite curved Region coordinate")
        chord = math.hypot(xx - x, yy - y)
        if chord <= 0:
            raise ValueError("zero-length curved Region edge")
        points.append((x, y))
        area += (x * yy - xx * y) / 2
        if bulge == 0:
            count += 1
            continue
        if abs(bulge) < 1e-8:
            raise ValueError("curved Region bulge below supported resolution")
        angle = 4 * math.atan(bulge)
        radius = chord / (2 * abs(math.sin(angle / 2)))
        if not math.isfinite(radius):
            raise ValueError("curved Region arc radius unresolved")
        # Directed midpoint offset from the chord. A positive bulge turns left.
        midpoint = ((x + xx) / 2, (y + yy) / 2)
        left = (-(yy - y) / chord, (xx - x) / chord)
        center_offset = chord * (1 / bulge - bulge) / 4
        if not math.isfinite(center_offset):
            raise ValueError("curved Region arc center unresolved")
        center = (midpoint[0] + left[0] * center_offset,
                  midpoint[1] + left[1] * center_offset)
        start = math.atan2(y - center[1], x - center[0])
        # The same sagitta bounds every subarc; cap count to fail predictably.
        max_angle = 2 * math.acos(max(-1.0, 1 - sagitta / radius))
        if max_angle <= 0:
            raise ValueError("curved Region arc exceeds numeric resolution")
        pieces = max(1, math.ceil(abs(angle) / max_angle))
        if pieces > 16384:
            raise ValueError("curved Region arc exceeds subdivision limit")
        step = angle / pieces
        for j in range(1, pieces):
            a = start + j * step
            points.append((center[0] + radius * math.cos(a),
                           center[1] + radius * math.sin(a)))
        count += pieces
        area += radius * radius * (angle - math.sin(angle)) / 2
        error += pieces * radius * radius * (abs(step) - math.sin(abs(step))) / 2
    return points, area, error, count


def approximate(shell, holes=(), *, sagitta_mm=0.001, max_segments=32768):
    """Return nominal, inward-safe and outward-covering regions in millimetres."""
    if not math.isfinite(sagitta_mm) or not 0 < sagitta_mm <= 0.001:
        raise ValueError("curved boundary sagitta must be within 0..0.001 mm")
    rings = [_ring(ring, sagitta_mm) for ring in (shell,) + tuple(holes)]
    segments = sum(row[3] for row in rings)
    if segments > max_segments:
        raise ValueError("curved Region exceeds segment budget")
    nominal = Polygon(rings[0][0], [row[0] for row in rings[1:]])
    if not nominal.is_valid or nominal.is_empty or nominal.area <= 0:
        raise ValueError("curved Region chord topology is invalid")
    # Native Region validation excludes source self intersections and contacts.
    # An offset that splits or loses a hole needs a separate access analysis.
    safe = nominal.buffer(-sagitta_mm, quad_segs=64)
    outer = nominal.buffer(sagitta_mm, quad_segs=64)
    if (safe.geom_type != "Polygon" or safe.is_empty or
            len(safe.interiors) != len(holes) or
            outer.geom_type != "Polygon" or
            len(outer.interiors) != len(holes)):
        raise ValueError("curved Region topology unresolved at error margin")
    exact_area = abs(rings[0][1]) - sum(abs(row[1]) for row in rings[1:])
    if exact_area <= 0:
        raise ValueError("curved Region analytic area is invalid")
    error = sum(row[2] for row in rings)
    if abs(nominal.area - exact_area) > error + 1e-7:
        raise ValueError("curved Region area error ledger inconsistent")
    return Approximation(nominal, safe, outer, sagitta_mm, error,
                         exact_area, segments)


@dataclass(frozen=True)
class RestResult:
    approximation: Approximation
    planned: polygon_rest.RestResult

    def rest_area(self, depth, *, final):
        cuts = self.planned.stock.cuts if final else self.planned.prior_stock.cuts
        outer_sweep = polygon_rest._cut_polygon(cuts, depth, radial_error=1e-6)
        inner_sweep = polygon_rest._cut_polygon(cuts, depth, radial_error=-1e-6)
        lower = max(0.0, self.approximation.safe.area -
                    self.approximation.safe.intersection(outer_sweep).area)
        upper = max(0.0, self.approximation.outer.area -
                    self.approximation.outer.intersection(inner_sweep).area)
        return lower, upper

    def protected_overcut_upper_area(self, depth):
        outer_sweep = polygon_rest._cut_polygon(
            self.planned.stock.cuts, depth, radial_error=1e-6)
        return outer_sweep.difference(self.approximation.safe).area

    def rest_volume(self, *, final):
        cuts = self.planned.stock.cuts if final else self.planned.prior_stock.cuts
        target_depth = self.planned.target.depth
        levels = sorted({0.0, target_depth} |
                        {-cut.bottom for cut in cuts if 0 < -cut.bottom < target_depth})
        values = [self.rest_area((a + b) / 2, final=final)
                  for a, b in zip(levels, levels[1:])]
        return tuple(sum((b - a) * row[i] for (a, b), row in
                         zip(zip(levels, levels[1:]), values)) for i in (0, 1))


def generate(approximation, prior_trace, cleanup_tool, *, expected_source,
             expected_motion, rough_allowance_mm):
    """Plan on the inward-safe target, then bound rest against the source arc."""
    if prior_trace.operations[0].target != approximation.target(
            prior_trace.operations[0].target.name,
            prior_trace.operations[0].target.depth):
        raise ValueError("curved prior target does not match approximation")
    planned = polygon_rest.generate(prior_trace, cleanup_tool,
                                    expected_source=expected_source,
                                    expected_motion=expected_motion,
                                    rough_allowance_mm=rough_allowance_mm)
    return RestResult(approximation, planned)
