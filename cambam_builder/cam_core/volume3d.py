"""Bounded layered 3D stock and decoded flat-endmill sweep evidence.

The section polygons are exact for the supplied stepped target. GEOS polygon
operations and circular buffer approximations still have floating point limits.
"""

from dataclasses import dataclass
import math
from numbers import Real
import time
import tracemalloc

from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


VERSION = "layered-volume-v1"
QUAD_SEGS = 24
SAGITTA_MM = 1 - math.cos(math.pi / (4 * QUAD_SEGS))
AREA_TOLERANCE_MM2 = 1e-7


def _rect(value):
    return (type(value) is tuple and len(value) == 4 and
            all(isinstance(v, Real) and not isinstance(v, bool) and
                math.isfinite(v) for v in value) and
            value[0] < value[2] and value[1] < value[3])


@dataclass(frozen=True)
class LayeredTarget:
    """Stock from Z=0 to -stock_depth; disjoint rectangular removal prisms."""

    stock_xy: tuple
    stock_depth: float
    prisms: tuple                 # (xmin, ymin, xmax, ymax, removal_depth)
    protected_xy: tuple          # thin feature that must remain solid

    def __post_init__(self):
        if (not _rect(self.stock_xy) or not _rect(self.protected_xy) or
                isinstance(self.stock_depth, bool) or
                not isinstance(self.stock_depth, Real) or
                not 0 < self.stock_depth < math.inf or
                type(self.prisms) is not tuple or not self.prisms or
                any(type(p) is not tuple or len(p) != 5 or
                    not _rect(p[:4]) or isinstance(p[4], bool) or
                    not isinstance(p[4], Real) or
                    not 0 < p[4] <= self.stock_depth
                    for p in self.prisms)):
            raise ValueError("invalid layered target")
        stock = box(*self.stock_xy)
        shapes = [box(*p[:4]) for p in self.prisms]
        feature = box(*self.protected_xy)
        if (not stock.is_valid or not stock.contains(feature) or
                any(not stock.contains(shape) or
                    shape.intersection(feature).area > AREA_TOLERANCE_MM2
                    for shape in shapes) or
                any(a.intersection(b).area > AREA_TOLERANCE_MM2
                    for i, a in enumerate(shapes)
                    for b in shapes[i + 1:])):
            raise ValueError("prisms overlap or touch protected stock")

    def section(self, depth):
        if not 0 < depth <= self.stock_depth:
            raise ValueError("section depth outside stock")
        return unary_union([box(*p[:4]) for p in self.prisms if depth <= p[4]])

    @property
    def target_volume_mm3(self):
        return sum(box(*p[:4]).area * p[4] for p in self.prisms)


@dataclass(frozen=True)
class VolumeOperation:
    name: str
    tool_id: str
    radius_mm: float
    cutting_length_mm: float
    target: LayeredTarget
    strategy: str

    def __post_init__(self):
        if (not self.name or not self.tool_id or
                self.strategy not in ("waterline", "rest") or
                isinstance(self.radius_mm, bool) or
                not isinstance(self.radius_mm, Real) or
                not 0 < self.radius_mm < math.inf or
                isinstance(self.cutting_length_mm, bool) or
                not isinstance(self.cutting_length_mm, Real) or
                not 0 < self.cutting_length_mm < math.inf or
                type(self.target) is not LayeredTarget):
            raise ValueError("invalid volume operation")


def _sweep(a, b, radius, *, outer):
    # Shapely's regular buffer is inscribed. The enlarged buffer has an
    # inradius at least the true cutter radius, so it bounds occupancy above.
    r = radius / math.cos(math.pi / (4 * QUAD_SEGS)) if outer else radius
    xy_a, xy_b = a[:2], b[:2]
    curve = Point(xy_a) if xy_a == xy_b else LineString((xy_a, xy_b))
    return curve.buffer(r, quad_segs=QUAD_SEGS)


def replay_stages(stages, decoded_moves):
    """Check complete decoded motion and return per-prefix residual intervals."""
    if (not stages or len(stages) != len(decoded_moves) or
            any(type(s.volume_operation) is not VolumeOperation for s in stages)):
        raise ValueError("volume stages required")
    target = stages[0].volume_operation.target
    if any(s.volume_operation.target != target for s in stages):
        raise ValueError("volume stages have different targets")
    cuts = []
    reports = []
    feature = box(*target.protected_xy)
    for stage, moves in zip(stages, decoded_moves):
        op = stage.volume_operation
        prior_cuts = tuple(cuts)
        for move in moves:
            a, b = move.start, move.end
            if move.role in ("rapid", "rapid_retract"):
                if move.role == "rapid" and min(a[2], b[2]) <= 0:
                    raise ValueError("rapid crosses initial stock")
                if a[:2] != b[:2] and min(a[2], b[2]) <= 0:
                    raise ValueError("lateral rapid below stock top")
                if move.role == "rapid_retract" and (a[:2] != b[:2] or b[2] <= a[2]):
                    raise ValueError("invalid rapid retract")
                continue
            if move.role in ("entry", "cleared_descent", "retract"):
                if a[:2] != b[:2] or (move.role in ("entry", "cleared_descent")
                                      and b[2] >= a[2]) or (
                        move.role == "retract" and b[2] <= a[2]):
                    raise ValueError("unsupported volume entry or retract")
            elif move.role != "cut" or a[2] != b[2]:
                raise ValueError("unsupported volume feed motion")
            depth = -min(a[2], b[2])
            if depth <= 0:
                continue
            if depth > min(target.stock_depth, op.cutting_length_mm):
                raise ValueError("tool exceeds stock or cutting length")
            outer = _sweep(a, b, op.radius_mm, outer=True)
            # Every cut path and plunge must fit the original target at its
            # deepest position; this also protects all shallower sections.
            if (outer.difference(target.section(depth)).area >
                    AREA_TOLERANCE_MM2 or outer.intersection(feature).area >
                    AREA_TOLERANCE_MM2):
                raise ValueError("cutter enters protected volume")
            if move.role == "cleared_descent":
                levels = {depth} | {p[4] for p in target.prisms if p[4] < depth} | {
                    d for d, _, _ in prior_cuts if d < depth}
                for level in levels:
                    if level <= 0:
                        continue
                    cleared = unary_union([inner for d, inner, _ in prior_cuts
                                           if d >= level])
                    if outer.difference(cleared).area > AREA_TOLERANCE_MM2:
                        raise ValueError("uncleared descent lacks predecessor stock")
            if move.role in ("entry", "cut"):
                cuts.append((depth, _sweep(a, b, op.radius_mm, outer=False), outer))
        levels = sorted({0.0, target.stock_depth} |
                        {p[4] for p in target.prisms} |
                        {depth for depth, _, _ in cuts})
        low_volume = high_volume = protected = 0.0
        sections = {}
        for lo, hi in zip(levels, levels[1:]):
            section = target.section((lo + hi) / 2)
            active = [(inner, outer) for depth, inner, outer in cuts if depth >= hi]
            inner = unary_union([pair[0] for pair in active]) if active else Point().buffer(0)
            outer = unary_union([pair[1] for pair in active]) if active else Point().buffer(0)
            residual_low = section.difference(outer).area
            residual_shape = section.difference(inner)
            residual_high = residual_shape.area
            low_volume += (hi - lo) * residual_low
            high_volume += (hi - lo) * residual_high
            protected += (hi - lo) * outer.difference(section).area
            sections[f"{lo:g}:{hi:g}"] = (residual_low, residual_high,
                                  len(residual_shape.geoms)
                                  if residual_shape.geom_type == "MultiPolygon"
                                  else int(not residual_shape.is_empty))
        if protected > AREA_TOLERANCE_MM2 * target.stock_depth:
            raise ValueError("protected material removed")
        reports.append({"stage": stage.id, "cuts": len(cuts),
                        "residual_volume_mm3": (low_volume, high_volume),
                        "protected_overcut_upper_mm3": protected,
                        "sections": sections})
    return {"status": "pass", "scope": "decoded layered 3D flat-endmill sweeps",
            "model": VERSION, "buffer_quad_segs": QUAD_SEGS,
            "buffer_radial_error_mm_per_mm": SAGITTA_MM,
            "area_tolerance_mm2": AREA_TOLERANCE_MM2,
            "target_volume_mm3": target.target_volume_mm3,
            "prefixes": tuple(reports)}


def compare_representations(target, pitches=(0.25, 0.125)):
    """Measure conservative XY column volume bounds against exact prisms.

    Peak bytes are Python allocations only; GEOS native allocation is excluded.
    Timing is diagnostic, not an acceptance threshold.
    """
    if type(target) is not LayeredTarget or not pitches or any(
            type(p) not in (int, float) or not math.isfinite(p) or p <= 0
            for p in pitches):
        raise ValueError("target and positive grid pitches required")
    xmin, ymin, xmax, ymax = target.stock_xy
    def measured(fn):
        tracemalloc.start()
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, round(elapsed, 3), peak

    exact, elapsed, peak = measured(lambda: tuple(
        (p[4], box(*p[:4]).area) for p in target.prisms))
    rows = [{"representation": "exact_rectangular_prisms",
             "volume_interval_mm3": (target.target_volume_mm3,
                                      target.target_volume_mm3),
             "elapsed_ms": elapsed, "python_peak_bytes": peak,
             "elements": len(exact)}]
    for pitch in pitches:
        nx = math.ceil((xmax - xmin) / pitch)
        ny = math.ceil((ymax - ymin) / pitch)
        def columns():
            cells = []
            for ix in range(nx):
                for iy in range(ny):
                    cell = box(xmin + ix * pitch, ymin + iy * pitch,
                               min(xmax, xmin + (ix + 1) * pitch),
                               min(ymax, ymin + (iy + 1) * pitch))
                    lower = upper = 0.0
                    for prism in target.prisms:
                        region = box(*prism[:4])
                        overlap = region.intersection(cell).area
                        if overlap > AREA_TOLERANCE_MM2:
                            upper += cell.area * prism[4]
                        if abs(overlap - cell.area) <= AREA_TOLERANCE_MM2:
                            lower += cell.area * prism[4]
                    cells.append((lower, upper))
            return (sum(cell[0] for cell in cells),
                    sum(cell[1] for cell in cells))
        interval, elapsed, peak = measured(columns)
        rows.append({"representation": "conservative_xy_columns",
                     "pitch_mm": pitch, "volume_interval_mm3": interval,
                     "elapsed_ms": elapsed, "python_peak_bytes": peak,
                     "elements": nx * ny})
    return tuple(rows)
