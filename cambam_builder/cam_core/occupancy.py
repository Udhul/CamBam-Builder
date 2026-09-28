"""Bounded continuous tool-body occupancy for decoded ordered motion.

Tools are coaxial cylinders above the programmed tip. Stock is conservatively
treated as its entire initial box for non-cutting components; this deliberately
does not credit cavities made by earlier cuts. Fixtures are closed boxes.
"""

from dataclasses import dataclass
import math

from . import replay


VERSION = "bounded-tool-occupancy-v1"
ARC_VERSION = "bounded-tool-occupancy-v2-planar-arcs"
GEOMETRY_TOLERANCE_MM = 1e-9


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Box:
    name: str
    bounds: tuple  # xmin, ymin, zmin, xmax, ymax, zmax in the program frame

    def __post_init__(self):
        v = self.bounds
        if (not self.name or type(v) is not tuple or len(v) != 6 or
                not all(_finite(n) for n in v) or
                any(v[i] >= v[i + 3] for i in range(3))):
            raise ValueError("invalid occupancy box")


@dataclass(frozen=True)
class ToolBand:
    kind: str  # cutter, shank, or holder
    bottom_mm: float  # tip-relative +Z
    top_mm: float
    radius_mm: float

    def __post_init__(self):
        if (self.kind not in ("cutter", "shank", "holder") or
                not all(_finite(n) for n in
                        (self.bottom_mm, self.top_mm, self.radius_mm)) or
                self.bottom_mm < 0 or self.bottom_mm >= self.top_mm or
                self.radius_mm <= 0):
            raise ValueError("invalid occupancy tool band")


@dataclass(frozen=True)
class ToolBody:
    tool_id: str
    bands: tuple

    def __post_init__(self):
        if (not self.tool_id or type(self.bands) is not tuple or
                len(self.bands) != 3 or
                any(type(b) is not ToolBand for b in self.bands) or
                tuple(b.kind for b in self.bands) !=
                ("cutter", "shank", "holder") or
                self.bands[0].bottom_mm != 0 or
                any(a.top_mm != b.bottom_mm for a, b in
                    zip(self.bands, self.bands[1:]))):
            raise ValueError("tool body needs contiguous cutter, shank and holder")


@dataclass(frozen=True)
class OccupancySetup:
    frame: str
    stock: Box
    fixtures: tuple
    tools: tuple

    def __post_init__(self):
        if (not self.frame or type(self.stock) is not Box or
                self.stock.bounds[5] != 0 or
                type(self.fixtures) is not tuple or
                any(type(f) is not Box for f in self.fixtures) or
                type(self.tools) is not tuple or not self.tools or
                any(type(t) is not ToolBody for t in self.tools) or
                len({t.tool_id for t in self.tools}) != len(self.tools) or
                len({f.name for f in self.fixtures}) != len(self.fixtures)):
            raise ValueError("invalid occupancy setup")


def _point_rect_distance2(x, y, v):
    dx = max(v[0] - x, 0, x - v[3])
    dy = max(v[1] - y, 0, y - v[4])
    return dx * dx + dy * dy


def _point_segment_distance2(x, y, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0 if length2 == 0 else max(0, min(1,
        ((x - a[0]) * dx + (y - a[1]) * dy) / length2))
    return (x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2


def _segment_rect_distance(a, b, v):
    # A segment intersects the rectangle iff a clipped parameter interval
    # remains. Otherwise the minimum occurs at a segment end or box corner.
    lo, hi = 0.0, 1.0
    for start, delta, lower, upper in (
            (a[0], b[0] - a[0], v[0], v[3]),
            (a[1], b[1] - a[1], v[1], v[4])):
        if delta == 0:
            if start < lower or start > upper:
                lo, hi = 1, 0
                break
        else:
            t0, t1 = (lower - start) / delta, (upper - start) / delta
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if lo <= hi:
        return 0.0
    return math.sqrt(min(
        _point_rect_distance2(*a[:2], v),
        _point_rect_distance2(*b[:2], v),
        *(_point_segment_distance2(x, y, a, b)
          for x in (v[0], v[3]) for y in (v[1], v[4]))))


def _overlap_parameter(a, b, band, box):
    z0, z1 = a[2], b[2]
    lower = box.bounds[2] - band.top_mm
    upper = box.bounds[5] - band.bottom_mm
    if not all(math.isfinite(v) for v in (lower, upper, z1 - z0)):
        raise ValueError("occupancy geometry arithmetic is unresolved")
    if z0 == z1:
        return (0.0, 1.0) if lower <= z0 <= upper else None
    t0, t1 = (lower - z0) / (z1 - z0), (upper - z0) / (z1 - z0)
    lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
    if not all(math.isfinite(v) for v in (t0, t1, lo, hi)):
        raise ValueError("occupancy geometry arithmetic is unresolved")
    return (lo, hi) if lo <= hi else None


def _at(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def verify(setup, stages, decoded_moves):
    """Reject continuous straight or supported level-arc band/box intersections."""
    if (type(setup) is not OccupancySetup or
            len(stages) != len(decoded_moves)):
        raise ValueError("occupancy setup and decoded stages required")
    tools = {tool.tool_id: tool for tool in setup.tools}
    if set(tools) != {stage.tool_id for stage in stages}:
        raise ValueError("occupancy tool list differs from ordered stages")
    checked = 0
    checked_arcs = 0
    maximum_arc_error = 0.0
    minimum_fixture_clearance = math.inf
    minimum_stock_clearance = math.inf
    for stage, moves in zip(stages, decoded_moves):
        for move in moves:
            replay._xyz(move.start)
            replay._xyz(move.end)
            if any(not math.isfinite(b - a) for a, b in
                   zip(move.start, move.end)):
                raise ValueError("occupancy geometry arithmetic is unresolved")
            if move.arc_g:
                # The stock replay uses this same bounded arc enclosure. Each
                # chord is within error of the continuous center path, so its
                # rectangle distance minus error is a clearance lower bound.
                segments, arc_error = replay.arc_segments(
                    move.start, move.end, move.center, move.arc_g)
                checked_arcs += 1
                maximum_arc_error = max(maximum_arc_error, arc_error)
            else:
                segments, arc_error = None, 0.0
            checked += 1
            for band in tools[stage.tool_id].bands:
                boxes = setup.fixtures + (() if band.kind == "cutter" else
                                          (setup.stock,))
                for box in boxes:
                    interval = _overlap_parameter(move.start, move.end,
                                                  band, box)
                    if interval is None:
                        continue
                    if segments is None:
                        a, b = (_at(move.start, move.end, t) for t in interval)
                        if not all(math.isfinite(v) for v in a + b):
                            raise ValueError("occupancy geometry arithmetic is unresolved")
                        distance = _segment_rect_distance(a, b, box.bounds)
                    else:
                        # Supported arcs are level, so axial overlap is the
                        # whole sweep. The arc helper rejects ramps/helices.
                        distance = min(_segment_rect_distance(a, b, box.bounds)
                                       for a, b in segments)
                    clearance = distance - band.radius_mm - arc_error
                    if not math.isfinite(clearance):
                        raise ValueError("occupancy geometry arithmetic is unresolved")
                    if clearance <= GEOMETRY_TOLERANCE_MM:
                        label = "stock" if box is setup.stock else f"fixture {box.name}"
                        raise ValueError(f"{band.kind} collision with {label}")
                    if box is setup.stock:
                        minimum_stock_clearance = min(minimum_stock_clearance,
                                                      clearance)
                    else:
                        minimum_fixture_clearance = min(
                            minimum_fixture_clearance, clearance)
    return {"status": "pass", "scope": (
                "decoded straight and planar-arc stage tool-body and box occupancy"
                if checked_arcs else "decoded straight stage tool-body and box occupancy"),
            "model": ARC_VERSION if checked_arcs else VERSION,
            "checked_moves": checked,
            **({"checked_arcs": checked_arcs,
                "maximum_arc_enclosure_mm": maximum_arc_error}
               if checked_arcs else {}),
            "fixture_count": len(setup.fixtures),
            "minimum_fixture_clearance_mm": (
                None if math.isinf(minimum_fixture_clearance) else
                minimum_fixture_clearance),
            "minimum_stock_clearance_mm": (
                None if math.isinf(minimum_stock_clearance) else
                minimum_stock_clearance),
            "geometry_tolerance_mm": GEOMETRY_TOLERANCE_MM}
