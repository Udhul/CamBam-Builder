"""Bounded flat-endmill rest cleanup on a straight-edge Region with holes.

The supplied roughing trace is the stock authority. GEOS constructs the nominal
center contours and bounded sweep polygons; replay checks every actual segment.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

from . import replay


CONTOUR_MARGIN_MM = 0.0005
QUAD_SEGS = 128


def _geometry(target):
    from shapely.geometry import Polygon
    return Polygon(target.region_shell, target.region_holes)


def _line(a, b):
    from shapely.geometry import LineString, Point
    return Point(a) if a == b else LineString((a, b))


def _xy(point):
    return tuple(round(float(v), 6) for v in point)


def _rings(geometry):
    if geometry.geom_type != "Polygon" or geometry.is_empty:
        raise ValueError("smaller endmill cannot traverse disconnected center region")
    return (geometry.exterior,) + tuple(geometry.interiors)


def _segments(geometry):
    if geometry.geom_type == "LineString":
        return (geometry,)
    if geometry.geom_type == "MultiLineString":
        return tuple(geometry.geoms)
    return ()


@lru_cache(maxsize=32)
def _cut_polygon(cuts, depth, *, radial_error=0):
    from shapely.geometry import GeometryCollection
    from shapely.ops import unary_union
    shapes = []
    seen = set()
    inflation = 1 / math.cos(math.pi / (4 * QUAD_SEGS))
    for cut in cuts:
        if cut.bottom <= -depth:
            key = (cut.tool.radius, min(cut.a, cut.b), max(cut.a, cut.b))
            if key in seen:
                continue
            seen.add(key)
            radius = cut.tool.radius + radial_error
            if radius <= 0:
                continue
            if radial_error > 0:
                radius *= inflation
            shapes.append(_line(cut.a, cut.b).buffer(radius, quad_segs=QUAD_SEGS))
    return unary_union(shapes) if shapes else GeometryCollection()


def _areas(target, cuts, depth):
    """Conservative disk-polygon area interval, conditional on GEOS topology."""
    if not 0 < depth <= target.depth:
        raise ValueError("section outside target depth")
    region = _geometry(target)
    outer = _cut_polygon(cuts, depth, radial_error=1e-6)
    inner = _cut_polygon(cuts, depth, radial_error=-1e-6)
    lower = max(0.0, region.area - region.intersection(outer).area)
    upper = max(0.0, region.area - region.intersection(inner).area)
    return (lower, upper)


def _volume(target, cuts):
    """Integrate the flat-cylinder section intervals between cut-tip levels."""
    levels = sorted({0.0, float(target.depth)} |
                    {-cut.bottom for cut in cuts if 0 < -cut.bottom < target.depth})
    lower = upper = 0.0
    for start, end in zip(levels, levels[1:]):
        a, b = _areas(target, cuts, (start + end) / 2)
        lower += (end - start) * a
        upper += (end - start) * b
    return (lower, upper)


@dataclass(frozen=True)
class RestResult:
    prior_trace: replay.Trace
    trace: replay.Trace
    prior_stock: replay.ReplayResult
    stock: replay.ReplayResult
    allowance_mm: float
    evidence_class: str = "conditional_polygonal_swept_stock"

    @property
    def target(self):
        return self.trace.operations[0].target

    def pure_rest_contains(self, x, y, depth):
        return self.prior_stock.residual_contains(self.target.name, x, y, depth)

    def residual_contains(self, x, y, depth):
        return self.stock.residual_contains(self.target.name, x, y, depth)

    def pure_rest_area(self, depth):
        return _areas(self.target, self.prior_stock.cuts, depth)

    def residual_area(self, depth):
        return _areas(self.target, self.stock.cuts, depth)

    def pure_rest_volume(self):
        return _volume(self.target, self.prior_stock.cuts)

    def residual_volume(self):
        return _volume(self.target, self.stock.cuts)

    def protected_overcut_upper_area(self, depth):
        if not 0 < depth <= self.target.depth:
            raise ValueError("section outside target depth")
        return _cut_polygon(self.stock.cuts, depth,
                            radial_error=1e-6).difference(_geometry(self.target)).area

    def residual_outside_ideal_envelope_area(self, depth, envelope_mm=0.05):
        if not 0 < depth <= self.target.depth or envelope_mm < 0:
            raise ValueError("invalid residual envelope")
        region = _geometry(self.target)
        radius = self.trace.operations[-1].tool.radius
        ideal = region.difference(region.buffer(-radius, quad_segs=QUAD_SEGS)
                                  .buffer(radius, quad_segs=QUAD_SEGS))
        residual = region.difference(_cut_polygon(self.stock.cuts, depth,
                                                   radial_error=-1e-6))
        allowed = ideal.buffer(envelope_mm).union(region.boundary.buffer(envelope_mm))
        return residual.difference(allowed).area

    def finite_tool_rest_area(self):
        region = _geometry(self.target)
        radius = self.trace.operations[-1].tool.radius
        reachable = region.buffer(-radius, quad_segs=QUAD_SEGS).buffer(
            radius, quad_segs=QUAD_SEGS)
        return max(0.0, region.area - region.intersection(reachable).area)

    @property
    def completion(self):
        return ("partial_target_completion" if self.finite_tool_rest_area() > 0
                else "target_reachable_by_tool")


def generate(prior_trace, cleanup_tool, *, expected_source, expected_motion,
             rough_allowance_mm=0.5):
    """Follow both original boundaries after verified roughing to the floor.

    The first T2 descent of each contour is at an actual T1 cut endpoint. A
    connector then cuts to the smaller-tool contour through the original target.
    All links between contours/depths retract above stock.
    """
    if (type(prior_trace) is not replay.Trace or
            type(cleanup_tool) is not replay.ToolProfile or
            cleanup_tool.kind != "cylinder" or
            not math.isfinite(float(rough_allowance_mm)) or
            rough_allowance_mm < 0 or
            prior_trace.motion_fingerprint != expected_motion):
        raise ValueError("stale or unsupported polygonal prior motion/setup")
    prior = replay.replay(prior_trace, expected_source=expected_source)
    if (not prior.cuts or type(prior_trace.items[-1]) is not replay.Event or
            prior_trace.items[-1].kind != "spindle_stop" or
            prior_trace.items[-1].tool not in
            {op.tool.name for op in prior_trace.operations}):
        raise ValueError("complete supplied roughing sequence required")
    target = prior_trace.operations[0].target
    if (not target.region_shell or
            any(op.target != target or op.tool.kind != "cylinder" or
                op.tool.radius <= cleanup_tool.radius or
                op.tool.cutting_length < target.depth or
                op.tool.name == cleanup_tool.name or op.name == "cleanup"
                for op in prior_trace.operations) or
            target.depth > cleanup_tool.cutting_length):
        raise ValueError("unsupported polygonal target or endmill pair")
    region = _geometry(target)
    if any(_line(cut.a, cut.b).distance(region.boundary) <
           cut.tool.radius + rough_allowance_mm - 1e-8 or
           cut.bottom < -target.depth for cut in prior.cuts):
        raise ValueError("supplied roughing violates original boundary allowance")
    depths = tuple(sorted({-cut.bottom for cut in prior.cuts}))
    if not depths or depths[-1] != target.depth or any(
            depth <= 0 or depth > target.depth for depth in depths):
        raise ValueError("supplied roughing does not reach every requested floor")
    feasible = region.buffer(-(cleanup_tool.radius + CONTOUR_MARGIN_MM),
                             quad_segs=64)
    rings = _rings(feasible)
    deep_ends = tuple(dict.fromkeys(
        point for cut in prior.cuts if cut.bottom == -target.depth
        for point in (cut.a, cut.b)))
    paths = []
    for ring in rings:
        points = tuple(dict.fromkeys(_xy(point) for point in ring.coords[:-1]))
        if len(points) < 3:
            raise ValueError("smaller endmill contour collapsed")
        paths.append((points, True))
    # Add interior rows only where the actual prior-plus-contour sweep still
    # leaves material away from the inevitable corner/boundary residual. A
    # row touching a thin boundary rest would otherwise recut the whole pocket.
    contour_cuts = tuple(
        replay.Sweep("cleanup", cleanup_tool, a, b, -target.depth)
        for points, _ in paths for a, b in
        zip(points, points[1:] + points[:1]))
    after_contours = region.difference(
        _cut_polygon(prior.cuts + contour_cuts, depths[0]))
    finite_ideal = region.difference(
        region.buffer(-cleanup_tool.radius, quad_segs=QUAD_SEGS)
        .buffer(cleanup_tool.radius, quad_segs=QUAD_SEGS))
    need_interior = after_contours.difference(
        finite_ideal.buffer(0.05).union(region.boundary.buffer(0.05)))
    y = math.floor(region.bounds[1]) + 0.5
    while y < region.bounds[3]:
        section = feasible.intersection(
            _line((region.bounds[0] - 1, y), (region.bounds[2] + 1, y)))
        for segment in _segments(section):
            if (segment.length > 0.05 and
                    segment.buffer(cleanup_tool.radius,
                                   quad_segs=32).intersection(need_interior).area > 0.001):
                points = tuple(_xy(p) for p in
                               (segment.coords[0], segment.coords[-1]))
                if points[0] != points[1]:
                    paths.append((points, False))
        y += 1.5
    contours = []
    for points, closed in paths:
        choices = []
        indices = range(len(points)) if closed else (0, len(points) - 1)
        for i in indices:
            point = points[i]
            for anchor in deep_ends:
                segment = _line(anchor, point)
                if feasible.covers(segment) and segment.distance(
                        region.boundary) >= cleanup_tool.radius + 1e-5:
                    choices.append((segment.length, i, anchor))
        if not choices:
            raise ValueError("no prior-cleared access to smaller endmill path")
        _, i, anchor = min(choices)
        ordered = (points[i:] + points[:i] + (points[i],) if closed else
                   points if i == 0 else tuple(reversed(points)))
        contours.append((anchor, ordered))
    at = prior_trace.items[-1].position
    if at[2] <= 0:
        raise ValueError("prior ends below stock clearance")
    cleanup = replay.Operation("cleanup", cleanup_tool, target)
    items = list(prior_trace.items)
    items.extend((replay.Event("tool_change", cleanup_tool.name, at),
                  replay.Event("spindle_start", cleanup_tool.name, at)))
    for depth in depths:
        for anchor, contour in contours:
            high = (anchor[0], anchor[1], at[2])
            low = (anchor[0], anchor[1], -depth)
            current = items[-1].position if type(items[-1]) is replay.Event else items[-1].end
            if current != high:
                items.append(replay.Motion("rapid", cleanup_tool.name, "cleanup",
                                           current, high))
            items.append(replay.Motion("cleared_descent", cleanup_tool.name,
                                       "cleanup", high, low, 60))
            current = low
            for point in contour:
                end = (point[0], point[1], -depth)
                if end != current:
                    items.append(replay.Motion("cut", cleanup_tool.name,
                                               "cleanup", current, end, 300))
                current = end
            items.append(replay.Motion("retract", cleanup_tool.name, "cleanup",
                                       current, (current[0], current[1], at[2]), 300))
    current = items[-1].end
    if current != at:
        items.append(replay.Motion("rapid", cleanup_tool.name, "cleanup",
                                   current, at))
    items.append(replay.Event("spindle_stop", cleanup_tool.name, at))
    source = hashlib.sha256(repr((prior_trace.motion_fingerprint, cleanup_tool,
                                  rough_allowance_mm, "polygon-rest-v1")).encode(
                                      "utf-8")).hexdigest()
    trace = replay.Trace(source, prior_trace.frame, prior_trace.initial_position,
                         prior_trace.operations + (cleanup,), tuple(items))
    stock = replay.replay(trace, expected_source=source)
    if stock.prefixes != prior.prefixes + (("cleanup", len(stock.cuts)),):
        raise ValueError("polygonal stock prefix order changed")
    return RestResult(prior_trace, trace, prior, stock, rough_allowance_mm)
