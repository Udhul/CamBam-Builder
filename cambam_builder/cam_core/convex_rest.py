"""One convex closed-region rest consumer of supplied pointed-cone motion.

This is nominal, fixed-axis geometry. The caller owns native input and output.
"""

from dataclasses import dataclass
import hashlib
import math

from . import replay, tapered_vcarve


def _section_area(polygon, depth):
    """Clip a convex polygon by its original edges shifted inward by depth."""
    vertices = list(polygon)
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        offset = depth * math.hypot(dx, dy)

        def signed(point):
            return dx * (point[1] - a[1]) - dy * (point[0] - a[0]) - offset

        clipped = []
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            f, s = signed(first), signed(second)
            if (f < 0) != (s < 0):
                u = f / (f - s)
                clipped.append((first[0] + u * (second[0] - first[0]),
                                first[1] + u * (second[1] - first[1])))
            if s >= 0:
                clipped.append(second)
        vertices = clipped
        if not vertices:
            return 0.0
    return abs(sum(a[0] * b[1] - a[1] * b[0]
                   for a, b in zip(vertices, vertices[1:] + vertices[:1]))) / 2


@dataclass(frozen=True)
class RestResult:
    prior_trace: replay.Trace
    trace: replay.Trace
    prior_stock: replay.ReplayResult
    stock: replay.ReplayResult
    evidence_class: str = "conditional_nominal_convex_rest"
    completion: str = "partial_target_completion"

    @property
    def target(self):
        return self.trace.operations[0].target

    def pure_rest_contains(self, x, y, depth):
        return self.prior_stock.residual_contains(self.target.name, x, y, depth)

    def residual_contains(self, x, y, depth):
        return self.stock.residual_contains(self.target.name, x, y, depth)

    def target_area(self, depth):
        if not 0 <= depth <= self.target.depth:
            raise ValueError("section outside target depth")
        return _section_area(self.target.polygon, depth)

    def pure_rest_area(self, depth):
        area = self.target_area(depth)
        radius = max(0.0, -self.prior_stock.cuts[0].bottom - depth)
        return max(0.0, area - math.pi * radius * radius)

    def residual_area(self, depth):
        area = self.target_area(depth)
        cut = self.stock.cuts[-1]
        d0, d1 = -cut.bottom_start, -cut.bottom
        length = math.dist(cut.a, cut.b)
        removed = tapered_vcarve.section_area((0, length, 0, d0, d1), depth)
        return max(0.0, area - removed)


def generate(prior_trace, end, *, expected_source, expected_motion):
    """Extend one supplied cleared cone column along a rising-clearance line.

    The prior trace is replayed as authority. Its pointed entry is retained in
    the combined ordered trace, and the cleanup starts at that cleared column.
    """
    if type(prior_trace) is not replay.Trace or type(end) is not tuple or len(end) != 2:
        raise ValueError("expected supplied prior trace and XY endpoint")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(float(v)) for v in end):
        raise ValueError("finite XY endpoint required")
    if prior_trace.motion_fingerprint != expected_motion:
        raise ValueError("stale prior motion")
    prior_stock = replay.replay(prior_trace, expected_source=expected_source)
    if (len(prior_trace.operations) != 1 or len(prior_stock.cuts) != 1 or
            prior_stock.prefixes != ((prior_trace.operations[0].name, 1),) or
            type(prior_trace.items[-1]) is not replay.Event or
            prior_trace.items[-1].kind != "spindle_stop"):
        raise ValueError("one complete prior cone column required")
    operation = prior_trace.operations[0]
    target, tool = operation.target, operation.tool
    prior = prior_stock.cuts[0]
    if (not target.polygon or tool.kind != "pointed_cone" or
            prior.a != prior.b or prior.bottom_start is not None or
            len(prior_trace.items) != 5 or
            tuple(item.role for item in prior_trace.items[2:4]) !=
            ("entry", "retract") or
            prior_trace.initial_position[2] <= 0):
        raise ValueError("unsupported prior cone motion or target")
    start = prior.a
    start_depth = -prior.bottom
    end_depth = replay._polygon_clearance(target.polygon, end)
    length = math.dist(start, end)
    if (not 0 < start_depth < end_depth <= target.depth or
            end_depth > tool.cutting_length or length == 0 or
            (end_depth - start_depth) >= length or
            abs(replay._polygon_clearance(target.polygon, start) -
                start_depth) > 1e-10):
        raise ValueError("cleanup has no supported rising original clearance")
    safe = prior_trace.items[-1].position
    if safe[:2] != start or safe[2] <= 0:
        raise ValueError("prior did not retract above its cleared column")
    low_start = (start[0], start[1], -start_depth)
    low_end = (end[0], end[1], -end_depth)
    high_end = (end[0], end[1], safe[2])
    cleanup = replay.Operation("cleanup", tool, target)
    if operation.name == cleanup.name:
        raise ValueError("prior operation name conflicts with cleanup")
    items = prior_trace.items[:-1] + (
        replay.Motion("cleared_descent", tool.name, cleanup.name, safe, low_start),
        replay.Motion("cut", tool.name, cleanup.name, low_start, low_end),
        replay.Motion("retract", tool.name, cleanup.name, low_end, high_end),
        replay.Event("spindle_stop", tool.name, high_end),
    )
    source = hashlib.sha256(repr((prior_trace.motion_fingerprint, end,
                                  "convex-rest-v1")).encode("utf-8")).hexdigest()
    trace = replay.Trace(source, prior_trace.frame, prior_trace.initial_position,
                         (operation, cleanup), items)
    stock = replay.replay(trace, expected_source=source)
    return RestResult(prior_trace, trace, prior_stock, stock)
