"""Bounded, ordered XYZ stock replay for cylindrical and pointed-cone sweeps.

The target and tool values are detached from generators and output formats.
Specialized job verifiers still own process limits, fixtures and residual oracles.
"""

from dataclasses import dataclass
import hashlib
import math
from numbers import Real


def _xyz(value):
    if not isinstance(value, tuple) or len(value) != 3 or any(
            isinstance(v, bool) or not isinstance(v, Real) or
            not math.isfinite(float(v)) for v in value):
        raise ValueError("finite XYZ tuple required")
    return value


def _tapered_cone_contains(spine, x, y, depth):
    """Membership in a linear-radius cone sweep along an increasing X spine."""
    x0, x1, cy, d0, d1 = spine
    slope = (d1 - d0) / (x1 - x0)
    if depth < 0 or depth > d1:
        return False
    radius0 = d0 - depth
    low = max(0.0, -radius0 / slope) if slope else 0.0
    length = x1 - x0
    if low > length:
        return False
    # The squared cone occupancy is a concave quadratic in spine distance.
    u = (x - x0 + slope * radius0) / (1 - slope * slope)
    u = min(length, max(low, u))
    radius = radius0 + slope * u
    return (x - x0 - u) ** 2 + (y - cy) ** 2 <= radius ** 2


@dataclass(frozen=True)
class ToolProfile:
    name: str
    kind: str
    radius: float
    cutting_length: float

    def __post_init__(self):
        if (not self.name or self.kind not in ("cylinder", "pointed_cone") or
                not math.isfinite(float(self.radius)) or
                not math.isfinite(float(self.cutting_length)) or
                self.radius <= 0 or self.cutting_length <= 0 or
                (self.kind == "pointed_cone" and
                 self.radius != self.cutting_length)):
            raise ValueError("unsupported replay tool profile")


@dataclass(frozen=True)
class Target:
    name: str
    bounds: tuple                 # xmin, ymin, xmax, ymax
    depth: float                 # positive depth below stock top Z=0
    inset_per_depth: bool = False
    island: tuple = ()
    cone_spine: tuple = ()       # x0, x1, center y, tip depth at x0/x1

    def __post_init__(self):
        if (not self.name or len(self.bounds) != 4 or
                any(not math.isfinite(float(v)) for v in self.bounds) or
                not self.bounds[0] < self.bounds[2] or
                not self.bounds[1] < self.bounds[3] or
                not math.isfinite(float(self.depth)) or self.depth <= 0 or
                (self.island and (len(self.island) != 4 or
                                  self.inset_per_depth)) or
                (self.cone_spine and (len(self.cone_spine) != 5 or
                    self.inset_per_depth or self.island))):
            raise ValueError("unsupported replay target")
        if self.cone_spine:
            x0, x1, cy, d0, d1 = self.cone_spine
            if (any(not math.isfinite(float(v)) for v in self.cone_spine) or
                    not x0 < x1 or not 0 < d0 < d1 <= self.depth or
                    not 0 < (d1 - d0) / (x1 - x0) < 1 or
                    x0 - d0 < self.bounds[0] or x1 + d1 > self.bounds[2] or
                    cy - d1 < self.bounds[1] or cy + d1 > self.bounds[3]):
                raise ValueError("unsupported tapered cone target")

    def contains(self, x, y, depth):
        if not 0 <= depth <= self.depth:
            return False
        if self.cone_spine:
            return _tapered_cone_contains(self.cone_spine, x, y, depth)
        a, b, c, d = self.bounds
        t = depth if self.inset_per_depth else 0
        inside = a + t <= x <= c - t and b + t <= y <= d - t
        if self.island:
            u, v, w, h = self.island
            inside = inside and not (u <= x <= w and v <= y <= h)
        return inside


@dataclass(frozen=True)
class Operation:
    name: str
    tool: ToolProfile
    target: Target


@dataclass(frozen=True)
class Motion:
    role: str
    tool: str
    operation: str
    start: tuple
    end: tuple
    feed: float = 0

    def __post_init__(self):
        if self.role not in ("rapid", "approach", "entry", "cleared_descent",
                             "cut", "retract"):
            raise ValueError("unsupported replay motion role")
        _xyz(self.start)
        _xyz(self.end)
        if self.start == self.end or not math.isfinite(float(self.feed)) or self.feed < 0:
            raise ValueError("invalid replay motion")


@dataclass(frozen=True)
class Event:
    kind: str
    tool: str
    position: tuple

    def __post_init__(self):
        if self.kind not in ("tool_change", "spindle_start", "spindle_stop"):
            raise ValueError("unsupported replay event")
        _xyz(self.position)


@dataclass(frozen=True)
class Trace:
    source_fingerprint: str
    frame: str
    initial_position: tuple
    operations: tuple
    items: tuple
    units: str = "mm"

    def __post_init__(self):
        if (not self.source_fingerprint or not self.frame or self.units != "mm" or
                not isinstance(self.operations, tuple) or not self.operations or
                not isinstance(self.items, tuple) or not self.items):
            raise ValueError("invalid replay trace")
        _xyz(self.initial_position)
        if (any(type(op) is not Operation for op in self.operations) or
                len({op.name for op in self.operations}) != len(self.operations)):
            raise ValueError("duplicate or invalid operations")

    @property
    def motion_fingerprint(self):
        return hashlib.sha256(repr((self.source_fingerprint, self.frame,
                                    self.initial_position, self.operations,
                                    self.items, self.units)).encode("utf-8")).hexdigest()


def _distance2(point, a, b):
    if a[0] == b[0]:
        return ((point[0] - a[0]) ** 2 +
                max(min(a[1], b[1]) - point[1], 0,
                    point[1] - max(a[1], b[1])) ** 2)
    if a[1] == b[1]:
        return ((point[1] - a[1]) ** 2 +
                max(min(a[0], b[0]) - point[0], 0,
                    point[0] - max(a[0], b[0])) ** 2)
    raise ValueError("replay supports axis-aligned stock cuts")


@dataclass(frozen=True)
class Sweep:
    operation: str
    tool: ToolProfile
    a: tuple
    b: tuple
    bottom: float
    bottom_start: float = None

    def radius_at(self, depth):
        if self.bottom_start is not None:
            raise ValueError("variable-depth sweep has no single section radius")
        if not 0 <= depth <= -self.bottom:
            return None
        return (self.tool.radius if self.tool.kind == "cylinder" else
                -self.bottom - depth)

    def contains(self, x, y, depth):
        if self.bottom_start is not None:
            return _tapered_cone_contains(
                (self.a[0], self.b[0], self.a[1],
                 -self.bottom_start, -self.bottom), x, y, depth)
        radius = self.radius_at(depth)
        return radius is not None and _distance2((x, y), self.a, self.b) <= radius ** 2


def _safe_cut(sweep, target):
    a, b = sweep.a, sweep.b
    if target.cone_spine:
        x0, x1, cy, d0, d1 = target.cone_spine
        if (sweep.tool.kind != "pointed_cone" or a[1] != cy or b[1] != cy or
                a[0] > b[0] or a[0] < x0 or b[0] > x1 or
                sweep.bottom_start is not None and a == b):
            raise ValueError("cut crosses tapered cone target")
        start_depth = -sweep.bottom_start if sweep.bottom_start is not None else -sweep.bottom
        end_depth = -sweep.bottom
        slope = (d1 - d0) / (x1 - x0)
        if (not 0 < start_depth <= end_depth <= sweep.tool.cutting_length or
                start_depth > d0 + slope * (a[0] - x0) or
                end_depth > d0 + slope * (b[0] - x0) or
                sweep.bottom_start is not None and
                not 0 < (end_depth - start_depth) / (b[0] - a[0]) < 1):
            raise ValueError("cut crosses tapered cone target or tool limit")
        return
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("replay supports axis-aligned stock cuts")
    if sweep.bottom < -target.depth or -sweep.bottom > sweep.tool.cutting_length:
        raise ValueError("cut exceeds target depth or cutting length")
    x0, y0, x1, y1 = target.bounds
    r = sweep.tool.radius if sweep.tool.kind == "cylinder" else -sweep.bottom
    if any(not (x0 + r <= x <= x1 - r and y0 + r <= y <= y1 - r)
           for x, y in (a, b)):
        raise ValueError("cut crosses protected target")
    if target.inset_per_depth and sweep.tool.kind != "pointed_cone":
        raise ValueError("unsupported inset target/tool combination")
    if target.island:
        u, v, w, h = target.island
        dx = max(u - max(a[0], b[0]), min(a[0], b[0]) - w, 0)
        dy = max(v - max(a[1], b[1]), min(a[1], b[1]) - h, 0)
        if dx * dx + dy * dy < r * r:
            raise ValueError("cut crosses protected island")


def _covered(cuts, move, tool):
    low = min(move.start[2], move.end[2])
    if low >= 0:
        return True
    a, b = move.start[:2], move.end[:2]
    if tool.kind == "cylinder":
        for prior in reversed(cuts):
            if prior.tool.kind != "cylinder" or prior.bottom > low:
                continue
            margin = prior.tool.radius - tool.radius
            if margin >= 0 and all(_distance2(p, prior.a, prior.b) <= margin ** 2
                                   for p in (a, b)):
                return True
    else:
        # The bounded cone route only retracts vertically along its immediately
        # preceding, full-depth cut endpoint. That cut contains every section of
        # the retracting cone. Broader cone access needs a separate proof.
        if (a == b and cuts and cuts[-1].tool == tool and
                cuts[-1].bottom <= low and
                _distance2(a, cuts[-1].a, cuts[-1].b) == 0):
            return True
    return False


@dataclass(frozen=True)
class ReplayResult:
    source_fingerprint: str
    motion_fingerprint: str
    cuts: tuple
    prefixes: tuple               # (operation, cumulative cut count)
    targets: tuple

    def removed_contains(self, x, y, depth, *, through=None):
        limit = len(self.cuts) if through is None else through
        if not isinstance(limit, int) or not 0 <= limit <= len(self.cuts):
            raise ValueError("invalid replay prefix")
        return any(c.contains(x, y, depth) for c in self.cuts[:limit])

    def residual_contains(self, target, x, y, depth, *, through=None):
        match = next((t for t in self.targets if t.name == target), None)
        if match is None:
            raise ValueError("unknown replay target")
        return (match.contains(x, y, depth) and
                not self.removed_contains(x, y, depth, through=through))


def replay(trace, *, expected_source):
    """Replay a complete ordered stream; never infer a missing cut or tool event."""
    if type(trace) is not Trace or trace.source_fingerprint != expected_source:
        raise ValueError("stale replay source")
    operations = {op.name: op for op in trace.operations}
    at, active, running, cuts, prefixes = trace.initial_position, None, False, [], []
    last_operation = None
    for index, item in enumerate(trace.items):
        if type(item) is Event:
            if item.position != at:
                raise ValueError(f"event {index}: position mismatch")
            if item.kind == "tool_change":
                if running or item.tool == active or not any(
                        op.tool.name == item.tool for op in trace.operations):
                    raise ValueError(f"event {index}: invalid tool change")
                active = item.tool
            elif item.kind == "spindle_start":
                if running or active != item.tool:
                    raise ValueError(f"event {index}: invalid spindle start")
                running = True
            elif not running or active != item.tool:
                raise ValueError(f"event {index}: invalid spindle stop")
            else:
                running = False
            continue
        if type(item) is not Motion or item.start != at or not running or item.tool != active:
            raise ValueError(f"move {index}: discontinuity or inactive tool")
        op = operations.get(item.operation)
        if op is None or op.tool.name != active:
            raise ValueError(f"move {index}: unknown operation/tool")
        if last_operation != op.name:
            if last_operation is not None:
                prefixes.append((last_operation, len(cuts)))
            last_operation = op.name
        if item.role in ("rapid", "approach"):
            if min(item.start[2], item.end[2]) < 0:
                raise ValueError(f"move {index}: low rapid/approach")
        elif item.role in ("entry", "cut"):
            if item.role == "entry" and item.start[:2] != item.end[:2]:
                raise ValueError(f"move {index}: entry changes XY")
            variable = (item.role == "cut" and item.start[2] != item.end[2] and
                        op.target.cone_spine and op.tool.kind == "pointed_cone")
            if item.role == "cut" and item.start[2] != item.end[2] and not variable:
                raise ValueError(f"move {index}: non-level cut")
            bottom = min(item.start[2], item.end[2])
            if bottom >= 0:
                raise ValueError(f"move {index}: cut misses stock")
            sweep = Sweep(op.name, op.tool, item.start[:2], item.end[:2], bottom,
                          item.start[2] if variable else None)
            _safe_cut(sweep, op.target)
            cuts.append(sweep)
        elif not _covered(cuts, item, op.tool):
            raise ValueError(f"move {index}: uncleared travel/access")
        at = item.end
    if running or last_operation is None:
        raise ValueError("incomplete replay sequence")
    prefixes.append((last_operation, len(cuts)))
    return ReplayResult(trace.source_fingerprint, trace.motion_fingerprint,
                        tuple(cuts), tuple(prefixes),
                        tuple(dict.fromkeys(op.target for op in trace.operations)))
