"""Analytic pointed-cone paths for one rectangular, flat-depth-capped slot.

This is stock geometry in millimetres, with the stock top at Z=0.  The target
depth at (x, y) is min(distance to the four slot sides, depth_cap).  Only a
90-degree pointed cone and horizontal straight passes are supported.  Entry,
retract and above-stock links are explicit; fixtures and machine dynamics are
outside this geometric certificate.
"""

from dataclasses import dataclass, field
import hashlib
import math

from . import replay as stock_replay


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite numeric value required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite numeric value required")
    return value


@dataclass(frozen=True)
class Slot:
    length: float = 12.0
    width: float = 4.0
    depth_cap: float = 2.0
    frame: str = "slot-drawing"

    def __post_init__(self):
        for name in ("length", "width", "depth_cap"):
            object.__setattr__(self, name, _finite(getattr(self, name)))
        if (self.length <= self.width or self.width <= 0 or
                not 0 < self.depth_cap <= self.width / 2 or
                not isinstance(self.frame, str) or not self.frame.strip()):
            raise ValueError("unsupported rectangular slot")

    def target_depth(self, x, y):
        x, y = _finite(x), _finite(y)
        return max(0.0, min(x, self.length - x, y,
                            self.width - y, self.depth_cap))

    def section_area(self, depth):
        depth = _finite(depth)
        if not 0 <= depth <= self.depth_cap:
            raise ValueError("section is outside target depth")
        return (self.length - 2 * depth) * (self.width - 2 * depth)


@dataclass(frozen=True)
class PointedCone:
    maximum_radius: float = 3.0
    conical_length: float = 3.0

    def __post_init__(self):
        for name in ("maximum_radius", "conical_length"):
            object.__setattr__(self, name, _finite(getattr(self, name)))
        if (self.maximum_radius <= 0 or
                self.conical_length != self.maximum_radius):
            raise ValueError("90-degree pointed cone requires radius = conical length")

    def radius_at_height(self, height):
        height = _finite(height)
        if not 0 <= height <= self.conical_length:
            raise ValueError("height exceeds conical cutting length")
        return height


@dataclass(frozen=True)
class Pass:
    x0: float
    x1: float
    y: float
    tip_depth: float

    def __post_init__(self):
        for name in ("x0", "x1", "y", "tip_depth"):
            object.__setattr__(self, name, _finite(getattr(self, name)))
        if self.x0 >= self.x1 or self.tip_depth <= 0:
            raise ValueError("pass needs a finite, positive-depth segment")


@dataclass(frozen=True)
class Motion:
    role: str
    start: tuple
    end: tuple

    def __post_init__(self):
        if self.role not in ("rapid", "plunge", "cut", "retract"):
            raise ValueError("unsupported slot motion role")
        for name in ("start", "end"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError("motion requires XYZ tuples")
            object.__setattr__(self, name, tuple(_finite(v) for v in value))
        if self.start == self.end:
            raise ValueError("zero-length motion")


def _motions(passes, safe_z):
    moves = []
    prior = None
    for path in passes:
        start = (path.x0, path.y, safe_z)
        if prior is not None and prior != start:
            moves.append(Motion("rapid", prior, start))
        low_start = (path.x0, path.y, -path.tip_depth)
        low_end = (path.x1, path.y, -path.tip_depth)
        end = (path.x1, path.y, safe_z)
        moves.extend((Motion("plunge", start, low_start),
                      Motion("cut", low_start, low_end),
                      Motion("retract", low_end, end)))
        prior = end
    return tuple(moves)


@dataclass(frozen=True)
class SlotPlan:
    slot: Slot
    tool: PointedCone
    passes: tuple
    motions: tuple
    safe_z: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "safe_z", _finite(self.safe_z))
        if (type(self.slot) is not Slot or type(self.tool) is not PointedCone or
                not isinstance(self.passes, tuple) or
                not isinstance(self.motions, tuple) or not self.passes or
                self.safe_z <= 0):
            raise ValueError("invalid slot plan")

    @property
    def fingerprint(self):
        values = (self.slot, self.tool, self.passes, self.motions, self.safe_z)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def replay_trace(plan, *, frame=None, offset=(0, 0)):
    """Adapt the bounded cone plan without changing its geometry oracle."""
    if len(offset) != 2 or any(not math.isfinite(float(v)) for v in offset):
        raise ValueError("finite XY placement required")
    ox, oy = offset
    tool = stock_replay.ToolProfile("cone", "pointed_cone",
                                    plan.tool.maximum_radius,
                                    plan.tool.conical_length)
    target = stock_replay.Target(
        "cone-slot", (ox, oy, ox + plan.slot.length, oy + plan.slot.width),
        plan.slot.depth_cap, inset_per_depth=True)
    operation = stock_replay.Operation("slot", tool, target)
    start = plan.motions[0].start
    position = (start[0] + ox, start[1] + oy, start[2])
    items = [stock_replay.Event("tool_change", tool.name, position),
             stock_replay.Event("spindle_start", tool.name, position)]
    for move in plan.motions:
        role = "entry" if move.role == "plunge" else move.role
        a = (move.start[0] + ox, move.start[1] + oy, move.start[2])
        b = (move.end[0] + ox, move.end[1] + oy, move.end[2])
        items.append(stock_replay.Motion(role, tool.name, "slot", a, b))
    items.append(stock_replay.Event("spindle_stop", tool.name, items[-1].end))
    return stock_replay.Trace(plan.fingerprint,
                              plan.slot.frame if frame is None else frame,
                              position, (operation,), tuple(items))


def generate_slot(slot=Slot(), tool=PointedCone()):
    """Generate the one full-depth V pass or three capped-depth clearing passes."""
    if type(slot) is not Slot or type(tool) is not PointedCone:
        raise ValueError("expected bounded slot and pointed cone")
    depth = slot.depth_cap
    if depth > tool.conical_length:
        raise ValueError("candidate penetration exceeds cone cutting radius/length")
    if depth == slot.width / 2:
        ys = (slot.width / 2,)
    else:
        # The accepted 4 mm opening with a 1 mm cap has a 2 mm top diameter.
        # Three rows cover its width at the surface while retaining a finite
        # pointed-tip floor residual between the rows.
        if slot.width != 4 or depth != 1:
            raise ValueError("capped generation supports the 4 mm / 1 mm case")
        ys = (1.0, 2.0, 3.0)
    paths = tuple(Pass(depth, slot.length - depth, y, depth) for y in ys)
    plan = SlotPlan(slot, tool, paths, _motions(paths, 1.0))
    verify_slot(plan)
    return plan


def _check_slot(plan):
    if type(plan) is not SlotPlan:
        raise ValueError("expected a slot plan")
    if (any(type(p) is not Pass for p in plan.passes) or
            any(type(m) is not Motion for m in plan.motions) or
            plan.motions != _motions(plan.passes, plan.safe_z)):
        raise ValueError("motions must be the complete ordered pass traversal")
    slot, tool = plan.slot, plan.tool
    first = plan.passes[0]
    for path in plan.passes:
        d = path.tip_depth
        if ((d, path.x0, path.x1) !=
                (first.tip_depth, first.x0, first.x1)):
            raise ValueError("slot verifier requires equal-depth, equal-span passes")
        if (d > slot.depth_cap or d > tool.conical_length or
                path.x0 < d or path.x1 > slot.length - d or
                path.y < d or path.y > slot.width - d):
            raise ValueError("cone sweep crosses target or tool limit")
        # At every stock depth t in [0,d], the cone radius is d-t.
        # The target section is [t,L-t] x [t,W-t].  The four endpoint
        # inequalities above are equivalent to containment for all t.
        # Vertical plunges/retracts are subsets of their deep endpoint disks;
        # rapid links have the tip strictly above the stock top.


def verify_slot(plan):
    """Check every swept cone section, not just the top silhouette or endpoints."""
    return SlotResult(plan)


def _disk_reach(radius, dy):
    return math.sqrt(max(0.0, radius * radius - dy * dy))


def _circle_primitive(radius, offset):
    offset = max(-radius, min(radius, offset))
    return (offset * _disk_reach(radius, offset) +
            radius * radius * math.asin(offset / radius)) / 2


@dataclass(frozen=True)
class SlotResult:
    plan: SlotPlan
    evidence_class: str = field(default="conditional_analytic_cone_slot", init=False)
    replay_result: stock_replay.ReplayResult = field(init=False, repr=False)

    def __post_init__(self):
        _check_slot(self.plan)
        trace = replay_trace(self.plan)
        object.__setattr__(self, "replay_result", stock_replay.replay(
            trace, expected_source=self.plan.fingerprint))

    @property
    def completion(self):
        # A positive section rest proves incompleteness. Zero measured sections
        # alone cannot prove complete volume removal.
        if (self.residual_area(0) > 1e-9 or
                self.residual_area(self.plan.slot.depth_cap) > 1e-9):
            return "partial_target_completion"
        return "undetermined"

    def _section(self, depth):
        depth = _finite(depth)
        if not 0 <= depth <= self.plan.slot.depth_cap:
            raise ValueError("section is outside target depth")
        return depth, tuple(Pass(s.a[0], s.b[0], s.a[1], -s.bottom)
                            for s in self.replay_result.cuts
                            if s.a != s.b and -s.bottom >= depth)

    def removed_contains(self, x, y, depth):
        x, y = _finite(x), _finite(y)
        depth, paths = self._section(depth)
        for path in paths:
            radius = path.tip_depth - depth
            dx = max(path.x0 - x, 0, x - path.x1)
            if dx * dx + (y - path.y) ** 2 <= radius * radius:
                return True
        return False

    def residual_contains(self, x, y, depth):
        x, y = _finite(x), _finite(y)
        depth, _ = self._section(depth)
        return (depth <= x <= self.plan.slot.length - depth and
                depth <= y <= self.plan.slot.width - depth and
                not self.removed_contains(x, y, depth))

    def residual_area(self, depth):
        """Analytic section area for generated equal-depth, equal-span passes."""
        depth, paths = self._section(depth)
        slot = self.plan.slot
        target = slot.section_area(depth)
        if not paths:
            return target
        first = paths[0]
        if any((p.tip_depth, p.x0, p.x1) !=
               (first.tip_depth, first.x0, first.x1) for p in paths):
            raise ValueError("area requires equal-depth, equal-span passes")
        radius = first.tip_depth - depth
        if radius == 0:
            return target  # Zero-width centerlines have zero area.
        centers = sorted({p.y for p in paths})
        removed = 0.0
        bottom, top = depth, slot.width - depth
        for index, center in enumerate(centers):
            left = bottom if index == 0 else (centers[index - 1] + center) / 2
            right = top if index == len(centers) - 1 else (center + centers[index + 1]) / 2
            low, high = max(left, center - radius), min(right, center + radius)
            if low < high:
                removed += (first.x1 - first.x0) * (high - low)
                removed += 2 * (_circle_primitive(radius, high - center) -
                                _circle_primitive(radius, low - center))
        return max(0.0, target - removed)

    def residual_volume_bounds(self, steps=4096):
        """Bound rest volume by nested target and cone sections.

        This geometric enclosure uses floating section areas, so its final digits
        are subject to ordinary floating roundoff; it is not interval arithmetic.
        Increasing steps tightens the geometric part of the enclosure.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        slot = self.plan.slot
        dz = slot.depth_cap / steps
        lower = upper = 0.0
        area_a = slot.section_area(0)
        rest_a = self.residual_area(0)
        for index in range(steps):
            area_b = slot.section_area((index + 1) * dz)
            rest_b = self.residual_area((index + 1) * dz)
            target_drop = max(0.0, area_a - area_b)
            removal_drop = max(0.0, (area_a - rest_a) - (area_b - rest_b))
            lower += dz * max(0.0, rest_a - target_drop,
                              rest_b - removal_drop)
            upper += dz * min(area_a, rest_a + removal_drop,
                              rest_b + target_drop)
            area_a, rest_a = area_b, rest_b
        return max(0.0, lower - 1e-10), upper + 1e-10
