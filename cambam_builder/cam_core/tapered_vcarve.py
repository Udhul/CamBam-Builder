"""Straight, increasing-depth pointed-cone grooves and section references."""

from dataclasses import dataclass, field, replace
import hashlib
import math
from numbers import Real

from . import replay
from .vcarve import Motion, PointedCone


TARGET_SPINE = (0.0, 12.0, 2.0, 1.0, 2.5)
STOCK_BOUNDS = (-2.0, -2.0, 16.0, 6.0)
OUTPUT_SETUP = (-10.0, -10.0, 5.0)
OUTPUT_TOOL = "T3"
OUTPUT_RPM = 12000
OUTPUT_FEEDS = {"approach": 120, "entry": 60, "cut": 300, "retract": 300}


@dataclass(frozen=True)
class TaperedRequest:
    """Detached finish target, tool and bounded planning controls."""

    target_spine: tuple = TARGET_SPINE
    stock_bounds: tuple = STOCK_BOUNDS
    stock_bottom: float = -3.0
    tool: PointedCone = field(default_factory=PointedCone)
    cut_interval: tuple = (2.0, 10.0)
    safe_z: float = 1.0


def standalone_request():
    return TaperedRequest()


def _numbers(values, size, label):
    if (type(values) not in (tuple, list) or len(values) != size or
            any(isinstance(value, bool) or not isinstance(value, Real) or
                not math.isfinite(float(value)) for value in values)):
        raise ValueError(f"{label} requires {size} finite numbers")
    return tuple(float(value) for value in values)


def _validated_request(request):
    if type(request) is not TaperedRequest or type(request.tool) is not PointedCone:
        raise ValueError("unsupported variable-depth V request/tool")
    spine = _numbers(request.target_spine, 5, "target spine")
    bounds = _numbers(request.stock_bounds, 4, "stock bounds")
    interval = _numbers(request.cut_interval, 2, "cut interval")
    bottom, safe = _numbers((request.stock_bottom, request.safe_z), 2,
                            "stock bottom and safe Z")
    x0, x1, y, d0, d1 = spine
    xmin, ymin, xmax, ymax = bounds
    if (not x0 < x1 or not 0 < d0 < d1 or
            not 0 < (d1 - d0) / (x1 - x0) < 1 or
            d1 > request.tool.conical_length or bottom > -d1 or safe <= 0 or
            not xmin < xmax or not ymin < ymax or
            x0 - d0 < xmin or x1 + d1 > xmax or
            y - d1 < ymin or y + d1 > ymax or
            not x0 < interval[0] < interval[1] < x1):
        raise ValueError("variable-depth V target, stock, tool or cut interval outside bounded family")
    return TaperedRequest(spine, bounds, bottom, request.tool, interval, safe)


def _depth_at(spine, x):
    x0, x1, _, d0, d1 = spine
    return d0 + (d1 - d0) * (x - x0) / (x1 - x0)


def _motions(spine, safe_z):
    x0, x1, y, d0, d1 = spine
    return (
        Motion("plunge", (x0, y, safe_z), (x0, y, -d0)),
        Motion("cut", (x0, y, -d0), (x1, y, -d1)),
        Motion("retract", (x1, y, -d1), (x1, y, safe_z)),
    )


@dataclass(frozen=True)
class TaperedPlan:
    target_spine: tuple
    cut_spine: tuple
    tool: PointedCone
    motions: tuple
    safe_z: float = 1.0
    stock_bounds: tuple = STOCK_BOUNDS
    stock_bottom: float = -3.0

    @property
    def fingerprint(self):
        values = (self.target_spine, self.cut_spine, self.tool,
                  self.motions, self.safe_z)
        # Preserve the accepted example's fingerprint and pinned output audits.
        if (self.stock_bounds, self.stock_bottom) != (STOCK_BOUNDS, -3.0):
            values += (self.stock_bounds, self.stock_bottom)
        return hashlib.sha256(repr(values)
                              .encode("utf-8")).hexdigest()


def generate(request=None):
    request = standalone_request() if request is None else request
    request = _validated_request(request)
    x0, x1 = request.cut_interval
    y = request.target_spine[2]
    cut = (x0, x1, y, _depth_at(request.target_spine, x0),
           _depth_at(request.target_spine, x1))
    plan = TaperedPlan(request.target_spine, cut, request.tool,
                       _motions(cut, request.safe_z), request.safe_z,
                       request.stock_bounds, request.stock_bottom)
    verify(plan)
    return plan


def trace_for(plan):
    tool = replay.ToolProfile("tapered-cone", "pointed_cone",
                              plan.tool.maximum_radius,
                              plan.tool.conical_length)
    target = replay.Target("tapered-groove", plan.stock_bounds,
                           plan.target_spine[4], cone_spine=plan.target_spine)
    op = replay.Operation("variable-v", tool, target)
    first = plan.motions[0].start
    items = [replay.Event("tool_change", tool.name, first),
             replay.Event("spindle_start", tool.name, first)]
    for move in plan.motions:
        role = "entry" if move.role == "plunge" else move.role
        items.append(replay.Motion(role, tool.name, op.name,
                                   move.start, move.end))
    items.append(replay.Event("spindle_stop", tool.name,
                              plan.motions[-1].end))
    return replay.Trace(plan.fingerprint, "tapered-groove-drawing", first,
                        (op,), tuple(items))


def output_trace(plan):
    """Resolved bounded process trace shared by CamBam and direct output."""
    verify(plan)
    original = trace_for(plan)
    first = plan.motions[0].start
    clearance = (first[0], first[1], OUTPUT_SETUP[2])
    items = [replay.Event("tool_change", OUTPUT_TOOL, OUTPUT_SETUP),
             replay.Event("spindle_start", OUTPUT_TOOL, OUTPUT_SETUP),
             replay.Motion("rapid", OUTPUT_TOOL, "variable-v",
                           OUTPUT_SETUP, clearance),
             replay.Motion("approach", OUTPUT_TOOL, "variable-v", clearance,
                           first, OUTPUT_FEEDS["approach"])]
    for move in plan.motions:
        role = "entry" if move.role == "plunge" else move.role
        items.append(replay.Motion(role, OUTPUT_TOOL, "variable-v",
                                   move.start, move.end,
                                   OUTPUT_FEEDS.get(role, 0)))
    items.extend((replay.Motion("rapid", OUTPUT_TOOL, "variable-v",
                                items[-1].end, OUTPUT_SETUP),
                  replay.Event("spindle_stop", OUTPUT_TOOL, OUTPUT_SETUP)))
    operation = replace(original.operations[0],
                        tool=replace(original.operations[0].tool,
                                     name=OUTPUT_TOOL))
    return replay.Trace(plan.fingerprint, original.frame, OUTPUT_SETUP,
                        (operation,), tuple(items))


def verify(plan):
    if type(plan) is not TaperedPlan:
        raise ValueError("invalid variable-depth plan")
    request = _validated_request(TaperedRequest(
        plan.target_spine, plan.stock_bounds, plan.stock_bottom, plan.tool,
        (plan.cut_spine[0], plan.cut_spine[1]) if
        type(plan.cut_spine) is tuple and len(plan.cut_spine) == 5 else (),
        plan.safe_z))
    if (plan.target_spine != request.target_spine or
            plan.stock_bounds != request.stock_bounds or
            plan.stock_bottom != request.stock_bottom or
            plan.motions != _motions(plan.cut_spine, plan.safe_z)):
        raise ValueError("invalid bounded variable-depth plan")
    if plan.cut_spine != (request.cut_interval[0], request.cut_interval[1],
                               request.target_spine[2],
                               _depth_at(request.target_spine, request.cut_interval[0]),
                               _depth_at(request.target_spine, request.cut_interval[1])):
        raise ValueError("variable-depth cut crosses target or tool limit")
    stock = replay.replay(trace_for(plan), expected_source=plan.fingerprint)
    return TaperedResult(plan, stock)


def section_area(spine, depth):
    """Exact ideal section area of a linearly growing disk union.

    Its boundary has two common-tangent sides and two endpoint circle arcs.
    At depths beyond the first tip, the active spine starts where radius is zero.
    """
    spine = _numbers(spine, 5, "section spine")
    if (isinstance(depth, bool) or not isinstance(depth, Real) or
            not math.isfinite(float(depth)) or not 0 <= depth):
        raise ValueError("unsupported section depth")
    x0, x1, _, d0, d1 = spine
    if not x0 < x1 or not 0 < d0 < d1:
        raise ValueError("unsupported tapered spine")
    slope = (d1 - d0) / (x1 - x0)
    if not 0 < slope < 1:
        raise ValueError("unsupported tapered spine")
    if depth >= d1:
        return 0.0
    r0, r1 = max(0.0, d0 - depth), d1 - depth
    length = (r1 - r0) / slope
    cosine = math.sqrt(1 - slope * slope)
    alpha = math.asin(slope)
    return (length * cosine * (r0 + r1) +
            math.pi * (r0 * r0 + r1 * r1) / 2 +
            alpha * (r1 * r1 - r0 * r0))


@dataclass(frozen=True)
class TaperedResult:
    plan: TaperedPlan
    stock: replay.ReplayResult = field(repr=False)
    evidence_class: str = "conditional_analytic_tapered_cone"

    @property
    def completion(self):
        return "partial_target_completion"

    def residual_area(self, depth):
        if not 0 <= depth <= self.plan.target_spine[4]:
            raise ValueError("section outside target depth")
        target = section_area(self.plan.target_spine, depth)
        cut = section_area(self.plan.cut_spine, depth)
        return max(0.0, target - cut)

    def residual_contains(self, x, y, depth):
        return self.stock.residual_contains("tapered-groove", x, y, depth)
