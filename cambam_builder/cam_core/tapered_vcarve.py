"""One bounded variable-depth V groove and its independent section reference.

The finish target is the swept envelope of a 90-degree pointed cone along a
12 mm center spine whose tip depth rises linearly from 1 to 2.5 mm. The generated
cut covers only x=2..10, so both finite ends retain measurable target stock.
"""

from dataclasses import dataclass, field
import hashlib
import math

from . import replay
from .vcarve import Motion, PointedCone


TARGET_SPINE = (0.0, 12.0, 2.0, 1.0, 2.5)
STOCK_BOUNDS = (-2.0, -2.0, 16.0, 6.0)


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

    @property
    def fingerprint(self):
        return hashlib.sha256(repr((self.target_spine, self.cut_spine,
                                    self.tool, self.motions, self.safe_z))
                              .encode("utf-8")).hexdigest()


def generate():
    x0, x1, y = 2.0, 10.0, TARGET_SPINE[2]
    cut = (x0, x1, y, _depth_at(TARGET_SPINE, x0),
           _depth_at(TARGET_SPINE, x1))
    plan = TaperedPlan(TARGET_SPINE, cut, PointedCone(),
                       _motions(cut, 1.0))
    verify(plan)
    return plan


def trace_for(plan):
    tool = replay.ToolProfile("tapered-cone", "pointed_cone",
                              plan.tool.maximum_radius,
                              plan.tool.conical_length)
    target = replay.Target("tapered-groove", STOCK_BOUNDS,
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


def verify(plan):
    if (type(plan) is not TaperedPlan or
            type(plan.tool) is not PointedCone or
            plan.target_spine != TARGET_SPINE or
            len(plan.cut_spine) != 5 or
            plan.motions != _motions(plan.cut_spine, plan.safe_z) or
            plan.safe_z <= 0):
        raise ValueError("invalid bounded variable-depth plan")
    x0, x1, y, d0, d1 = plan.cut_spine
    tx0, tx1, ty, _, _ = plan.target_spine
    if (not tx0 <= x0 < x1 <= tx1 or y != ty or
            d0 != _depth_at(plan.target_spine, x0) or
            d1 != _depth_at(plan.target_spine, x1) or
            d1 > plan.tool.conical_length):
        raise ValueError("variable-depth cut crosses target or tool limit")
    stock = replay.replay(trace_for(plan), expected_source=plan.fingerprint)
    return TaperedResult(plan, stock)


def section_area(spine, depth):
    """Exact ideal section area of a linearly growing disk union.

    Its boundary has two common-tangent sides and two endpoint circle arcs.
    At depths beyond the first tip, the active spine starts where radius is zero.
    """
    if len(spine) != 5 or not 0 <= depth:
        raise ValueError("unsupported section depth")
    x0, x1, _, d0, d1 = spine
    if depth >= d1:
        return 0.0
    slope = (d1 - d0) / (x1 - x0)
    if not 0 < slope < 1:
        raise ValueError("unsupported tapered spine")
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
