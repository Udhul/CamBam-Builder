"""Profile-aware ornamental pairs using existing fixed-V stock enclosures.

Targets describe surfaces, independently of cutters. Assembly uses only verified
component sweeps and monotone endpoint slabs, never nominal removal as stock.
"""
from dataclasses import dataclass
import hashlib
import math

from shapely.affinity import translate
from shapely.ops import unary_union

from . import ornamental_inlay as straight, v_region


@dataclass(frozen=True)
class Design:
    """Paired offset surfaces with side fit referenced at the plug tip.

    Receiver opening is the motif. The retained plug tip is the motif eroded
    by seating*tan(angle/2)+side_fit; its clearing Regions shrink with depth.
    Thus the retained plug grows towards its backing. Offsets around corners
    need not be inverse operations: no fit follows from the nominal geometry.
    Overtravel is explicit allowed removal beyond the minimum requested gaps.
    """
    allowances: straight.Design
    angle_degrees: float
    receiver_overtravel_mm: float = 0.0
    plug_overtravel_mm: float = 0.0

    def __post_init__(self):
        if type(self.allowances) is not straight.Design:
            raise ValueError("straight-wall allowance Design required")
        for name in ("angle_degrees", "receiver_overtravel_mm", "plug_overtravel_mm"):
            object.__setattr__(self, name, straight._number(getattr(self, name), name))
        q = self.allowances
        if (not 0 < self.angle_degrees < 180 or
                min(self.receiver_overtravel_mm, self.plug_overtravel_mm) < 0 or
                self.depth("receiver") >= q.receiver_thickness_mm or
                self.plug_overtravel_mm >= q.backing_mm):
            raise ValueError("invalid taper or overtravel stock allowance")
        # Smallest sections protect the bridge and retained plug topology.
        receiver = self.targets("receiver")[0].section(self.depth("receiver"))
        for shape in (receiver, self.plug_tip):
            straight._polygon(shape, "tapered motif")
            core = shape.buffer(-q.minimum_web_mm / 2, quad_segs=128)
            if (core.is_empty or core.geom_type != "Polygon" or
                    len(core.interiors) != len(q.motif.interiors) or
                    len(shape.interiors) != len(q.motif.interiors)):
                raise ValueError("taper collapses bridge or changes motif topology")
        # Even the largest retained nominal section must remain inside the blank.
        outer = q.stock_xy.difference(straight._flip(unary_union([
            t.section(self.depth("plug")) for t in self.targets("plug")])))
        if not q.stock_xy.boundary.intersection(outer).is_empty:
            raise ValueError("edge access leaves a retained stock rim")

    @property
    def tangent(self):
        return math.tan(math.radians(self.angle_degrees) / 2)

    @property
    def plug_tip(self):
        q = self.allowances
        return q.motif.buffer(-(q.seating_mm * self.tangent + q.side_fit_mm),
                              quad_segs=128)

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(("tapered-inlay-v1", self.allowances.fingerprint,
            self.angle_degrees, self.receiver_overtravel_mm,
            self.plug_overtravel_mm)).encode()).hexdigest()

    def depth(self, side):
        return self.allowances.depth(side) + (self.receiver_overtravel_mm
            if side == "receiver" else self.plug_overtravel_mm)

    def source(self, side):
        self.depth(side)
        return f"{self.fingerprint}:{side}"

    def frame(self, side):
        return self.allowances.frame(side)

    def targets(self, side):
        q = self.allowances
        depth = self.depth(side)
        shape = q.motif if side == "receiver" else straight._flip(
            q.stock_xy.buffer(q.edge_access_mm, join_style="mitre")
            .difference(self.plug_tip))
        return tuple(v_region.VTarget(self.source(side), p, p, depth,
            design_angle_degrees=self.angle_degrees, frame=self.frame(side))
            for p in straight._parts(shape))


@dataclass(frozen=True)
class PartStock:
    """Independent component compositions; missing clearing earns no credit."""
    components: tuple

    def __post_init__(self):
        if (type(self.components) is not tuple or not self.components or
                any(type(c) is not v_region.VComposition for c in self.components)):
            raise ValueError("nonempty tuple of V stock compositions required")

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(("inlay-part-stock-v1",
            tuple(c.fingerprint for c in self.components))).encode()).hexdigest()


def _checked(design, side, stock, expected):
    if type(stock) is not PartStock or stock.fingerprint != expected:
        raise ValueError("stale part stock/tool/motion binding")
    targets = {t.fingerprint for t in design.targets(side)}
    for component in stock.components:
        if component.target.fingerprint not in targets:
            raise ValueError("part target, revision or frame differs from design")
        v_region.VComposition(component.target, component.stages)
    return stock


def section(stock, depth):
    """Actual removed-section inner/outer enclosures, before finite-blank clipping."""
    if type(stock) is not PartStock:
        raise ValueError("PartStock required")
    evidence = [v_region.section_evidence(c, depth) for c in stock.components]
    return (unary_union([e.known_free_inner for e in evidence]),
            unary_union([e.known_free_outer for e in evidence]))


@dataclass(frozen=True)
class Verification:
    design_fingerprint: str
    receiver_stock: str
    plug_stock: str
    registration_xy_mm: tuple
    status: str
    collision_volume_lower_mm3: float
    collision_volume_upper_mm3: float
    minimum_bottom_gap_mm: float
    minimum_surface_gap_mm: float
    sections: tuple


def verify_pair(design, receiver, plug, *, expected_receiver_stock,
                expected_plug_stock, registration_xy=(0.0, 0.0), slabs=16):
    """Bound seated collision and every earlier fixed-XY insertion pose.

    All supported cutter radii grow with height, so removal shrinks with depth
    and retained plug grows with machining depth. For each seated slab, the
    largest plug and smallest cavity bound every intermediate section. Earlier
    insertion has a smaller plug at the same receiver depth. Endpoint bounds
    also enclose continuously varying Z sweeps and cutter-depth discontinuities.
    More slabs may resolve conservative overlap; no area tolerance grants fit.
    """
    if type(design) is not Design:
        raise ValueError("tapered Design required")
    if type(slabs) is not int or slabs <= 0:
        raise ValueError("positive integer slab count required")
    if type(registration_xy) is not tuple or len(registration_xy) != 2:
        raise ValueError("finite assembly XY registration required")
    dx, dy = (straight._number(v, "registration") for v in registration_xy)
    _checked(design, "receiver", receiver, expected_receiver_stock)
    _checked(design, "plug", plug, expected_plug_stock)
    q = design.allowances
    e = q.seating_mm
    levels = [e * i / slabs for i in range(slabs)] + [e]
    rs = [section(receiver, z) for z in levels]
    ps = [section(plug, e-z) for z in levels]

    def retained(removed):
        return translate(straight._flip(straight._flip(q.stock_xy)
                                       .difference(removed)), dx, dy)

    rows = []
    lower = upper = 0.0
    for i, (lo, hi) in enumerate(zip(levels, levels[1:])):
        # Plug at u=e-hi is smallest; cavity at z=lo is largest.
        definite = retained(ps[i+1][1]).intersection(q.stock_xy.difference(rs[i][1]))
        possible = retained(ps[i][0]).intersection(q.stock_xy.difference(rs[i+1][0]))
        lower += definite.area * (hi-lo)
        upper += possible.area * (hi-lo)
        rows.append((lo, hi, definite.area, possible.area))
    floor, _ = section(receiver, q.receiver_depth)
    bottom_ok = floor.covers(retained(ps[-1][0]))
    shoulder, _ = section(plug, q.plug_depth)
    surface_ok = retained(shoulder).intersection(q.stock_xy.difference(rs[0][0])).is_empty
    status = "pass" if upper == 0 and bottom_ok and surface_ok else (
        "collision" if lower > 0 else "unresolved")
    return Verification(design.fingerprint, receiver.fingerprint, plug.fingerprint,
        (dx, dy), status, lower, upper, q.bottom_gap_mm if bottom_ok else 0.0,
        q.surface_gap_mm if surface_ok else 0.0, tuple(rows))
