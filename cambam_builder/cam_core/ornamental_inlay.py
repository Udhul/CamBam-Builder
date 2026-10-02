"""Straight-wall ornamental inlays from two independently replayed part stocks.

All depths are positive into a blank. Assembly is the proper rigid flip
``(x, y, -u) -> (-x, y, u - insertion)`` in physical XYZ coordinates.
Only level cylindrical sweeps are supported. No nominal target is treated as
machined stock, and a finishing envelope is not a verified facing toolpath.
"""
from dataclasses import dataclass
import hashlib
import math

from shapely.affinity import scale, translate
from shapely.geometry import Polygon

from . import replay
from .polygon_rest import _cut_polygon


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            (positive and value <= 0)):
        raise ValueError(f"finite {'positive ' if positive else ''}{name} required")
    return float(value)


def _polygon(value, name):
    if (type(value) is not Polygon or value.has_z or value.is_empty or not value.is_valid or
            not math.isfinite(value.area) or value.area <= 0):
        raise ValueError(f"valid nonempty {name} Polygon required")


def _flip(shape):
    return scale(shape, xfact=-1, yfact=1, origin=(0, 0))


def _parts(shape):
    return (shape,) if shape.geom_type == "Polygon" else tuple(shape.geoms)


@dataclass(frozen=True)
class Design:
    """Tool-independent, parallel straight-wall paired targets in millimetres.

    Positive fit erodes the retained plug by a lateral disk offset; negative
    fit requests interference, bounded by ``fit_limit_mm``. The polygonal GEOS
    offset (128 segments/quadrant) is the declared nominal geometry, not an
    exact curved surface. Minimum web uses an erosion/topology test, not a
    material-strength prediction. Stock XY is in assembly coordinates.
    """
    revision: str
    motif: Polygon
    stock_xy: Polygon
    seating_mm: float
    bottom_gap_mm: float
    surface_gap_mm: float
    backing_mm: float
    receiver_thickness_mm: float
    side_fit_mm: float
    fit_limit_mm: float
    minimum_web_mm: float
    edge_access_mm: float

    def __post_init__(self):
        if type(self.revision) is not str or not self.revision:
            raise ValueError("design revision required")
        _polygon(self.motif, "motif")
        _polygon(self.stock_xy, "stock")
        for name in ("seating_mm", "backing_mm", "receiver_thickness_mm",
                     "fit_limit_mm", "minimum_web_mm", "edge_access_mm"):
            object.__setattr__(self, name, _number(getattr(self, name), name,
                                                  positive=True))
        for name in ("bottom_gap_mm", "surface_gap_mm", "side_fit_mm"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if (min(self.bottom_gap_mm, self.surface_gap_mm) < 0 or
                abs(self.side_fit_mm) > self.fit_limit_mm or
                self.receiver_depth >= self.receiver_thickness_mm):
            raise ValueError("invalid independent fit, gap or stock allowances")
        for shape in (self.motif, self.plug_xy):
            _polygon(shape, "retained motif")
            core = shape.buffer(-self.minimum_web_mm / 2, quad_segs=128)
            if (core.is_empty or core.geom_type != "Polygon" or
                    len(core.interiors) != len(self.motif.interiors) or
                    len(shape.interiors) != len(self.motif.interiors)):
                raise ValueError("fragile bridge or changed motif topology")
            if (not self.stock_xy.contains(shape) or
                    self.stock_xy.boundary.distance(shape) <= 0):
                raise ValueError("motif must be strictly inside bounded stock")

    @property
    def receiver_depth(self):
        return self.seating_mm + self.bottom_gap_mm

    @property
    def plug_depth(self):
        return self.seating_mm + self.surface_gap_mm

    @property
    def plug_xy(self):
        return (self.motif if self.side_fit_mm == 0 else
                self.motif.buffer(-self.side_fit_mm, quad_segs=128))

    @property
    def fingerprint(self):
        values = tuple((name, value.wkb_hex if type(value) is Polygon else value)
                       for name, value in vars(self).items())
        return hashlib.sha256(repr(("straight-inlay-v1", values)).encode()).hexdigest()

    def source(self, side):
        self.depth(side)
        return f"{self.fingerprint}:{side}"

    def frame(self, side):
        self.depth(side)
        return f"inlay-{side}-{'flipped-x' if side == 'plug' else 'assembly'}"

    def depth(self, side):
        if side not in ("receiver", "plug"):
            raise ValueError("receiver or plug side required")
        return self.receiver_depth if side == "receiver" else self.plug_depth

    def targets(self, side):
        """Finite removal Regions; hole pockets are separate plug components.

        Plug clearing may run beyond the blank into the explicitly bounded
        edge-access envelope. This is not fixture/body clearance permission.
        """
        depth = self.depth(side)
        shape = self.motif if side == "receiver" else _flip(
            self.stock_xy.buffer(self.edge_access_mm, join_style="mitre")
            .difference(self.plug_xy))
        return tuple(replay.Target(f"{side}-{i}", tuple(p.bounds), depth,
                     region_shell=tuple(p.exterior.coords)[:-1],
                     region_holes=tuple(tuple(r.coords)[:-1] for r in p.interiors))
                     for i, p in enumerate(_parts(shape)))


def _stock(design, side, trace, expected_motion):
    if (type(trace) is not replay.Trace or
            trace.motion_fingerprint != expected_motion or
            trace.frame != design.frame(side)):
        raise ValueError("stale motion/tool or incorrect part frame")
    targets = design.targets(side)
    if any(op.target not in targets or op.tool.kind != "cylinder"
           for op in trace.operations):
        raise ValueError("part target or tool differs from straight-wall design")
    result = replay.replay(trace, expected_source=design.source(side))
    if not result.cuts or any(c.bottom_start is not None or c.path_error_mm
                              for c in result.cuts):
        raise ValueError("only level linear cylindrical part stock is supported")
    return result.cuts


def _removed(cuts, depth):
    # Existing cylinder enclosure owner: true disks lie between these polygons.
    return (_cut_polygon(cuts, depth, radial_error=-1e-7),
            _cut_polygon(cuts, depth, radial_error=1e-7))


@dataclass(frozen=True)
class Verification:
    design_fingerprint: str
    receiver_motion: str
    plug_motion: str
    registration_xy_mm: tuple
    status: str
    collision_volume_lower_mm3: float
    collision_volume_upper_mm3: float
    minimum_bottom_gap_mm: float
    minimum_surface_gap_mm: float
    sections: tuple


def verify_pair(design, receiver, plug, *, expected_receiver_motion,
                expected_plug_motion, registration_xy=(0.0, 0.0)):
    """Verify rigid insertion from first contact to seating, using actual stock.

    Stock from top-down, level cylinder cuts grows monotonically with depth.
    Thus at every receiver depth the seated plug contains every earlier plug
    cross-section. Checking all finite seated slabs proves the full insertion
    at fixed XY registration. Positive possible collision is unresolved, never
    accepted by an area tolerance. Negative-fit interference cannot pass.
    """
    if type(design) is not Design:
        raise ValueError("ornamental Design required")
    if type(registration_xy) is not tuple or len(registration_xy) != 2:
        raise ValueError("finite assembly XY registration required")
    dx, dy = (_number(v, "registration") for v in registration_xy)
    rc = _stock(design, "receiver", receiver, expected_receiver_motion)
    pc = _stock(design, "plug", plug, expected_plug_motion)
    e = design.seating_mm
    levels = sorted({0.0, e} | {-c.bottom for c in rc if 0 < -c.bottom < e} |
                    {e + c.bottom for c in pc if 0 < e + c.bottom < e})
    rows = []
    lower = upper = 0.0
    for lo, hi in zip(levels, levels[1:]):
        z = (lo + hi) / 2
        ri, ro = _removed(rc, z)
        pi, po = _removed(pc, e - z)
        # Remaining plug is blank minus removal, with enclosure order reversed.
        inner = translate(_flip(_flip(design.stock_xy).difference(po)), dx, dy)
        outer = translate(_flip(_flip(design.stock_xy).difference(pi)), dx, dy)
        # Receiver stock occupies its finite blank only.
        definite = inner.intersection(design.stock_xy.difference(ro))
        possible = outer.intersection(design.stock_xy.difference(ri))
        lower += definite.area * (hi - lo)
        upper += possible.area * (hi - lo)
        rows.append((lo, hi, definite.area, possible.area))
    # The actual cleared floor under the entire retained plug controls glue gap.
    # Insist on full requested cavity depth, including its floor corners.
    deepest_inner, _ = _removed(rc, design.receiver_depth)
    pi, _ = _removed(pc, 0)
    tip_outer = translate(_flip(_flip(design.stock_xy).difference(pi)), dx, dy)
    bottom_ok = deepest_inner.covers(tip_outer)
    # A shallow exterior clearance may seat but leave a premature shoulder
    # above the receiver. The deepest gap section encloses all shallower ones.
    gap_cut, _ = _removed(pc, design.plug_depth)
    gap_outer = translate(_flip(_flip(design.stock_xy).difference(gap_cut)), dx, dy)
    top_free, _ = _removed(rc, 0)
    surface_ok = gap_outer.intersection(design.stock_xy.difference(top_free)).is_empty
    status = "pass" if upper == 0 and bottom_ok and surface_ok else (
        "collision" if lower > 0 else "unresolved")
    return Verification(design.fingerprint, receiver.motion_fingerprint,
                        plug.motion_fingerprint, (dx, dy), status, lower, upper,
                        design.bottom_gap_mm if bottom_ok else 0.0,
                        design.surface_gap_mm if surface_ok else 0.0, tuple(rows))


def finish_envelope(design, receiver, plug, *, expected_receiver_motion,
                    expected_plug_motion, removal_mm, minimum_retained_mm,
                    assembly_token, cure_token, renewed_setup_token,
                    registration_xy=(0.0, 0.0)):
    """Audit a declared full-plane removal envelope, not cutting execution.

    Remove everything above any plane between receiver depths 0 and removal_mm.
    Report worst visible plug error and retained thickness over that range.
    Tokens are caller process declarations; they are not observation/telemetry.
    """
    removal = _number(removal_mm, "finish removal")
    minimum = _number(minimum_retained_mm, "retained thickness", positive=True)
    if (removal < 0 or design.seating_mm - removal < minimum or
            any(type(t) is not str or not t.strip() for t in
                (assembly_token, cure_token, renewed_setup_token))):
        raise ValueError("finish thickness or declared assembly/cure/setup missing")
    checked = verify_pair(design, receiver, plug,
        expected_receiver_motion=expected_receiver_motion,
        expected_plug_motion=expected_plug_motion,
        registration_xy=registration_xy)
    if checked.status != "pass":
        raise ValueError("verified collision-free paired stock required before finishing")
    cuts = _stock(design, "plug", plug, expected_plug_motion)
    low, high = design.seating_mm - removal, design.seating_mm
    levels = sorted({low, high} | {-c.bottom for c in cuts
                                 if low < -c.bottom < high})
    samples = set(levels) | {(a + b) / 2 for a, b in zip(levels, levels[1:])}
    missing = excess = 0.0
    topology_ok = True
    for u in samples:
        inner_cut, outer_cut = _removed(cuts, u)
        inner = translate(_flip(_flip(design.stock_xy).difference(outer_cut)),
                          *registration_xy)
        outer = translate(_flip(_flip(design.stock_xy).difference(inner_cut)),
                          *registration_xy)
        missing = max(missing, design.plug_xy.difference(inner).area)
        excess = max(excess, outer.difference(design.plug_xy).area)
        topology_ok &= all(s.geom_type == "Polygon" and not s.is_empty and
                           len(s.interiors) == len(design.plug_xy.interiors)
                           for s in (inner, outer))
    return {"evidence": "declared_full_plane_removal_envelope",
            "design_fingerprint": design.fingerprint,
            "receiver_motion": expected_receiver_motion,
            "plug_motion": expected_plug_motion,
            "process_declarations": (assembly_token, cure_token, renewed_setup_token),
            "plane_depth_interval_mm": (0.0, removal),
            "registration_xy_mm": checked.registration_xy_mm,
            "nominal_core_minimum_retained_mm": design.seating_mm - removal,
            "minimum_receiver_floor_mm": (design.receiver_thickness_mm -
                                           design.receiver_depth),
            "backing_removed_mm": design.backing_mm,
            "motif_missing_upper_mm2": missing,
            "motif_excess_upper_mm2": excess,
            "section_enclosure_topology_matches": topology_ok}
