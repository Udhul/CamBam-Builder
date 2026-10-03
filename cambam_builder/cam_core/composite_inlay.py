"""Bounded assembled stock and executable flat facing of ornamental inlays.

Section depths use the receiver top. Facing motion instead uses the assembled
backing top as Z=0. Process tokens declare assembly/cure/setup, not observations.
"""
from dataclasses import dataclass
import hashlib
import math

from shapely.affinity import translate
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

from . import ornamental_inlay as straight, tapered_inlay as tapered, replay


def _hash(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


@dataclass(frozen=True)
class Assembly:
    design: object
    receiver: object
    plug: object
    expected_receiver: str
    expected_plug: str
    registration_xy: tuple = (0.0, 0.0)
    slabs: int = 16

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.design) is straight.Design:
            result = straight.verify_pair(self.design, self.receiver, self.plug,
                expected_receiver_motion=self.expected_receiver,
                expected_plug_motion=self.expected_plug,
                registration_xy=self.registration_xy)
        elif type(self.design) is tapered.Design:
            result = tapered.verify_pair(self.design, self.receiver, self.plug,
                expected_receiver_stock=self.expected_receiver,
                expected_plug_stock=self.expected_plug,
                registration_xy=self.registration_xy, slabs=self.slabs)
        else:
            raise ValueError("supported ornamental design required")
        if result.status != "pass":
            raise ValueError("verified collision-free paired stock required")
        return result

    @property
    def allowances(self):
        return (self.design if type(self.design) is straight.Design else
                self.design.allowances)

    @property
    def top_mm(self):
        return self.allowances.surface_gap_mm + self.allowances.backing_mm

    @property
    def footprint(self):
        q = self.allowances
        return q.stock_xy.union(translate(q.stock_xy, *self.registration_xy))

    @property
    def fingerprint(self):
        return _hash(("composite-inlay-v1", self.design.fingerprint,
            self.expected_receiver, self.expected_plug, self.registration_xy,
            self.slabs))

    def section(self, depth):
        """Return receiver and plug retained (inner, outer) section bounds."""
        depth = straight._number(depth, "section depth")
        q = self.allowances

        def body(side, u, thickness, blank):
            if u < 0 or u > thickness:
                return GeometryCollection(), GeometryCollection()
            stock = self.receiver if side == "receiver" else self.plug
            if type(self.design) is straight.Design:
                cuts = straight._stock(self.design, side, stock,
                    self.expected_receiver if side == "receiver" else self.expected_plug)
                inner, outer = straight._removed(cuts, u)
            else:
                inner, outer = tapered.section(stock, u)
            return blank.difference(outer), blank.difference(inner)

        receiver = body("receiver", depth, q.receiver_thickness_mm, q.stock_xy)
        plug = body("plug", q.seating_mm-depth,
                    q.plug_depth+q.backing_mm, straight._flip(q.stock_xy))
        plug = tuple(translate(straight._flip(s), *self.registration_xy) for s in plug)
        return receiver, plug


@dataclass(frozen=True)
class Finish:
    assembly: Assembly
    removal_mm: float
    plane_tolerance_mm: float
    motif_tolerance_mm: float
    minimum_plug_core_mm: float
    minimum_receiver_floor_mm: float
    edge_access_mm: float
    assembly_token: str
    cure_token: str
    renewed_setup_token: str

    def __post_init__(self):
        if type(self.assembly) is not Assembly:
            raise ValueError("Assembly required")
        for name in ("removal_mm", "plane_tolerance_mm", "motif_tolerance_mm",
                     "minimum_plug_core_mm", "minimum_receiver_floor_mm", "edge_access_mm"):
            object.__setattr__(self, name, straight._number(getattr(self, name), name,
                positive=name not in ("removal_mm", "plane_tolerance_mm", "motif_tolerance_mm")))
        if (min(self.plane_tolerance_mm, self.motif_tolerance_mm) < 0 or
                self.removal_mm < self.plane_tolerance_mm or
                self.removal_mm+self.plane_tolerance_mm >= self.assembly.allowances.seating_mm):
            raise ValueError("finish plane must remain inside seated plug")
        if any(type(t) is not str or not t.strip() for t in
               (self.assembly_token, self.cure_token, self.renewed_setup_token)):
            raise ValueError("declared assembly/cure/renewed setup required")

    @property
    def fingerprint(self):
        return _hash(("composite-finish-v1", self.assembly.fingerprint,
                      tuple((k, v) for k, v in vars(self).items() if k != "assembly")))

    @property
    def frame(self):
        return "assembled-backing-top:" + self.fingerprint

    @property
    def target(self):
        x0, y0, x1, y1 = self.assembly.footprint.bounds
        e = self.edge_access_mm
        return replay.Target("composite-face", (x0-e, y0-e, x1+e, y1+e),
            self.assembly.top_mm+self.removal_mm+self.plane_tolerance_mm)

    @property
    def nominal_motif(self):
        a = self.assembly
        if type(a.design) is straight.Design:
            return a.design.plug_xy
        removed = unary_union([t.section(a.allowances.seating_mm-self.removal_mm)
                               for t in a.design.targets("plug")])
        return a.allowances.stock_xy.difference(straight._flip(removed))


def generate(finish, tool, *, stepover_mm, stepdown_mm, clearance_mm):
    """Generate bounded raster passes with high retracts and explicit controls."""
    if type(finish) is not Finish or type(tool) is not replay.ToolProfile:
        raise ValueError("Finish and cylindrical ToolProfile required")
    pitch = straight._number(stepover_mm, "stepover", positive=True)
    step = straight._number(stepdown_mm, "stepdown", positive=True)
    high = straight._number(clearance_mm, "clearance", positive=True)
    depth = finish.assembly.top_mm + finish.removal_mm
    if (tool.kind != "cylinder" or tool.radius > finish.edge_access_mm or
            tool.cutting_length < depth or pitch >= 2*tool.radius):
        raise ValueError("tool reach, edge access or facing overlap invalid")
    x0, y0, x1, y1 = finish.assembly.footprint.bounds
    rows, passes = max(1, math.ceil((y1-y0)/pitch)), max(1, math.ceil(depth/step))
    at = initial = (x0, y0, high)
    op = replay.Operation("face", tool, finish.target)
    items = [replay.Event("tool_change", tool.name, at),
             replay.Event("spindle_start", tool.name, at)]
    for k in range(1, passes+1):
        z = -depth*k/passes
        for i in range(rows+1):
            y = y0+(y1-y0)*i/rows
            start, end = ((x0, x1) if i % 2 == 0 else (x1, x0))
            top = (start, y, high)
            if at != top:
                items.append(replay.Motion("rapid", tool.name, op.name, at, top))
            down = (start, y, z)
            across = (end, y, z)
            at = (end, y, high)
            items.extend((replay.Motion("entry", tool.name, op.name, top, down),
                          replay.Motion("cut", tool.name, op.name, down, across),
                          replay.Motion("retract", tool.name, op.name, across, at)))
    items.append(replay.Event("spindle_stop", tool.name, at))
    trace = replay.Trace(finish.fingerprint, finish.frame, initial, (op,), tuple(items))
    replay.replay(trace, expected_source=finish.fingerprint)
    return trace


def verify(finish, trace, *, expected_motion):
    """Check actual removal, final plane, motif bounds and retained core/floor.

    Thickness applies to the nominal tip core, not tapered edge ledges. Matching
    inner/outer topology does not certify topology inside their uncertainty band.
    """
    if type(finish) is not Finish:
        raise ValueError("Finish required")
    a = finish.assembly
    a.validate()
    if (type(trace) is not replay.Trace or trace.motion_fingerprint != expected_motion or
            trace.frame != finish.frame or any(op.target != finish.target or
            op.tool.kind != "cylinder" for op in trace.operations)):
        raise ValueError("stale facing motion/tool/target/setup binding")
    cuts = replay.replay(trace, expected_source=finish.fingerprint).cuts
    if not cuts or any(c.bottom_start is not None or c.path_error_mm for c in cuts):
        raise ValueError("level linear cylindrical facing required")
    lo = finish.removal_mm-finish.plane_tolerance_mm
    hi = max(-c.bottom for c in cuts)-a.top_mm
    cleared, _ = straight._removed(cuts, a.top_mm+lo)
    uncovered = a.footprint.difference(cleared)
    plane_ok = uncovered.is_empty and hi <= finish.removal_mm+finish.plane_tolerance_mm
    # Plug shrinks monotonically with receiver depth. These endpoint bounds
    # contain every section exposed by a plane anywhere in the accepted band.
    lower_plug = a.section(max(lo, hi))[1][0]
    upper_plug = a.section(lo)[1][1]
    nominal = finish.nominal_motif
    tol = finish.motif_tolerance_mm
    required = nominal.buffer(-tol, quad_segs=128) if tol else nominal
    allowed = nominal.buffer(tol, quad_segs=128) if tol else nominal
    topology = all(s.geom_type == "Polygon" and not s.is_empty and
                   len(s.interiors) == len(nominal.interiors)
                   for s in (required, lower_plug, upper_plug))
    motif_ok = plane_ok and topology and lower_plug.covers(required) and allowed.covers(upper_plug)
    core = a.allowances.plug_xy if type(a.design) is straight.Design else a.design.plug_tip
    core = translate(core, *a.registration_xy)
    tip_inner = a.section(a.allowances.seating_mm)[1][0]
    core_ok = tip_inner.covers(core)
    thickness = max(0.0, a.allowances.seating_mm-max(lo, hi)) if core_ok else 0.0
    floor = a.allowances.receiver_thickness_mm-max(a.design.depth("receiver"), hi)
    thickness_ok = (thickness >= finish.minimum_plug_core_mm and
                    floor >= finish.minimum_receiver_floor_mm)
    return {"status": "pass" if plane_ok and motif_ok and thickness_ok else "unresolved",
            "evidence": "replayed_composite_facing", "finish_fingerprint": finish.fingerprint,
            "assembly_fingerprint": a.fingerprint, "facing_motion": trace.motion_fingerprint,
            "process_declarations": (finish.assembly_token, finish.cure_token,
                                     finish.renewed_setup_token),
            "plane_depth_interval_mm": (lo, max(lo, hi)) if plane_ok else None, "plane_ok": plane_ok,
            "unfaced_upper_mm2": uncovered.area, "motif_ok": motif_ok,
            "motif_missing_upper_mm2": nominal.difference(lower_plug).area if plane_ok else None,
            "motif_excess_upper_mm2": upper_plug.difference(nominal).area if plane_ok else None,
            "section_enclosure_topology_matches": plane_ok and topology,
            "nominal_core_minimum_retained_mm": thickness,
            "minimum_receiver_floor_mm": floor, "thickness_ok": thickness_ok,
            "backing_removed": plane_ok}
