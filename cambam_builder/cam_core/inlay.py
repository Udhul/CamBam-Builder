"""One bounded circular, pointed-V pocket/plug pair with separate stock states.

Depth coordinates are positive into each part.  The plug is flipped at assembly:
its machining depth ``u`` maps to receiver depth ``engagement - u``.  The final
cut depth exceeds engagement, leaving a facing allowance behind the seated face.
"""

from dataclasses import dataclass
import hashlib
import math

from .v_region import VProfile


TOL = 0.00011                 # four-decimal controller coordinates


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"finite {name} required")
    return float(value)


@dataclass(frozen=True)
class InlayRequest:
    contour_radius_mm: float
    stock_radius_mm: float
    cut_depth_mm: float
    engagement_mm: float
    clearance_mm: float
    tool: VProfile
    ring_step_mm: float = 0.08
    assembly_xy_mm: tuple = (0.0, 0.0)
    flipped: bool = True

    def __post_init__(self):
        for name in ("contour_radius_mm", "stock_radius_mm", "cut_depth_mm",
                     "engagement_mm", "clearance_mm", "ring_step_mm"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (type(self.tool) is not VProfile or self.tool.kind != "pointed" or
                type(self.assembly_xy_mm) is not tuple or
                len(self.assembly_xy_mm) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in self.assembly_xy_mm) or type(self.flipped) is not bool):
            raise ValueError("pointed V tool and finite assembly frame required")
        t = self.tool.tangent
        if (not self.flipped or self.clearance_mm < 0 or
                not 0 < self.engagement_mm < self.cut_depth_mm <=
                self.tool.cutting_length or
                self.contour_radius_mm <= self.cut_depth_mm * t or
                self.contour_radius_mm - self.engagement_mm * t -
                self.clearance_mm <= 0 or
                self.stock_radius_mm <= self.contour_radius_mm +
                (self.cut_depth_mm - self.engagement_mm) * t or
                self.ring_step_mm <= 0 or
                self.ring_step_mm > 2 * (self.cut_depth_mm -
                                       self.engagement_mm) * t - TOL or
                self.tool.radius(self.cut_depth_mm) > self.tool.maximum_radius):
            raise ValueError("inlay clearance, stock, depth or path spacing impossible")
        if math.hypot(*self.assembly_xy_mm) > self.clearance_mm + TOL:
            raise ValueError("assembly registration exceeds side clearance")

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InlayOperation:
    name: str
    tool_id: str
    side: str
    request: InlayRequest
    rings_mm: tuple

    def __post_init__(self):
        if (self.side not in ("female", "male") or not self.name or
                not self.tool_id or not self.rings_mm or
                any(type(r) not in (int, float) or r < 0
                    for r in self.rings_mm)):
            raise ValueError("invalid inlay operation")

    @property
    def wall_center_mm(self):
        q = self.request
        if self.side == "female":
            return q.contour_radius_mm - q.cut_depth_mm * q.tool.tangent
        return (q.contour_radius_mm +
                (q.cut_depth_mm - q.engagement_mm) * q.tool.tangent -
                q.clearance_mm)

    def target_interval(self, depth):
        q = self.request
        if not 0 <= depth <= q.cut_depth_mm:
            raise ValueError("section outside inlay stock")
        if self.side == "female":
            return (0.0, q.contour_radius_mm - depth * q.tool.tangent)
        return (self.wall_center_mm -
                (q.cut_depth_mm - depth) * q.tool.tangent,
                q.stock_radius_mm)


@dataclass(frozen=True)
class InlayPair:
    request: InlayRequest
    female: object
    male: object

    @property
    def fingerprint(self):
        return hashlib.sha256(repr((self.request.fingerprint,
            self.female.fingerprint, self.male.fingerprint)).encode("utf-8")).hexdigest()


def _rings(first, last, step):
    count = math.ceil((last - first) / step)
    if count > 1000:
        raise ValueError("inlay ring budget exceeded")
    return tuple(first + (last - first) * i / count for i in range(count + 1))


def generate(request):
    """Derive two independent one-tool jobs from one circular design contour."""
    if type(request) is not InlayRequest:
        raise ValueError("inlay request required")
    from .ordered_job import Job, JobMove, Stage

    depth, step = request.cut_depth_mm, request.ring_step_mm
    female_edge = request.contour_radius_mm - depth * request.tool.tangent
    male_edge = (request.contour_radius_mm +
                 (depth - request.engagement_mm) * request.tool.tangent -
                 request.clearance_mm)
    definitions = (("female", _rings(0, female_edge, step)),
                   ("male", _rings(male_edge, request.stock_radius_mm, step)))
    jobs = []
    for side, rings in definitions:
        op = InlayOperation(side, "T1", side, request, rings)
        safe = (0.0, 0.0, 2.0)
        at = safe
        motions = []
        for radius in rings:
            high = (radius, 0.0, 2.0)
            if high != at:
                motions.append(JobMove("rapid", at, high))
            low = (radius, 0.0, -depth)
            motions.append(JobMove("entry", high, low, 60))
            if radius:
                far = (-radius, 0.0, -depth)
                motions.append(JobMove("cut", low, far, 240, 3, (0.0, 0.0)))
                motions.append(JobMove("cut", far, low, 240, 3, (0.0, 0.0)))
            motions.append(JobMove("retract", low, high, 120))
            at = high
        if at != safe:
            motions.append(JobMove("rapid", at, safe))
        stage = Stage(side, "T1", tuple(motions), 10000,
                      operation=op, source_revision=request.fingerprint)
        jobs.append(Job(request.fingerprint, (stage,), safe,
                        program_frame=f"{side}-machining"))
    return InlayPair(request, *jobs)


def _union(intervals):
    merged = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + 1e-8:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return tuple((a, b) for a, b in merged)


def _area(intervals):
    return math.pi * sum(b*b-a*a for a, b in intervals)


def _intersection(intervals, lo, hi):
    return _union((max(a, lo), min(b, hi)) for a, b in intervals
                  if min(b, hi) > max(a, lo))


def _decoded_rings(operation, motions):
    """Use decoded arcs, including their centers, for the section stock replay."""
    rings, arcs, entries = [], [], []
    for move in motions:
        if move.arc_g:
            if (move.arc_g != 3 or move.center is None or
                    abs(move.start[2] + operation.request.cut_depth_mm) > TOL or
                    abs(move.end[2] + operation.request.cut_depth_mm) > TOL or
                    math.hypot(*move.center) > TOL):
                raise ValueError("unsupported inlay arc or depth")
            arcs.append(move)
            if len(arcs) == 2:
                a, b = arcs
                radius = math.hypot(a.start[0]-a.center[0],
                                    a.start[1]-a.center[1])
                if (math.dist(a.end, b.start) > TOL or
                        math.dist(b.end, a.start) > TOL or
                        abs(math.hypot(b.start[0]-b.center[0],
                                       b.start[1]-b.center[1])-radius) > TOL):
                    raise ValueError("inlay arcs do not close a full circle")
                rings.append(radius)
                arcs = []
        elif move.role == "entry" and move.end[2] < 0:
            if move.start[:2] != move.end[:2] or move.start[2] <= 0:
                raise ValueError("inlay entry requires vertical access from safe height")
            if abs(move.end[2] + operation.request.cut_depth_mm) > TOL:
                raise ValueError("inlay entry depth differs")
            entries.append(math.hypot(*move.end[:2]))
        elif move.role in ("rapid", "retract"):
            if move.role == "rapid" and min(move.start[2], move.end[2]) <= 0:
                raise ValueError("inlay rapid enters stock")
            if move.role == "retract" and (
                    move.start[:2] != move.end[:2] or move.end[2] <= 0 or
                    abs(move.start[2] + operation.request.cut_depth_mm) > TOL):
                raise ValueError("inlay retract requires vertical return from cut depth")
        else:
            raise ValueError("unsupported inlay motion")
    if (arcs or len(entries) != len(operation.rings_mm) or
            any(abs(a-b) > TOL for a, b in zip(entries,
                                                operation.rings_mm)) or
            len(rings) != len([r for r in operation.rings_mm if r])):
        raise ValueError("inlay ring motion incomplete")
    if any(abs(a-b) > TOL for a, b in zip(rings,
                                          (r for r in operation.rings_mm if r))):
        raise ValueError("decoded inlay rings differ")
    return (0.0,) + tuple(rings) if operation.side == "female" else tuple(rings)


def section(operation, radii, depth):
    q = operation.request
    cutter = (q.cut_depth_mm - depth) * q.tool.tangent
    removed = _union((max(0.0, r-cutter), min(q.stock_radius_mm, r+cutter))
                     for r in radii if r-cutter < q.stock_radius_mm)
    target_lo, target_hi = operation.target_interval(depth)
    required = _area(((target_lo, target_hi),))
    covered = _area(_intersection(removed, target_lo, target_hi))
    overcut = _area(_intersection(removed, 0, target_lo)) + _area(
        _intersection(removed, target_hi, q.stock_radius_mm))
    if operation.side == "female":
        contiguous = next((b for a, b in removed if a <= TOL), 0.0)
        wall = contiguous
    else:
        contiguous = next((a for a, b in removed if b >= q.stock_radius_mm-TOL),
                          q.stock_radius_mm)
        wall = contiguous
    return {"depth_mm": depth, "required_area_mm2": required,
            "residual_area_mm2": max(0.0, required-covered),
            "protected_overcut_mm2": overcut, "actual_wall_radius_mm": wall,
            "target_wall_radius_mm": target_hi if operation.side == "female"
                                      else target_lo}


def replay_stages(stages, actual):
    if len(stages) != 1 or len(actual) != 1:
        raise ValueError("each inlay part requires separate stock replay")
    op = stages[0].operation
    if (type(op) is not InlayOperation or
            stages[0].source_revision != op.request.fingerprint):
        raise ValueError("stale inlay source or operation")
    rings = _decoded_rings(op, actual[0])
    q = op.request
    if (max(b-a for a, b in zip(rings, rings[1:])) >
            2 * (q.cut_depth_mm-q.engagement_mm) * q.tool.tangent - TOL or
            abs(rings[0]-(0 if op.side == "female" else op.wall_center_mm)) > TOL or
            abs(rings[-1]-(op.wall_center_mm if op.side == "female"
                           else q.stock_radius_mm)) > TOL):
        raise ValueError("inlay rings leave an insertion-envelope gap")
    depths = (0.0, q.engagement_mm/2, q.engagement_mm,
              (q.engagement_mm+q.cut_depth_mm)/2, q.cut_depth_mm)
    rows = tuple(section(op, rings, d) for d in depths)
    for row in rows[:3]:
        if (row["residual_area_mm2"] > 0.002 or
                row["protected_overcut_mm2"] > 0.002 or
                abs(row["actual_wall_radius_mm"]-
                    row["target_wall_radius_mm"]) > TOL):
            raise ValueError("inlay insertion-envelope stock or wall fails")
    return {"status": "pass", "model": "circular-pointed-v-inlay-v1",
            "side": op.side, "independent_stock": True,
            "decoded_ring_count": len(rings), "sections": rows,
            "engagement_mm": q.engagement_mm,
            "bottom_gap_mm": q.cut_depth_mm-q.engagement_mm}


def assembly(request):
    """Independent circle cross-section and insertion oracle."""
    if type(request) is not InlayRequest:
        raise ValueError("inlay request required")
    offset = math.hypot(*request.assembly_xy_mm)
    min_gap = request.clearance_mm-offset
    if min_gap < -TOL:
        raise ValueError("inlay parts collide after registration")
    # At insertion s, radial gap is clearance + (engagement-s)*tan(alpha).
    samples = tuple((s, min_gap +
                     (request.engagement_mm-s)*request.tool.tangent)
                    for s in (0.0, request.engagement_mm/2,
                              request.engagement_mm))
    return {"status": "pass", "flip": "parallel-plane Z reversal",
            "registration_xy_mm": request.assembly_xy_mm,
            "side_gap_mm": min_gap, "nominal_side_clearance_mm":
            request.clearance_mm,
            "bottom_gap_mm": request.cut_depth_mm-request.engagement_mm,
            "backing_face_contact_depth_mm": 0.0,
            "insertion_gap_samples_mm": samples,
            "side_contact": abs(min_gap) <= TOL}


def audit_pair(pair, female_files, male_files, dialect,
               *, female_hashes=None, male_hashes=None,
               expected_fingerprint=None):
    """Reaudit both complete programs and bind the two stocks to one assembly."""
    if type(pair) is not InlayPair:
        raise ValueError("paired inlay required")
    fresh = generate(pair.request)
    if (pair.fingerprint != fresh.fingerprint or
            (expected_fingerprint is not None and
             expected_fingerprint != pair.fingerprint)):
        raise ValueError("stale paired inlay evidence")
    from ..integrations.ordered_output import audit_files
    reports = (audit_files(pair.female, dialect, female_files,
                           expected_hashes=female_hashes),
               audit_files(pair.male, dialect, male_files,
                           expected_hashes=male_hashes))
    if any(r["stock_access_residual"]["status"] != "pass" for r in reports):
        raise ValueError("paired stock replay failed")
    female = reports[0]["stock_access_residual"]["sections"]
    male = reports[1]["stock_access_residual"]["sections"]
    offset = math.hypot(*pair.request.assembly_xy_mm)
    actual_gaps = tuple(female[i]["actual_wall_radius_mm"] -
                        male[2-i]["actual_wall_radius_mm"] - offset
                        for i in range(3))
    if min(actual_gaps) < -TOL:
        raise ValueError("decoded inlay stock collides in assembly")
    return {"status": "paired_inlay_pass", "pair_fingerprint": pair.fingerprint,
            "assembly": assembly(pair.request),
            "decoded_side_gap_samples_mm": actual_gaps,
            "female": reports[0]["stock_access_residual"],
            "male": reports[1]["stock_access_residual"],
            "program_sha256": (reports[0]["motion_equivalence"]["program_sha256"],
                               reports[1]["motion_equivalence"]["program_sha256"])}
