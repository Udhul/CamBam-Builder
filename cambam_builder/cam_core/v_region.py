"""Conservative variable-depth V paths on a Region with optional curved bounds.

The finish target at section depth ``t`` is the source Region eroded by
``t * tan(design_angle / 2)``. A path is accepted only when its entire cutting profile
stays in that target at every height.  Polygonal and curved source bounds are
kept separately so an arc chord cannot silently become the protected boundary.
"""

from dataclasses import dataclass, replace
import hashlib
import math

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import replay


MAX_CUT_SLOPE = 2.0           # tip Z change per XY millimetre


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class VProfile:
    kind: str                         # pointed, flat, rounded
    angle_degrees: float
    tip_radius: float
    maximum_radius: float
    cutting_length: float

    def __post_init__(self):
        if self.kind not in ("pointed", "flat", "rounded"):
            raise ValueError("unsupported V profile")
        for name in ("angle_degrees", "tip_radius", "maximum_radius", "cutting_length"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (not 0 < self.angle_degrees < 180 or self.maximum_radius <= 0 or
                self.cutting_length <= 0 or self.tip_radius < 0 or
                (self.kind == "pointed" and self.tip_radius != 0) or
                (self.kind != "pointed" and self.tip_radius <= 0) or
                self.radius(self.cutting_length) > self.maximum_radius + 1e-10):
            raise ValueError("V profile exceeds its geometry or cutting envelope")

    @property
    def tangent(self):
        return math.tan(math.radians(self.angle_degrees) / 2)

    @property
    def join_height(self):
        if self.kind != "rounded":
            return 0.0
        angle = math.radians(self.angle_degrees) / 2
        return self.tip_radius * (1 - math.sin(angle))

    def radius(self, height):
        height = _finite(height, "profile height")
        if height < 0 or height > self.cutting_length + 1e-10:
            raise ValueError("height exceeds V cutting length")
        if self.kind == "pointed":
            return height * self.tangent
        if self.kind == "flat":
            return self.tip_radius + height * self.tangent
        angle = math.radians(self.angle_degrees) / 2
        if height <= self.join_height:
            return math.sqrt(max(0.0, 2 * self.tip_radius * height - height * height))
        return (self.tip_radius * math.cos(angle) +
                (height - self.join_height) * self.tangent)

    def depth_for_radius(self, radius):
        radius = _finite(radius, "available radius")
        if radius < self.radius(0):
            return None
        if radius >= self.radius(self.cutting_length):
            return self.cutting_length
        if self.kind == "pointed":
            return radius / self.tangent
        if self.kind == "flat":
            return (radius - self.tip_radius) / self.tangent
        join_radius = self.tip_radius * math.cos(math.radians(self.angle_degrees) / 2)
        if radius <= join_radius:
            # Rationalize R - sqrt(R^2-r^2) to retain tiny positive depths.
            return radius * radius / (self.tip_radius + math.sqrt(
                max(0.0, self.tip_radius ** 2 - radius ** 2)))
        return self.join_height + (radius - join_radius) / self.tangent

    def occupancy_radius(self, penetration, section_depth):
        """Radius consumed from the original boundary at one stock section."""
        if not 0 <= section_depth <= penetration <= self.cutting_length:
            raise ValueError("section outside cutter penetration")
        return section_depth * self.tangent + self.radius(penetration - section_depth)


@dataclass(frozen=True)
class VTarget:
    """Immutable ideal pointed V envelope with a flat floor at the depth cap.

    An omitted angle is a legacy tool-defined input, resolved by ``freeze`` or
    VPlan construction. Compare cutters using that resolved target. ``frame=None``
    retains the legacy unbound frame; explicit polygon/curved designs default to
    the program frame. No relief allowance or known-free stock is inferred.
    """
    source_id: str
    safe: Polygon
    outer: Polygon
    cap_depth: float
    sagitta_mm: float = 0.0
    design_angle_degrees: float = None
    frame: str = None

    def __post_init__(self):
        if (not self.source_id or self.safe.geom_type != "Polygon" or
                self.outer.geom_type != "Polygon" or not self.safe.is_valid or
                not self.outer.is_valid or self.safe.is_empty or
                not self.outer.covers(self.safe)):
            raise ValueError("invalid V Region bounds")
        object.__setattr__(self, "cap_depth", _finite(self.cap_depth, "cap depth"))
        object.__setattr__(self, "sagitta_mm", _finite(self.sagitta_mm, "sagitta"))
        if self.cap_depth <= 0 or self.sagitta_mm < 0:
            raise ValueError("invalid V depth or arc error")
        if self.design_angle_degrees is not None:
            angle = _finite(self.design_angle_degrees, "design angle")
            if not 0 < angle < 180:
                raise ValueError("invalid V design angle")
            object.__setattr__(self, "design_angle_degrees", angle)
        if self.frame is not None and (type(self.frame) is not str or not self.frame):
            raise ValueError("invalid V design frame")

    @classmethod
    def polygon(cls, source_id, shell, holes, cap_depth, *,
                design_angle_degrees=None, frame=None):
        shape = Polygon(shell, holes)
        return cls(source_id, shape, shape, cap_depth,
                   design_angle_degrees=design_angle_degrees,
                   frame=("program" if frame is None and design_angle_degrees is not None
                          else frame))

    @classmethod
    def curved(cls, source_id, approximation, cap_depth, *,
               design_angle_degrees=None, frame=None):
        return cls(source_id, approximation.safe, approximation.outer,
                   cap_depth, approximation.sagitta_mm, design_angle_degrees,
                   ("program" if frame is None and design_angle_degrees is not None
                    else frame))

    def freeze(self, tool):
        """Map a legacy opening to a fixed design once, before tool comparison."""
        if type(tool) is not VProfile:
            raise ValueError("V profile required for compatibility mapping")
        return (replace(self, design_angle_degrees=tool.angle_degrees)
                if self.design_angle_degrees is None else self)

    @property
    def tangent(self):
        if self.design_angle_degrees is None:
            raise ValueError("freeze legacy V target before design queries")
        return math.tan(math.radians(self.design_angle_degrees) / 2)

    @property
    def fingerprint(self):
        values = (self.source_id, self.safe.wkb_hex, self.outer.wkb_hex,
                  self.cap_depth, self.sagitta_mm, self.design_angle_degrees,
                  self.frame, "ideal-pointed-capped-v1")
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()

    def section(self, depth, tangent=None, *, outer=False):
        depth = _finite(depth, "section depth")
        if not 0 <= depth <= self.cap_depth:
            raise ValueError("section outside V target")
        if self.design_angle_degrees is not None:
            if tangent is not None and not math.isclose(
                    _finite(tangent, "section tangent"), self.tangent,
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("section tangent differs from fixed V design")
            tangent = self.tangent
        elif tangent is None:
            raise ValueError("freeze legacy V target before design queries")
        tangent = _finite(tangent, "section tangent")
        if tangent <= 0:
            raise ValueError("section tangent must be positive")
        shape = self.outer if outer else self.safe
        # Round hole/concave-corner joins are inscribed: nominal erosion is
        # an outer enclosure. Enlarge the inner erosion's disk to compensate
        # for chord loss, including GEOS rounding of non-quadrant fillet counts.
        distance = depth * tangent
        if not outer:
            distance /= math.cos(math.pi / 128)
        return shape.buffer(-distance, quad_segs=64) if depth else shape

    def roughing_centers(self, depth, radius, *, margin_mm=0.01):
        """Derived cylinder-center boundary; never a replacement finish target.

        Required removal is the section itself, allowed removal is the same
        ideal envelope (with separate geometric bounds), and known-free stock
        comes only from replay. This boundary grants no pre-cleared access.
        """
        radius = _finite(radius, "roughing radius")
        margin_mm = _finite(margin_mm, "roughing margin")
        if radius <= 0 or margin_mm <= 0:
            raise ValueError("positive roughing radius and margin required")
        return self.section(depth).buffer(
            -(radius + margin_mm) / math.cos(math.pi / 128), quad_segs=64)

    def volume_bounds(self, slabs=8):
        """Design removal volume, independent of any candidate cutter/stock."""
        if type(slabs) is not int or slabs <= 0:
            raise ValueError("positive integer V volume slabs required")
        dz = self.cap_depth / slabs
        return (dz * sum(self.section(self.cap_depth if i == slabs else i * dz).area
                         for i in range(1, slabs + 1)),
                dz * sum(self.section(i * dz, outer=True).area for i in range(slabs)))


@dataclass(frozen=True)
class VPath:
    role: str                       # edge or fill
    points: tuple                    # ordered (x, y, positive penetration)


@dataclass(frozen=True)
class VMotion:
    role: str                       # rapid, entry, cut, retract
    start: tuple
    end: tuple


@dataclass(frozen=True)
class VPlan:
    target: VTarget
    tool: VProfile
    paths: tuple
    motions: tuple
    safe_z: float
    margin_mm: float
    stepover_mm: float
    status: str
    reason: str
    fill_pattern: str = "raster"

    def __post_init__(self):
        if type(self.target) is not VTarget or type(self.tool) is not VProfile:
            raise ValueError("V Region target and tool required")
        object.__setattr__(self, "target", self.target.freeze(self.tool))

    @property
    def fingerprint(self):
        values = (self.target.fingerprint, self.tool, self.paths, self.motions,
                  self.safe_z, self.margin_mm, self.stepover_mm, self.status)
        if self.fill_pattern != "raster":
            values += (self.fill_pattern,)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VSequence:
    """Ordered verified V sweeps against one fixed design, from virgin stock.

    Stock is the union of actual per-plan cutter sweeps, never a sum of their
    areas. No cached geometry or inferred pre-cleared opening is admitted.
    """
    plans: tuple

    def __post_init__(self):
        if (type(self.plans) is not tuple or not self.plans or
                any(type(plan) is not VPlan for plan in self.plans)):
            raise ValueError("nonempty V plan tuple required")
        for plan in self.plans:
            verify(plan)
            if plan.target.fingerprint != self.target.fingerprint:
                raise ValueError("V sequence design targets differ")

    @property
    def target(self):
        return self.plans[0].target

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(("v-sequence-v1", self.target.fingerprint,
            tuple(plan.fingerprint for plan in self.plans))).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VRest:
    """A source-bound cylindrical stock prefix followed by a V path plan."""
    plan: VPlan
    prior_trace: replay.Trace
    prior_stock: replay.ReplayResult

    @property
    def fingerprint(self):
        return hashlib.sha256(repr((self.plan.fingerprint,
            self.prior_trace.motion_fingerprint)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VComposition:
    """Actual cylindrical/V stages sharing one fixed design and virgin stock.

    Cylinder traces are independently executable: they do not infer access
    through another stage's cavity. Section stock unions every supplied sweep.
    """
    target: VTarget
    stages: tuple

    def __post_init__(self):
        if (type(self.target) is not VTarget or
                self.target.design_angle_degrees is None or
                type(self.stages) is not tuple or not self.stages):
            raise ValueError("fixed design and nonempty composed stages required")
        for stage in self.stages:
            if type(stage) is VPlan:
                verify(stage)
                if stage.target.fingerprint != self.target.fingerprint:
                    raise ValueError("composed design targets differ")
            elif type(stage) is replay.Trace:
                _cylindrical_stock(self.target, stage)
            else:
                raise ValueError("composed stock requires V plans or cylinder traces")

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(("v-composition-v1", self.target.fingerprint,
            tuple(s.fingerprint if type(s) is VPlan else s.motion_fingerprint
                  for s in self.stages))).encode("utf-8")).hexdigest()


def _cylindrical_stock(target, trace, margin_mm=0):
    """Re-establish source, original boundary and full-height cylinder safety."""
    if (type(trace) is not replay.Trace or
            trace.source_fingerprint != target.source_id or
            (target.frame is not None and trace.frame != target.frame)):
        raise ValueError("stale V prior source")
    stock = replay.replay(trace, expected_source=target.source_id)
    if not stock.cuts or len(trace.operations) != 1:
        raise ValueError("complete V prior cut required")
    op = trace.operations[0]
    if (op.tool.kind != "cylinder" or not op.target.region_shell or
            op.target.depth != target.cap_depth or
            not Polygon(op.target.region_shell, op.target.region_holes).equals(target.safe)):
        raise ValueError("V prior tool or target differs from source")
    for cut in stock.cuts:
        depth = -cut.bottom
        line = LineString((cut.a, cut.b)) if cut.a != cut.b else Point(cut.a)
        required = cut.tool.radius + cut.path_error_mm + depth * target.tangent
        if (cut.tool != op.tool or not 0 < depth <= target.cap_depth or
                not target.safe.covers(line) or
                line.distance(target.safe.boundary) + 1e-8 < required + margin_mm * .5):
            raise ValueError("V prior cut crosses capped finish target")
    return stock


def with_prior(plan, prior_trace):
    """Replay supplied prior motion and prove its entire cylinder fits the V target."""
    verify(plan)
    stock = _cylindrical_stock(plan.target, prior_trace, plan.margin_mm)
    return VRest(plan, prior_trace, stock)


def _polygons(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type == "MultiPolygon":
        return tuple(geometry.geoms)
    return ()


def _segments(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "LineString":
        return (geometry,)
    if geometry.geom_type == "MultiLineString":
        return tuple(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return tuple(g for part in geometry.geoms for g in _segments(part))
    return ()


def _sample(line, step):
    count = max(1, math.ceil(line.length / step))
    if count > 50000:
        raise ValueError("V path vertex budget exceeded")
    return tuple(tuple(round(c, 7) for c in line.interpolate(i / count,
                        normalized=True).coords[0]) for i in range(count + 1))


def contact_radius(target, tool, penetration):
    """All-height boundary occupancy of a cutter in the fixed ideal V design.

    Maximize t*q + rho(d-t) analytically over 0 <= t <= d. Pointed/flat
    profiles are linear. A rounded tip additionally has a concave spherical
    segment with one stationary height. This is a continuous certificate,
    including entry/retract at every intermediate penetration.
    """
    if type(target) is not VTarget or type(tool) is not VProfile:
        raise ValueError("fixed V target and profile required")
    penetration = _finite(penetration, "penetration")
    if not 0 <= penetration <= min(target.cap_depth, tool.cutting_length):
        raise ValueError("penetration outside design/cutter limits")
    q = target.tangent
    heights = [0.0, penetration]
    if tool.kind == "rounded":
        heights.append(min(penetration, tool.join_height))
        stationary = tool.tip_radius * (1 - q / math.hypot(1, q))
        heights.append(min(penetration, tool.join_height, stationary))
    return max((penetration - h) * q + tool.radius(h) for h in heights)


def _reachable_depth(target, tool, clearance, cap):
    if clearance <= contact_radius(target, tool, 0):
        return 0.0
    if contact_radius(target, tool, cap) <= clearance:
        return cap
    # A monotone envelope permits bounded deterministic inversion. Return
    # the safe lower endpoint; coordinate rounding is covered by the margin.
    low, high = 0.0, cap
    for _ in range(48):
        middle = (low + high) / 2
        if contact_radius(target, tool, middle) <= clearance:
            low = middle
        else:
            high = middle
    return low


def _depth_path(target, tool, xy, cap, margin, role):
    boundary = target.safe.boundary
    edge_caps = []
    for a, b in zip(xy, xy[1:]):
        clearance = LineString((a, b)).distance(boundary) - margin
        edge_caps.append(_reachable_depth(target, tool, max(0.0, clearance), cap))
    depths = []
    for index in range(len(xy)):
        adjacent = edge_caps[max(0, index - 1):min(len(edge_caps), index + 1)]
        depths.append(min(adjacent) if adjacent else 0.0)
    if min(depths) <= 1e-6:
        return None
    points = tuple((p[0], p[1], round(depth, 7)) for p, depth in zip(xy, depths))
    return VPath(role, points)


def _motions(paths, safe_z):
    items = []
    at = None
    for path in paths:
        x, y, depth = path.points[0]
        high = (x, y, safe_z)
        if at is not None and at != high:
            items.append(VMotion("rapid", at, high))
        low = (x, y, -depth)
        items.append(VMotion("entry", high, low))
        for point in path.points[1:]:
            end = (point[0], point[1], -point[2])
            if end != low:
                items.append(VMotion("cut", low, end))
            low = end
        at = (low[0], low[1], safe_z)
        items.append(VMotion("retract", low, at))
    return tuple(items)


def complete_motion(plan, initial_tip):
    """Include safe travel to and from one verified V plan."""
    verify(plan)
    replay._xyz(initial_tip)
    if not plan.motions:
        raise ValueError("V plan has no executable motion")
    first = plan.motions[0].start
    moves = []
    if first != initial_tip:
        moves.append(VMotion("rapid", initial_tip, first))
    moves.extend(plan.motions)
    last = moves[-1].end
    if last != initial_tip:
        moves.append(VMotion("rapid", last, initial_tip))
    return tuple(moves)


def plan(target, tool, *, stepover_mm=1.0, xy_step_mm=1.0,
         margin_mm=0.01, safe_z=2.0, max_paths=1000,
         fill_pattern="raster"):
    """Trace a fixed V design with varying-Z rows or offset contours.

    Return partial paths when flute length or finite spacing leaves target rest.
    """
    if type(target) is not VTarget or type(tool) is not VProfile:
        raise ValueError("V Region target and tool required")
    target = target.freeze(tool)
    stepover_mm = _finite(stepover_mm, "stepover")
    xy_step_mm = _finite(xy_step_mm, "XY step")
    margin_mm = _finite(margin_mm, "clearance margin")
    safe_z = _finite(safe_z, "safe Z")
    if (stepover_mm <= 0 or xy_step_mm <= 0 or margin_mm <= 1e-5 or
            safe_z <= 0 or type(max_paths) is not int or max_paths <= 0 or
            fill_pattern not in ("raster", "offset")):
        raise ValueError("V path controls exceed tool/setup limits")
    # A short flute can still remove the upper part of a deeper target.  Keep
    # the target cap intact for residual reporting and limit only penetration.
    cap = min(target.cap_depth, tool.cutting_length)
    deep = target.safe.buffer(-(contact_radius(target, tool, cap) + margin_mm), quad_segs=32)
    start_depth = min(0.05, cap)
    reachable = target.safe.buffer(-(contact_radius(target, tool, start_depth) + margin_mm),
                                   quad_segs=32)
    # Fixed 0.05 mm seeding can miss a legitimate shallower path in a narrow
    # opening.  Depths below the planner's 1e-6 mm path threshold are not
    # executable at its seven-decimal coordinate precision.
    while reachable.is_empty and start_depth > 1e-6:
        start_depth /= 2
        reachable = target.safe.buffer(-(contact_radius(target, tool, start_depth) + margin_mm),
                                       quad_segs=32)
    if reachable.is_empty:
        result = VPlan(target, tool, (), (), safe_z, margin_mm, stepover_mm,
                       "infeasible", "no cutter center fits at supported depth",
                       fill_pattern)
        verify(result)
        return result
    paths = []
    for polygon in _polygons(deep):
        for ring in (polygon.exterior,) + tuple(polygon.interiors):
            xy = _sample(LineString(ring.coords), xy_step_mm)
            candidate = _depth_path(target, tool, xy, cap, margin_mm, "edge")
            if candidate is not None:
                paths.append(candidate)
    if fill_pattern == "raster":
        xmin, ymin, xmax, ymax = reachable.bounds
        if math.ceil((ymax - ymin) / stepover_mm) > max_paths:
            raise ValueError("V fill row budget exceeded")
        row = ymin + stepover_mm / 2
        while row < ymax:
            horizontal = LineString(((xmin - 1, row), (xmax + 1, row)))
            for line in _segments(reachable.intersection(horizontal)):
                if line.length < xy_step_mm / 10:
                    continue
                candidate = _depth_path(target, tool, _sample(line, xy_step_mm),
                                        cap, margin_mm, "fill")
                if candidate is not None:
                    paths.append(candidate)
            if len(paths) > max_paths:
                raise ValueError("V path budget exceeded")
            row += stepover_mm
    else:
        inset = stepover_mm / 2
        # Every ring comes from a smaller center region. Disconnected islands
        # and holes stay separate, so no low feed bridges protected stock.
        for _ in range(max_paths):
            offset = reachable.buffer(-inset, quad_segs=32)
            if offset.is_empty:
                break
            for polygon in _polygons(offset):
                for ring in (polygon.exterior,) + tuple(polygon.interiors):
                    candidate = _depth_path(target, tool,
                        _sample(LineString(ring.coords), xy_step_mm),
                        cap, margin_mm, "fill")
                    if candidate is not None:
                        paths.append(candidate)
            if len(paths) > max_paths:
                raise ValueError("V path budget exceeded")
            inset += stepover_mm
        else:
            raise ValueError("V offset fill budget exceeded")
    # A finite raster pitch or first offset can miss an entire short or
    # disconnected reachable component.  Give each untouched component one
    # interior row; established paths for other components are unchanged.
    xmin, _, xmax, _ = reachable.bounds
    for polygon in _polygons(reachable):
        if any(polygon.covers(Point(path.points[0][:2])) for path in paths):
            continue
        row = polygon.representative_point().y
        horizontal = LineString(((xmin - 1, row), (xmax + 1, row)))
        for line in _segments(polygon.intersection(horizontal)):
            candidate = _depth_path(target, tool, _sample(line, xy_step_mm),
                                    cap, margin_mm, "fill")
            if candidate is not None:
                paths.append(candidate)
        if len(paths) > max_paths:
            raise ValueError("V path budget exceeded")
    status = "partial" if paths else "infeasible"
    reason = ("finite stepover and finite tip leave measured residual" if paths else
              "no admissible positive-depth path found")
    result = VPlan(target, tool, tuple(paths), _motions(paths, safe_z),
                   safe_z, margin_mm, stepover_mm, status, reason, fill_pattern)
    verify(result)
    return result


def depth_passes(plan, max_stepdown_mm, *, depth_cap_mm=None):
    """Clip a verified candidate into explicit axial passes on unchanged XY paths.

    An optional shallower cap leaves an axial allowance without changing the
    desired part. Entry permission and engagement remain caller-owned policies.
    """
    verify(plan)
    step = _finite(max_stepdown_mm, "maximum stepdown")
    if step <= 0 or not plan.paths:
        raise ValueError("positive stepdown and executable V paths required")
    cap = max(p[2] for path in plan.paths for p in path.points)
    if depth_cap_mm is not None:
        limit = _finite(depth_cap_mm, "pass depth cap")
        if not 0 < limit <= plan.target.cap_depth:
            raise ValueError("pass depth cap outside fixed design")
        cap = min(cap, limit)
    result = []
    for i in range(1, math.ceil(cap / step) + 1):
        depth = min(cap, i * step)
        paths = tuple(replace(path, points=tuple((x, y, min(d, depth))
                      for x, y, d in path.points)) for path in plan.paths)
        result.append(verify(replace(plan, paths=paths,
                                    motions=_motions(paths, plan.safe_z))))
    return tuple(result)


def verify(result):
    """Check the full profile between vertices and complete ordered motion."""
    if (type(result) is not VPlan or type(result.target) is not VTarget or
            type(result.tool) is not VProfile or
            result.status not in ("partial", "infeasible") or
            result.fill_pattern not in ("raster", "offset")):
        raise ValueError("invalid V plan")
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (result.safe_z, result.margin_mm, result.stepover_mm)) or
            result.safe_z <= 0 or result.margin_mm <= 1e-5 or
            result.stepover_mm <= 0):
        raise ValueError("V path controls exceed tool/setup limits")
    if result.motions != _motions(result.paths, result.safe_z):
        raise ValueError("V motion differs from paths")
    if not result.paths:
        if result.status != "infeasible":
            raise ValueError("missing V paths")
        return result
    if result.status == "infeasible":
        raise ValueError("infeasible V plan carries executable paths")
    boundary = result.target.safe.boundary
    tool = result.tool
    for path in result.paths:
        if path.role not in ("edge", "fill") or len(path.points) < 2:
            raise ValueError("invalid V path")
        for x, y, depth in path.points:
            if (not all(math.isfinite(v) for v in (x, y, depth)) or
                    not 0 < depth <= result.target.cap_depth or
                    tool.radius(depth) > tool.maximum_radius):
                raise ValueError("V path exceeds depth/profile limits")
        for a, b in zip(path.points, path.points[1:]):
            segment = LineString((a[:2], b[:2]))
            if segment.length <= 0:
                raise ValueError("zero-length V segment")
            if abs(a[2] - b[2]) > MAX_CUT_SLOPE * segment.length + 1e-8:
                raise ValueError("V cut slope exceeds process limit")
            # The maximum-depth envelope encloses every intermediate Z on
            # this whole segment, even when design and cutter angles differ.
            if (not result.target.safe.covers(segment) or
                    segment.distance(boundary) + 1e-8 <
                    contact_radius(result.target, tool, max(a[2], b[2])) +
                    result.margin_mm * 0.5):
                raise ValueError("full V cutter crosses protected Region")
    for motion in result.motions:
        if motion.role == "rapid" and min(motion.start[2], motion.end[2]) <= 0:
            raise ValueError("low V rapid")
    return result


@dataclass(frozen=True)
class VSectionEvidence:
    """Located section bounds; allowed removal equals this ideal design.

    Inner/outer values are geometric enclosures, not an extra relief allowance.
    Known-free bounds come from supplied replay/verified cutter sweeps only.
    """
    design_fingerprint: str
    depth: float
    required_inner: object
    required_outer: object
    known_free_inner: object
    known_free_outer: object
    residual_inner: object
    residual_outer: object
    possible_overcut: object

    @property
    def report(self):
        return (self.residual_inner.area, self.residual_outer.area,
                self.possible_overcut.area)


def section_evidence(result, depth, *, final=True):
    """Located conditional GEOS section evidence against the original design."""
    if type(result) is VComposition:
        result = VComposition(result.target, result.stages)
        target = result.target
        plans = tuple(s for s in result.stages if type(s) is VPlan) if final else ()
        prior_cuts = tuple(c for s in result.stages if type(s) is replay.Trace
                           for c in _cylindrical_stock(target, s).cuts) if final else ()
    elif type(result) is VRest:
        # A frozen container is still replaceable. Re-establish source/frame,
        # full-cylinder containment and replay identity before trusting stock.
        checked = with_prior(result.plan, result.prior_trace)
        if checked.prior_stock != result.prior_stock:
            raise ValueError("stale V prior stock evidence")
        plan, prior_cuts = checked.plan, checked.prior_stock.cuts
    elif type(result) is VSequence:
        # Re-establish validity at consumption, as for the other value types.
        result = VSequence(result.plans)
        plan, prior_cuts = result.plans[0], ()
    else:
        plan, prior_cuts = result, ()
    if type(result) is not VComposition:
        verify(plan)
        target = plan.target
        plans = (result.plans if type(result) is VSequence else (plan,)) if final else ()
    depth = _finite(depth, "section depth")
    if not 0 <= depth <= target.cap_depth:
        raise ValueError("section outside V target")
    inner, outer = [], []
    # Each primitive is a disk or straight capsule, so its round caps span
    # exact half/full circles with 32 segments per quadrant.
    inflation = 1 / math.cos(math.pi / 128)
    for cut in prior_cuts:
        section = cut.section_segment(depth)
        if section is not None:
            a, b = section
            line = LineString((a, b)) if a != b else Point(a)
            inner.append(line.buffer(max(0, cut.tool.radius -
                                         cut.path_error_mm - 1e-6),
                                     quad_segs=32))
            outer.append(line.buffer((cut.tool.radius + cut.path_error_mm +
                                      1e-6) * inflation, quad_segs=32))
    for swept_plan in plans:
        for path in swept_plan.paths:
            for a, b in zip(path.points, path.points[1:]):
                low, high = min(a[2], b[2]), max(a[2], b[2])
                line = LineString((a[:2], b[:2]))
                if high >= depth:
                    radius = swept_plan.tool.radius(high - depth) + 1e-6
                    outer.append(line.buffer(radius * inflation, quad_segs=32))
                if low >= depth:
                    radius = max(0.0, swept_plan.tool.radius(low - depth) - 1e-6)
                    if radius:
                        inner.append(line.buffer(radius, quad_segs=32))
    outer_sweep = unary_union(outer) if outer else Polygon()
    inner_sweep = unary_union(inner) if inner else Polygon()
    safe = target.section(depth)
    cover = target.section(depth, outer=True)
    return VSectionEvidence(target.fingerprint, depth, safe, cover,
        inner_sweep, outer_sweep, safe.difference(outer_sweep),
        cover.difference(inner_sweep), outer_sweep.difference(safe))


def section_report(result, depth, *, final=True):
    """Conditional GEOS residual interval for one fixed V design section."""
    return section_evidence(result, depth, final=final).report


def volume_bounds(result, slabs=8, *, final=True):
    """Conservative geometric slab bounds for remaining V-target volume."""
    if type(slabs) is not int or slabs <= 0:
        raise ValueError("positive integer V volume slabs required")
    target = (result.target if type(result) is VComposition else
              result.plans[0].target if type(result) is VSequence else
              result.plan.target if type(result) is VRest else result.target)
    levels = [target.cap_depth * i / slabs for i in range(slabs + 1)]
    levels[-1] = target.cap_depth
    sections = [section_report(result, t, final=final) for t in levels]
    target_safe = [target.section(t).area for t in levels]
    target_outer = [target.section(t, outer=True).area
                    for t in levels]
    lower = upper = 0.0
    for i in range(slabs):
        dz = levels[i + 1] - levels[i]
        removed_a = target_outer[i] - sections[i][0]
        removed_b = target_safe[i + 1] - sections[i + 1][1]
        lower += dz * max(0.0, target_safe[i + 1] - removed_a)
        upper += dz * max(0.0, target_outer[i] - removed_b)
    return lower, upper
