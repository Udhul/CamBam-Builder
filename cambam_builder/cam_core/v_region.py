"""Conservative variable-depth V paths on a Region with optional curved bounds.

The finish target at section depth ``t`` is the source Region eroded by
``t * tan(angle / 2)``.  A path is accepted only when its entire cutting profile
stays in that target at every height.  Polygonal and curved source bounds are
kept separately so an arc chord cannot silently become the protected boundary.
"""

from dataclasses import dataclass
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
        if radius < self.radius(0) - 1e-10:
            return None
        if radius >= self.radius(self.cutting_length):
            return self.cutting_length
        if self.kind == "pointed":
            return radius / self.tangent
        if self.kind == "flat":
            return (radius - self.tip_radius) / self.tangent
        join_radius = self.radius(self.join_height)
        if radius <= join_radius:
            return self.tip_radius - math.sqrt(max(0.0, self.tip_radius ** 2 - radius ** 2))
        return self.join_height + (radius - join_radius) / self.tangent

    def occupancy_radius(self, penetration, section_depth):
        """Radius consumed from the original boundary at one stock section."""
        if not 0 <= section_depth <= penetration <= self.cutting_length:
            raise ValueError("section outside cutter penetration")
        return section_depth * self.tangent + self.radius(penetration - section_depth)


@dataclass(frozen=True)
class VTarget:
    source_id: str
    safe: Polygon
    outer: Polygon
    cap_depth: float
    sagitta_mm: float = 0.0

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

    @classmethod
    def polygon(cls, source_id, shell, holes, cap_depth):
        shape = Polygon(shell, holes)
        return cls(source_id, shape, shape, cap_depth)

    @classmethod
    def curved(cls, source_id, approximation, cap_depth):
        return cls(source_id, approximation.safe, approximation.outer,
                   cap_depth, approximation.sagitta_mm)

    @property
    def fingerprint(self):
        values = (self.source_id, self.safe.wkb_hex, self.outer.wkb_hex,
                  self.cap_depth, self.sagitta_mm)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()

    def section(self, depth, tangent, *, outer=False):
        depth = _finite(depth, "section depth")
        if not 0 <= depth <= self.cap_depth:
            raise ValueError("section outside V target")
        shape = self.outer if outer else self.safe
        return shape.buffer(-depth * tangent, quad_segs=64) if depth else shape


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

    @property
    def fingerprint(self):
        values = (self.target.fingerprint, self.tool, self.paths, self.motions,
                  self.safe_z, self.margin_mm, self.stepover_mm, self.status)
        if self.fill_pattern != "raster":
            values += (self.fill_pattern,)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


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


def with_prior(plan, prior_trace):
    """Replay supplied prior motion and prove its entire cylinder fits the V target."""
    verify(plan)
    if (type(prior_trace) is not replay.Trace or
            prior_trace.source_fingerprint != plan.target.source_id):
        raise ValueError("stale V prior source")
    stock = replay.replay(prior_trace,
                          expected_source=plan.target.source_id)
    if not stock.cuts or len(prior_trace.operations) != 1:
        raise ValueError("complete V prior cut required")
    op = prior_trace.operations[0]
    if (op.tool.kind != "cylinder" or not op.target.region_shell or
            op.target.depth != plan.target.cap_depth or
            not Polygon(op.target.region_shell,
                        op.target.region_holes).equals(plan.target.safe)):
        raise ValueError("V prior tool or target differs from source")
    boundary = plan.target.safe.boundary
    for cut in stock.cuts:
        depth = -cut.bottom
        line = LineString((cut.a, cut.b)) if cut.a != cut.b else Point(cut.a)
        required = cut.tool.radius + depth * plan.tool.tangent
        if (cut.tool != op.tool or not 0 < depth <= plan.target.cap_depth or
                not plan.target.safe.covers(line) or
                line.distance(boundary) + 1e-8 < required + plan.margin_mm * 0.5):
            raise ValueError("V prior cut crosses capped finish target")
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


def _depth_path(target, tool, xy, cap, margin, role):
    boundary = target.safe.boundary
    edge_caps = []
    for a, b in zip(xy, xy[1:]):
        clearance = LineString((a, b)).distance(boundary) - margin
        available = tool.depth_for_radius(max(0.0, clearance))
        edge_caps.append(min(cap, available or 0.0))
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
    """Trace cap boundaries and fill with varying-Z rows or offset contours."""
    if type(target) is not VTarget or type(tool) is not VProfile:
        raise ValueError("V Region target and tool required")
    stepover_mm = _finite(stepover_mm, "stepover")
    xy_step_mm = _finite(xy_step_mm, "XY step")
    margin_mm = _finite(margin_mm, "clearance margin")
    safe_z = _finite(safe_z, "safe Z")
    if (stepover_mm <= 0 or xy_step_mm <= 0 or margin_mm <= 1e-5 or
            safe_z <= 0 or type(max_paths) is not int or max_paths <= 0 or
            target.cap_depth > tool.cutting_length or
            tool.radius(target.cap_depth) > tool.maximum_radius or
            fill_pattern not in ("raster", "offset")):
        raise ValueError("V path controls exceed tool/setup limits")
    cap = target.cap_depth
    deep = target.safe.buffer(-(tool.radius(cap) + margin_mm), quad_segs=32)
    start_depth = min(0.05, cap)
    reachable = target.safe.buffer(-(tool.radius(start_depth) + margin_mm),
                                   quad_segs=32)
    if reachable.is_empty:
        result = VPlan(target, tool, (), (), safe_z, margin_mm, stepover_mm,
                       "infeasible", "no positive-depth cutter center fits",
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
    status = "partial" if paths else "infeasible"
    reason = ("finite stepover and finite tip leave measured residual" if paths else
              "no admissible positive-depth path")
    result = VPlan(target, tool, tuple(paths), _motions(paths, safe_z),
                   safe_z, margin_mm, stepover_mm, status, reason, fill_pattern)
    verify(result)
    return result


def verify(result):
    """Check the full profile between vertices and complete ordered motion."""
    if (type(result) is not VPlan or result.status not in ("partial", "infeasible") or
            result.fill_pattern not in ("raster", "offset")):
        raise ValueError("invalid V plan")
    if result.motions != _motions(result.paths, result.safe_z):
        raise ValueError("V motion differs from paths")
    if not result.paths:
        if result.status != "infeasible":
            raise ValueError("missing V paths")
        return result
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
            # For all t <= d, t*tan(alpha) + rho(d-t) <= rho(d).
            # Therefore this one continuous-line distance check covers every
            # cutter height, including a changing tip Z between vertices.
            if (not result.target.safe.covers(segment) or
                    segment.distance(boundary) + 1e-8 <
                    tool.radius(max(a[2], b[2])) + result.margin_mm * 0.5):
                raise ValueError("full V cutter crosses protected Region")
    for motion in result.motions:
        if motion.role == "rapid" and min(motion.start[2], motion.end[2]) <= 0:
            raise ValueError("low V rapid")
    return result


def section_report(result, depth, *, final=True):
    """Conditional GEOS residual interval for one V target section."""
    if type(result) is VRest:
        plan, prior_cuts = result.plan, result.prior_stock.cuts
    else:
        plan, prior_cuts = result, ()
    verify(plan)
    depth = _finite(depth, "section depth")
    if not 0 <= depth <= plan.target.cap_depth:
        raise ValueError("section outside V target")
    inner, outer = [], []
    for cut in prior_cuts:
        if -cut.bottom >= depth:
            line = LineString((cut.a, cut.b)) if cut.a != cut.b else Point(cut.a)
            inner.append(line.buffer(max(0, cut.tool.radius - 1e-6), quad_segs=32))
            outer.append(line.buffer(cut.tool.radius + 1e-6, quad_segs=32))
    for path in plan.paths if final else ():
        for a, b in zip(path.points, path.points[1:]):
            low, high = min(a[2], b[2]), max(a[2], b[2])
            line = LineString((a[:2], b[:2]))
            if high >= depth:
                radius = plan.tool.radius(high - depth) + 1e-6
                outer.append(line.buffer(radius, quad_segs=32))
            if low >= depth:
                radius = max(0.0, plan.tool.radius(low - depth) - 1e-6)
                if radius:
                    inner.append(line.buffer(radius, quad_segs=32))
    outer_sweep = unary_union(outer) if outer else Polygon()
    inner_sweep = unary_union(inner) if inner else Polygon()
    safe = plan.target.section(depth, plan.tool.tangent)
    cover = plan.target.section(depth, plan.tool.tangent, outer=True)
    lower = max(0.0, safe.area - safe.intersection(outer_sweep).area)
    upper = max(0.0, cover.area - cover.intersection(inner_sweep).area)
    return (lower, upper, outer_sweep.difference(safe).area)


def volume_bounds(result, slabs=8, *, final=True):
    """Conservative geometric slab bounds for remaining V-target volume."""
    if type(slabs) is not int or slabs <= 0:
        raise ValueError("positive integer V volume slabs required")
    plan = result.plan if type(result) is VRest else result
    levels = [plan.target.cap_depth * i / slabs for i in range(slabs + 1)]
    sections = [section_report(result, t, final=final) for t in levels]
    target_safe = [plan.target.section(t, plan.tool.tangent).area for t in levels]
    target_outer = [plan.target.section(t, plan.tool.tangent, outer=True).area
                    for t in levels]
    lower = upper = 0.0
    for i in range(slabs):
        dz = levels[i + 1] - levels[i]
        removed_a = target_outer[i] - sections[i][0]
        removed_b = target_safe[i + 1] - sections[i + 1][1]
        lower += dz * max(0.0, target_safe[i + 1] - removed_a)
        upper += dz * max(0.0, target_outer[i] - removed_b)
    return lower, upper
