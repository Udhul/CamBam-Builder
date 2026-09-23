"""Bounded, standalone RC01 two-tool motion and continuous-height verification.

All coordinates are millimetres in the named drawing frame.  This module emits
motion values, not controller code or a claim about CamBam's generated motion.
"""

from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import math
from typing import Optional

from .stock import SectionRectangle, SectionTarget


def _q(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Q)):
        raise ValueError("finite numeric coordinate required")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("finite numeric coordinate required")
    return Q(value)


@dataclass(frozen=True)
class Tool:
    name: str
    radius: Q
    cutting_length: Q = Q(6)
    shank_radius: Q = Q(3)
    shank_top: Q = Q(20)
    holder_radius: Q = Q(10)
    holder_start: Q = Q(20)
    holder_top: Q = Q(40)
    rpm: int = 12000

    def __post_init__(self):
        for field in ("radius", "cutting_length", "shank_radius", "shank_top",
                      "holder_radius", "holder_start", "holder_top"):
            object.__setattr__(self, field, _q(getattr(self, field)))
        if not self.name or not (0 < self.radius and 0 < self.cutting_length <
                                 self.shank_top <= self.holder_top) or not (
                0 <= self.holder_start < self.holder_top) or min(
                self.shank_radius, self.holder_radius) <= 0 or self.rpm <= 0:
            raise ValueError("invalid tool components")


@dataclass(frozen=True)
class Job:
    frame: str = "rc01-drawing"
    stock: SectionRectangle = SectionRectangle(-5, -5, 45, 35)
    outer: SectionRectangle = SectionRectangle(0, 0, 40, 30)
    island: SectionRectangle = SectionRectangle(16, 11, 24, 19)
    floor: Q = Q(-3)
    stock_bottom: Q = Q(-10)
    fixture_bottom: Q = Q(-15)
    tools: tuple = (Tool("T1", Q(3)), Tool("T2", Q(1)))
    position_error: Q = Q(0)
    radius_error: Q = Q(0)

    def __post_init__(self):
        SectionTarget(self.stock, self.outer, self.island)
        for field in ("floor", "stock_bottom", "fixture_bottom",
                      "position_error", "radius_error"):
            object.__setattr__(self, field, _q(getattr(self, field)))
        if (not self.frame or self.floor >= 0 or self.stock_bottom >= self.floor
                or self.fixture_bottom >= self.stock_bottom or
                self.position_error < 0 or self.radius_error < 0 or
                len(self.tools) != 2 or tuple(t.name for t in self.tools) != ("T1", "T2")):
            raise ValueError("unsupported RC01 job")

    @property
    def fingerprint(self):
        values = (self.frame, self.stock, self.outer, self.island, self.floor,
                  self.stock_bottom, self.fixture_bottom, self.tools,
                  self.position_error, self.radius_error)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Move:
    role: str
    tool: str
    start: tuple
    end: tuple
    feed: int
    operation: str

    def __post_init__(self):
        if self.role not in ("rapid", "approach", "entry", "cleared_descent",
                             "cut", "retract"):
            raise ValueError("unknown move role")
        object.__setattr__(self, "start", tuple(map(_q, self.start)))
        object.__setattr__(self, "end", tuple(map(_q, self.end)))
        if len(self.start) != 3 or len(self.end) != 3 or self.start == self.end:
            raise ValueError("move needs distinct XYZ endpoints")
        if self.feed < 0 or (self.role == "rapid") != (self.feed == 0):
            raise ValueError("invalid feed for role")


@dataclass(frozen=True)
class Event:
    kind: str
    tool: str
    position: tuple
    operation: str
    rpm: int = 0
    direction: str = "stopped"
    coolant: str = "off"

    def __post_init__(self):
        if self.kind not in ("tool_change", "spindle_start", "spindle_stop"):
            raise ValueError("unknown event")
        object.__setattr__(self, "position", tuple(map(_q, self.position)))
        if (len(self.position) != 3 or self.coolant != "off" or self.rpm < 0
                or self.direction not in ("CW", "stopped")):
            raise ValueError("invalid spindle/setup event")


@dataclass(frozen=True)
class Program:
    job_fingerprint: str
    items: tuple
    frame: str = "rc01-drawing"
    units: str = "mm"
    tip_datum: str = "flat_endmill_bottom"

    @property
    def motion_fingerprint(self):
        return hashlib.sha256(repr((self.job_fingerprint, self.items, self.frame,
                                    self.units, self.tip_datum)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Certificate:
    fingerprint: str
    motion_fingerprint: str
    moves: int
    rough_cuts: tuple
    final_cuts: tuple
    rough_rest_area: tuple
    final_rest_area: tuple
    rough_rest_volume: tuple
    final_rest_volume: tuple
    rough_rest_by_depth: tuple
    final_rest_by_depth: tuple
    area_coordinate_enclosure_mm: float
    location_polygon_sagitta_mm: float
    location_numeric_enclosure_mm: Optional[float]
    status: str = "partial_target_completion"


SETUP = (Q(-10), Q(-10), Q(5))


def _rows(radius, step, window):
    x0, y0, x1, y1 = window
    n = round(float((y1 - y0 - 2 * radius) / step))
    return tuple(y0 + radius + i * step for i in range(n + 1))


def _rough_runs():
    # The wall traverses remove raster endpoint scallops.  Every cut is a
    # horizontal or vertical analytic capsule; the island corner endpoints
    # round away from its forbidden radius-3 offset.
    yield ((Q(5), Q(5)), (Q(3), Q(5)))
    yield ((Q(3), Q(3)), (Q(3), Q(27)))
    yield ((Q(37), Q(3)), (Q(37), Q(27)))
    for i in range(49):
        y = Q(3) + Q(i, 2)
        if y in (Q(8), Q(22)) or not Q(8) < y < Q(22):
            yield ((Q(3), y), (Q(37), y))
            continue
        distance = min(abs(float(y - 11)), abs(float(y - 19)))
        reach = 3.0 if 11 <= y <= 19 else math.sqrt(9 - distance * distance)
        left = Q(math.floor((16 - reach) * 1_000_000), 1_000_000)
        right = 40 - left
        yield ((Q(3), y), (left, y))
        yield ((right, y), (Q(37), y))


def _cleanup_runs():
    for x0, y0, x1, y1 in ((0, 0, 7, 7), (33, 0, 40, 7),
                           (33, 23, 40, 30), (0, 23, 7, 30)):
        for i in range(21):
            y = Q(y0 + 1) + Q(i, 4)
            yield ((Q(x0 + 1), y), (Q(x1 - 1), y)), (Q(x0 + 5) if x0 == 0 else Q(x0 + 2),
                                                   Q(y0 + 5) if y0 == 0 else Q(y0 + 2))


def generate(job=Job()):
    """Generate deterministic T1 roughing and four-window T2 cleanup at 3 levels."""
    if type(job) is not Job or job != Job():
        raise ValueError("generator currently supports the exact nominal RC01 job")
    items, at = [], SETUP
    for tool, op, runs in ((job.tools[0], "rough", tuple(_rough_runs())),
                           (job.tools[1], "cleanup", tuple(_cleanup_runs()))):
        items.append(Event("tool_change", tool.name, at, op))
        items.append(Event("spindle_start", tool.name, at, op, tool.rpm, "CW"))
        for depth in (-1, -2, -3):
            for run in runs:
                (a, b), column = (run, None) if op == "rough" else run
                # T2 always enters through one of the four T1-cleared columns,
                # then cuts from that point to the corner-window row start.
                if column is not None:
                    a = column
                def add(role, end, feed):
                    nonlocal at
                    end = tuple(map(_q, end))
                    if at != end:
                        items.append(Move(role, tool.name, at, end, feed, op))
                        at = end
                add("rapid", (a[0], a[1], 5), 0)
                add("approach", (a[0], a[1], 1), 120)
                add("entry" if op == "rough" else "cleared_descent",
                    (a[0], a[1], depth), 60 if op == "rough" else 120)
                if op == "cleanup":
                    add("cut", (run[0][0][0], a[1], depth), 300)
                    add("cut", (run[0][0][0], run[0][0][1], depth), 300)
                    add("cut", (run[0][1][0], run[0][1][1], depth), 300)
                else:
                    add("cut", (b[0], b[1], depth), 300)
                add("retract", (at[0], at[1], 5), 300)
        if at != SETUP:
            items.append(Move("rapid", tool.name, at, SETUP, 0, op))
            at = SETUP
        items.append(Event("spindle_stop", tool.name, at, op))
    return Program(job.fingerprint, tuple(items), job.frame)


def _segment_distance2(point, a, b):
    x, y = point
    if a[1] == b[1]:
        return (max(min(a[0], b[0]) - x, 0, x - max(a[0], b[0])) ** 2 +
                (y - a[1]) ** 2)
    if a[0] == b[0]:
        return ((x - a[0]) ** 2 +
                max(min(a[1], b[1]) - y, 0, y - max(a[1], b[1])) ** 2)
    raise ValueError("RC01 proof supports axis-aligned stock motions")


def _allowed_cut(job, a, b, radius):
    o, island = job.outer, job.island
    if any(not (o.xmin + radius <= x <= o.xmax - radius and
                o.ymin + radius <= y <= o.ymax - radius) for x, y in (a, b)):
        return False
    if a[1] == b[1]:
        dx = max(island.xmin - max(a[0], b[0]), min(a[0], b[0]) - island.xmax, 0)
        dy = max(island.ymin - a[1], a[1] - island.ymax, 0)
    elif a[0] == b[0]:
        dx = max(island.xmin - a[0], a[0] - island.xmax, 0)
        dy = max(island.ymin - max(a[1], b[1]), min(a[1], b[1]) - island.ymax, 0)
    else:
        raise ValueError("diagonal cutting is outside the RC01 proof")
    return dx * dx + dy * dy >= radius * radius


def _covered_by_single(cuts, a, b, radius, bottom):
    """Exact sufficient clearance proof from one earlier swept cylinder."""
    for prior_bottom, prior_top, pa, pb, pr, _ in cuts:
        margin = pr - radius
        if prior_bottom <= bottom and prior_top >= 0 and margin >= 0 and all(
                _segment_distance2(p, pa, pb) <= margin * margin for p in (a, b)):
            return True
    return False


def _capsule_polygon(a, b, radius, resolution):
    from shapely.geometry import LineString
    return LineString((tuple(map(float, a)), tuple(map(float, b)))).buffer(
        float(radius), quad_segs=resolution)


def _exact_area_interval(job, cuts, strip_nm=1_000_000):
    """Rational strip enclosure of the union of complete axis-aligned capsules.

    Integer nanometre endpoints and directed integer square roots enclose every
    circle intersection over a whole Y strip.  The returned area bounds do not
    depend on GEOS, point sampling, or the generator's feasibility calculation.
    """
    scale = 1_000_000_000
    def nm(value):
        number = _q(value) * scale
        if number.denominator != 1:
            raise ValueError("RC01 coordinate finer than integer nanometres")
        return int(number)
    def union_length(intervals, spans):
        total = 0
        for lo, hi in spans:
            clipped = sorted((max(lo, a), min(hi, b)) for a, b in intervals
                             if a < hi and b > lo)
            end = lo
            for a, b in clipped:
                if b > end:
                    total += b - max(a, end)
                    end = b
        return total
    capsules = []
    for _, _, a, b, r, _ in cuts:
        capsules.append((min(nm(a[0]), nm(b[0])), max(nm(a[0]), nm(b[0])),
                         min(nm(a[1]), nm(b[1])), max(nm(a[1]), nm(b[1])),
                         nm(r)))
    capsules = tuple(dict.fromkeys(capsules))
    y_min, y_max = nm(job.outer.ymin), nm(job.outer.ymax)
    if (y_max - y_min) % strip_nm:
        raise ValueError("strip step must divide RC01 target height")
    x_min, x_max = nm(job.outer.xmin), nm(job.outer.xmax)
    island_x = (nm(job.island.xmin), nm(job.island.xmax))
    island_y = (nm(job.island.ymin), nm(job.island.ymax))
    strip_count = (y_max - y_min) // strip_nm
    starts = [[] for _ in range(strip_count + 1)]
    stops = [[] for _ in range(strip_count + 1)]
    for index, (_, _, ya, yb, radius) in enumerate(capsules):
        first = max(0, (ya - radius - y_min) // strip_nm)
        last = min(strip_count - 1, (yb + radius - y_min) // strip_nm)
        if first <= last:
            starts[first].append(index)
            stops[last + 1].append(index)
    removed_inner = removed_outer = 0
    active = set()
    for strip_index in range(strip_count):
        active.difference_update(stops[strip_index])
        active.update(starts[strip_index])
        y0 = y_min + strip_index * strip_nm
        y1 = y0 + strip_nm
        spans = ((x_min, island_x[0]), (island_x[1], x_max)) if (
            island_y[0] <= y0 and y1 <= island_y[1]) else ((x_min, x_max),)
        possible, guaranteed = [], []
        for capsule_index in active:
            xa, xb, ya, yb, radius = capsules[capsule_index]
            near = max(ya - y1, y0 - yb, 0)
            if near > radius:
                continue
            outer_sq = radius * radius - near * near
            outer_extent = math.isqrt(outer_sq)
            if outer_extent * outer_extent < outer_sq:
                outer_extent += 1
            possible.append((xa - outer_extent, xb + outer_extent))
            far = max(ya - y0, y0 - yb, ya - y1, y1 - yb, 0)
            if far <= radius:
                guaranteed_extent = math.isqrt(radius * radius - far * far)
                guaranteed.append((xa - guaranteed_extent,
                                   xb + guaranteed_extent))
        removed_inner += union_length(guaranteed, spans) * strip_nm
        removed_outer += union_length(possible, spans) * strip_nm
    target_nm2 = int(job.outer.area - job.island.area) * scale * scale
    lo = (target_nm2 - removed_outer) / (scale * scale)
    hi = (target_nm2 - removed_inner) / (scale * scale)
    return (math.nextafter(max(0.0, lo), -math.inf),
            math.nextafter(hi, math.inf))


def _rest_interval(job, cuts, radius, resolution=1024):
    """Exact area enclosure plus independent polygonal residual-location check."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    target = box(*map(float, (job.outer.xmin, job.outer.ymin,
                              job.outer.xmax, job.outer.ymax))).difference(
        box(*map(float, (job.island.xmin, job.island.ymin,
                          job.island.xmax, job.island.ymax))))
    unique = tuple(dict.fromkeys((a, b, r) for _, _, a, b, r, _ in cuts))
    inner = unary_union([_capsule_polygon(a, b, r, resolution)
                         for a, b, r in unique])
    rest = target.difference(inner)
    # A separate location oracle catches narrow missed strips whose area alone
    # would fit the budget.  Only the four finite-tool outer corner rests are
    # ideal; the original island and outer walls remain boundary authorities.
    corners = ((job.outer.xmin, job.outer.ymin, 1, 1),
               (job.outer.xmax, job.outer.ymin, -1, 1),
               (job.outer.xmax, job.outer.ymax, -1, -1),
               (job.outer.xmin, job.outer.ymax, 1, -1))
    ideals = []
    for x, y, sx, sy in corners:
        x, y, r = float(x), float(y), float(radius)
        square = box(min(x, x + sx*r), min(y, y + sy*r),
                     max(x, x + sx*r), max(y, y + sy*r))
        disk = _capsule_polygon((Q(x + sx*r), Q(y + sy*r)),
                                (Q(x + sx*r), Q(y + sy*r)), radius, resolution)
        ideals.append(square.difference(disk))
    allowed_rest = unary_union(ideals).buffer(0.05).union(target.boundary.buffer(0.05))
    if rest.difference(allowed_rest).area > 1e-7:
        raise ValueError("residual location exceeds 0.05 mm bound")
    return _exact_area_interval(job, cuts)


def verify(program, job=Job(), *, measure_rest=True):
    """Replay ordered whole-tool motion over continuous Z intervals.

    Access is deliberately conservative: below-stock non-cutting occupancy must
    fit one prior guaranteed cylinder across every affected height.  Unsupported
    diagonal/between-volume routes fail closed instead of gaining clearance from
    a sampled section.  The nominal RC01 generator needs no such union routes.
    """
    if type(program) is not Program or type(job) is not Job or (
            program.job_fingerprint != job.fingerprint or program.frame != job.frame
            or program.units != "mm" or program.tip_datum != "flat_endmill_bottom"):
        raise ValueError("stale or incompatible RC01 motion evidence")
    baseline = Job()
    if (job.frame, job.stock, job.outer, job.island, job.floor,
            job.stock_bottom, job.fixture_bottom) != (
            baseline.frame, baseline.stock, baseline.outer, baseline.island,
            baseline.floor, baseline.stock_bottom, baseline.fixture_bottom):
        raise ValueError("unsupported RC01 geometry/setup change")
    tools = {t.name: t for t in job.tools}
    if job.position_error or job.radius_error:
        raise ValueError("positive physical uncertainty requires regenerated clearance")
    at, active, running, cuts, rough = SETUP, None, False, [], None
    tool_changes = []
    for index, item in enumerate(program.items):
        if isinstance(item, Event):
            if item.position != at or at != SETUP:
                raise ValueError(f"event {index}: tool/setup position mismatch")
            if item.rpm != (tools[item.tool].rpm if item.kind == "spindle_start"
                            and item.tool in tools else 0):
                raise ValueError(f"event {index}: spindle speed mismatch")
            if item.direction != ("CW" if item.kind == "spindle_start" else "stopped"):
                raise ValueError(f"event {index}: spindle direction mismatch")
            if item.operation != ("rough" if item.tool == "T1" else "cleanup"):
                raise ValueError(f"event {index}: operation mismatch")
            if item.kind == "tool_change":
                if running or item.tool not in tools or item.tool == active:
                    raise ValueError(f"event {index}: invalid tool change")
                if item.tool == "T2":
                    if active != "T1":
                        raise ValueError("T2 requires T1 predecessor")
                    rough = tuple(cuts)
                active = item.tool
                tool_changes.append(active)
            elif item.kind == "spindle_start":
                if running or active != item.tool:
                    raise ValueError(f"event {index}: invalid spindle start")
                running = True
            else:
                if not running or active != item.tool:
                    raise ValueError(f"event {index}: invalid spindle stop")
                running = False
            continue
        if type(item) is not Move or active != item.tool or not running or item.start != at:
            raise ValueError(f"move {index}: discontinuity or inactive tool")
        t = tools[active]
        a, b = item.start[:2], item.end[:2]
        z0, z1 = item.start[2], item.end[2]
        if (any(not (-15 <= x <= 55 and -15 <= y <= 45) for x, y in (a, b))
                or not (-3 <= z0 <= 10 and -3 <= z1 <= 10)):
            raise ValueError(f"move {index}: travel bound")
        if item.operation != ("rough" if active == "T1" else "cleanup"):
            raise ValueError(f"move {index}: operation mismatch")
        expected_feed = {"rapid": 0, "approach": 120, "entry": 60,
                         "cleared_descent": 120, "cut": 300, "retract": 300}[item.role]
        if item.feed != expected_feed:
            raise ValueError(f"move {index}: process feed mismatch")
        if item.role == "rapid" and (z0 != 5 or z1 != 5):
            raise ValueError(f"move {index}: rapid below clearance")
        if item.role in ("approach", "entry", "cleared_descent", "retract") and a != b:
            raise ValueError(f"move {index}: vertical role changes XY")
        if item.role == "approach" and (z0, z1) != (5, 1):
            raise ValueError(f"move {index}: invalid approach")
        if item.role == "retract" and z1 != 5:
            raise ValueError(f"move {index}: invalid retract")
        prior_cuts = tuple(cuts)
        if item.role in ("entry", "cut"):
            if item.role == "entry" and (active != "T1" or z0 != 1 or z1 >= 0):
                raise ValueError(f"move {index}: invalid cutting entry")
            if item.role == "cut" and (z0 != z1 or z1 >= 0):
                raise ValueError(f"move {index}: invalid level cut")
            bottom = min(z0, z1)
            if bottom < job.floor or not _allowed_cut(job, a, b,
                                                     t.radius + job.radius_error + job.position_error):
                raise ValueError(f"move {index}: protected target/floor overcut")
            # A cut at a new level can engage at most 1 mm of new stock.  Its
            # entire disk sweep above that layer must match guaranteed earlier
            # removal; this is exact containment, not a height sample.
            if bottom < -1 and not _covered_by_single(cuts, a, b, t.radius, bottom + 1):
                raise ValueError(f"move {index}: axial engagement exceeds 1 mm")
            cuts.append((bottom, Q(0) if item.role == "entry" else
                         min(Q(0), bottom + t.cutting_length),
                         a, b, t.radius, active))
        elif item.role in ("cleared_descent", "retract") or (
                item.role == "rapid" and min(z0, z1) < 0):
            low = min(z0, z1)
            if low < 0 and not _covered_by_single(cuts, a, b, t.radius, low):
                raise ValueError(f"move {index}: uncleared travel/access")
        # The entire cutting cylinder, shank and holder Z intervals are checked
        # against stock/fixture slabs.  Nominal RC01 non-cutting parts are above
        # the top; changed component reach cannot inherit that clearance.
        low_tip = min(z0, z1)
        if low_tip < job.fixture_bottom:
            raise ValueError(f"move {index}: fixture collision")
        for lower, upper, radius in ((t.cutting_length, t.shank_top, t.shank_radius),
                                     (t.holder_start, t.holder_top, t.holder_radius)):
            if low_tip + lower < 0 and max(z0, z1) + upper > job.stock_bottom:
                if not _covered_by_single(prior_cuts, a, b, radius, low_tip + lower):
                    raise ValueError(f"move {index}: non-cutting component stock collision")
        at = item.end
    if running or at != SETUP or rough is None or tool_changes != ["T1", "T2"] or not program.items or not isinstance(
            program.items[-1], Event) or program.items[-1].kind != "spindle_stop":
        raise ValueError("incomplete setup/tool sequence")
    # Process stepovers are computed from actual cut centerlines, separately
    # for each disconnected cleanup window and depth.
    for tool_name, limit, windows in (("T1", Q(12, 5), ((0, 40, 0, 30),)),
                                      ("T2", Q(4, 5), ((0, 7, 0, 7),
                                                      (33, 40, 0, 7),
                                                      (33, 40, 23, 30),
                                                      (0, 7, 23, 30)))):
        for depth in (-1, -2, -3):
            for x0, x1, y0, y1 in windows:
                rows = sorted({a[1] for bottom, _, a, b, _, name in cuts
                               if name == tool_name and bottom == depth and
                               a[1] == b[1] and x0 <= a[0] <= x1 and
                               y0 <= a[1] <= y1})
                if len(rows) < 2 or any(b - a > limit for a, b in zip(rows, rows[1:])):
                    raise ValueError("lateral stepover exceeds process bound")
    for corner in ((5, 5), (35, 5), (35, 25), (5, 25)):
        p = tuple(map(Q, corner))
        if not _covered_by_single(rough, p, p, Q(1), Q(-3)):
            raise ValueError("missing T1-cleared T2 corner column")
    area_cache = {}
    def by_depth(prefix, radius):
        if not measure_rest:
            return ((float("nan"), float("nan")),) * 3
        result = []
        for bottom in (-1, -2, -3):
            section = tuple(c for c in prefix if c[0] <= bottom and c[1] >= 0)
            key = (tuple(dict.fromkeys((a, b, r) for _, _, a, b, r, _
                                       in section)), radius)
            if key not in area_cache:
                try:
                    area_cache[key] = _rest_interval(job, section, radius)
                except ValueError as exc:
                    raise ValueError(f"residual at Z={bottom}: {exc}") from exc
            area = area_cache[key]
            ideal = (4 - math.pi) * float(radius) ** 2
            if area[1] < ideal or area[1] > ideal + 0.5:
                raise ValueError(f"residual at Z={bottom} outside RC01 budget: {area}")
            result.append(area)
        return tuple(result)
    rough_layers = by_depth(rough, Q(3))
    final_layers = by_depth(cuts, Q(1))
    rough_area = (min(a[0] for a in rough_layers), max(a[1] for a in rough_layers))
    final_area = (min(a[0] for a in final_layers), max(a[1] for a in final_layers))
    rough_volume = (sum(a[0] for a in rough_layers), sum(a[1] for a in rough_layers))
    final_volume = (sum(a[0] for a in final_layers), sum(a[1] for a in final_layers))
    if measure_rest and any(rough_layer[0] - final_layer[1] <
                            (4 - math.pi) * 8 - 0.5
                            for rough_layer, final_layer in zip(rough_layers, final_layers)):
        raise ValueError("cleanup benefit below RC01 budget")
    return Certificate(job.fingerprint, program.motion_fingerprint,
                       sum(isinstance(i, Move) for i in program.items),
                       rough, tuple(cuts), rough_area, final_area,
                       rough_volume, final_volume, rough_layers, final_layers,
                       1e-9 if measure_rest else float("nan"),
                       3 * (1 - math.cos(math.pi / (4 * 1024)))
                       if measure_rest else float("nan"), None,
                       "partial_target_completion" if measure_rest else "motion_only")
