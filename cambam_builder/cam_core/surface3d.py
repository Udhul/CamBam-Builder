"""Bounded analytic height-field stock and ball-end motion evidence.

Exact affine or spherical-bowl surfaces supply contact and protected-material
tests. XY cells bound remaining volume for decoded straight XYZ ball-center
sweeps. No mesh or material outside the declared rectangular stock is represented.
"""

from dataclasses import dataclass
import math
from numbers import Real
import time
import tracemalloc


VERSION = "sloped-ball-v1"
BOWL_VERSION = "spherical-bowl-ball-v1"
ACCESS_MATCH_TOLERANCE_MM = 0.00006  # decoded G-code coordinates use 4 decimals
PLANE_TOLERANCE_MM = 0.0
BOWL_TOLERANCE_MM = 0.0


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SlopedTarget:
    """Stock top Z=0; requested floor Z=-(intercept + slope*x)."""

    stock_xy: tuple
    stock_depth_mm: float
    intercept_mm: float
    slope: float

    def __post_init__(self):
        rect = self.stock_xy
        if (type(rect) is not tuple or len(rect) != 4 or
                not all(_number(v) for v in rect) or
                rect[0] >= rect[2] or rect[1] >= rect[3] or
                not all(_number(v) for v in (self.stock_depth_mm,
                                              self.intercept_mm, self.slope)) or
                self.stock_depth_mm <= 0 or self.slope == 0):
            raise ValueError("invalid sloped target")
        depths = (self.depth_at(rect[0]), self.depth_at(rect[2]))
        if min(depths) <= 0 or max(depths) > self.stock_depth_mm:
            raise ValueError("slope lies outside initial stock")

    def depth_at(self, x):
        return self.intercept_mm + self.slope * x

    @property
    def target_volume_mm3(self):
        x0, y0, x1, y1 = self.stock_xy
        return (x1 - x0) * (y1 - y0) * self.depth_at((x0 + x1) / 2)

    def section_area_mm2(self, depth):
        if not _number(depth) or depth < 0 or depth > self.stock_depth_mm:
            raise ValueError("section outside stock")
        x0, y0, x1, y1 = self.stock_xy
        crossing = (depth - self.intercept_mm) / self.slope
        width = (x1 - max(x0, crossing) if self.slope > 0 else
                 min(x1, crossing) - x0)
        return max(0.0, min(x1 - x0, width)) * (y1 - y0)


@dataclass(frozen=True)
class SphericalBowlTarget:
    """Spherical-cap recess with a protected flat rim inside rectangular stock."""

    stock_xy: tuple
    stock_depth_mm: float
    center_xy: tuple
    rim_radius_mm: float
    depth_mm: float

    def __post_init__(self):
        v = self.stock_xy
        c = self.center_xy
        if (type(v) is not tuple or len(v) != 4 or
                not all(_number(n) for n in v) or
                v[0] >= v[2] or v[1] >= v[3] or
                type(c) is not tuple or len(c) != 2 or
                not all(_number(n) for n in c) or
                not all(_number(n) for n in
                        (self.stock_depth_mm, self.rim_radius_mm,
                         self.depth_mm)) or
                self.stock_depth_mm <= 0 or self.rim_radius_mm <= 0 or
                not 0 < self.depth_mm < min(self.stock_depth_mm,
                                             self.rim_radius_mm) or
                c[0] - self.rim_radius_mm <= v[0] or
                c[0] + self.rim_radius_mm >= v[2] or
                c[1] - self.rim_radius_mm <= v[1] or
                c[1] + self.rim_radius_mm >= v[3]):
            raise ValueError("invalid spherical bowl and protected rim")
        try:
            representable = math.isfinite(self.sphere_radius_mm)
        except (OverflowError, ZeroDivisionError):
            representable = False
        if not representable:
            raise ValueError("spherical bowl radius is not representable")

    @property
    def sphere_radius_mm(self):
        return (self.rim_radius_mm ** 2 + self.depth_mm ** 2) / (2 * self.depth_mm)

    @property
    def rim_plane_offset_mm(self):
        return self.sphere_radius_mm - self.depth_mm

    def depth_at(self, x, y):
        radial = math.hypot(x - self.center_xy[0], y - self.center_xy[1])
        if radial >= self.rim_radius_mm:
            return 0.0
        sphere = self.sphere_radius_mm
        rim_plane = self.rim_plane_offset_mm
        # Rationalize the cap height: subtracting two nearly equal radii
        # erases shallow bowls even when their input depth is representable.
        return ((self.rim_radius_mm - radial) *
                ((self.rim_radius_mm + radial) / sphere) /
                (math.sqrt(1 - (radial / sphere) ** 2) +
                 rim_plane / sphere))

    @property
    def target_volume_mm3(self):
        r, h = self.rim_radius_mm, self.depth_mm
        return math.pi * h * (3 * r * r + h * h) / 6

    def section_area_mm2(self, depth):
        if not _number(depth) or depth < 0 or depth > self.stock_depth_mm:
            raise ValueError("section outside stock")
        if depth >= self.depth_mm:
            return 0.0
        cap_height = self.depth_mm - depth
        sphere = self.sphere_radius_mm
        return math.pi * (cap_height * sphere) * (
            2 - cap_height / sphere)


@dataclass(frozen=True)
class SurfaceOperation:
    name: str
    tool_id: str
    radius_mm: float
    cutting_length_mm: float
    target: SlopedTarget
    strategy: str

    def __post_init__(self):
        if (not self.name or not self.tool_id or
                self.strategy not in ("surface_follow", "rest") or
                not _number(self.radius_mm) or self.radius_mm <= 0 or
                not _number(self.cutting_length_mm) or
                self.cutting_length_mm <= 0 or
                type(self.target) not in (SlopedTarget, SphericalBowlTarget)):
            raise ValueError("invalid ball surface operation")


def contact_tip_z(target, radius_mm, x, *, clearance_mm=0):
    """Ball-tip Z for tangency to the infinite design plane at center X."""
    if (type(target) is not SlopedTarget or not _number(radius_mm) or
            radius_mm <= 0 or not _number(x) or
            not _number(clearance_mm) or clearance_mm < 0):
        raise ValueError("invalid contact query")
    return (-target.depth_at(x) +
            radius_mm * (math.sqrt(1 + target.slope ** 2) - 1) +
            clearance_mm)


def contact_x(target, radius_mm, center_x):
    """Exact point of tangent contact on the infinite plane."""
    if (type(target) is not SlopedTarget or not _number(radius_mm) or
            radius_mm <= 0 or not _number(center_x)):
        raise ValueError("invalid contact query")
    return center_x - radius_mm * target.slope / math.sqrt(1 + target.slope ** 2)


def bowl_contact_tip_z(target, radius_mm, x, y, *, clearance_mm=0):
    """Safe ball-tip Z from the exact concentric-sphere contact offset."""
    if (type(target) is not SphericalBowlTarget or
            not all(_number(v) for v in (radius_mm, x, y, clearance_mm)) or
            radius_mm <= 0 or radius_mm >= target.rim_radius_mm or
            clearance_mm < 0):
        raise ValueError("invalid bowl contact query")
    radial = math.hypot(x - target.center_xy[0], y - target.center_xy[1])
    if radial > target.rim_radius_mm - radius_mm:
        raise ValueError("ball center enters protected rim")
    offset_radius = target.sphere_radius_mm - radius_mm
    # The exact offset is -depth at the center; preserve that value when
    # sphere and offset radii are too close for direct subtraction.
    return (-target.depth_mm + radial * (radial / offset_radius) /
            (1 + math.sqrt(1 - (radial / offset_radius) ** 2)) +
            clearance_mm)


def straight_pass_volume_mm3(target, radius_mm, center_x, tip_z):
    """Analytic removed volume for an edge-to-edge Y pass inside X bounds.

    The ball center is below Z=0 and the cutter disk fits inside stock X.
    This oracle does not derive volume from the grid evaluator.
    """
    if (type(target) not in (SlopedTarget, SphericalBowlTarget) or
            not all(_number(v) for v in (radius_mm, center_x, tip_z)) or
            radius_mm <= 0):
        raise ValueError("invalid straight pass query")
    x0, y0, x1, y1 = target.stock_xy
    center_z = tip_z + radius_mm
    if not (x0 + radius_mm <= center_x <= x1 - radius_mm and center_z < 0):
        raise ValueError("straight pass outside oracle scope")
    return (y1 - y0) * (-2 * radius_mm * center_z +
                        math.pi * radius_mm ** 2 / 2)


def _point_depth(point, a, b, radius, *, upper=False):
    """Deepest ball envelope at XY point along one linear center move."""
    if radius <= 0:
        return 0.0
    px, py = point
    ax, ay, az = a
    vx, vy, vz = b[0] - ax, b[1] - ay, b[2] - az
    length2 = vx * vx + vy * vy
    if length2 == 0:
        distance2 = (px - ax) ** 2 + (py - ay) ** 2
        return max(0.0, -min(az, b[2]) + math.sqrt(max(0, radius ** 2 - distance2))) if distance2 <= radius ** 2 else 0.0
    projection = ((px - ax) * vx + (py - ay) * vy) / length2
    perpendicular2 = max(0.0, (px - ax) ** 2 + (py - ay) ** 2 -
                         projection ** 2 * length2)
    if perpendicular2 > radius ** 2:
        return 0.0
    reach = math.sqrt(max(0.0, radius ** 2 - perpendicular2) / length2)
    lo, hi = max(0.0, projection - reach), min(1.0, projection + reach)
    if lo > hi:
        return 0.0

    def depth(t):
        dx, dy = px - ax - t * vx, py - ay - t * vy
        return -az - t * vz + math.sqrt(max(0.0, radius ** 2 - dx * dx - dy * dy))

    # The lower spherical envelope plus a linear Z term is concave here.
    for _ in range(42):
        left, right = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
        if depth(left) < depth(right):
            lo = left
        else:
            hi = right
    sampled = max(0.0, depth(lo), depth(hi))
    if not upper:
        return sampled
    # The optimum remains in [lo, hi]. The square-root modulus and linear Z
    # bound enclose the unevaluated maximum, including near the disk rim.
    width = hi - lo
    return sampled + math.sqrt(2 * radius * math.sqrt(length2) * width) + abs(vz) * width


def _cells(target, pitch):
    x0, y0, x1, y1 = target.stock_xy
    nx, ny = math.ceil((x1 - x0) / pitch), math.ceil((y1 - y0) / pitch)
    for ix in range(nx):
        left, right = x0 + ix * pitch, min(x1, x0 + (ix + 1) * pitch)
        for iy in range(ny):
            bottom, top = y0 + iy * pitch, min(y1, y0 + (iy + 1) * pitch)
            if type(target) is SlopedTarget:
                design_low = min(target.depth_at(left), target.depth_at(right))
                design_high = max(target.depth_at(left), target.depth_at(right))
            else:
                cx, cy = target.center_xy
                near_x = max(left - cx, 0, cx - right)
                near_y = max(bottom - cy, 0, cy - top)
                far_x = max(abs(left - cx), abs(right - cx))
                far_y = max(abs(bottom - cy), abs(top - cy))
                near_r = math.hypot(near_x, near_y)
                far_r = math.hypot(far_x, far_y)
                design_low = target.depth_at(cx + far_r, cy)
                design_high = target.depth_at(cx + near_r, cy)
            yield ((left + right) / 2, (bottom + top) / 2,
                   math.hypot(right - left, top - bottom) / 2,
                   (right - left) * (top - bottom),
                   design_low, design_high)


def _residual(target, cuts, pitch, *, prior_count=0):
    low = high = gain_lower = 0.0
    cells = 0
    for x, y, delta, area, design_low, design_high in _cells(target, pitch):
        cut_low = cut_high = prior_high = 0.0
        for index, (a, b, radius) in enumerate(cuts):
            cut_low = max(cut_low, _point_depth(
                (x, y), a, b, max(0.0, radius - delta)))
            upper = _point_depth((x, y), a, b, radius + delta, upper=True)
            cut_high = max(cut_high, upper)
            if index < prior_count:
                prior_high = max(prior_high, upper)
        low += area * max(0.0, design_low - cut_high)
        high += area * max(0.0, design_high - cut_low)
        if prior_count:
            # On this same cell the new removed depth is at least the final
            # lower removal minus the prior upper removal. Subtracting whole
            # job intervals would lose this local correlation.
            gain_lower += area * max(0.0,
                min(design_low, cut_low) - min(design_high, prior_high))
        cells += 1
    return (low, high), gain_lower, cells


def replay_stages(stages, decoded_moves, *, pitch_mm=0.125):
    """Check decoded ball motion, prior access, surface safety and stock bounds."""
    if (not stages or len(stages) != len(decoded_moves) or
            any(type(s.surface_operation) is not SurfaceOperation for s in stages)
            or not _number(pitch_mm) or pitch_mm <= 0):
        raise ValueError("surface stages and positive pitch required")
    target = stages[0].surface_operation.target
    if any(s.surface_operation.target != target for s in stages):
        raise ValueError("surface stages have different targets")
    cuts, prior_entries, reports = [], [], []
    holder_clearance = math.inf
    for stage, moves in zip(stages, decoded_moves):
        op = stage.surface_operation
        prior_count = len(cuts)
        stage_entries = []
        for move in moves:
            a, b = move.start, move.end
            if move.role in ("rapid", "rapid_retract"):
                if min(a[2], b[2]) <= 0 or move.role == "rapid_retract" and (
                        a[:2] != b[:2] or b[2] <= a[2]):
                    raise ValueError("rapid crosses initial stock")
                continue
            if move.role in ("entry", "cleared_descent", "retract"):
                if a[:2] != b[:2] or (move.role == "retract") != (b[2] > a[2]):
                    raise ValueError("unsupported surface feed motion")
            elif move.role != "cut":
                raise ValueError("unsupported surface feed motion")
            if move.role == "retract":
                continue
            if min(a[2], b[2]) >= 0:
                continue
            if -min(a[2], b[2]) > min(target.stock_depth_mm,
                                      op.cutting_length_mm):
                raise ValueError("ball exceeds stock or cutting length")
            holder_clearance = min(holder_clearance,
                                   op.cutting_length_mm + min(a[2], b[2]))
            if move.role == "cleared_descent" and not any(
                    xy == a[:2] and radius >= op.radius_mm and
                    tip_z <= b[2] + ACCESS_MATCH_TOLERANCE_MM
                    for xy, tip_z, radius in prior_entries):
                raise ValueError("uncleared descent lacks predecessor ball sweep")
            for point in (a, b):
                if type(target) is SlopedTarget:
                    if point[2] < contact_tip_z(target, op.radius_mm, point[0]) - PLANE_TOLERANCE_MM:
                        raise ValueError("ball enters protected slope")
                else:
                    safe_tip = bowl_contact_tip_z(target, op.radius_mm,
                                                   point[0], point[1])
                    if point[2] < safe_tip - BOWL_TOLERANCE_MM:
                        raise ValueError("ball enters protected bowl")
            center_a = (a[0], a[1], a[2] + op.radius_mm)
            center_b = (b[0], b[1], b[2] + op.radius_mm)
            cuts.append((center_a, center_b, op.radius_mm))
            if move.role == "entry":
                stage_entries.append((a[:2], b[2], op.radius_mm))
        prior_entries.extend(stage_entries)
        interval, gain_lower, count = _residual(target, cuts, pitch_mm,
                                                prior_count=prior_count)
        reports.append({"stage": stage.id, "cuts": len(cuts),
                        "residual_volume_mm3": interval,
                        "newly_removed_volume_lower_mm3": gain_lower,
                        "protected_overcut_upper_mm3": 0.0,
                        "minimum_holder_clearance_mm": holder_clearance,
                        "cells": count})
    bowl = type(target) is SphericalBowlTarget
    return {"status": "pass", "scope": (
                "decoded spherical bowl and ball sweeps" if bowl else
                "decoded planar slope and ball sweeps"),
            "model": BOWL_VERSION if bowl else VERSION, "pitch_mm": pitch_mm,
            "access_match_tolerance_mm": ACCESS_MATCH_TOLERANCE_MM,
            **({"bowl_tolerance_mm": BOWL_TOLERANCE_MM} if bowl else
               {"plane_tolerance_mm": PLANE_TOLERANCE_MM}),
            "target_volume_mm3": target.target_volume_mm3,
            "prefixes": tuple(reports)}


def compare_representations(target, pitches=(0.25, 0.125)):
    """Time exact analytic volume against conservative cell enclosures."""
    if (type(target) not in (SlopedTarget, SphericalBowlTarget) or not pitches or
            any(not _number(p) or p <= 0 for p in pitches)):
        raise ValueError("target and positive pitches required")

    def measured(fn):
        tracemalloc.start()
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, round(elapsed, 3), peak

    exact, elapsed, peak = measured(lambda: target.target_volume_mm3)
    rows = [{"representation": ("exact_affine_plane" if type(target) is SlopedTarget
                               else "exact_spherical_cap"),
             "volume_interval_mm3": (exact, exact),
             "elapsed_ms": elapsed, "python_peak_bytes": peak,
             "elements": 1}]
    for pitch in pitches:
        def columns():
            low = high = count = 0
            for _, _, _, area, dmin, dmax in _cells(target, pitch):
                low += area * dmin
                high += area * dmax
                count += 1
            return (low, high), count
        (interval, count), elapsed, peak = measured(columns)
        rows.append({"representation": "conservative_xy_columns",
                     "pitch_mm": pitch, "volume_interval_mm3": interval,
                     "elapsed_ms": elapsed, "python_peak_bytes": peak,
                     "elements": count})
    return tuple(rows)
