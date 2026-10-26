"""Continuous ball motion and cumulative stock bounds for bounded reliefs.

Stock starts at Z=0. Each XY cell encloses the union of complete straight
ball sweeps, rather than summing pass volumes. A clamped extension of the
design supplies a conservative cutter-contact guard at stock edges. Shank
clearance is against the entire initial stock; fixtures/holders are checked
separately by the ordered-job occupancy owner.
"""

from dataclasses import dataclass
import math

from . import relief


VERSION = "bounded-relief-ball-stock-v1"
MAX_CELLS = 250_000
MAX_PROOF_SPLITS = 20_000
ROUNDING_RELATIVE = 1e-10


def _finite(value: float) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _xyz(point: tuple[float, float, float]) -> None:
    if (type(point) is not tuple or len(point) != 3 or
            not all(_finite(value) for value in point)):
        raise ValueError("finite relief XYZ tuple required")


@dataclass(frozen=True)
class ReliefOperation:
    """A ball cutter and independent relief intent in millimetres.

    ``strategy`` is descriptive, never stock evidence. The pitch bounds the
    residual query, while contact tolerance only controls contact refinement.
    Actual motions must still pass the continuous protected-surface proof.
    """

    name: str
    tool_id: str
    radius_mm: float
    cutting_length_mm: float
    target: relief.CubicRelief | relief.HeightField
    strategy: str
    pitch_mm: float = 0.1
    contact_tolerance_mm: float = 0.001

    def __post_init__(self) -> None:
        if (type(self.name) is not str or not self.name or
                type(self.tool_id) is not str or not self.tool_id or
                type(self.target) not in (relief.CubicRelief, relief.HeightField) or
                self.strategy not in ("parallel", "waterline", "guided", "rest") or
                not all(_finite(value) and value > 0 for value in
                        (self.radius_mm, self.cutting_length_mm, self.pitch_mm,
                         self.contact_tolerance_mm)) or
                self.cutting_length_mm < self.radius_mm):
            raise ValueError("invalid bounded relief operation")
        x0, y0, x1, y1 = self.target.stock_xy
        try:
            cells = math.ceil((x1 - x0) / self.pitch_mm) * math.ceil(
                (y1 - y0) / self.pitch_mm)
        except OverflowError as exc:
            raise ValueError("relief residual grid is not representable") from exc
        if cells > MAX_CELLS:
            raise ValueError("relief residual cell budget exceeded")


def _roundoff(*values: float) -> float:
    return ROUNDING_RELATIVE * max(1.0, *(abs(value) for value in values))


def _design_guard(target: relief.CubicRelief | relief.HeightField) -> float:
    values = [*target.stock_xy, target.stock_depth_mm]
    if type(target) is relief.CubicRelief:
        xmax = max(abs(target.stock_xy[0]), abs(target.stock_xy[2]))
        values.extend(abs(coefficient) * xmax ** degree
                      for degree, coefficient in enumerate(target.coefficients))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("relief design arithmetic is not representable")
    return _roundoff(*values)


def _sweep_depth(point: tuple[float, float], a: tuple[float, float, float],
                 b: tuple[float, float, float], radius: float) -> float:
    """Maximum removed column depth under a straight sphere-centre sweep.

    The stationary point follows by differentiating the concave lower sphere
    envelope. The clipping interval includes both sphere-footprint endpoints.
    Callers add/subtract the declared floating-point guard to bound roundoff.
    """
    if radius <= 0:
        return 0.0
    px, py = point
    vx, vy, vz = (b[index] - a[index] for index in range(3))
    dx, dy = px - a[0], py - a[1]
    length2 = vx * vx + vy * vy
    radius2 = radius * radius
    if not all(math.isfinite(value) for value in (length2, radius2, vz)):
        raise ValueError("relief sweep arithmetic is not representable")
    if length2 == 0:
        distance2 = dx * dx + dy * dy
        if distance2 > radius2:
            return 0.0
        return max(0.0, -min(a[2], b[2]) +
                   math.sqrt(max(0.0, radius2 - distance2)))
    projection = (dx * vx + dy * vy) / length2
    # The cross-product form avoids subtracting nearly equal squared lengths.
    cross = dx * vy - dy * vx
    perpendicular2 = cross * cross / length2
    if not all(math.isfinite(value) for value in (projection, perpendicular2)):
        raise ValueError("relief sweep arithmetic is not representable")
    if perpendicular2 > radius2:
        return 0.0
    cap2 = max(0.0, radius2 - perpendicular2)
    reach = math.sqrt(cap2 / length2)
    lo, hi = max(0.0, projection - reach), min(1.0, projection + reach)
    if lo > hi:
        return 0.0
    stationary = projection - (vz / math.hypot(math.sqrt(length2), vz)) * reach
    t = max(lo, min(hi, stationary))

    def depth(at: float) -> float:
        qx, qy = dx - at * vx, dy - at * vy
        return -a[2] - at * vz + math.sqrt(
            max(0.0, radius2 - qx * qx - qy * qy))

    result = max(0.0, depth(lo), depth(t), depth(hi))
    if not math.isfinite(result):
        raise ValueError("relief sweep arithmetic is not representable")
    return result


def _grid(target: relief.CubicRelief | relief.HeightField,
          pitch_mm: float):
    x0, y0, x1, y1 = target.stock_xy
    guard = _design_guard(target)
    nx, ny = math.ceil((x1 - x0) / pitch_mm), math.ceil((y1 - y0) / pitch_mm)
    if nx * ny > MAX_CELLS:
        raise ValueError("relief residual cell budget exceeded")
    for ix in range(nx):
        left, right = x0 + ix * pitch_mm, min(x1, x0 + (ix + 1) * pitch_mm)
        for iy in range(ny):
            bottom, top = y0 + iy * pitch_mm, min(y1, y0 + (iy + 1) * pitch_mm)
            rect = (left, bottom, right, top)
            height_low, height_high = target.height_bounds(rect)
            yield (rect, ((left + right) / 2, (bottom + top) / 2),
                   math.hypot(right - left, top - bottom) / 2,
                   (right - left) * (top - bottom),
                   max(0.0, -height_high - guard),
                   min(target.stock_depth_mm, -height_low + guard))


def _depth_bounds(point: tuple[float, float], delta: float,
                  cuts: list[tuple], *, count: int | None = None) -> tuple[float, float]:
    lower = upper = 0.0
    selected = cuts if count is None else cuts[:count]
    for a, b, radius in selected:
        guard = _roundoff(*a, *b, radius, *point, delta)
        reach = radius + delta + guard
        if (point[0] < min(a[0], b[0]) - reach or
                point[0] > max(a[0], b[0]) + reach or
                point[1] < min(a[1], b[1]) - reach or
                point[1] > max(a[1], b[1]) + reach):
            continue
        lower = max(lower, max(0.0, _sweep_depth(
            point, a, b, radius - delta - guard) - guard))
        upper = max(upper, _sweep_depth(point, a, b, radius + delta + guard) + guard)
    return lower, upper


def residual_at(target: relief.CubicRelief | relief.HeightField,
                cuts: list[tuple], x: float, y: float) -> tuple[float, float]:
    """Residual thickness bracket at a design point for sphere-centre cuts.

    ``cuts`` contains ``(start_xyz, end_xyz, radius_mm)`` in the program frame.
    This geometric query does not certify gouge, process or motion provenance.
    """
    if (type(target) not in (relief.CubicRelief, relief.HeightField) or
            not all(_finite(value) for value in (x, y)) or
            not target.stock_xy[0] <= x <= target.stock_xy[2] or
            not target.stock_xy[1] <= y <= target.stock_xy[3]):
        raise ValueError("relief residual point outside stock")
    _validate_cuts(cuts)
    low, high = _depth_bounds((x, y), 0.0, cuts)
    design = -target.height_at(x, y)
    guard = _design_guard(target)
    return max(0.0, design - guard - high), max(0.0, design + guard - low)


def _validate_cuts(cuts: list[tuple]) -> None:
    if type(cuts) not in (list, tuple):
        raise ValueError("relief sweep sequence required")
    for cut in cuts:
        if type(cut) is not tuple or len(cut) != 3:
            raise ValueError("invalid relief sweep")
        a, b, radius = cut
        _xyz(a)
        _xyz(b)
        if not _finite(radius) or radius <= 0:
            raise ValueError("positive relief sweep radius required")


def _residual(target: relief.CubicRelief | relief.HeightField,
              cuts: list[tuple], pitch_mm: float, prior_count: int) -> dict:
    volume_low = volume_high = maximum_low = maximum_high = gain_low = 0.0
    worst_rect = None
    count = 0
    for rect, point, delta, area, design_low, design_high in _grid(target, pitch_mm):
        cut_low, cut_high = _depth_bounds(point, delta, cuts)
        residual_low = max(0.0, design_low - cut_high)
        residual_high = max(0.0, design_high - cut_low)
        volume_low += area * residual_low
        volume_high += area * residual_high
        maximum_low = max(maximum_low, residual_low)
        if residual_high > maximum_high or worst_rect is None:
            maximum_high, worst_rect = residual_high, rect
        if prior_count:
            _, prior_high = _depth_bounds(point, delta, cuts, count=prior_count)
            gain_low += area * max(0.0,
                min(design_low, cut_low) - min(design_high, prior_high))
        else:
            gain_low += area * min(design_low, cut_low)
        count += 1
    return {"residual_volume_mm3": (volume_low, volume_high),
            "maximum_residual_thickness_mm": (maximum_low, maximum_high),
            "worst_cell_xy": worst_rect,
            "newly_removed_volume_lower_mm3": gain_low,
            "protected_overcut_upper_mm3": 0.0,
            "cells": count}


def replay_stages(stages: tuple, decoded_moves: tuple, *,
                  max_proof_splits: int = MAX_PROOF_SPLITS) -> dict:
    """Verify decoded continuous motion and report every cumulative stock prefix.

    Only straight fixed-axis ball motion is supported. Supplied axial limits
    conservatively cap absolute virgin-stock depth; no prior retrace credit or
    engagement/load model is inferred. Unresolved contact refinement rejects.
    """
    if (type(stages) is not tuple or not stages or
            type(decoded_moves) is not tuple or len(stages) != len(decoded_moves) or
            any(type(stage.operation) is not ReliefOperation for stage in stages) or
            type(max_proof_splits) is not int or max_proof_splits < 0):
        raise ValueError("relief stages and complete decoded motion required")
    target = stages[0].operation.target
    pitch = stages[0].operation.pitch_mm
    if any(stage.operation.target != target or stage.operation.pitch_mm != pitch
           for stage in stages):
        raise ValueError("relief stages need the same design and residual grid")
    if any(stage.source_revision != target.fingerprint for stage in stages):
        raise ValueError("relief stage design revision is stale")
    cache, cuts, reports = {}, [], []
    proof_splits = 0
    minimum_shank = math.inf
    previous_end = None

    def contact_at(op: ReliefOperation, point: tuple) -> tuple[float, float]:
        key = (op.radius_mm, op.contact_tolerance_mm, point[0], point[1])
        if key not in cache:
            query = relief.contact(target, op.radius_mm, point[0], point[1],
                                   tolerance_mm=op.contact_tolerance_mm)
            cache[key] = (query.lower_tip_z_mm, query.upper_tip_z_mm)
        return cache[key]

    def prove(op: ReliefOperation, a: tuple, b: tuple) -> None:
        nonlocal proof_splits
        pending = [(a, b)]
        lx, ly = target.gradient_bounds
        while pending:
            start, end = pending.pop()
            low_a, high_a = contact_at(op, start)
            low_b, high_b = contact_at(op, end)
            guard = _roundoff(*start, *end, op.radius_mm)
            if start[2] < low_a - guard or end[2] < low_b - guard:
                raise ValueError("ball enters protected relief")
            directional_change = (lx * abs(end[0] - start[0]) +
                                  ly * abs(end[1] - start[1]))
            if min(start[2] - high_a, end[2] - high_b) >= directional_change / 2 + guard:
                continue
            if proof_splits >= max_proof_splits:
                raise ValueError("continuous relief contact proof budget exhausted")
            middle = tuple(x + (y - x) / 2 for x, y in zip(start, end))
            if middle == start or middle == end:
                raise ValueError("continuous relief contact proof is unresolved")
            proof_splits += 1
            pending.extend(((start, middle), (middle, end)))

    for stage, moves in zip(stages, decoded_moves):
        op = stage.operation
        if op.name != stage.id or op.tool_id != stage.tool_id:
            raise ValueError("relief stage operation identity differs")
        if type(moves) not in (tuple, list) or not moves:
            raise ValueError("complete relief stage motion required")
        prior_count = len(cuts)
        maximum_depth = maximum_entry = 0.0
        at = None
        for move in moves:
            a, b = move.start, move.end
            _xyz(a)
            _xyz(b)
            if a == b or (at is not None and a != at) or (
                    at is None and previous_end is not None and a != previous_end):
                raise ValueError("relief decoded motion is discontinuous")
            at = b
            if move.arc_g:
                raise ValueError("relief replay supports straight motion only")
            if move.role in ("rapid", "rapid_retract"):
                if min(a[2], b[2]) <= 0 or (
                        move.role == "rapid_retract" and (a[:2] != b[:2] or b[2] <= a[2])):
                    raise ValueError("rapid crosses initial relief stock")
                continue
            if move.role in ("entry", "retract"):
                if a[:2] != b[:2] or (move.role == "retract") != (b[2] > a[2]):
                    raise ValueError("relief feed entry/retract must be vertical and monotonic")
            elif move.role != "cut":
                raise ValueError("unsupported relief feed motion")
            low_tip = min(a[2], b[2])
            depth = max(0.0, -low_tip)
            if depth > target.stock_depth_mm:
                raise ValueError("relief cutter exceeds stock depth")
            shank = low_tip + op.cutting_length_mm
            if shank <= 0:
                raise ValueError("relief shank contacts initial stock")
            minimum_shank = min(minimum_shank, shank)
            limits = getattr(stage, "axial_limits", None)
            maximum_depth = max(maximum_depth, depth)
            if move.role == "entry":
                maximum_entry = max(maximum_entry, depth)
                if limits is not None and depth > limits.max_entry_depth_mm:
                    raise ValueError("relief entry exceeds declared depth limit")
            if limits is not None and depth > limits.max_stepdown_mm:
                raise ValueError("relief absolute depth exceeds declared virgin stepdown limit")
            if low_tip >= 0:
                continue
            if move.role in ("entry", "retract"):
                lowest = a if a[2] < b[2] else b
                prove(op, lowest, lowest)
            else:
                prove(op, a, b)
            # Entries and retracts are full union members, so stock evidence
            # remains faithful even when a supplied path contains no cut row.
            cuts.append(((*a[:2], a[2] + op.radius_mm),
                         (*b[:2], b[2] + op.radius_mm), op.radius_mm))
        if moves[0].start[2] <= 0 or at[2] <= 0:
            raise ValueError("relief stage requires above-stock endpoints")
        previous_end = at
        prefix = _residual(target, cuts, pitch, prior_count)
        prefix.update({"stage": stage.id, "cuts": len(cuts),
                       "minimum_shank_clearance_mm": minimum_shank,
                       "axial_limits": {
                           "status": "pass" if getattr(stage, "axial_limits", None) else "not_evaluated",
                           "scope": "absolute virgin-stock depth; no prior retrace credit",
                           "maximum_depth_mm": maximum_depth,
                           "maximum_entry_depth_mm": maximum_entry,
                           "engagement_and_load": "not_evaluated"}})
        reports.append(prefix)
    return {"status": "pass", "scope": "decoded bounded non-overhanging ball relief",
            "model": VERSION, "design_fingerprint": target.fingerprint,
            "pitch_mm": pitch, "rounding_relative": ROUNDING_RELATIVE,
            "continuous_proof_splits": proof_splits,
            "prefixes": tuple(reports)}
