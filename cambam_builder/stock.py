"""Exact directional section bounds for a supplied horizontal disk sweep.

This is conditional geometric evidence, not an executed-toolpath certificate.
All inputs use one caller-named Cartesian frame, millimetres, and fixed Z.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Optional, Tuple, Union


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("expected a finite int, float or Fraction")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("expected a finite number")
    return Fraction(value)


@dataclass(frozen=True)
class SectionRectangle:
    xmin: Fraction
    ymin: Fraction
    xmax: Fraction
    ymax: Fraction

    def __post_init__(self):
        for name in ("xmin", "ymin", "xmax", "ymax"):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("rectangle must have positive width and height")

    def contains(self, x, y):
        x, y = _number(x), _number(y)
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclass(frozen=True)
class SectionTarget:
    """Exact required removal inside stock, with one protected island."""

    stock: SectionRectangle
    outer: SectionRectangle
    island: SectionRectangle

    def __post_init__(self):
        if any(type(rect) is not SectionRectangle for rect in
               (self.stock, self.outer, self.island)):
            raise ValueError("target requires exact section rectangles")
        s, o, i = self.stock, self.outer, self.island
        if (o.xmin < s.xmin or o.xmax > s.xmax or
                o.ymin < s.ymin or o.ymax > s.ymax):
            raise ValueError("target outer rectangle must lie inside stock")
        if not (o.xmin < i.xmin < i.xmax < o.xmax and
                o.ymin < i.ymin < i.ymax < o.ymax):
            raise ValueError("protected island must lie strictly inside target")

    def contains(self, x, y):
        """Outer and island boundary are in the target; island interior is not."""
        x, y = _number(x), _number(y)
        i = self.island
        return self.outer.contains(x, y) and not (
            i.xmin < x < i.xmax and i.ymin < y < i.ymax)

    @property
    def area(self):
        return self.outer.area - self.island.area


def _capsule_within_target(target, capsule):
    """Closed disk sweep may touch target walls but cannot enter protected area."""
    s, r, o, i = capsule.sweep, capsule.radius, target.outer, target.island
    if (s.xmin - r < o.xmin or s.xmax + r > o.xmax or
            s.y - r < o.ymin or s.y + r > o.ymax):
        return False
    dx = max(i.xmin - s.xmax, s.xmin - i.xmax, 0)
    dy = max(i.ymin - s.y, s.y - i.ymax, 0)
    if r == 0:
        return not (s.xmin < i.xmax and s.xmax > i.xmin and
                    i.ymin < s.y < i.ymax)
    # Equality is wall tangency. The open island interior remains untouched.
    return dx * dx + dy * dy >= r * r


@dataclass(frozen=True)
class HorizontalSweep:
    """Entire segment is traversed; xmin == xmax denotes one disk placement."""

    xmin: Fraction
    xmax: Fraction
    y: Fraction

    def __post_init__(self):
        for name in ("xmin", "xmax", "y"):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if self.xmin > self.xmax:
            raise ValueError("segment endpoints must be ordered")


@dataclass(frozen=True)
class Capsule:
    sweep: HorizontalSweep
    radius: Fraction

    def __post_init__(self):
        if type(self.sweep) is not HorizontalSweep:
            raise ValueError("expected an analytic horizontal sweep")
        object.__setattr__(self, "radius", _number(self.radius))
        if self.radius < 0:
            raise ValueError("radius must be nonnegative")

    def contains(self, x, y):
        x, y = _number(x), _number(y)
        s = self.sweep
        dx = max(s.xmin - x, 0, x - s.xmax)
        return dx * dx + (y - s.y) ** 2 <= self.radius ** 2

    @property
    def area_interval(self):
        # Classical strict rational brackets for pi; no floating arithmetic.
        linear = 2 * self.radius * (self.sweep.xmax - self.sweep.xmin)
        square = self.radius ** 2
        return (linear + Fraction(333, 106) * square,
                linear + Fraction(355, 113) * square)


@dataclass(frozen=True)
class RemainingSection:
    """Exact set difference, including its boundary membership convention."""

    stock: SectionRectangle
    removed: Optional[Union[Capsule, "CapsuleUnion"]]

    def __post_init__(self):
        if type(self.stock) is not SectionRectangle:
            raise ValueError("expected an exact section rectangle")
        if self.removed is not None:
            if type(self.removed) is CapsuleUnion:
                if self.removed.stock != self.stock:
                    raise ValueError("union must use the same stock")
                return
            if type(self.removed) is not Capsule:
                raise ValueError("expected an analytic capsule or None")
            s, r = self.removed.sweep, self.removed.radius
            if (s.xmin - r < self.stock.xmin or s.xmax + r > self.stock.xmax
                    or s.y - r < self.stock.ymin or s.y + r > self.stock.ymax):
                raise ValueError("removed capsule must lie wholly inside stock")

    def contains(self, x, y):
        return self.stock.contains(x, y) and (
            self.removed is None or not self.removed.contains(x, y))

    @property
    def area_interval(self):
        lo, hi = (0, 0) if self.removed is None else self.removed.area_interval
        return max(Fraction(0), self.stock.area - hi), self.stock.area - lo


@dataclass(frozen=True)
class SweepBounds:
    frame_id: str
    section_z_mm: Fraction
    stock: SectionRectangle
    sweep: HorizontalSweep
    radius_min_mm: Fraction
    radius_max_mm: Fraction
    position_error_mm: Fraction
    removal_lower: Optional[Capsule]
    removal_upper: Capsule
    remaining_lower: RemainingSection
    remaining_upper: RemainingSection
    evidence_class: str = "conditional_analytic_section_bounds"


def bound_horizontal_sweep(stock, sweep, *, frame_id, section_z_mm,
                           radius_min_mm, radius_max_mm, position_error_mm=0):
    """Bound removal and remaining stock inside an exact rectangle.

    At every nominal segment parameter, actual center error has Euclidean norm
    <= position_error_mm and the cutting disk radius lies in [min, max]. The
    entire segment is covered, with no extra cutting motion in this section.
    Exterior stock is protected: the outer capsule must stay in the rectangle.
    Boundary contact is allowed. Invalid/unsupported/unsafe input raises ValueError
    and returns no partial certificate. The standalone rest target is this stock;
    compose_target_rest_bounds validates a separate target. Nominal planar values
    are not admitted.
    """
    if type(stock) is not SectionRectangle or type(sweep) is not HorizontalSweep:
        raise ValueError("only exact section rectangles and horizontal sweeps are supported")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("a nonempty caller-owned frame identity is required")
    z, rmin, rmax, error = map(_number, (
        section_z_mm, radius_min_mm, radius_max_mm, position_error_mm))
    if rmin <= 0 or rmax < rmin or error < 0:
        raise ValueError("require 0 < radius_min <= radius_max and position_error >= 0")
    outer = Capsule(sweep, rmax + error)
    r = outer.radius
    if (sweep.xmin - r < stock.xmin or sweep.xmax + r > stock.xmax
            or sweep.y - r < stock.ymin or sweep.y + r > stock.ymax):
        raise ValueError("outer occupancy crosses protected rectangle boundary")
    inner = None if error > rmin else Capsule(sweep, rmin - error)
    return SweepBounds(frame_id, z, stock, sweep, rmin, rmax, error,
                       inner, outer, RemainingSection(stock, outer),
                       RemainingSection(stock, inner))


@dataclass(frozen=True)
class CapsuleUnion:
    """Exact membership; conservative area on a fixed rational stock grid.

    Fully covered cells contribute to the lower area; possibly touched cells
    contribute to the upper area. A cell is counted at most once in each bound.
    Grid size controls cost/precision, not membership or physical uncertainty.
    """

    stock: SectionRectangle
    capsules: Tuple[Capsule, ...]
    grid_size: int = 32

    def __post_init__(self):
        if type(self.stock) is not SectionRectangle:
            raise ValueError("expected an exact section rectangle")
        if type(self.grid_size) is not int or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        try:
            capsules = tuple(self.capsules)
        except TypeError as exc:
            raise ValueError("expected a finite capsule iterable") from exc
        for capsule in capsules:
            if type(capsule) is not Capsule:
                raise ValueError("expected analytic capsules")
            RemainingSection(self.stock, capsule)
        object.__setattr__(self, "capsules", capsules)

    def contains(self, x, y):
        x, y = _number(x), _number(y)
        return any(c.contains(x, y) for c in self.capsules)

    @property
    def area_interval(self):
        stock, n = self.stock, self.grid_size
        dx = (stock.xmax - stock.xmin) / n
        dy = (stock.ymax - stock.ymin) / n
        lower = upper = 0
        # Zero-radius segments affect membership but have zero planar area.
        capsules = tuple(c for c in self.capsules if c.radius > 0)
        if not capsules:
            return Fraction(0), Fraction(0)
        for i in range(n):
            x0, x1 = stock.xmin + i * dx, stock.xmin + (i + 1) * dx
            for j in range(n):
                y0, y1 = stock.ymin + j * dy, stock.ymin + (j + 1) * dy
                touched = covered = False
                for c in capsules:
                    s = c.sweep
                    near_x = max(s.xmin - x1, x0 - s.xmax, 0)
                    near_y = max(y0 - s.y, s.y - y1, 0)
                    touched |= near_x ** 2 + near_y ** 2 <= c.radius ** 2
                    # Convexity: all four corners inside implies whole cell inside.
                    far_x = max(s.xmin - x0, x1 - s.xmax, 0)
                    far_y = max(abs(y0 - s.y), abs(y1 - s.y))
                    if far_x ** 2 + far_y ** 2 <= c.radius ** 2:
                        covered = True
                        break
                lower += covered
                upper += touched
        return lower * dx * dy, upper * dx * dy


@dataclass(frozen=True)
class ComposedSweepBounds:
    frame_id: str
    section_z_mm: Fraction
    stock: SectionRectangle
    sources: Tuple[SweepBounds, ...]
    removal_lower: CapsuleUnion
    removal_upper: CapsuleUnion
    remaining_lower: RemainingSection
    remaining_upper: RemainingSection
    evidence_class: str = "conditional_analytic_section_bounds"


def compose_sweep_bounds(stock, sources, *, frame_id, section_z_mm, grid_size=32):
    """Compose a finite ordered collection of complete supplied sweep bounds.

    Sources must share stock, frame and section. Revalidate their certificates;
    retain order, duplicates and uncertainty as provenance. Empty input removes
    nothing. Prefixes on the same grid give monotonic remaining-area intervals.
    No independence assumption about errors between passes is required.
    """
    if type(stock) is not SectionRectangle:
        raise ValueError("expected an exact section rectangle")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("a nonempty caller-owned frame identity is required")
    z = _number(section_z_mm)
    try:
        sources = tuple(sources)
    except TypeError as exc:
        raise ValueError("expected a finite sweep bounds iterable") from exc
    for source in sources:
        if type(source) is not SweepBounds:
            raise ValueError("expected analytic SweepBounds sources")
        if (source.stock != stock or source.frame_id != frame_id
                or source.section_z_mm != z):
            raise ValueError("sources must share stock, frame and section Z")
        checked = bound_horizontal_sweep(
            stock, source.sweep, frame_id=frame_id, section_z_mm=z,
            radius_min_mm=source.radius_min_mm,
            radius_max_mm=source.radius_max_mm,
            position_error_mm=source.position_error_mm)
        if checked != source:
            raise ValueError("source differs from validated analytic bounds")
    lower = CapsuleUnion(stock, tuple(s.removal_lower for s in sources
                                     if s.removal_lower is not None), grid_size)
    upper = CapsuleUnion(stock, tuple(s.removal_upper for s in sources), grid_size)
    return ComposedSweepBounds(frame_id, z, stock, sources, lower, upper,
                               RemainingSection(stock, upper),
                               RemainingSection(stock, lower))


@dataclass(frozen=True)
class TargetRestSection:
    """Required target material still present after a bounded removal union."""

    target: SectionTarget
    removed: CapsuleUnion

    def __post_init__(self):
        if type(self.target) is not SectionTarget or type(self.removed) is not CapsuleUnion:
            raise ValueError("expected exact target and analytic capsule union")
        if self.removed.stock != self.target.stock or any(
                not _capsule_within_target(self.target, c)
                for c in self.removed.capsules):
            raise ValueError("removal enters protected target material")

    def contains(self, x, y):
        return self.target.contains(x, y) and not self.removed.contains(x, y)

    @property
    def area_interval(self):
        lo, hi = self.removed.area_interval
        return max(Fraction(0), self.target.area - hi), self.target.area - lo


@dataclass(frozen=True)
class TargetStockBounds:
    """Original target rest and cumulative whole-stock state at one section."""

    target: SectionTarget
    stock_bounds: ComposedSweepBounds
    rest_lower: TargetRestSection
    rest_upper: TargetRestSection
    evidence_class: str = "conditional_analytic_section_bounds"


def compose_target_rest_bounds(target, sources, *, frame_id, section_z_mm,
                               grid_size=32):
    """Bound cumulative rest in an original holed target, preserving stock state.

    Every supplied source is revalidated by compose_sweep_bounds. Its outer
    occupancy must fit the target, regardless of earlier clearing; the original
    exterior and island remain protected. Passes may cross the artificial
    rest/cleared-space interface. No connecting travel is inferred.
    """
    if type(target) is not SectionTarget:
        raise ValueError("expected an exact rectangular section target")
    stock_bounds = compose_sweep_bounds(
        target.stock, sources, frame_id=frame_id, section_z_mm=section_z_mm,
        grid_size=grid_size)
    if any(not _capsule_within_target(target, c)
           for c in stock_bounds.removal_upper.capsules):
        raise ValueError("outer occupancy enters protected target material")
    return TargetStockBounds(
        target, stock_bounds,
        TargetRestSection(target, stock_bounds.removal_upper),
        TargetRestSection(target, stock_bounds.removal_lower))


@dataclass(frozen=True)
class SectionMotionSegment:
    """Directed horizontal cut or non-cutting travel at the declared section."""

    kind: str
    x_start: Fraction
    x_end: Fraction
    y: Fraction
    cover_source_index: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("cut", "travel"):
            raise ValueError("motion kind must be cut or travel")
        for name in ("x_start", "x_end", "y"):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if self.kind == "cut" and self.cover_source_index is not None:
            raise ValueError("cut motion cannot claim prior clearance")
        if self.kind == "travel" and self.cover_source_index is not None and (
                type(self.cover_source_index) is not int or self.cover_source_index < 0):
            raise ValueError("cover_source_index must be a nonnegative integer")

    @property
    def sweep(self):
        return HorizontalSweep(min(self.x_start, self.x_end),
                               max(self.x_start, self.x_end), self.y)


@dataclass(frozen=True)
class SectionMotionPath:
    """One tool envelope and one explicit section entry followed by contiguous moves."""

    entry_kind: str
    radius_min_mm: Fraction
    radius_max_mm: Fraction
    position_error_mm: Fraction
    segments: Tuple[SectionMotionSegment, ...]
    entry_cover_source_index: Optional[int] = None

    def __post_init__(self):
        if self.entry_kind not in ("cutting", "cleared", "outside_stock"):
            raise ValueError("entry_kind must be cutting, cleared or outside_stock")
        for name in ("radius_min_mm", "radius_max_mm", "position_error_mm"):
            object.__setattr__(self, name, _number(getattr(self, name)))
        if (self.radius_min_mm <= 0 or self.radius_max_mm < self.radius_min_mm
                or self.position_error_mm < 0):
            raise ValueError("require 0 < radius_min <= radius_max and position_error >= 0")
        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise ValueError("expected a finite motion segment iterable") from exc
        if not segments or any(type(s) is not SectionMotionSegment for s in segments):
            raise ValueError("path requires analytic motion segments")
        if any((a.x_end, a.y) != (b.x_start, b.y)
               for a, b in zip(segments, segments[1:])):
            raise ValueError("path motion must be contiguous at this section")
        if self.entry_kind == "cutting":
            if segments[0].kind != "cut" or self.entry_cover_source_index is not None:
                raise ValueError("cutting entry must begin with a cut")
        elif self.entry_kind == "cleared":
            if (type(self.entry_cover_source_index) is not int
                    or self.entry_cover_source_index < 0):
                raise ValueError("cleared entry requires a prior source index")
        elif self.entry_cover_source_index is not None:
            raise ValueError("outside-stock entry cannot claim prior clearance")
        object.__setattr__(self, "segments", segments)


def _capsule_inside_capsule(inner, outer):
    """Exact sufficient and necessary test for parallel horizontal capsules."""
    available = outer.radius - inner.radius
    if available < 0:
        return False
    s, p = inner.sweep, outer.sweep
    return all((max(p.xmin - x, 0, x - p.xmax) ** 2 +
                (s.y - p.y) ** 2 <= available ** 2)
               for x in (s.xmin, s.xmax))


def _capsule_outside_stock(capsule, stock):
    s = capsule.sweep
    dx = max(stock.xmin - s.xmax, s.xmin - stock.xmax, 0)
    dy = max(stock.ymin - s.y, s.y - stock.ymax, 0)
    return dx * dx + dy * dy > capsule.radius * capsule.radius


@dataclass(frozen=True)
class VerifiedSectionMotion:
    """Ordered conditional section proof; no higher-Z entry or tool-change proof."""

    paths: Tuple[SectionMotionPath, ...]
    target_bounds: TargetStockBounds
    evidence_class: str = "conditional_analytic_section_motion"


def verify_section_motion(target, paths, *, frame_id, section_z_mm, grid_size=32):
    """Verify supplied cuts and explicit non-cutting motion against prior stock.

    A travel/cleared-entry capsule must fit one cited earlier guaranteed-removal
    capsule, or be wholly disjoint from initial stock. Splitting a connector lets
    different pieces cite different sources. Cutting entries are checked at the
    section, but arrival from another height and tool changes are not modeled.
    """
    if type(target) is not SectionTarget:
        raise ValueError("expected an exact rectangular section target")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("a nonempty caller-owned frame identity is required")
    z = _number(section_z_mm)
    try:
        paths = tuple(paths)
    except TypeError as exc:
        raise ValueError("expected a finite motion path iterable") from exc
    sources = []
    for path in paths:
        if type(path) is not SectionMotionPath:
            raise ValueError("expected analytic section motion paths")
        try:
            segments = tuple(path.segments)
        except TypeError as exc:
            raise ValueError("expected a finite motion segment iterable") from exc
        if any(type(segment) is not SectionMotionSegment for segment in segments):
            raise ValueError("expected analytic motion segments")
        checked = SectionMotionPath(
            path.entry_kind, path.radius_min_mm, path.radius_max_mm,
            path.position_error_mm,
            tuple(SectionMotionSegment(segment.kind, segment.x_start,
                                       segment.x_end, segment.y,
                                       segment.cover_source_index)
                  for segment in segments), path.entry_cover_source_index)
        if checked != path:
            raise ValueError("motion path differs from validated analytic input")
        first = path.segments[0]
        entry = Capsule(HorizontalSweep(first.x_start, first.x_start, first.y),
                        path.radius_max_mm + path.position_error_mm)
        if path.entry_kind == "cutting":
            if not _capsule_within_target(target, entry):
                raise ValueError("cutting entry enters protected target material")
        elif path.entry_kind == "cleared":
            index = path.entry_cover_source_index
            if index >= len(sources) or sources[index].removal_lower is None or not (
                    _capsule_inside_capsule(entry, sources[index].removal_lower)):
                raise ValueError("entry is not inside cited prior guaranteed free space")
        elif not _capsule_outside_stock(entry, target.stock):
            raise ValueError("outside-stock entry touches initial stock")
        for segment in path.segments:
            occupancy = Capsule(segment.sweep,
                                path.radius_max_mm + path.position_error_mm)
            if segment.kind == "travel":
                index = segment.cover_source_index
                if index is None:
                    if not _capsule_outside_stock(occupancy, target.stock):
                        raise ValueError("travel touches stock without prior clearance")
                elif index >= len(sources) or sources[index].removal_lower is None or not (
                        _capsule_inside_capsule(occupancy, sources[index].removal_lower)):
                    raise ValueError("travel leaves cited prior guaranteed free space")
            else:
                source = bound_horizontal_sweep(
                    target.stock, segment.sweep, frame_id=frame_id,
                    section_z_mm=z, radius_min_mm=path.radius_min_mm,
                    radius_max_mm=path.radius_max_mm,
                    position_error_mm=path.position_error_mm)
                if not _capsule_within_target(target, source.removal_upper):
                    raise ValueError("cutting occupancy enters protected target material")
                sources.append(source)
    bounds = compose_target_rest_bounds(
        target, sources, frame_id=frame_id, section_z_mm=z, grid_size=grid_size)
    return VerifiedSectionMotion(paths, bounds)
