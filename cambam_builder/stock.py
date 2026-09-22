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
    """Bound removal/rest within an exact rectangular stock == required target.

    At every nominal segment parameter, actual center error has Euclidean norm
    <= position_error_mm and the cutting disk radius lies in [min, max]. The
    entire segment is covered, with no extra cutting motion in this section.
    Exterior stock is protected: the outer capsule must stay in the rectangle.
    Boundary contact is allowed. Invalid/unsupported/unsafe input raises ValueError
    and returns no partial certificate. Nominal planar values are not admitted.
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
