"""Exact directional section bounds for a supplied horizontal disk sweep.

This is conditional geometric evidence, not an executed-toolpath certificate.
All inputs use one caller-named Cartesian frame, millimetres, and fixed Z.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Optional


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
    removed: Optional[Capsule]

    def __post_init__(self):
        if type(self.stock) is not SectionRectangle:
            raise ValueError("expected an exact section rectangle")
        if self.removed is not None:
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
