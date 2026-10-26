"""Bounded fixed-axis relief geometry and ball contact in millimeters.

The design is a graph over rectangular stock, independent of its cutter.  The
analytic cubic is extruded along Y; HeightField is an exact piecewise bilinear
surface, not an approximation claim about an imported mesh.  Contact uses the
surface extended by clamping XY to the stock rectangle.  This Lipschitz extension
is conservative at stock edges and makes continuous-motion bounds possible.

Intervals include a 64-ulp arithmetic allowance at the coordinate, height and
polynomial-term scale.  They describe ordinary finite floating-point geometry, not exact-real or
arbitrarily scaled arithmetic.  Unrepresentable geometry and exhausted contact
refinement raise ValueError rather than returning an uncertified result.
"""

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import heapq
import math


VERSION = "bounded-relief-contact-v1"


def _number(value: float) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("relief arithmetic is not representable")


def _rectangle(rect: tuple[float, float, float, float]) -> None:
    if (type(rect) is not tuple or len(rect) != 4 or
            not all(_number(v) for v in rect) or
            rect[0] >= rect[2] or rect[1] >= rect[3]):
        raise ValueError("finite nondegenerate relief rectangle required")
    _finite(rect[2] - rect[0], rect[3] - rect[1])


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _xy(x: float, y: float) -> None:
    if not _number(x) or not _number(y):
        raise ValueError("finite relief XY required")


def _padding(*values: float) -> float:
    return 64 * math.ulp(max(1.0, *(abs(v) for v in values)))


def _gradient_bound(value: float) -> float:
    bound = value + _padding(value) if value else 0.0
    _finite(bound)
    return bound


@dataclass(frozen=True)
class SurfaceSample:
    """Differential data at one regular point of the actual design surface."""

    height_mm: float
    normal_xyz: tuple[float, float, float]
    principal_curvatures_per_mm: tuple[float, float]


@dataclass(frozen=True)
class CubicRelief:
    """Analytic moulding h(x,y)=a+b*x+c*x**2+d*x**3, extruded along Y.

    Stock top is Z=0.  Coefficients use the declared absolute X coordinates;
    heights, including extrema between endpoints, must stay within the stock.
    """

    stock_xy: tuple[float, float, float, float]
    stock_depth_mm: float
    coefficients: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _rectangle(self.stock_xy)
        if (not _number(self.stock_depth_mm) or self.stock_depth_mm <= 0 or
                type(self.coefficients) is not tuple or
                len(self.coefficients) != 4 or
                not all(_number(v) for v in self.coefficients)):
            raise ValueError("invalid cubic relief")
        lo, hi = self.height_bounds(self.stock_xy)
        if lo < -self.stock_depth_mm or hi > 0:
            raise ValueError("relief lies outside stock")
        self.gradient_bounds

    def _height(self, x: float) -> float:
        a, b, c, d = self.coefficients
        height = ((d * x + c) * x + b) * x + a
        _finite(height)
        return height

    def _critical_x(self) -> tuple[float, ...]:
        _, b, c, d = self.coefficients
        aa, bb = 3 * d, 2 * c
        _finite(aa, bb)
        if aa == 0:
            if bb == 0:
                return ()
            root = -b / bb
            _finite(root)
            return (root,)
        scale = max(abs(aa), abs(bb), abs(b))
        aa, bb, cc = aa / scale, bb / scale, b / scale
        if aa == 0:
            raise ValueError("cubic derivative arithmetic is not representable")
        discriminant = bb * bb - 4 * aa * cc
        if discriminant < 0:
            return ()
        q = -0.5 * (bb + math.copysign(math.sqrt(discriminant), bb))
        if q == 0:
            return (0.0,)
        roots = (q / aa, cc / q)
        _finite(*roots)
        return roots

    def height_at(self, x: float, y: float) -> float:
        """Height of the conservative clamped extension, including outside XY."""
        _xy(x, y)
        return self._height(_clamp(x, self.stock_xy[0], self.stock_xy[2]))

    def height_bounds(self, rect: tuple[float, float, float, float]) -> tuple[float, float]:
        """Exact-polynomial extrema on a rectangle, before arithmetic allowance."""
        _rectangle(rect)
        x0 = _clamp(rect[0], self.stock_xy[0], self.stock_xy[2])
        x1 = _clamp(rect[2], self.stock_xy[0], self.stock_xy[2])
        points = (x0, x1) + tuple(x for x in self._critical_x() if x0 < x < x1)
        heights = tuple(self._height(x) for x in points)
        return min(heights), max(heights)

    @property
    def gradient_bounds(self) -> tuple[float, float]:
        """Global bounds on absolute graph derivatives, including extension."""
        _, b, c, d = self.coefficients
        x0, _, x1, _ = self.stock_xy
        points = [x0, x1]
        if d != 0:
            root = -c / (3 * d)
            _finite(root)
            if x0 < root < x1:
                points.append(root)
        values = tuple(abs((3 * d * x + 2 * c) * x + b) for x in points)
        _finite(*values)
        return _gradient_bound(max(values)), 0.0

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def sample(self, x: float, y: float) -> SurfaceSample:
        """Height, upward unit normal and signed graph principal curvatures."""
        _xy(x, y)
        x0, y0, x1, y1 = self.stock_xy
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError("surface sample outside relief")
        _, b, c, d = self.coefficients
        hx, hxx = (3 * d * x + 2 * c) * x + b, 6 * d * x + 2 * c
        _finite(hx, hxx)
        length = math.hypot(1, hx)
        curvature = hxx / length / length / length
        return SurfaceSample(self._height(x), (-hx / length, 0.0, 1 / length),
                             tuple(sorted((0.0, curvature))))


@dataclass(frozen=True)
class HeightField:
    """Caller-owned, exact piecewise bilinear non-overhanging relief.

    ``heights[j][i]`` is Z at ``(xs[i], ys[j])``.  No interpolation-error claim
    is made about another source representation.  Differential samples on an
    interior cell seam reject because a unique normal need not exist there.
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    heights: tuple[tuple[float, ...], ...]
    stock_depth_mm: float

    def __post_init__(self) -> None:
        for axis in (self.xs, self.ys):
            if (type(axis) is not tuple or len(axis) < 2 or
                    not all(_number(v) for v in axis) or
                    any(a >= b for a, b in zip(axis, axis[1:]))):
                raise ValueError("strictly increasing finite relief axes required")
            _finite(axis[-1] - axis[0])
        if (not _number(self.stock_depth_mm) or self.stock_depth_mm <= 0 or
                type(self.heights) is not tuple or len(self.heights) != len(self.ys) or
                any(type(row) is not tuple or len(row) != len(self.xs)
                    for row in self.heights) or
                any(not _number(h) or not -self.stock_depth_mm <= h <= 0
                    for row in self.heights for h in row)):
            raise ValueError("invalid relief heights or stock depth")
        self.gradient_bounds

    @property
    def stock_xy(self) -> tuple[float, float, float, float]:
        return self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]

    def _cell(self, x: float, y: float) -> tuple[int, int, float, float]:
        x = _clamp(x, self.xs[0], self.xs[-1])
        y = _clamp(y, self.ys[0], self.ys[-1])
        i = min(len(self.xs) - 2, bisect_right(self.xs, x) - 1)
        j = min(len(self.ys) - 2, bisect_right(self.ys, y) - 1)
        u = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        v = (y - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        return i, j, u, v

    def height_at(self, x: float, y: float) -> float:
        """Bilinear height, extended outside stock by clamping XY."""
        _xy(x, y)
        i, j, u, v = self._cell(x, y)
        z00, z10 = self.heights[j][i:i + 2]
        z01, z11 = self.heights[j + 1][i:i + 2]
        return (1 - v) * ((1 - u) * z00 + u * z10) + v * ((1 - u) * z01 + u * z11)

    def height_bounds(self, rect: tuple[float, float, float, float]) -> tuple[float, float]:
        """Extrema at clipped rectangle corners and crossed grid coordinates."""
        _rectangle(rect)
        x0, y0 = _clamp(rect[0], self.xs[0], self.xs[-1]), _clamp(rect[1], self.ys[0], self.ys[-1])
        x1, y1 = _clamp(rect[2], self.xs[0], self.xs[-1]), _clamp(rect[3], self.ys[0], self.ys[-1])
        xs = (x0, x1) + tuple(x for x in self.xs if x0 < x < x1)
        ys = (y0, y1) + tuple(y for y in self.ys if y0 < y < y1)
        values = tuple(self.height_at(x, y) for x in xs for y in ys)
        return min(values), max(values)

    @property
    def gradient_bounds(self) -> tuple[float, float]:
        hx = max(abs((row[i + 1] - row[i]) / (self.xs[i + 1] - self.xs[i]))
                 for row in self.heights for i in range(len(self.xs) - 1))
        hy = max(abs((self.heights[j + 1][i] - self.heights[j][i]) /
                     (self.ys[j + 1] - self.ys[j]))
                 for j in range(len(self.ys) - 1) for i in range(len(self.xs)))
        _finite(hx, hy)
        return _gradient_bound(hx), _gradient_bound(hy)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def sample(self, x: float, y: float) -> SurfaceSample:
        _xy(x, y)
        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1]):
            raise ValueError("surface sample outside relief")
        if x in self.xs[1:-1] or y in self.ys[1:-1]:
            raise ValueError("relief seam has no unique differential sample")
        i, j, u, v = self._cell(x, y)
        z00, z10 = self.heights[j][i:i + 2]
        z01, z11 = self.heights[j + 1][i:i + 2]
        dx, dy = self.xs[i + 1] - self.xs[i], self.ys[j + 1] - self.ys[j]
        hx = ((1 - v) * (z10 - z00) + v * (z11 - z01)) / dx
        hy = ((1 - u) * (z01 - z00) + u * (z11 - z10)) / dy
        hxy = ((z11 - z01) - (z10 - z00)) / dx / dy
        length = math.hypot(1, hx, hy)
        mean = -(hx / length) * (hy / length) * (hxy / length)
        gaussian = -(hxy / length / length) ** 2
        discriminant = mean * mean - gaussian
        _finite(hx, hy, hxy, mean, gaussian, discriminant)
        root = math.sqrt(max(0.0, discriminant))
        return SurfaceSample(self.height_at(x, y),
                             (-hx / length, -hy / length, 1 / length),
                             (mean - root, mean + root))


Relief = CubicRelief | HeightField


@dataclass(frozen=True)
class Contact:
    """Bracket on ball-tip Z for no penetration of the extended design."""

    lower_tip_z_mm: float
    upper_tip_z_mm: float
    witness_xy: tuple[float, float]
    evaluated_cells: int


def _tangent_upper(surface: Relief, rect: tuple[float, float, float, float],
                   x: float, y: float, radius: float, extruded: bool) -> float:
    """Offset a local supporting plane plus its rigorous Taylor remainder.

    The plane's maximum over the whole disk is an upper bound over this cell.
    The ordinary height hull remains useful when that maximum is outside it.
    At grid or clamping seams this optional accelerator returns infinity.
    """
    left, bottom, right, top = rect
    sx0, sy0, sx1, sy1 = surface.stock_xy
    if ((left < sx0 < right) or (left < sx1 < right) or
            (not extruded and ((bottom < sy0 < top) or (bottom < sy1 < top)))):
        return math.inf
    mx, my = left + (right - left) / 2, bottom + (top - bottom) / 2
    cx, cy = _clamp(mx, sx0, sx1), _clamp(my, sy0, sy1)
    varying_x = sx0 <= left and right <= sx1
    varying_y = not extruded and sy0 <= bottom and top <= sy1
    if type(surface) is CubicRelief:
        _, b, c, d = surface.coefficients
        hx = (3 * d * cx + 2 * c) * cx + b if varying_x else 0.0
        hy = 0.0
        hxx = max(abs(6 * d * left + 2 * c), abs(6 * d * right + 2 * c)) if varying_x else 0.0
        remainder = 0.5 * hxx * ((right - left) / 2) ** 2
    else:
        i, j, u, v = surface._cell(cx, cy)
        if ((varying_x and not surface.xs[i] <= left <= right <= surface.xs[i + 1]) or
                (varying_y and not surface.ys[j] <= bottom <= top <= surface.ys[j + 1])):
            return math.inf
        z00, z10 = surface.heights[j][i:i + 2]
        z01, z11 = surface.heights[j + 1][i:i + 2]
        dx, dy = surface.xs[i + 1] - surface.xs[i], surface.ys[j + 1] - surface.ys[j]
        hx = ((1 - v) * (z10 - z00) + v * (z11 - z01)) / dx if varying_x else 0.0
        hy = ((1 - u) * (z01 - z00) + u * (z11 - z10)) / dy if varying_y else 0.0
        hxy = ((z11 - z01) - (z10 - z00)) / dx / dy if varying_x and varying_y else 0.0
        remainder = abs(hxy) * ((right - left) / 2) * ((top - bottom) / 2)
    upper = (surface.height_at(mx, my) + hx * (x - mx) + hy * (y - my) +
             radius * (math.hypot(1, hx, hy) - 1) + remainder)
    _finite(hx, hy, remainder, upper)
    return upper


def contact(surface: Relief, radius_mm: float, x: float, y: float, *,
            tolerance_mm: float = 0.001, max_cells: int = 20000) -> Contact:
    """Bound ball contact by rectangular branch-and-bound over its XY disk.

    The lower endpoint has a surface witness.  The upper endpoint is a hull
    bound; only it grants non-penetration.  The interval includes the stated
    arithmetic allowance and must fit the requested tolerance before returning.
    Contact is a geometric query, without cutting-length, stock or body proof.
    """
    if (type(surface) not in (CubicRelief, HeightField) or
            not _number(radius_mm) or radius_mm <= 0 or
            not _number(tolerance_mm) or tolerance_mm <= 0 or
            type(max_cells) is not int or max_cells < 1):
        raise ValueError("invalid relief contact query")
    _xy(x, y)
    radius = radius_mm
    x0, x1, y0, y1 = x - radius, x + radius, y - radius, y + radius
    radius2 = radius * radius
    _finite(x0, x1, y0, y1, radius2)
    if x0 >= x1 or y0 >= y1 or radius2 == 0:
        raise ValueError("contact footprint is not representable")
    arithmetic_scale = surface.stock_depth_mm
    if type(surface) is CubicRelief:
        # Horner cancellation can hide large polynomial terms behind a shallow
        # resulting height.  Include that scale rather than trusting the result.
        xmax = max(abs(surface.stock_xy[0]), abs(surface.stock_xy[2]))
        a, b, c, d = surface.coefficients
        terms = (abs(a), abs(b) * xmax, abs(c) * xmax * xmax,
                 abs(d) * xmax * xmax * xmax)
        _finite(*terms)
        arithmetic_scale = max(arithmetic_scale, *terms)
    allowance = _padding(x0, x1, y0, y1, radius, arithmetic_scale)
    if 2 * allowance >= tolerance_mm:
        raise ValueError("contact tolerance is below arithmetic allowance")
    # An extruded surface's maximum is attained at y=center_y.  This removes
    # a redundant search dimension without changing the geometric query.
    extruded = surface.gradient_bounds[1] == 0
    lower = surface.height_at(x, y)
    witness = (x, y)
    pending: list[tuple[float, int, tuple[float, float, float, float]]] = []
    evaluated = 0

    def evaluate(rect: tuple[float, float, float, float]) -> None:
        nonlocal lower, witness, evaluated
        if evaluated >= max_cells:
            raise ValueError("relief contact refinement budget exhausted")
        evaluated += 1
        left, bottom, right, top = rect
        nearest_x, nearest_y = _clamp(x, left, right), _clamp(y, bottom, top)
        nearest2 = (nearest_x - x) ** 2 + (nearest_y - y) ** 2
        if nearest2 > radius2:
            return
        height_upper = surface.height_bounds(rect)[1]
        upper = height_upper - radius + math.sqrt(max(0.0, radius2 - nearest2))
        upper = min(upper, _tangent_upper(surface, rect, x, y, radius, extruded))
        middle_x = left + (right - left) / 2
        middle_y = bottom + (top - bottom) / 2
        candidates = ((nearest_x, nearest_y), (middle_x, middle_y),
                      (left, bottom), (left, top), (right, bottom), (right, top))
        if extruded:
            candidates += ((left, y), (middle_x, y), (right, y))
        for qx, qy in candidates:
            distance2 = (qx - x) ** 2 + (qy - y) ** 2
            if distance2 <= radius2:
                value = surface.height_at(qx, qy) - radius + math.sqrt(max(0.0, radius2 - distance2))
                if value > lower:
                    lower, witness = value, (qx, qy)
        _finite(upper, lower)
        if upper + allowance > lower - allowance:
            heapq.heappush(pending, (-upper, evaluated, rect))

    evaluate((x0, y0, x1, y1))
    while pending:
        upper = max(lower, -pending[0][0])
        if upper - lower + 2 * allowance <= tolerance_mm:
            return Contact(lower - allowance, upper + allowance, witness, evaluated)
        _, _, rect = heapq.heappop(pending)
        left, bottom, right, top = rect
        if extruded or right - left >= top - bottom:
            middle = left + (right - left) / 2
            if not left < middle < right:
                raise ValueError("contact refinement is not representable")
            evaluate((left, bottom, middle, top))
            evaluate((middle, bottom, right, top))
        else:
            middle = bottom + (top - bottom) / 2
            if not bottom < middle < top:
                raise ValueError("contact refinement is not representable")
            evaluate((left, bottom, right, middle))
            evaluate((left, middle, right, top))
    return Contact(lower - allowance, lower + allowance, witness, evaluated)
