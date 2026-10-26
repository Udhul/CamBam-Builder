"""Compact authored swept reliefs and explicit, bounded source reconstruction.

Coordinates and lengths are millimetres. Sources retain analytic construction;
height fields, triangles and tracing polylines are disposable derived views.
Supported surfaces are non-overhanging max-unions of swept raised profiles.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relief import HeightField


SCHEMA = "constructive-sweeps-v1"
MAX_SAMPLES = 250_000


def _finite(value: float) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _numbers(values: tuple[float, ...], count: int, label: str) -> None:
    if type(values) is not tuple or len(values) != count or not all(_finite(v) for v in values):
        raise ValueError(f"finite {label} tuple required")


def _positive(value: float, label: str) -> None:
    if not _finite(value) or value <= 0:
        raise ValueError(f"positive finite {label} required")


def _segments(distance: float, step: float) -> int:
    ratio = distance / step
    if not math.isfinite(ratio) or ratio > MAX_SAMPLES:
        raise ValueError("constructive trace budget exceeded")
    return max(1, math.ceil(ratio))


def _pad(value: float) -> float:
    return 64 * math.ulp(max(1.0, abs(value)))


def _sin_bounds(a: float, b: float) -> tuple[float, float]:
    """Range of sin on an ordered, finite interval including interior extrema."""
    if b - a >= 2 * math.pi:
        return -1.0, 1.0
    values = [math.sin(a), math.sin(b)]
    first = math.ceil((a - math.pi / 2) / math.pi)
    last = math.floor((b - math.pi / 2) / math.pi)
    values.extend(1.0 if n % 2 == 0 else -1.0 for n in range(first, last + 1))
    return min(values), max(values)


def _linear(pair: tuple[float, float], u: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * u


@dataclass(frozen=True)
class SweptRidge:
    """Raised profile swept along an analytic spine in template coordinates.

    ``widths_mm`` are half widths. Center, width and height endpoints vary
    linearly with local X. A sine term bends the spine; cycles and phase remain
    editable descriptors. ``cap_ends`` multiplies height by sin(pi*u)**2.
    Cosine profiles have smooth support boundaries; tent profiles retain a sharp
    ridge and explicit support creases. Max-union crossings retain crease intent.
    """

    name: str
    x_interval_mm: tuple[float, float]
    centers_mm: tuple[float, float]
    widths_mm: tuple[float, float]
    heights_mm: tuple[float, float]
    wave_amplitude_mm: float = 0.0
    wave_cycles: float = 1.0
    wave_phase_rad: float = 0.0
    profile: str = "cosine"
    cap_ends: bool = True

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("ridge name required")
        for pair, label in ((self.x_interval_mm, "ridge X"), (self.centers_mm, "centers"),
                            (self.widths_mm, "widths"), (self.heights_mm, "heights")):
            _numbers(pair, 2, label)
        if (self.x_interval_mm[0] >= self.x_interval_mm[1] or
                not math.isfinite(self.x_interval_mm[1] - self.x_interval_mm[0])):
            raise ValueError("increasing ridge X interval required")
        if min(self.widths_mm) <= 0 or min(self.heights_mm) < 0:
            raise ValueError("positive widths and nonnegative heights required")
        if not all(_finite(v) for v in (self.wave_amplitude_mm, self.wave_cycles,
                                        self.wave_phase_rad)) or self.wave_cycles < 0:
            raise ValueError("finite spine wave and nonnegative cycles required")
        if self.profile not in ("cosine", "tent") or type(self.cap_ends) is not bool:
            raise ValueError("unsupported sweep profile or cap")
        if max(abs(self.wave_phase_rad), abs(2 * math.pi * self.wave_cycles)) > 1e6:
            raise ValueError("spine phase exceeds bounded arithmetic domain")
        if not all(math.isfinite(v) for v in self.derivative_bounds):
            raise ValueError("ridge derivative arithmetic is not representable")
        # Cancellation between a large baseline and opposite wave can quantize
        # the spine into jumps much larger than the intended profile width.
        # Such a descriptor cannot support our continuous derivative contract.
        center_scale = max(1.0, *(abs(v) for v in self.centers_mm), abs(self.wave_amplitude_mm))
        phase_scale = max(1.0, abs(2 * math.pi * self.wave_cycles), abs(self.wave_phase_rad))
        center_guard = (64 * math.ulp(center_scale) +
                        abs(self.wave_amplitude_mm) * 64 * math.ulp(phase_scale))
        if center_guard > min(self.widths_mm) * 1e-6:
            raise ValueError("ill-conditioned spine descriptor relative to section width")

    @cached_property
    def length_mm(self) -> float:
        return self.x_interval_mm[1] - self.x_interval_mm[0]

    def center_at(self, u: float) -> float:
        return (_linear(self.centers_mm, u) + self.wave_amplitude_mm *
                math.sin(2 * math.pi * self.wave_cycles * u + self.wave_phase_rad))

    def width_at(self, u: float) -> float:
        return _linear(self.widths_mm, u)

    def _profile(self, q: float) -> float:
        if abs(q) >= 1:
            return 0.0
        if self.profile == "tent":
            return 1 - abs(q)
        return (1 + math.cos(math.pi * q)) / 2

    def height_at(self, x: float, y: float) -> float:
        if not self.x_interval_mm[0] <= x <= self.x_interval_mm[1]:
            return 0.0
        u = (x - self.x_interval_mm[0]) / self.length_mm
        cap = math.sin(math.pi * u) ** 2 if self.cap_ends else 1.0
        return (_linear(self.heights_mm, u) * cap *
                self._profile((y - self.center_at(u)) / self.width_at(u)))

    @cached_property
    def derivative_bounds(self) -> tuple[float, float]:
        """Global absolute X/Y derivative bounds, including support creases."""
        length = self.length_mm
        width = min(self.widths_mm)
        amplitude = max(self.heights_mm)
        center_rate = (abs(self.centers_mm[1] - self.centers_mm[0]) +
                       abs(self.wave_amplitude_mm * 2 * math.pi * self.wave_cycles)) / length
        width_rate = abs(self.widths_mm[1] - self.widths_mm[0]) / length
        profile_rate = math.pi / 2 if self.profile == "cosine" else 1.0
        envelope_rate = abs(self.heights_mm[1] - self.heights_mm[0]) / length
        if self.cap_ends:
            envelope_rate += amplitude * math.pi / length
        return (envelope_rate + amplitude * profile_rate * (center_rate + width_rate) / width,
                amplitude * profile_rate / width)

    @cached_property
    def arithmetic_height_allowance_mm(self) -> float:
        """Conservative finite-arithmetic envelope for admitted descriptors."""
        center_scale = max(1.0, *(abs(v) for v in self.centers_mm), abs(self.wave_amplitude_mm))
        phase_scale = max(1.0, abs(2 * math.pi * self.wave_cycles), abs(self.wave_phase_rad))
        center_guard = (64 * math.ulp(center_scale) +
                        abs(self.wave_amplitude_mm) * 64 * math.ulp(phase_scale))
        width_guard = 64 * math.ulp(max(1.0, *self.widths_mm))
        x_guard = 64 * math.ulp(max(1.0, *(abs(v) for v in self.x_interval_mm)))
        amplitude = max(self.heights_mm)
        profile_rate = math.pi / 2 if self.profile == "cosine" else 1.0
        return (self.derivative_bounds[0] * x_guard +
                amplitude * profile_rate * (center_guard + width_guard) / min(self.widths_mm) +
                64 * math.ulp(max(1.0, amplitude)))

    def height_bounds(self, rect: tuple[float, float, float, float]) -> tuple[float, float]:
        """Conservative interval extension; narrows to its arithmetic envelope."""
        x0, y0, x1, y1 = rect
        a, b = max(x0, self.x_interval_mm[0]), min(x1, self.x_interval_mm[1])
        if a > b:
            return 0.0, 0.0
        u0, u1 = ((v - self.x_interval_mm[0]) / self.length_mm for v in (a, b))
        linear_centers = (_linear(self.centers_mm, u0), _linear(self.centers_mm, u1))
        sine = _sin_bounds(2 * math.pi * self.wave_cycles * u0 + self.wave_phase_rad,
                           2 * math.pi * self.wave_cycles * u1 + self.wave_phase_rad)
        waves = tuple(self.wave_amplitude_mm * v for v in sine)
        c0, c1 = min(linear_centers) + min(waves), max(linear_centers) + max(waves)
        widths = (_linear(self.widths_mm, u0), _linear(self.widths_mm, u1))
        numerators = (y0 - c1, y1 - c0)
        quotients = tuple(n / w for n in numerators for w in widths)
        q0, q1 = min(quotients), max(quotients)
        profile_values = [self._profile(q0), self._profile(q1)]
        if q0 <= 0 <= q1:
            profile_values.append(1.0)
        p0, p1 = min(profile_values), max(profile_values)
        amplitudes = (_linear(self.heights_mm, u0), _linear(self.heights_mm, u1))
        cap0, cap1 = 1.0, 1.0
        if self.cap_ends:
            cap_values = [math.sin(math.pi * u0) ** 2, math.sin(math.pi * u1) ** 2]
            if u0 <= 0.5 <= u1:
                cap_values.append(1.0)
            cap0, cap1 = min(cap_values), max(cap_values)
        low, high = min(amplitudes) * cap0 * p0, max(amplitudes) * cap1 * p1
        if x0 < self.x_interval_mm[0] or x1 > self.x_interval_mm[1]:
            low = 0.0
        return max(0.0, low - _pad(low)), high + _pad(high)


@dataclass(frozen=True)
class SampledSource:
    """A disposable height field bound to its compact source revision.

    ``deviation_bound_mm`` bounds vertical error to the original source for
    both bilinear interpolation and either triangle diagonal on each grid cell.
    It is a conservative Lipschitz bound, not observed sample agreement.
    """

    target: "HeightField"
    source_id: str
    source_fingerprint: str
    deviation_bound_mm: float

    def require_current(self, source: "ConstructiveSource") -> None:
        if self.source_id != source.source_id or self.source_fingerprint != source.fingerprint:
            raise ValueError("derived geometry is stale after source or modifier edit")


@dataclass(frozen=True)
class ConstructiveSource:
    """Editable max-union of analytic swept profiles on rectangular stock.

    Repetition operates in X before XYZ scaling. ``origin_xy`` is the final
    world-space origin; heights and template depth scale in Z. All dimensions
    in ridge definitions remain template values. Source identity survives edits;
    the canonical fingerprint changes with every parameter/provenance revision.
    """

    source_id: str
    template_length_mm: float
    template_width_mm: float
    template_depth_mm: float
    base_height_mm: float
    ridges: tuple[SweptRidge, ...]
    repeat_count: int = 1
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin_xy: tuple[float, float] = (0.0, 0.0)
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source_id) is not str or not self.source_id:
            raise ValueError("source identity required")
        for value, label in ((self.template_length_mm, "template length"),
                             (self.template_width_mm, "template width"),
                             (self.template_depth_mm, "template depth")):
            _positive(value, label)
        if not _finite(self.base_height_mm) or not -self.template_depth_mm <= self.base_height_mm <= 0:
            raise ValueError("base must lie within template stock")
        if (type(self.ridges) is not tuple or
                any(type(r) is not SweptRidge for r in self.ridges) or
                len({r.name for r in self.ridges}) != len(self.ridges)):
            raise ValueError("uniquely named swept ridges required")
        if type(self.repeat_count) is not int or not 1 <= self.repeat_count <= 1000:
            raise ValueError("repeat count must be an integer in 1..1000")
        _numbers(self.scale_xyz, 3, "XYZ scale")
        _numbers(self.origin_xy, 2, "origin XY")
        if min(self.scale_xyz) <= 0:
            raise ValueError("positive XYZ scale required")
        if type(self.provenance) is not tuple or any(type(v) is not str for v in self.provenance):
            raise ValueError("string provenance tuple required")
        for ridge in self.ridges:
            a, b = ridge.x_interval_mm
            if a < 0 or b > self.template_length_mm:
                raise ValueError("ridge X interval lies outside template")
            if self.base_height_mm + max(ridge.heights_mm) > 0:
                raise ValueError("raised profile exceeds stock top")
            if not ridge.cap_ends:
                if (a != 0 or b != self.template_length_mm):
                    raise ValueError("uncapped ridge must span entire template")
                if self.repeat_count > 1 and (ridge.centers_mm[0] != ridge.centers_mm[1] or
                        ridge.widths_mm[0] != ridge.widths_mm[1] or
                        ridge.heights_mm[0] != ridge.heights_mm[1] or
                        ridge.wave_cycles != int(ridge.wave_cycles)):
                    raise ValueError("uncapped repeat requires matching periodic sections")
        if not all(math.isfinite(v) for v in (*self.stock_xy, self.stock_depth_mm,
                                               *self.gradient_bounds, self.arithmetic_height_allowance_mm)):
            raise ValueError("source arithmetic is not representable")
        if self.stock_xy[0] >= self.stock_xy[2] or self.stock_xy[1] >= self.stock_xy[3]:
            raise ValueError("source dimensions are not representable")

    @cached_property
    def stock_xy(self) -> tuple[float, float, float, float]:
        x, y = self.origin_xy
        return (x, y, x + self.template_length_mm * self.repeat_count * self.scale_xyz[0],
                y + self.template_width_mm * self.scale_xyz[1])

    @cached_property
    def stock_depth_mm(self) -> float:
        return self.template_depth_mm * self.scale_xyz[2]

    @cached_property
    def gradient_bounds(self) -> tuple[float, float]:
        gx = max((r.derivative_bounds[0] for r in self.ridges), default=0.0) * self.scale_xyz[2] / self.scale_xyz[0]
        gy = max((r.derivative_bounds[1] for r in self.ridges), default=0.0) * self.scale_xyz[2] / self.scale_xyz[1]
        return gx + _pad(gx) if gx else 0.0, gy + _pad(gy) if gy else 0.0

    def _template_xy(self, x: float, y: float) -> tuple[float, float]:
        if not _finite(x) or not _finite(y):
            raise ValueError("finite source XY required")
        x0, y0, x1, y1 = self.stock_xy
        tx = (min(x1, max(x0, x)) - x0) / self.scale_xyz[0]
        ty = (min(y1, max(y0, y)) - y0) / self.scale_xyz[1]
        return min(self.template_length_mm, tx % self.template_length_mm
                   if tx < self.template_length_mm * self.repeat_count else self.template_length_mm), ty

    def height_at(self, x: float, y: float) -> float:
        """Analytic source height with conservative stock-edge clamping."""
        tx, ty = self._template_xy(x, y)
        return (self.base_height_mm + max((r.height_at(tx, ty) for r in self.ridges), default=0.0)) * self.scale_xyz[2]

    def height_bounds(self, rect: tuple[float, float, float, float]) -> tuple[float, float]:
        """Conservative local source range including repeated joins and max seams."""
        _numbers(rect, 4, "rectangle")
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError("nondegenerate source rectangle required")
        x0, y0, x1, y1 = self.stock_xy
        a = (max(x0, min(x1, rect[0])) - x0) / self.scale_xyz[0]
        b = (max(x0, min(x1, rect[2])) - x0) / self.scale_xyz[0]
        c = (max(y0, min(y1, rect[1])) - y0) / self.scale_xyz[1]
        d = (max(y0, min(y1, rect[3])) - y0) / self.scale_xyz[1]
        first = min(self.repeat_count - 1, int(a / self.template_length_mm))
        last = min(self.repeat_count - 1, int(b / self.template_length_mm))
        ranges = []
        for repeat in range(first, last + 1):
            offset = repeat * self.template_length_mm
            local = (max(0.0, a - offset), c, min(self.template_length_mm, b - offset), d)
            bounds = tuple(r.height_bounds(local) for r in self.ridges)
            ranges.append((max((v[0] for v in bounds), default=0.0),
                           max((v[1] for v in bounds), default=0.0)))
        low = (self.base_height_mm + min(v[0] for v in ranges)) * self.scale_xyz[2]
        high = (self.base_height_mm + max(v[1] for v in ranges)) * self.scale_xyz[2]
        arithmetic = self.arithmetic_height_allowance_mm
        return low - _pad(low) - arithmetic, high + _pad(high) + arithmetic

    def canonical_json(self) -> str:
        """Canonical editable source; no derived tessellation or toolpath data."""
        return json.dumps({"schema": SCHEMA, **asdict(self)}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @cached_property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "ConstructiveSource":
        """Load this explicit schema, rejecting unknown keys and invalid geometry."""
        try:
            data = json.loads(text)
            if type(data) is not dict or data.pop("schema") != SCHEMA:
                raise ValueError("unsupported constructive source schema")
            ridges = []
            for record in data["ridges"]:
                for key in ("x_interval_mm", "centers_mm", "widths_mm", "heights_mm"):
                    record[key] = tuple(record[key])
                ridges.append(SweptRidge(**record))
            data["ridges"] = tuple(ridges)
            for key in ("scale_xyz", "origin_xy", "provenance"):
                data[key] = tuple(data[key])
            return cls(**data)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid constructive source document") from exc

    @cached_property
    def arithmetic_height_allowance_mm(self) -> float:
        """World-coordinate and descriptor arithmetic allowance for evaluations."""
        x0, y0, x1, y1 = self.stock_xy
        gx, gy = self.gradient_bounds
        descriptors = max((r.arithmetic_height_allowance_mm for r in self.ridges), default=0.0)
        return (64 * math.ulp(max(1.0, abs(x0), abs(x1))) * gx +
                64 * math.ulp(max(1.0, abs(y0), abs(y1))) * gy +
                descriptors * self.scale_xyz[2] + _pad(self.stock_depth_mm))

    def sample_heightfield(self, *, x_samples: int, y_samples: int) -> SampledSource:
        """Lower to a regular grid with a certified vertical approximation bound.

        The bound follows the global Lipschitz derivatives and convex corner
        interpolation, so it remains valid at tent creases, max seams and repeats.
        Both bilinear and triangle interpolation reproduce affine XY coordinates.
        """
        from .relief import HeightField

        if (type(x_samples) is not int or type(y_samples) is not int or
                min(x_samples, y_samples) < 2 or x_samples * y_samples > MAX_SAMPLES):
            raise ValueError("heightfield sample counts exceed supported budget")
        x0, y0, x1, y1 = self.stock_xy
        xs = tuple(x0 + (x1 - x0) * i / (x_samples - 1) for i in range(x_samples))
        ys = tuple(y0 + (y1 - y0) * i / (y_samples - 1) for i in range(y_samples))
        heights = tuple(tuple(self.height_at(x, y) for x in xs) for y in ys)
        target = HeightField(xs, ys, heights, self.stock_depth_mm)
        gx, gy = self.gradient_bounds
        # Floating-point translated grids are not uniformly spaced in their
        # actual coordinates. Bound the cells that were emitted, not nominal
        # template increments. Coordinate subtraction/division and trigonometric
        # evaluation receive a separate conservative arithmetic envelope.
        dx = max(b - a for a, b in zip(xs, xs[1:]))
        dy = max(b - a for a, b in zip(ys, ys[1:]))
        arithmetic = self.arithmetic_height_allowance_mm
        error = gx * dx / 2 + gy * dy / 2 + arithmetic
        if not math.isfinite(error):
            raise ValueError("source approximation bound is not representable")
        return SampledSource(target, self.source_id, self.fingerprint, error + _pad(error))

    def trace_xy(self, family: str, *, spacing_mm: float,
                 sample_step_mm: float) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Feature paths in world XY; complete-area coverage is a CAM responsibility.

        Spine traces follow constant normalized section coordinates, retaining
        ridge and support curves. Section traces cross the sweep at X stations.
        Spacing controls normalized offsets/stations; sample_step bounds X or Y
        parameter increments, not surface arc length or machining scallop.
        """
        if family not in ("spine", "section"):
            raise ValueError("supported trace families are spine and section")
        _positive(spacing_mm, "trace spacing")
        _positive(sample_step_mm, "trace sample step")
        sx, sy, _ = self.scale_xyz
        x0, y0 = self.origin_xy
        output = []
        points = 0
        for repeat in range(self.repeat_count):
            offset = repeat * self.template_length_mm
            for ridge in self.ridges:
                along_count = _segments(ridge.length_mm * sx, sample_step_mm)
                cross_count = max(2, _segments(2 * max(ridge.widths_mm) * sy, spacing_mm))
                if family == "spine":
                    cross_count += cross_count % 2  # Explicit center trace.
                    if (cross_count + 1) * (along_count + 1) + points > MAX_SAMPLES:
                        raise ValueError("constructive trace budget exceeded")
                    paths = [[(i / along_count, -1 + 2 * j / cross_count)
                              for i in range(along_count + 1)] for j in range(cross_count + 1)]
                else:
                    station_count = _segments(ridge.length_mm * sx, spacing_mm)
                    section_count = _segments(2 * max(ridge.widths_mm) * sy, sample_step_mm)
                    if (station_count + 1) * (section_count + 1) + points > MAX_SAMPLES:
                        raise ValueError("constructive trace budget exceeded")
                    paths = [[(i / station_count, -1 + 2 * j / section_count)
                              for j in range(section_count + 1)] for i in range(station_count + 1)]
                for path in paths:
                    xy = tuple((x0 + (offset + ridge.x_interval_mm[0] + u * ridge.length_mm) * sx,
                                min(self.stock_xy[3], max(y0, y0 +
                                    (ridge.center_at(u) + q * ridge.width_at(u)) * sy)))
                               for u, q in path)
                    points += len(xy)
                    if points > MAX_SAMPLES:
                        raise ValueError("constructive trace budget exceeded")
                    if len(set(xy)) > 1:
                        output.append(xy)
        return tuple(output)


@dataclass(frozen=True)
class SpineFit:
    """One declared sinusoidal candidate, with located observation residuals."""

    cycles: float
    centers_mm: tuple[float, float]
    wave_amplitude_mm: float
    wave_phase_rad: float
    maximum_residual_mm: float
    rms_residual_mm: float
    residuals_mm: tuple[float, ...]
    rank: int
    identifiable: bool


@dataclass(frozen=True)
class Reconstruction:
    """Explicit fitting evidence; never a claim of recovered author intent.

    Error metrics concern supplied correspondences only. They do not certify
    unsampled geometry or arbitrary raster/mesh reconstruction. All observations
    are retained exactly as values and bound by a canonical input fingerprint.
    """

    observations_xy: tuple[tuple[float, float], ...]
    input_fingerprint: str
    x_interval_mm: tuple[float, float]
    candidates: tuple[SpineFit, ...]
    selected_index: int | None
    ambiguous: bool
    diagnostics: tuple[str, ...]

    @property
    def recovered_author_intent(self) -> bool:
        return False


def reconstruct_spine(observations_xy: tuple[tuple[float, float], ...], *,
                      x_interval_mm: tuple[float, float],
                      candidate_cycles: tuple[float, ...] = (0.0, 1.0, 2.0),
                      residual_limit_mm: float = 0.05,
                      ambiguity_margin_mm: float = 0.001) -> Reconstruction:
    """Fit an explicitly enumerated linear-plus-sine spine model using NumPy.

    Each observation carries known X correspondence. Profile, width, material,
    topology and construction intent are not inferred. A candidate is selected
    only when its design matrix has full rank, its residual is within the caller
    limit and no competing admissible model is within the ambiguity margin.
    """
    import numpy as np

    _numbers(x_interval_mm, 2, "reconstruction X")
    if x_interval_mm[0] >= x_interval_mm[1]:
        raise ValueError("increasing reconstruction X interval required")
    _positive(residual_limit_mm, "fit residual limit")
    if not _finite(ambiguity_margin_mm) or ambiguity_margin_mm < 0:
        raise ValueError("nonnegative fit ambiguity margin required")
    if (type(observations_xy) is not tuple or len(observations_xy) < 2 or
            len(observations_xy) > MAX_SAMPLES):
        raise ValueError("bounded reconstruction observations required")
    for point in observations_xy:
        _numbers(point, 2, "observation XY")
        if not x_interval_mm[0] <= point[0] <= x_interval_mm[1]:
            raise ValueError("observation X outside declared correspondence")
    if (type(candidate_cycles) is not tuple or not candidate_cycles or
            len(candidate_cycles) > 100 or len(set(candidate_cycles)) != len(candidate_cycles) or
            any(not _finite(v) or not 0 <= v <= 1000 for v in candidate_cycles)):
        raise ValueError("finite unique candidate cycles in 0..1000 required")
    length = x_interval_mm[1] - x_interval_mm[0]
    us = np.array([(x - x_interval_mm[0]) / length for x, _ in observations_xy])
    ys = np.array([y for _, y in observations_xy])
    candidates = []
    for cycles in candidate_cycles:
        columns = [np.ones(len(us)), us]
        if cycles:
            columns.extend((np.sin(2 * math.pi * cycles * us), np.cos(2 * math.pi * cycles * us)))
        matrix = np.column_stack(columns)
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, ys, rcond=None)
        if not np.isfinite(coefficients).all():
            raise ValueError("reconstruction arithmetic is not representable")
        residuals = tuple(float(v) for v in matrix @ coefficients - ys)
        if not all(math.isfinite(value) for value in residuals):
            raise ValueError("reconstruction residuals are not representable")
        amplitude = float(math.hypot(*coefficients[2:])) if cycles else 0.0
        phase = float(math.atan2(coefficients[3], coefficients[2])) if cycles and amplitude > 1e-14 else 0.0
        centers = (float(coefficients[0]), float(coefficients[0] + coefficients[1]))
        maximum = max(abs(v) for v in residuals)
        rms = (maximum * math.sqrt(sum((v / maximum) ** 2 for v in residuals) / len(residuals))
               if maximum else 0.0)
        if not all(math.isfinite(v) for v in (*centers, amplitude, phase, maximum, rms)):
            raise ValueError("fitted descriptor arithmetic is not representable")
        candidates.append(SpineFit(float(cycles), centers, amplitude, phase, maximum, rms,
                          residuals, int(rank), bool(rank == len(columns))))
    candidates.sort(key=lambda fit: (fit.maximum_residual_mm, fit.cycles))
    admissible = [i for i, fit in enumerate(candidates)
                  if fit.identifiable and fit.maximum_residual_mm <= residual_limit_mm]
    diagnostics = ["Fit errors bound observations only; supplied X correspondence is assumed.",
                   "Profile, topology and original author intent are not reconstructed."]
    ambiguous = not admissible
    selected = admissible[0] if admissible else None
    if selected is not None:
        best = candidates[selected].maximum_residual_mm
        ambiguous = any(candidates[i].maximum_residual_mm <= best + ambiguity_margin_mm
                        for i in admissible[1:])
    if any(not fit.identifiable for fit in candidates):
        diagnostics.append("At least one candidate is rank deficient and cannot establish unique parameters.")
        # An underdetermined admissible model is evidence of ambiguity too.
        if any(not f.identifiable and f.maximum_residual_mm <= residual_limit_mm for f in candidates):
            ambiguous = True
    if ambiguous:
        selected = None
        diagnostics.append("No unique admissible construction in the declared candidate space.")
    serialized = json.dumps(observations_xy, separators=(",", ":"), allow_nan=False)
    return Reconstruction(observations_xy, hashlib.sha256(serialized.encode()).hexdigest(),
                          x_interval_mm, tuple(candidates), selected, ambiguous, tuple(diagnostics))
