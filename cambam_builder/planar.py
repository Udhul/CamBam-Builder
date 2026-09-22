"""Detached nominal planar geometry, in local millimetres.

No value here is a stock, clearance, toolpath, or containment certificate.
Shapely is optional and confined to the private adapter. Analytic feasible
centers do not need it. See the implemented contract in docs/structure_spec.md.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from typing import Optional, Tuple, Union


XY = Tuple[float, float]


def _xy(value):
    return tuple(value)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


@dataclass(frozen=True)
class PlanarFrame:
    """Coordinates use source units; origin is subtracted before conversion.

    frame_id names the coordinate system, not the shape. origin is an explicit
    numerical anchor in that system. Output local XY maps back by /scale+origin.
    """

    units: str
    frame_id: str
    origin: XY
    section_z: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "origin", _xy(self.origin))


@dataclass(frozen=True)
class ErrorBudget:
    boundary_mm: float
    area_mm2: float
    max_segments: int = 100000
    require_certified: bool = False


@dataclass(frozen=True)
class Polygon:
    """Explicitly closed line rings; hole ownership does not depend on winding."""

    shell: Tuple[XY, ...]
    holes: Tuple[Tuple[XY, ...], ...] = ()
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "shell", tuple(_xy(p) for p in self.shell))
        object.__setattr__(self, "holes", tuple(tuple(_xy(p) for p in h) for h in self.holes))


@dataclass(frozen=True)
class Rectangle:
    center: XY
    width: float
    height: float
    angle_radians: float = 0.0
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "center", _xy(self.center))


@dataclass(frozen=True)
class Circle:
    center: XY
    radius: float
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "center", _xy(self.center))


@dataclass(frozen=True)
class RegionSet:
    frame: PlanarFrame
    components: Tuple[Union[Polygon, Rectangle, Circle], ...]

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))


@dataclass(frozen=True)
class ErrorTerm:
    stage: str
    boundary_mm: Optional[float]
    area_mm2: Optional[float]
    kind: str  # bound, allowance, or unknown; allowances are not proofs


@dataclass(frozen=True)
class SourceSpan:
    source: str
    component: int
    ring: int
    segment: int
    parameter_start: float
    parameter_end: float
    start_mm: XY
    end_mm: XY

    def __post_init__(self):
        object.__setattr__(self, "start_mm", _xy(self.start_mm))
        object.__setattr__(self, "end_mm", _xy(self.end_mm))


@dataclass(frozen=True)
class Provenance:
    input_fingerprints: Tuple[str, ...]
    policy_fingerprint: str
    parameters: Tuple[Tuple[str, str], ...]
    adapter_revision: str
    backend_versions: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("input_fingerprints", "parameters", "backend_versions"):
            object.__setattr__(self, name, _tuples(getattr(self, name)))


@dataclass(frozen=True)
class PlanarApproximation:
    """Normalized polygons and input lineage; no backend objects escape.

    source_spans maps original tessellated segments at normalization. Derived
    operations retain it as input lineage, not a claimed output-edge attribution.
    """

    frame: PlanarFrame
    components: Tuple[Polygon, ...]
    source_spans: Tuple[SourceSpan, ...]
    fingerprint: str
    error_ledger: Tuple[ErrorTerm, ...]
    provenance: Provenance

    def __post_init__(self):
        for name in ("components", "source_spans", "error_ledger"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class FeasibleSet:
    """Exact rational local-mm centers, placed by the original analytic frame.

    Areas are tagged ('rectangle', (half_width, half_height)) or
    ('disk', (radius,)). Segments and points use primitive-centered local axes.
    center_mm and angle_radians place these axes relative to frame.origin.
    Rational coordinates prevent rounding a near fit into a point/segment.
    """

    frame: PlanarFrame
    center_mm: Tuple[Fraction, Fraction]
    angle_radians: float
    areas: Tuple[Tuple[str, Tuple[Fraction, ...]], ...] = ()
    segments: Tuple[Tuple[Tuple[Fraction, Fraction], ...], ...] = ()
    points: Tuple[Tuple[Fraction, Fraction], ...] = ()

    def __post_init__(self):
        for name in ("center_mm", "areas", "segments", "points"):
            object.__setattr__(self, name, _tuples(getattr(self, name)))

    @property
    def dimensions(self):
        return frozenset(d for d, values in ((2, self.areas), (1, self.segments),
                                             (0, self.points)) if values)


@dataclass(frozen=True)
class PlanarResult:
    status: str
    operation: str
    value: Optional[Union[PlanarApproximation, FeasibleSet]]
    budget: ErrorBudget
    error_ledger: Tuple[ErrorTerm, ...] = ()
    topology_events: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()
    provenance: Optional[Provenance] = None
    evidence_class: str = "nominal_geometry"
    budget_certified: bool = False

    def __post_init__(self):
        for name in ("error_ledger", "topology_events", "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


class _Failure(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _fail(status, message):
    raise _Failure(status, message)


def _finite(value, name, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("invalid_input", name + ": expected a finite number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0) or (nonnegative and value < 0):
        _fail("invalid_input", name + ": invalid numeric value")
    return Fraction(value)


def _scale(units):
    if units not in ("mm", "inch"):
        _fail("invalid_input", "units must be mm or inch")
    return Fraction(1) if units == "mm" else Fraction(127, 5)


def _point(point, name):
    if len(point) != 2:
        _fail("invalid_input", name + ": expected XY, no implicit Z projection")
    return tuple(_finite(v, name) for v in point)


def _frame(frame):
    if not isinstance(frame, PlanarFrame) or not isinstance(frame.frame_id, str) or not frame.frame_id:
        _fail("invalid_input", "explicit PlanarFrame identity required")
    scale = _scale(frame.units)
    origin = _point(frame.origin, "frame.origin")
    z = None if frame.section_z is None else _finite(frame.section_z, "section_z") * scale
    return scale, origin, (frame.frame_id, tuple(v * scale for v in origin), z)


def _budget(budget):
    if not isinstance(budget, ErrorBudget):
        _fail("invalid_input", "ErrorBudget required")
    _finite(budget.boundary_mm, "boundary_mm", positive=True)
    _finite(budget.area_mm2, "area_mm2", positive=True)
    if type(budget.max_segments) is not int or budget.max_segments < 4:
        _fail("invalid_input", "max_segments must be an integer >= 4")
    if type(budget.require_certified) is not bool:
        _fail("invalid_input", "require_certified must be boolean")


def _hash(value):
    def exact(item):
        if isinstance(item, (int, float, Fraction)) and not isinstance(item, bool):
            return Fraction(item)
        if isinstance(item, (tuple, list)):
            return tuple(exact(v) for v in item)
        if isinstance(item, PlanarFrame):
            return (item.units, item.frame_id, exact(item.origin), exact(item.section_z))
        return item
    return hashlib.sha256(repr(exact(value)).encode("utf-8")).hexdigest()


def _policy(budget):
    return _hash((budget.boundary_mm, budget.area_mm2, budget.max_segments,
                  budget.require_certified, "nominal-planar-v1"))


def _canonical_ring(ring):
    vertices = tuple(ring[:-1])
    if not vertices:
        return ()
    # Minimum coordinate is unique for a valid simple ring; no quadratic rotations.
    variants = []
    for seq in (vertices, vertices[::-1]):
        i = min(range(len(seq)), key=seq.__getitem__)
        variants.append(seq[i:] + seq[:i])
    return min(variants)


def _source_key(component):
    if isinstance(component, Polygon):
        return ("polygon", _canonical_ring(component.shell),
                tuple(sorted(_canonical_ring(h) for h in component.holes)))
    if isinstance(component, Rectangle):
        return ("rectangle", component.center, component.width, component.height,
                component.angle_radians)
    if isinstance(component, Circle):
        return ("circle", component.center, component.radius)
    _fail("unsupported", "only line-ring polygons and analytic rectangles/circles are supported")


def _primitive(component):
    if not isinstance(component.source, str):
        _fail("invalid_input", "source must be a string")
    if isinstance(component, (Circle, Rectangle)):
        _point(component.center, "primitive.center")
        if isinstance(component, Circle):
            _finite(component.radius, "circle.radius", positive=True)
        else:
            _finite(component.width, "rectangle.width", positive=True)
            _finite(component.height, "rectangle.height", positive=True)
            _finite(component.angle_radians, "rectangle.angle_radians")


def _float(value):
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail("unresolved", "coordinate conversion overflow")
    if not math.isfinite(result) or (value != 0 and result == 0):
        _fail("unresolved", "coordinate conversion loses a feature")
    return result


def _resolution(values, scale, budget):
    allowance = max((4 * math.ulp(float(v)) * float(scale) for v in values), default=0.0)
    if not math.isfinite(allowance) or allowance > budget.boundary_mm:
        _fail("unresolved", "coordinate resolution exhausts boundary budget")
    return allowance


def _chords(radius, boundary, area, limit):
    # Inscribed circle: stable sagitta and stable small-angle area deficit.
    count = 4
    while count <= limit:
        angle = 2 * math.pi / count
        delta = (angle ** 3 / 6 - angle ** 5 / 120 + angle ** 7 / 5040
                 if angle < 0.01 else angle - math.sin(angle))
        distance = 2 * radius * math.sin(angle / 4) ** 2
        deficit = count * radius * radius * delta / 2
        if distance <= boundary and deficit <= area:
            return count, distance, deficit
        count *= 2
    _fail("unresolved", "circle refinement exceeds max_segments")


def _ledger_check(ledger, budget):
    for attr, limit in (("boundary_mm", budget.boundary_mm), ("area_mm2", budget.area_mm2)):
        known = [getattr(term, attr) for term in ledger if getattr(term, attr) is not None]
        if any(not math.isfinite(v) or v < 0 for v in known) or sum(known) > limit:
            _fail("unresolved", "known error contributions exhaust " + attr)
    if budget.require_certified:
        _fail("unresolved", "certified bounds unavailable for this nominal slice")


def _result_failure(exc, operation, budget, ledger=(), provenance=None):
    return PlanarResult(exc.status, operation, None, budget, ledger,
                        diagnostics=(str(exc),), provenance=provenance)


def normalize(region, budget):
    """Validate detached source and tessellate single analytic circles nominally.

    Multiple components containing circles are unsupported: chord topology alone
    cannot establish source contact/disjointness. General arc predicates remain
    outside this first slice. Line rings and rigid rectangles compose freely.
    """
    from . import _planar_shapely as adapter

    ledger = ()
    provenance = None
    try:
        _budget(budget)
        if not isinstance(region, RegionSet):
            _fail("invalid_input", "RegionSet required")
        scale, origin, _ = _frame(region.frame)
        if any(not isinstance(c, (Polygon, Circle, Rectangle)) for c in region.components):
            _fail("unsupported", "general arcs/curves and native entities require a separate adapter")
        raw = []
        spans = []
        numeric = list(region.frame.origin)
        if region.frame.section_z is not None:
            numeric.append(region.frame.section_z)
        boundary_error = area_error = 0.0
        segment_count = 0

        def local(point):
            point = _point(point, "coordinate")
            numeric.extend(point)
            return tuple(_float((v - o) * scale) for v, o in zip(point, origin))

        for index, component in enumerate(region.components):
            _primitive(component)
            if isinstance(component, Polygon):
                rings = (component.shell,) + component.holes
                converted = []
                for ri, ring in enumerate(rings):
                    if len(ring) < 4 or ring[0] != ring[-1]:
                        _fail("invalid_input", "component %d ring %d: explicit closure required" % (index, ri))
                    for point in ring:
                        _point(point, "component %d ring %d coordinate" % (index, ri))
                    if len(set(ring[:-1])) < 3 or any(a == b for a, b in zip(ring, ring[1:])):
                        _fail("invalid_input", "component %d ring %d: degenerate source edge/ring" % (index, ri))
                    coords = tuple(local(p) for p in ring)
                    if len(set(coords)) != len(set(ring)):
                        _fail("unresolved", "component %d ring %d: conversion loses a feature" % (index, ri))
                    converted.append(coords)
                    spans.extend(SourceSpan(component.source, index, ri, j, 0.0, 1.0, coords[j], coords[j+1])
                                 for j in range(len(ring) - 1))
                raw.append((converted[0], tuple(converted[1:])))
                segment_count += sum(len(r) - 1 for r in converted)
            else:
                center = local(component.center)
                if isinstance(component, Rectangle):
                    numeric.extend((component.width, component.height))
                    w, h = (_float(Fraction(v) * scale / 2) for v in (component.width, component.height))
                    co, si = math.cos(component.angle_radians), math.sin(component.angle_radians)
                    coords = tuple((center[0] + co*x - si*y, center[1] + si*x + co*y)
                                   for x, y in ((-w, -h), (w, -h), (w, h), (-w, h)))
                    count = 4
                else:
                    if len(region.components) != 1:
                        _fail("unsupported", "analytic circle component contacts need source topology predicates")
                    numeric.append(component.radius)
                    radius = _float(Fraction(component.radius) * scale)
                    count, boundary_error, area_error = _chords(
                        radius, budget.boundary_mm / 2, budget.area_mm2 / 2, budget.max_segments)
                    coords = tuple((center[0] + radius * math.cos(2*math.pi*j/count),
                                    center[1] + radius * math.sin(2*math.pi*j/count)) for j in range(count))
                raw.append((coords + (coords[0],), ()))
                if len(set(coords)) != count:
                    _fail("unresolved", "analytic placement loses a feature")
                spans.extend(SourceSpan(component.source, index, 0,
                                        j if isinstance(component, Rectangle) else 0,
                                        0.0 if isinstance(component, Rectangle) else j/count,
                                        1.0 if isinstance(component, Rectangle) else (j+1)/count,
                                        coords[j], coords[(j+1) % count])
                             for j in range(count))
                segment_count += count
            if segment_count > budget.max_segments:
                _fail("unresolved", "input exceeds max_segments")
        allowance = _resolution(numeric, scale, budget)
        output_coords = [v for shell, holes in raw for ring in (shell,) + holes for p in ring for v in p]
        allowance += _resolution(output_coords, Fraction(1), budget)
        ledger = (ErrorTerm("input_approximation", boundary_error, area_error, "bound"),
                  ErrorTerm("coordinate_resolution", allowance, None, "allowance"),
                  ErrorTerm("numeric_computation", None, None, "unknown"),
                  ErrorTerm("output_conversion", None, None, "unknown"))
        _ledger_check(ledger, budget)
        components, contacts = adapter.validate(tuple(raw))
        fingerprint = _hash((region.frame, tuple(sorted(_source_key(c) for c in region.components)), _policy(budget)))
        provenance = Provenance((fingerprint,), _policy(budget), (), "planar-v1", adapter.versions())
        value = PlanarApproximation(region.frame, tuple(Polygon(s, h) for s, h in components),
                                    tuple(spans), fingerprint, ledger, provenance)
        return PlanarResult("ok", "normalize", value, budget, ledger, contacts, provenance=provenance)
    except (_Failure, adapter.AdapterError) as exc:
        return _result_failure(exc, "normalize", budget, ledger, provenance)


def _counts(components):
    return (len(components), sum(len(c.holes) for c in components))


def _operate(operation, left, budget, right=None, radius_mm=0.0):
    from . import _planar_shapely as adapter

    ledger = ()
    provenance = None
    try:
        _budget(budget)
        inputs = (left,) if right is None else (left, right)
        if any(not isinstance(v, PlanarApproximation) for v in inputs):
            _fail("invalid_input", "operations require normalized PlanarApproximation values")
        for value in inputs:
            if (any(not isinstance(c, Polygon) for c in value.components)
                    or any(not isinstance(t, ErrorTerm) for t in value.error_ledger)
                    or any(not isinstance(s, SourceSpan) for s in value.source_spans)
                    or not isinstance(value.provenance, Provenance)):
                _fail("invalid_input", "malformed normalized approximation")
        if any(_frame(v.frame)[2] != _frame(left.frame)[2] for v in inputs):
            _fail("invalid_input", "frame, numerical anchor or section mismatch; explicit mapping required")
        _finite(radius_mm, "radius_mm", nonnegative=True)
        ledger = tuple(term for v in inputs for term in v.error_ledger) + (
            ErrorTerm("operation_approximation", None, None, "unknown"),
            ErrorTerm("operation_numeric", None, None, "unknown"),
            ErrorTerm("output_conversion", None, None, "unknown"))
        _ledger_check(ledger, budget)
        raw = [tuple((c.shell, c.holes) for c in v.components) for v in inputs]
        for components in raw:
            if sum(len(r)-1 for s, hs in components for r in (s,) + hs) > budget.max_segments:
                _fail("unresolved", "operation input exceeds max_segments")
            adapter.validate(components)
        # This subdivision bounds only the kernel's circle approximation, not
        # offset error propagation near topology events (explicitly unknown).
        quad = 1
        if operation == "nominal_area_erosion" and radius_mm:
            count, _, _ = _chords(radius_mm, budget.boundary_mm / 2,
                                  budget.area_mm2 / 2, budget.max_segments)
            quad = count // 4
        parameters = (("radius_mm", repr(radius_mm)),) if operation == "nominal_area_erosion" else ()
        provenance = Provenance(tuple(v.fingerprint for v in inputs), _policy(budget),
                                parameters, "planar-v1", adapter.versions())
        components, contacts = adapter.operate(operation, raw[0], raw[1] if right is not None else (),
                                               radius=radius_mm, quad_segs=quad)
        if sum(len(r)-1 for s, hs in components for r in (s,) + hs) > budget.max_segments:
            _fail("unresolved", "operation output exceeds max_segments")
        converted = tuple(Polygon(s, h) for s, h in components)
        fingerprint = _hash((operation, provenance, components))
        value = PlanarApproximation(left.frame, converted,
                                    tuple(s for v in inputs for s in v.source_spans),
                                    fingerprint, ledger, provenance)
        event = "components/holes: %s -> %s" % (tuple(_counts(v.components) for v in inputs), _counts(converted))
        return PlanarResult("ok", operation, value, budget, ledger, (event,) + contacts,
                            provenance=provenance)
    except (_Failure, adapter.AdapterError) as exc:
        return _result_failure(exc, operation, budget, ledger, provenance)


def union(left, right, budget):
    return _operate("union", left, budget, right)


def difference(left, right, budget):
    return _operate("difference", left, budget, right)


def nominal_area_erosion(value, radius_mm, budget):
    """Regularized area only; emptiness is not proof of no feasible centers."""
    return _operate("nominal_area_erosion", value, budget, radius_mm=radius_mm)


def feasible_centers(primitive, frame, tool_radius, tool_units, budget):
    """Closed-set disk erosion of one analytic rectangle/disk, no backend.

    Classifies exact supplied numbers before conversion/placement. Uncertain
    sizes, general polygons and collapsed mixed strata are unsupported.
    """
    ledger = ()
    provenance = None
    try:
        _budget(budget)
        scale, origin, _ = _frame(frame)
        radius = _finite(tool_radius, "tool_radius", nonnegative=True) * _scale(tool_units)
        if not isinstance(primitive, (Rectangle, Circle)):
            _fail("unsupported", "complete feasible centers require an analytic rectangle or circle")
        _primitive(primitive)
        center = tuple((v - o) * scale for v, o in zip(_point(primitive.center, "center"), origin))
        angle = primitive.angle_radians if isinstance(primitive, Rectangle) else 0.0
        values = (*frame.origin, *primitive.center, primitive.width, primitive.height) if isinstance(primitive, Rectangle) else (*frame.origin, *primitive.center, primitive.radius)
        allowance = _resolution(values, scale, budget)
        ledger = (ErrorTerm("analytic_predicate", 0.0, 0.0, "bound"),
                  ErrorTerm("coordinate_resolution", allowance, None, "allowance"),
                  ErrorTerm("placed_output_conversion", None, None, "unknown"))
        _ledger_check(ledger, budget)
        areas, segments, points = (), (), ()
        zero = Fraction(0)
        if isinstance(primitive, Circle):
            remaining = Fraction(primitive.radius) * scale - radius
            if remaining > 0:
                areas = (("disk", (remaining,)),)
            elif remaining == 0:
                points = ((zero, zero),)
        else:
            w, h = (Fraction(v) * scale / 2 - radius for v in (primitive.width, primitive.height))
            if w >= 0 and h >= 0:
                if w > 0 and h > 0:
                    areas = (("rectangle", (w, h)),)
                elif w == h == 0:
                    points = ((zero, zero),)
                else:
                    segments = (((-w, -h), (w, h)),)
        provenance = Provenance((_hash((frame, _source_key(primitive))),), _policy(budget),
                                (("tool_radius", repr(tool_radius)), ("tool_units", tool_units)), "analytic-v1")
        value = FeasibleSet(frame, center, angle, areas, segments, points)
        return PlanarResult("ok", "feasible_centers", value, budget, ledger, provenance=provenance)
    except _Failure as exc:
        return _result_failure(exc, "feasible_centers", budget, ledger, provenance)
