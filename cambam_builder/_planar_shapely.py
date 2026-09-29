"""Private, lazy GEOS boundary for detached nominal filled-area geometry.

Only owned immutable coordinate tuples cross this boundary. No precision grid,
repair, pruning, or feasibility inference is performed here.
"""

from functools import wraps
import math


class AdapterError(Exception):
    """A classified failure which the owned core can expose without GEOS types."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _boundary(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                "backend_failure", f"{function.__name__}: {type(exc).__name__}: {exc}"
            ) from exc

    return guarded


def _backend():
    try:
        import shapely
        from shapely.geometry import LinearRing, Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise AdapterError("unsupported", "The optional planar Shapely backend is unavailable") from exc
    return shapely, LinearRing, Polygon, unary_union


@_boundary
def versions():
    shapely, _, _, _ = _backend()
    return str(shapely.__version__), str(shapely.geos_version_string)


def _fail(status, location, reason):
    raise AdapterError(status, f"{location}: {reason}")


def _ring(value, location, status, LinearRing, Polygon):
    try:
        ring = tuple(tuple(float(v) for v in point) for point in value)
    except (TypeError, ValueError, OverflowError):
        _fail(status, location, "ring requires finite XY coordinate pairs")
    if any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in ring):
        _fail(status, location, "ring requires finite XY coordinate pairs")
    # IEEE signed zeros represent the same coordinate and must share a key.
    ring = tuple(tuple(0.0 if v == 0 else v for v in point) for point in ring)
    if len(ring) < 4 or ring[0] != ring[-1]:
        _fail(status, location, "ring requires explicit closure and at least three vertices")
    if len(set(ring[:-1])) < 3:
        _fail(status, location, "ring requires at least three distinct vertices")
    if any(a == b for a, b in zip(ring, ring[1:])):
        _fail(status, location, "zero-length ring edge")
    line = LinearRing(ring)
    polygon = Polygon(ring)
    if not line.is_simple or not polygon.is_valid:
        _fail(status, location, "self-crossing, overlapping or pinched ring")
    if not math.isfinite(polygon.area) or polygon.area == 0:
        _fail(status, location, "ring has zero or nonfinite area")
    return ring, line, polygon


def _canonical_ring(ring, is_ccw, wanted_ccw):
    points = ring[:-1]
    if is_ccw != wanted_ccw:
        points = tuple(reversed(points))
    start = min(range(len(points)), key=points.__getitem__)
    points = points[start:] + points[:start]
    return points + (points[0],)


def _contact_details(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Point":
        return (f"point {tuple(geometry.coords[0])!r}",)
    if geometry.geom_type in ("LineString", "LinearRing"):
        coords = tuple(tuple(point) for point in geometry.coords)
        return tuple(
            f"edge {tuple(sorted((a, b)))!r}" for a, b in zip(coords, coords[1:])
        )
    if hasattr(geometry, "geoms"):
        return tuple(detail for part in geometry.geoms for detail in _contact_details(part))
    raise AdapterError("backend_failure", "Unexpected dimension in a boundary contact")


def _validated(components, status, backend):
    _, LinearRing, Polygon, _ = backend
    owned = []
    for index, (shell, holes) in enumerate(components):
        location = f"component {index}"
        shell, shell_line, shell_polygon = _ring(
            shell, location + " shell", status, LinearRing, Polygon
        )
        canonical_holes = []
        hole_polygons = []
        raw_holes = []
        for hole_index, hole in enumerate(holes):
            hole_location = f"{location} hole {hole_index}"
            hole, hole_line, hole_polygon = _ring(hole, hole_location, status, LinearRing, Polygon)
            contact = shell_line.intersection(hole_line)
            if not contact.is_empty:
                _fail(status, hole_location, "shell/hole contact: " + "; ".join(_contact_details(contact)))
            if not shell_polygon.contains(hole_polygon):
                _fail(status, hole_location, "hole is not strictly inside its assigned shell")
            for previous_index, previous in enumerate(hole_polygons):
                if previous.intersects(hole_polygon):
                    contact = previous.boundary.intersection(hole_line)
                    details = "; ".join(_contact_details(contact))
                    _fail(status, hole_location, f"touching, nested or overlapping hole {previous_index}: {details}")
            hole_polygons.append(hole_polygon)
            raw_holes.append(hole)
            canonical_holes.append(_canonical_ring(hole, hole_line.is_ccw, False))
        polygon = Polygon(shell, raw_holes)
        if not polygon.is_valid or not math.isfinite(polygon.area):
            _fail(status, location, "invalid or nonfinite filled polygon")
        component = (
            _canonical_ring(shell, shell_line.is_ccw, True), tuple(sorted(canonical_holes))
        )
        owned.append((component, polygon))
    owned.sort(key=lambda item: item[0])
    contacts = []
    for index, (_, polygon) in enumerate(owned):
        for other_index in range(index):
            previous = owned[other_index][1]
            if polygon.relate_pattern(previous, "T********"):
                _fail(status, f"components {other_index}, {index}", "filled interiors overlap")
            contact = polygon.boundary.intersection(previous.boundary)
            contacts.extend(
                f"components {other_index}, {index}: {detail}"
                for detail in _contact_details(contact)
            )
    return tuple(item[0] for item in owned), tuple(sorted(set(contacts))), tuple(item[1] for item in owned)


@_boundary
def validate(components):
    """Admit strict filled components; preserve independent boundary contacts."""
    canonical, contacts, _ = _validated(components, "invalid_input", _backend())
    return canonical, contacts


def _result_components(geometry):
    if geometry.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
        raise AdapterError("backend_failure", "Unexpected non-area backend result")
    if not geometry.is_valid or not math.isfinite(geometry.area):
        raise AdapterError("backend_failure", "Invalid or nonfinite backend result")
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        rings = (geometry.exterior,) + tuple(geometry.interiors)
        if any(not all(math.isfinite(v) for v in point) for ring in rings for point in ring.coords):
            raise AdapterError("backend_failure", "Nonfinite backend coordinates")
        return ((tuple(geometry.exterior.coords), tuple(tuple(hole.coords) for hole in geometry.interiors)),)
    return tuple(component for part in geometry.geoms for component in _result_components(part))


@_boundary
def operate(operation, left, right=(), radius=0.0, quad_segs=16):
    """Run explicit regularized union/difference or nominal area erosion.

    Input component sets may share boundaries. Unary union implements the filled
    operand semantics here; it is never used to admit or repair an input.
    """
    backend = _backend()
    _, _, Polygon, unary_union = backend
    _, _, left_polygons = _validated(left, "invalid_input", backend)
    _, _, right_polygons = _validated(right, "invalid_input", backend)
    left_geometry = unary_union(left_polygons) if left_polygons else Polygon()
    right_geometry = unary_union(right_polygons) if right_polygons else Polygon()
    if operation == "union":
        result = left_geometry.union(right_geometry)
    elif operation == "difference":
        result = left_geometry.difference(right_geometry)
    elif operation == "nominal_area_erosion":
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
            raise AdapterError("invalid_input", "Erosion radius must be finite and nonnegative")
        if isinstance(quad_segs, bool) or not isinstance(quad_segs, int) or quad_segs < 1:
            raise AdapterError("invalid_input", "quad_segs must be a positive integer")
        # Zero erosion is identity, not an implicit buffer(0) repair.
        result = left_geometry if radius == 0 else left_geometry.buffer(-radius, quad_segs=quad_segs)
    else:
        raise AdapterError("unsupported", f"Unknown planar operation: {operation}")
    components = _result_components(result)
    canonical, contacts, _ = _validated(components, "unsupported", backend)
    return canonical, contacts
