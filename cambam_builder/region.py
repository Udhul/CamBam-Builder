"""CamBam Region primitive and typed-Region contour XML helpers.

A Region owns its contour Plines. Their canonical Vertex records are never
serialized as independently identifiable project primitives.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .cad_transformations import (
    apply_transform,
    from_cambam_matrix_str,
    identity_matrix,
    to_cambam_matrix_str,
)
from .cambam_entities import BoundingBox, PLINE_BULGE_TOLERANCE, Pline, Primitive, Vertex


_TOPOLOGY_REL_TOLERANCE = 1e-10
_Z_TOLERANCE = 1e-10
_TWO_PI = 2.0 * math.pi


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class _Segment:
    p0: Point2D
    p1: Point2D
    center: Optional[Point2D] = None
    radius: float = 0.0
    start: float = 0.0
    sweep: float = 0.0

    @property
    def is_arc(self) -> bool:
        return self.center is not None

    def point_at(self, parameter: float) -> Point2D:
        # Adjacent segments must use the same stored endpoints for the
        # half-open ray rule. Trig reconstruction can move Y across the ray.
        if parameter == 0.0:
            return self.p0
        if parameter == 1.0:
            return self.p1
        if not self.is_arc:
            return (
                self.p0[0] + (self.p1[0] - self.p0[0]) * parameter,
                self.p0[1] + (self.p1[1] - self.p0[1]) * parameter,
            )
        angle = self.start + self.sweep * parameter
        assert self.center is not None
        return (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle),
        )


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _finite_affine(matrix, name: str) -> np.ndarray:
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be a real affine 3x3 matrix")
    try:
        result = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite affine 3x3 matrix") from exc
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.array_equal(result[2], (0.0, 0.0, 1.0))
    ):
        raise ValueError(f"{name} must be a finite affine 3x3 matrix")
    linear = result[:2, :2]
    magnitude = float(np.max(np.abs(linear)))
    if magnitude == 0.0 or abs(float(np.linalg.det(linear / magnitude))) <= np.finfo(float).eps:
        raise ValueError(f"{name} must be nonsingular for Region topology")
    return result


def _is_similarity(matrix: np.ndarray) -> bool:
    linear = matrix[:2, :2]
    magnitude = float(np.max(np.abs(linear)))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        return False
    normalized = linear / magnitude
    gram = normalized.T @ normalized
    scale_squared = float((gram[0, 0] + gram[1, 1]) / 2.0)
    return scale_squared > 0.0 and np.allclose(
        gram,
        np.identity(2) * scale_squared,
        rtol=1e-10,
        atol=1e-12 * max(1.0, scale_squared),
    )


def _has_curves(contour: Pline) -> bool:
    return any(
        abs(_finite_float(vertex.bulge, "contour bulge")) > PLINE_BULGE_TOLERANCE
        for vertex in contour._validated_vertices()
    )


def _owned_contour(contour: Pline, name: str) -> Pline:
    if not isinstance(contour, Pline):
        raise TypeError(f"{name} must be a Pline")
    result = deepcopy(contour)
    if contour.get_project() is not None:
        # Capture a selected project Pline's complete pose before detaching.
        # Region transforms subsequently act on this copied geometry.
        result.effective_transform = contour.get_total_transform().copy()
        result.local_z_offset = contour.get_total_z_offset()
    # An owned contour is geometry within the Region, not a registry entity.
    result._project_ref = None
    return result


def _point_distance(first: Point2D, second: Point2D) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _subtract(first: Point2D, second: Point2D) -> Point2D:
    return first[0] - second[0], first[1] - second[1]


def _deduplicate(points: Sequence[Point2D], tolerance: float) -> List[Point2D]:
    result: List[Point2D] = []
    for point in points:
        if not any(_point_distance(point, existing) <= tolerance for existing in result):
            result.append(point)
    return result


def _arc_parameter(segment: _Segment, point: Point2D, tolerance: float) -> Optional[float]:
    assert segment.center is not None
    if _point_distance(point, segment.p0) <= tolerance:
        return 0.0
    if _point_distance(point, segment.p1) <= tolerance:
        return 1.0
    radial_error = abs(_point_distance(point, segment.center) - segment.radius)
    if radial_error > tolerance:
        return None
    angle = math.atan2(point[1] - segment.center[1], point[0] - segment.center[0])
    if segment.sweep > 0.0:
        distance = (angle - segment.start) % _TWO_PI
        parameter = distance / segment.sweep
    else:
        distance = (segment.start - angle) % _TWO_PI
        parameter = distance / -segment.sweep
    parameter_tolerance = tolerance / max(segment.radius * abs(segment.sweep), tolerance)
    if parameter <= 1.0 + parameter_tolerance:
        return min(1.0, max(0.0, parameter))
    return None


def _line_line(first: _Segment, second: _Segment, tolerance: float):
    r = _subtract(first.p1, first.p0)
    s = _subtract(second.p1, second.p0)
    qp = _subtract(second.p0, first.p0)
    denominator = _cross(r, s)
    cross_tolerance = tolerance * max(1.0, math.hypot(*r), math.hypot(*s))
    if abs(denominator) <= cross_tolerance:
        if abs(_cross(qp, r)) > cross_tolerance:
            return False, []
        axis = 0 if abs(r[0]) >= abs(r[1]) else 1
        first_interval = sorted((first.p0[axis], first.p1[axis]))
        second_interval = sorted((second.p0[axis], second.p1[axis]))
        low = max(first_interval[0], second_interval[0])
        high = min(first_interval[1], second_interval[1])
        if high < low - tolerance:
            return False, []
        if high > low + tolerance:
            return True, []
        candidates = [
            point
            for point in (first.p0, first.p1, second.p0, second.p1)
            if (
                min(first_interval) - tolerance <= point[axis] <= max(first_interval) + tolerance
                and min(second_interval) - tolerance <= point[axis] <= max(second_interval) + tolerance
            )
        ]
        return False, _deduplicate(candidates, tolerance)

    t = _cross(qp, s) / denominator
    u = _cross(qp, r) / denominator
    parameter_tolerance = tolerance / max(1.0, math.hypot(*r), math.hypot(*s))
    if (
        -parameter_tolerance <= t <= 1.0 + parameter_tolerance
        and -parameter_tolerance <= u <= 1.0 + parameter_tolerance
    ):
        return False, [(first.p0[0] + t * r[0], first.p0[1] + t * r[1])]
    return False, []


def _line_arc(line: _Segment, arc: _Segment, tolerance: float):
    assert arc.center is not None
    direction = _subtract(line.p1, line.p0)
    from_center = _subtract(line.p0, arc.center)
    length_squared = direction[0] ** 2 + direction[1] ** 2
    projection = -(
        from_center[0] * direction[0] + from_center[1] * direction[1]
    ) / length_squared
    closest = (
        from_center[0] + projection * direction[0],
        from_center[1] + projection * direction[1],
    )
    height_squared = arc.radius ** 2 - (closest[0] ** 2 + closest[1] ** 2)
    radial_tolerance = tolerance * max(1.0, arc.radius)
    if height_squared < -radial_tolerance:
        return False, []
    delta = math.sqrt(max(0.0, height_squared) / length_squared)
    parameter_tolerance = tolerance / max(1.0, math.sqrt(length_squared))
    points = []
    for parameter in _deduplicate_scalar((projection - delta, projection + delta), parameter_tolerance):
        if -parameter_tolerance <= parameter <= 1.0 + parameter_tolerance:
            point = (
                line.p0[0] + parameter * direction[0],
                line.p0[1] + parameter * direction[1],
            )
            if _arc_parameter(arc, point, tolerance) is not None:
                points.append(point)
    return False, _deduplicate(points, tolerance)


def _deduplicate_scalar(values: Sequence[float], tolerance: float) -> List[float]:
    result: List[float] = []
    for value in values:
        if not any(abs(value - existing) <= tolerance for existing in result):
            result.append(value)
    return result


def _arc_arc(first: _Segment, second: _Segment, tolerance: float):
    assert first.center is not None and second.center is not None
    center_delta = _subtract(second.center, first.center)
    distance = math.hypot(*center_delta)
    if distance <= tolerance and abs(first.radius - second.radius) <= tolerance:
        # Coincident circles are valid for complementary arcs (for example a
        # two-semicircle contour), but any shared arc interior is overlap.
        for endpoint in (first.p0, first.p1):
            parameter = _arc_parameter(second, endpoint, tolerance)
            if parameter is not None and tolerance < parameter < 1.0 - tolerance:
                return True, []
        for endpoint in (second.p0, second.p1):
            parameter = _arc_parameter(first, endpoint, tolerance)
            if parameter is not None and tolerance < parameter < 1.0 - tolerance:
                return True, []
        for midpoint, other in (
            (first.point_at(0.5), second),
            (second.point_at(0.5), first),
        ):
            parameter = _arc_parameter(other, midpoint, tolerance)
            if parameter is not None and tolerance < parameter < 1.0 - tolerance:
                return True, []
        common = [
            point
            for point in (first.p0, first.p1)
            if _arc_parameter(second, point, tolerance) is not None
        ]
        return False, _deduplicate(common, tolerance)

    if (
        distance > first.radius + second.radius + tolerance
        or distance < abs(first.radius - second.radius) - tolerance
        or distance <= tolerance
    ):
        return False, []
    along = (
        first.radius ** 2 - second.radius ** 2 + distance ** 2
    ) / (2.0 * distance)
    height_squared = first.radius ** 2 - along ** 2
    if height_squared < -tolerance * max(1.0, first.radius):
        return False, []
    height = math.sqrt(max(0.0, height_squared))
    unit = center_delta[0] / distance, center_delta[1] / distance
    base = (
        first.center[0] + along * unit[0],
        first.center[1] + along * unit[1],
    )
    perpendicular = -unit[1], unit[0]
    candidates = [
        (base[0] + height * perpendicular[0], base[1] + height * perpendicular[1]),
        (base[0] - height * perpendicular[0], base[1] - height * perpendicular[1]),
    ]
    points = [
        point
        for point in candidates
        if _arc_parameter(first, point, tolerance) is not None
        and _arc_parameter(second, point, tolerance) is not None
    ]
    return False, _deduplicate(points, tolerance)


def _intersections(first: _Segment, second: _Segment, tolerance: float):
    if not first.is_arc and not second.is_arc:
        return _line_line(first, second, tolerance)
    if not first.is_arc:
        return _line_arc(first, second, tolerance)
    if not second.is_arc:
        return _line_arc(second, first, tolerance)
    return _arc_arc(first, second, tolerance)


def _arc_from_bulge(p0: Point2D, p1: Point2D, bulge: float) -> _Segment:
    half_chord = ((p1[0] - p0[0]) / 2.0, (p1[1] - p0[1]) / 2.0)
    half_length = math.hypot(*half_chord)
    midpoint = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    left_normal = -half_chord[1] / half_length, half_chord[0] / half_length
    inverse_bulge = 1.0 / bulge
    offset = half_length * ((inverse_bulge - bulge) / 2.0)
    center = (
        midpoint[0] + left_normal[0] * offset,
        midpoint[1] + left_normal[1] * offset,
    )
    radius = half_length * ((abs(bulge) + abs(inverse_bulge)) / 2.0)
    start = math.atan2(p0[1] - center[1], p0[0] - center[0])
    return _Segment(p0, p1, center, radius, start, 4.0 * math.atan(bulge))


def _contour_segments(contour: Pline, name: str):
    if contour.closed is not True:
        raise ValueError(f"{name} must be closed")
    vertices = contour._validated_vertices()
    if len(vertices) < 2:
        raise ValueError(f"{name} must contain at least two vertices")
    matrix = _finite_affine(contour.effective_transform, f"{name} transform")
    curved = _has_curves(contour)
    if curved and not _is_similarity(matrix):
        raise ValueError(
            f"{name} has bulges under a non-similarity contour transform; "
            "elliptical Region topology is unsupported"
        )

    local_points: List[Point2D] = []
    bulges: List[float] = []
    for index, vertex in enumerate(vertices):
        try:
            x = _finite_float(vertex.x, f"{name} point {index} X")
            y = _finite_float(vertex.y, f"{name} point {index} Y")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} vertex {index} must contain finite coordinates"
            ) from exc
        local_points.append((x, y))
        bulges.append(_finite_float(vertex.bulge, f"{name} point {index} bulge"))
    transformed = apply_transform(local_points, matrix)
    if len(transformed) != len(local_points):
        raise ValueError(f"{name} transform did not produce finite affine points")
    points = [(float(point[0]), float(point[1])) for point in transformed]
    extent_scale = max(
        1.0,
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    )
    coordinate_scale = max(1.0, *(abs(value) for point in points for value in point))
    # Translation does not change geometric validity.  The ULP floor accounts
    # only for precision already lost by representing very large coordinates.
    tolerance = max(
        _TOPOLOGY_REL_TOLERANCE * extent_scale,
        float(np.spacing(coordinate_scale)) * 8.0,
        1e-12,
    )
    orientation = -1.0 if float(np.linalg.det(matrix[:2, :2])) < 0.0 else 1.0
    bulges = [value * orientation for value in bulges]

    segments: List[_Segment] = []
    for index, p0 in enumerate(points):
        p1 = points[(index + 1) % len(points)]
        if _point_distance(p0, p1) <= tolerance:
            raise ValueError(f"{name} contains a zero-length segment")
        bulge = bulges[index]
        segments.append(
            _arc_from_bulge(p0, p1, bulge)
            if abs(bulge) > PLINE_BULGE_TOLERANCE
            else _Segment(p0, p1)
        )

    _validate_simple(segments, tolerance, name)
    area = _signed_area(segments)
    if (
        not math.isfinite(area)
        or abs(area) <= _TOPOLOGY_REL_TOLERANCE * extent_scale * extent_scale
    ):
        raise ValueError(f"{name} must enclose a finite nonzero area")
    return segments, tolerance


def _validate_simple(segments: Sequence[_Segment], tolerance: float, name: str) -> None:
    count = len(segments)
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            adjacent = (
                second_index == first_index + 1
                or (first_index == 0 and second_index == count - 1)
            )
            overlap, points = _intersections(
                segments[first_index], segments[second_index], tolerance
            )
            if overlap:
                raise ValueError(f"{name} has overlapping segments")
            if adjacent:
                if count == 2:
                    expected = (segments[first_index].p0, segments[first_index].p1)
                else:
                    expected = ((
                        segments[first_index].p0
                        if first_index == 0 and second_index == count - 1
                        else segments[first_index].p1
                    ),)
                if any(
                    not any(_point_distance(point, candidate) <= tolerance for candidate in expected)
                    for point in points
                ):
                    raise ValueError(f"{name} self-intersects")
            elif points:
                raise ValueError(f"{name} self-intersects")


def _signed_area(segments: Sequence[_Segment]) -> float:
    origin = segments[0].p0
    twice_area = 0.0
    for segment in segments:
        p0 = _subtract(segment.p0, origin)
        p1 = _subtract(segment.p1, origin)
        if not segment.is_arc:
            twice_area += _cross(p0, p1)
            continue
        assert segment.center is not None
        center = _subtract(segment.center, origin)
        end = segment.start + segment.sweep
        twice_area += (
            segment.radius * center[0] * (math.sin(end) - math.sin(segment.start))
            - segment.radius * center[1] * (math.cos(end) - math.cos(segment.start))
            + segment.radius ** 2 * segment.sweep
        )
    return twice_area / 2.0


def _contours_intersect(
    first: Sequence[_Segment], second: Sequence[_Segment], tolerance: float
) -> bool:
    for first_segment in first:
        for second_segment in second:
            overlap, points = _intersections(first_segment, second_segment, tolerance)
            if overlap or points:
                return True
    return False


def _point_inside(point: Point2D, segments: Sequence[_Segment]) -> bool:
    """Return strict even-odd containment using exact line/circle crossings."""
    crossings = 0
    px, py = point
    for segment in segments:
        if not segment.is_arc:
            y0, y1 = segment.p0[1], segment.p1[1]
            if (y0 > py) != (y1 > py):
                parameter = (py - y0) / (y1 - y0)
                x = segment.p0[0] + parameter * (segment.p1[0] - segment.p0[0])
                if x > px:
                    crossings += 1
            continue

        # Split at Y extrema.  Each resulting arc is monotonic in Y, so the
        # same half-open rule used for line segments gives stable vertex counts.
        split_parameters = [0.0, 1.0]
        for critical in (math.pi / 2.0, 3.0 * math.pi / 2.0):
            if segment.sweep > 0.0:
                distance = (critical - segment.start) % _TWO_PI
                parameter = distance / segment.sweep
            else:
                distance = (segment.start - critical) % _TWO_PI
                parameter = distance / -segment.sweep
            if 0.0 < parameter < 1.0:
                split_parameters.append(parameter)
        split_parameters.sort()
        for lower, upper in zip(split_parameters, split_parameters[1:]):
            lower_point = segment.point_at(lower)
            upper_point = segment.point_at(upper)
            if (lower_point[1] > py) == (upper_point[1] > py):
                continue
            lo, hi = lower, upper
            if lower_point[1] == py:
                parameter = lower
            elif upper_point[1] == py:
                parameter = upper
            else:
                increasing = upper_point[1] > lower_point[1]
                for _ in range(64):
                    midpoint = (lo + hi) / 2.0
                    mid_y = segment.point_at(midpoint)[1]
                    if (mid_y < py) == increasing:
                        lo = midpoint
                    else:
                        hi = midpoint
                parameter = (lo + hi) / 2.0
            if segment.point_at(parameter)[0] > px:
                crossings += 1
    return bool(crossings % 2)


def _validate_region_topology(outer_curve: Pline, hole_curves: Sequence[Pline]) -> None:
    named = [("outer_curve", outer_curve)] + [
        (f"hole_curves[{index}]", contour)
        for index, contour in enumerate(hole_curves)
    ]
    built = [_contour_segments(contour, name) for name, contour in named]
    all_z = []
    for name, contour in named:
        local_offset = _finite_float(
            getattr(contour, "local_z_offset", 0.0), f"{name} local_z_offset"
        )
        all_z.extend(
            _finite_float(local_offset + value, f"{name} effective vertex Z")
            for value in (vertex.z for vertex in contour._validated_vertices())
        )
    reference_z = all_z[0]
    if any(not math.isclose(value, reference_z, rel_tol=0.0, abs_tol=_Z_TOLERANCE) for value in all_z[1:]):
        raise ValueError("Region contours must be coplanar in Z")

    outer_segments, outer_tolerance = built[0]
    holes = built[1:]
    for index, (hole_segments, hole_tolerance) in enumerate(holes):
        tolerance = max(outer_tolerance, hole_tolerance)
        if _contours_intersect(outer_segments, hole_segments, tolerance):
            raise ValueError(f"hole_curves[{index}] touches or crosses outer_curve")
        if not _point_inside(hole_segments[0].p0, outer_segments):
            raise ValueError(f"hole_curves[{index}] must be strictly inside outer_curve")

    for first_index in range(len(holes)):
        for second_index in range(first_index + 1, len(holes)):
            first, first_tolerance = holes[first_index]
            second, second_tolerance = holes[second_index]
            tolerance = max(first_tolerance, second_tolerance)
            if _contours_intersect(first, second, tolerance):
                raise ValueError(
                    f"hole_curves[{first_index}] and hole_curves[{second_index}] touch or cross"
                )
            if _point_inside(first[0].p0, second) or _point_inside(second[0].p0, first):
                raise ValueError("Region holes must not be nested")


def _rounded(value: float, decimals: Optional[int]):
    return round(value, decimals) if decimals is not None else value


def _append_contour_xml(parent: ET.Element, tag: str, contour: Pline, decimals: Optional[int]) -> ET.Element:
    element = ET.SubElement(parent, tag, {"Closed": "true"})
    ET.SubElement(element, "ModificationCount").text = "0"
    matrix_text = to_cambam_matrix_str(
        contour.effective_transform,
        output_decimals=decimals,
        z_offset=_finite_float(getattr(contour, "local_z_offset", 0.0), "contour local_z_offset"),
    )
    ET.SubElement(element, "mat", {"m": matrix_text})
    points_element = ET.SubElement(element, "pts")
    for vertex in contour._validated_vertices():
        x = _rounded(_finite_float(vertex.x, "contour X"), decimals)
        y = _rounded(_finite_float(vertex.y, "contour Y"), decimals)
        z = _rounded(_finite_float(vertex.z, "contour Z"), decimals)
        bulge = _rounded(
            _finite_float(vertex.bulge, "contour bulge"),
            decimals,
        )
        ET.SubElement(points_element, "p", {"b": str(bulge)}).text = f"{x},{y},{z}"
    return element


@dataclass
class Region(Primitive):
    """One project primitive containing a closed outer Pline and owned holes."""

    outer_curve: Pline = field(default_factory=Pline)
    hole_curves: List[Pline] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.outer_curve = _owned_contour(self.outer_curve, "outer_curve")
        if self.hole_curves is None:
            self.hole_curves = []
        try:
            self.hole_curves = [
                _owned_contour(contour, f"hole_curves[{index}]")
                for index, contour in enumerate(self.hole_curves)
            ]
        except TypeError as exc:
            raise TypeError("hole_curves must be a sequence of Plines") from exc
        _finite_affine(self.effective_transform, "Region transform")
        self.validate_geometry()

    @property
    def contours(self) -> Tuple[Pline, ...]:
        return (self.outer_curve, *self.hole_curves)

    def validate_geometry(self) -> None:
        _finite_affine(self.effective_transform, "Region transform")
        _validate_region_topology(self.outer_curve, self.hole_curves)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray):
        result = []
        for contour in self.contours:
            combined = total_transform @ contour.effective_transform
            vertices = contour._validated_vertices()
            xy = apply_transform(
                [(vertex.x, vertex.y) for vertex in vertices], combined
            )
            result.append([
                (
                    float(point[0]),
                    float(point[1]),
                    vertices[index].bulge,
                )
                for index, point in enumerate(xy)
            ])
        return {"outer_curve": result[0], "hole_curves": result[1:]}

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ):
        result = []
        for contour in self.contours:
            combined = total_transform @ contour.effective_transform
            if _has_curves(contour) and not _is_similarity(combined):
                raise ValueError("Bulged Region XYZ queries require an XY similarity transform")
            orientation = -1.0 if np.linalg.det(combined[:2, :2]) < 0.0 else 1.0
            vertices = contour._validated_vertices()
            xy = apply_transform(
                [(vertex.x, vertex.y) for vertex in vertices], combined
            )
            contour_z = total_z_offset + _finite_float(
                getattr(contour, "local_z_offset", 0.0), "contour local_z_offset"
            )
            result.append([
                (
                    float(point[0]),
                    float(point[1]),
                    _finite_float(vertices[index].z + contour_z, "world Region Z"),
                    vertices[index].bulge * orientation,
                )
                for index, point in enumerate(xy)
            ])
        return {"outer_curve": result[0], "hole_curves": result[1:]}

    def _calculate_bounding_box(self, absolute_geometry) -> BoundingBox:
        points = [
            (point[0], point[1])
            for contour in [
                absolute_geometry["outer_curve"],
                *absolute_geometry["hole_curves"],
            ]
            for point in contour
        ]
        return BoundingBox.from_points(points)

    def get_bounding_box(self) -> BoundingBox:
        bounds = BoundingBox()
        total_transform = self.get_total_transform()
        for contour in self.contours:
            posed = deepcopy(contour)
            posed._project_ref = None
            posed.effective_transform = total_transform @ contour.effective_transform
            bounds = bounds.union(posed.get_bounding_box())
        return bounds

    def get_geometric_center(self) -> Point2D:
        bounds = self.get_bounding_box()
        if not bounds.is_valid():
            return 0.0, 0.0
        return (bounds.min_x + bounds.max_x) / 2.0, (bounds.min_y + bounds.max_y) / 2.0

    def shift_geometry_z(self, dz: float) -> None:
        shift = _finite_float(dz, "dz")
        shifted = [
            [Vertex(vertex.x, vertex.y,
                    _finite_float(vertex.z + shift, "shifted Region vertex Z"),
                    bulge=vertex.bulge)
             for vertex in contour._validated_vertices()]
            for contour in self.contours
        ]
        for contour, vertices in zip(self.contours, shifted):
            contour.vertices = vertices

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """Bake XY geometry and owned contour poses; retain the Region Z offset.

        Use the project full-bake API to bake the Region's local Z offset and
        compensate descendants. This matches the other Primitive XY bake methods.
        """
        transform = self.effective_transform if transform_to_bake is None else transform_to_bake
        transform = _finite_affine(transform, "Region bake transform")
        reset_transform = transform_to_bake is None
        staged = [_owned_contour(contour, f"contour {index}") for index, contour in enumerate(self.contours)]
        for index, contour in enumerate(staged):
            combined = transform @ _finite_affine(
                contour.effective_transform, f"contour {index} transform"
            )
            if _has_curves(contour) and not _is_similarity(combined):
                raise ValueError(
                    "Cannot bake a non-similarity transform into a bulged Region contour"
                )
            vertices = contour._validated_vertices()
            xy = apply_transform([(vertex.x, vertex.y) for vertex in vertices], combined)
            reflected = float(np.linalg.det(combined[:2, :2])) < 0.0
            contour_shift = _finite_float(
                getattr(contour, "local_z_offset", 0.0),
                f"contour {index} local_z_offset",
            )
            new_vertices = []
            for point_index, point in enumerate(xy):
                original = vertices[point_index]
                new_vertices.append(Vertex(
                    float(point[0]), float(point[1]),
                    _finite_float(original.z + contour_shift, "baked Region vertex Z"),
                    bulge=-original.bulge if reflected else original.bulge,
                ))
            contour.vertices = new_vertices
            contour.effective_transform = identity_matrix()
            contour.local_z_offset = 0.0

        _validate_region_topology(staged[0], staged[1:])
        self.outer_curve = staged[0]
        self.hole_curves = staged[1:]
        if reset_transform:
            self.effective_transform = identity_matrix()

    def to_xml_element(
        self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]
    ) -> ET.Element:
        self.validate_geometry()
        _finite_affine(self.get_total_transform(), "Region world transform")
        element = ET.Element("entity", {"xsi:type": "Region"})
        ET.SubElement(element, "ModificationCount").text = "0"
        self._add_common_xml_attributes(element, xml_primitive_id, parent_uuid)
        _append_contour_xml(element, "OuterCurve", self.outer_curve, self.output_decimals)
        holes = ET.SubElement(element, "HoleCurves")
        for contour in self.hole_curves:
            _append_contour_xml(holes, "Polyline", contour, self.output_decimals)
        # Precision is part of the interchange contract: reject a rounded
        # contour that closes a gap or otherwise changes valid topology.
        try:
            rounded = parse_region_geometry(element)
            _validate_region_topology(rounded["outer_curve"], rounded["hole_curves"])
            rounded_world, _ = from_cambam_matrix_str(element.find("mat").get("m"), return_z=True)
            _finite_affine(rounded_world, "Rounded Region world transform")
        except ValueError as exc:
            raise ValueError(
                f"Region XML precision {self.output_decimals} cannot preserve valid geometry: {exc}"
            ) from exc
        return element


def _children_named(element: ET.Element, local_name: str) -> List[ET.Element]:
    return [child for child in element if child.tag.rsplit("}", 1)[-1] == local_name]


def _reject_unknown_children(element: ET.Element, allowed: Sequence[str], name: str) -> None:
    unknown = [
        child.tag.rsplit("}", 1)[-1]
        for child in element
        if child.tag.rsplit("}", 1)[-1] not in allowed
    ]
    if unknown:
        raise ValueError(f"{name} contains unsupported child element {unknown[0]!r}")


def _parse_contour(element: ET.Element, name: str) -> Pline:
    if element.get("Closed", "").strip().lower() != "true":
        raise ValueError(f"{name} must declare Closed='true'")
    _reject_unknown_children(element, ("ModificationCount", "mat", "pts"), name)
    point_containers = _children_named(element, "pts")
    if len(point_containers) != 1:
        raise ValueError(f"{name} must contain exactly one pts element")
    _reject_unknown_children(point_containers[0], ("p",), f"{name}/pts")
    vertices = []
    for index, point_element in enumerate(_children_named(point_containers[0], "p")):
        fields = (point_element.text or "").split(",")
        if len(fields) != 3:
            raise ValueError(f"{name} point {index} must contain x,y,z")
        x, y, z = [
            _finite_float(value.strip(), f"{name} point {index}") for value in fields
        ]
        bulge = _finite_float(point_element.get("b", "0"), f"{name} point {index} bulge")
        vertices.append(Vertex(x, y, z, bulge=bulge))

    matrices = _children_named(element, "mat")
    if len(matrices) > 1:
        raise ValueError(f"{name} must contain at most one mat element")
    matrix_text = matrices[0].get("m", "Identity") if matrices else "Identity"
    matrix, z_offset = from_cambam_matrix_str(matrix_text, return_z=True)
    return Pline(
        vertices=vertices,
        closed=True,
        effective_transform=matrix,
        local_z_offset=z_offset,
    )


def parse_region_geometry(element: ET.Element):
    """Parse a typed CamBam Region entity into Region constructor geometry kwargs."""
    type_values = [
        value for key, value in element.attrib.items() if key.rsplit("}", 1)[-1].split(":")[-1] == "type"
    ]
    if type_values and type_values[0].strip().lower() != "region":
        raise ValueError("Expected an entity with xsi:type='Region'")
    outer_elements = _children_named(element, "OuterCurve")
    if len(outer_elements) != 1:
        raise ValueError("Region must contain exactly one OuterCurve")
    hole_containers = _children_named(element, "HoleCurves")
    if len(hole_containers) > 1:
        raise ValueError("Region must contain at most one HoleCurves element")
    if hole_containers:
        _reject_unknown_children(hole_containers[0], ("Polyline",), "HoleCurves")
    holes = (
        _children_named(hole_containers[0], "Polyline")
        if hole_containers
        else []
    )
    return {
        "outer_curve": _parse_contour(outer_elements[0], "OuterCurve"),
        "hole_curves": [
            _parse_contour(contour, f"HoleCurves/Polyline[{index}]")
            for index, contour in enumerate(holes)
        ],
    }
