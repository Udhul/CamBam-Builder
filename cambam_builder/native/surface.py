"""Native indexed triangle meshes, without implicit relief or solid semantics."""

from dataclasses import dataclass, field
import math
import re
from typing import TypedDict
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .core import (
    BoundingBox, Primitive, Vertex, _affine_matrix_or_none, _finite_float,
    _normalize_vertices, _validate_stored_vertices,
)
from .transformations import identity_matrix


class SurfaceGeometry(TypedDict):
    """World XYZ vertices and oriented zero-based triangle indices."""

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


class SurfaceDefinition(TypedDict):
    """Constructor fields decoded from native Surface XML."""

    vertices: list[Vertex]
    faces: list[tuple[int, int, int]]
    display_edges: bool


def _mesh(
    vertices: list[Vertex], faces: list[tuple[int, int, int]],
) -> tuple[list[Vertex], list[tuple[int, int, int]]]:
    validated = _validate_stored_vertices(vertices, allow_bulge=False)
    points = [Vertex(v.x, v.y, v.z) for v in validated]
    if len(points) < 3:
        raise ValueError("Surface requires at least three vertices")
    try:
        triangles = list(faces)
    except TypeError as exc:
        raise ValueError("Surface faces must be triangle index triples") from exc
    if not triangles:
        raise ValueError("Surface requires at least one triangle face")
    normalized = []
    for face in triangles:
        if not isinstance(face, (list, tuple)) or len(face) != 3:
            raise ValueError("Surface faces must be triangle index triples")
        if any(isinstance(i, bool) or not isinstance(i, int) for i in face):
            raise ValueError("Surface face indices must be integers")
        if len(set(face)) != 3 or any(i < 0 or i >= len(points) or i > 2**31 - 1 for i in face):
            raise ValueError("Surface face indices must be distinct and in range")
        a, b, c = (points[i] for i in face)
        u = (b.x - a.x, b.y - a.y, b.z - a.z)
        v = (c.x - a.x, c.y - a.y, c.z - a.z)
        scale = max(abs(value) for value in (*u, *v))
        if not math.isfinite(scale) or scale == 0:
            raise ValueError("Surface triangle has degenerate or overflowing geometry")
        # Scaling avoids overflow/underflow in the cross product for ordinary
        # very large/small meshes without an arbitrary unit-dependent cutoff.
        u, v = tuple(value / scale for value in u), tuple(value / scale for value in v)
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0])
        if not any(cross):
            raise ValueError("Surface triangle must have nonzero area")
        normalized.append(tuple(face))
    return points, normalized


@dataclass
class Surface(Primitive):
    """Editable native indexed triangle mesh in document units.

    Vertices accept XY/XYZ tuples or zero-bulge Vertex records at construction;
    stored records are mutable Vertex values. Faces are oriented, zero-based
    triples. Queries and baking support XY affine transforms and Z translation;
    reflection reverses triangle winding as native CamBam does. No manifold,
    closed-solid, upward-facing or non-overhanging interpretation is implied.

    Invalid/nonfinite points, invalid indices and degenerate triangles raise
    ValueError on construction/use/export. DisplayEdges is native display state;
    CamBam's volatile ModificationCount is always emitted as zero.
    """

    vertices: list[Vertex] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    display_edges: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vertices = _normalize_vertices(self.vertices, allow_bulge=False)
        self.vertices, self.faces = self._validated_mesh()

    def _validated_mesh(self) -> tuple[list[Vertex], list[tuple[int, int, int]]]:
        if not isinstance(self.display_edges, bool):
            raise ValueError("Surface display_edges must be a bool")
        return _mesh(self.vertices, self.faces)

    def _transformed_mesh(
        self, transform: np.ndarray, z_offset: float = 0.0,
    ) -> tuple[list[Vertex], list[tuple[int, int, int]]]:
        matrix = _affine_matrix_or_none(transform)
        if matrix is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        points, faces = self._validated_mesh()
        result = []
        offset = _finite_float(z_offset, "z_offset")
        for point in points:
            xy = matrix @ np.array((point.x, point.y, 1.0))
            result.append(Vertex(xy[0], xy[1], point.z + offset))
        if np.linalg.slogdet(matrix[:2, :2])[0] < 0:
            faces = [(a, c, b) for a, b, c in faces]
        return _mesh(result, faces)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> SurfaceGeometry:
        return self._calculate_absolute_geometry_xyz(total_transform, self.get_total_z_offset())

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float,
    ) -> SurfaceGeometry:
        points, faces = self._transformed_mesh(total_transform, total_z_offset)
        return {"vertices": [(v.x, v.y, v.z) for v in points], "faces": faces}

    def _calculate_bounding_box(self, absolute_geometry: SurfaceGeometry) -> BoundingBox:
        return BoundingBox.from_points([(v[0], v[1]) for v in absolute_geometry["vertices"]])

    def get_geometric_center(self) -> tuple[float, float]:
        bounds = self.get_bounding_box()
        if not bounds.is_valid():
            raise ValueError("Cannot determine the center of invalid Surface geometry")
        return ((bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2)

    def shift_geometry_z(self, dz: float) -> None:
        points, faces = self._transformed_mesh(identity_matrix(), _finite_float(dz, "dz"))
        self.vertices, self.faces = points, faces

    def bake_geometry(self, transform_to_bake: np.ndarray | None = None) -> None:
        """Bake XY placement atomically, retaining intrinsic Z and local Z offset."""
        points, faces = self._transformed_mesh(
            self.effective_transform if transform_to_bake is None else transform_to_bake)
        self.vertices, self.faces = points, faces
        if transform_to_bake is None:
            self.effective_transform = identity_matrix()

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: uuid.UUID | None) -> ET.Element:
        points, faces = self._validated_mesh()
        if self.output_decimals is not None:
            points = [Vertex(*(round(value, self.output_decimals) for value in (v.x, v.y, v.z)))
                      for v in points]
        # Export must not turn a valid small triangle into a degenerate one.
        points, faces = _mesh(points, faces)
        self._transformed_mesh(self.get_total_transform(), self.get_total_z_offset())
        element = ET.Element("surface")
        ET.SubElement(element, "ModificationCount").text = "0"
        self._add_common_xml_attributes(element, xml_primitive_id, parent_uuid)
        vertices_element = ET.SubElement(element, "verts")
        for point in points:
            ET.SubElement(vertices_element, "v").text = ",".join(str(v) for v in (point.x, point.y, point.z))
        faces_element = ET.SubElement(element, "faces")
        for face in faces:
            ET.SubElement(faces_element, "f").text = ",".join(str(i) for i in face)
        if self.display_edges:
            ET.SubElement(element, "DisplayEdges").text = "true"
        return element


def surface_geometry_from_xml(element: ET.Element) -> SurfaceDefinition:
    """Decode the supported native arrays/state, rejecting lossy encodings.

    The reader owns id, Tag and mat parsing. ModificationCount is validated as
    native Int64 metadata but deliberately not exposed or preserved as state.
    """
    allowed = {"ModificationCount", "Tag", "mat", "verts", "faces", "DisplayEdges"}
    if element.tag != "surface" or set(element.attrib) - {"id"}:
        raise ValueError("Unsupported Surface element or attributes")
    if element.text and element.text.strip():
        raise ValueError("Unsupported Surface text")
    counts: dict[str, int] = {}
    for child in element:
        if child.tag not in allowed or (child.tail and child.tail.strip()):
            raise ValueError("Unsupported Surface child or text")
        counts[child.tag] = counts.get(child.tag, 0) + 1
        if counts[child.tag] > 1:
            raise ValueError(f"Duplicate Surface {child.tag}")
        if child.tag not in {"Tag", "mat"} and child.attrib:
            raise ValueError(f"Unsupported Surface {child.tag} attributes")
    if counts.get("verts") != 1 or counts.get("faces") != 1:
        raise ValueError("Surface requires verts and faces arrays")
    vertices = []
    faces = []
    for name, item_name in (("verts", "v"), ("faces", "f")):
        container = element.find(name)
        assert container is not None
        if container.text and container.text.strip():
            raise ValueError(f"Unsupported Surface {name} text")
        for item in container:
            if (item.tag != item_name or item.attrib or len(item) or
                    (item.tail and item.tail.strip())):
                raise ValueError(f"Unsupported Surface {name} encoding")
            values = (item.text or "").split(",")
            if len(values) != 3:
                raise ValueError(f"Surface {item_name} requires three values")
            if name == "verts":
                vertices.append(Vertex(*(_finite_float(v, "Surface coordinate") for v in values)))
            else:
                if any(re.fullmatch(r"[+-]?\d+", v.strip()) is None for v in values):
                    raise ValueError("Surface face indices must be integers")
                faces.append(tuple(int(v) for v in values))
    state = element.find("DisplayEdges")
    display_edges = False
    if state is not None:
        value = (state.text or "").strip()
        if len(state) or value not in {"true", "false", "1", "0"}:
            raise ValueError("Invalid Surface DisplayEdges")
        display_edges = value in {"true", "1"}
    counter = element.find("ModificationCount")
    if counter is not None:
        value = (counter.text or "").strip()
        if len(counter) or re.fullmatch(r"[+-]?\d+", value) is None or not -(2**63) <= int(value) < 2**63:
            raise ValueError("Invalid Surface ModificationCount")
    vertices, faces = _mesh(vertices, faces)
    return {"vertices": vertices, "faces": faces, "display_edges": display_edges}
