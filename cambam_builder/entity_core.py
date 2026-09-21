"""Shared identity, geometry, and primitive foundations."""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import weakref
import uuid
import logging
import json
import math
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Optional, Set, Any, Sequence, Type, TypeVar, TYPE_CHECKING

import numpy as np

# Assuming cad_transformations provides these:
from .cad_transformations import (
    identity_matrix, apply_transform, get_transformed_point, to_cambam_matrix_str, from_cambam_matrix_str
)

# Type hint for the project class without circular import
if TYPE_CHECKING:
    from .cambam_project import CamBamProject

logger = logging.getLogger(__name__)

# Curved bounds are evaluated from their local parametric geometry and the
# complete affine matrix.  These tolerances are deliberately small: they only
# classify values which are numerically indistinguishable from the associated
# exact case, rather than changing ordinary CAD geometry.
ARC_SWEEP_TOLERANCE_DEGREES = 1e-9
ARC_ANGLE_TOLERANCE_RADIANS = 1e-12
PLINE_BULGE_TOLERANCE = 1e-12
CURVE_POINT_TOLERANCE = 1e-12


def _finite_float(value: Any, field_name: str) -> float:
    """Return ``value`` as a finite float, with a field-specific error."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


@dataclass(init=False)
class Vertex:
    """One intrinsic Pline/Points vertex.

    ``bulge`` is keyword-only so a three-value tuple always means XYZ. Tuple
    shorthand is normalized by Pline and Points; curved input therefore uses an
    explicit ``Vertex(..., bulge=value)`` record.
    """

    x: float
    y: float
    z: float = 0.0
    bulge: float = 0.0

    def __init__(self, x: float, y: float, z: float = 0.0, *, bulge: float = 0.0):
        self.x = _finite_float(x, "Vertex.x")
        self.y = _finite_float(y, "Vertex.y")
        self.z = _finite_float(z, "Vertex.z")
        self.bulge = _finite_float(bulge, "Vertex.bulge")


VertexInput = Union[Vertex, Tuple[float, float], Tuple[float, float, float]]


def _normalize_vertices(values: Sequence[VertexInput], *, allow_bulge: bool) -> List[Vertex]:
    """Copy and normalize public vertex input into the sole stored representation."""
    try:
        inputs = list(values)
    except TypeError as exc:
        raise ValueError("vertices must be a sequence of Vertex or coordinate tuples") from exc

    result: List[Vertex] = []
    for index, value in enumerate(inputs):
        if isinstance(value, Vertex):
            vertex = Vertex(value.x, value.y, value.z, bulge=value.bulge)
        elif isinstance(value, tuple) and len(value) in (2, 3):
            vertex = Vertex(*value)
        else:
            raise ValueError(
                f"vertices[{index}] must be Vertex, (x, y), or (x, y, z)"
            )
        if not allow_bulge and vertex.bulge != 0.0:
            raise ValueError(f"Points vertex {index} must have zero bulge")
        result.append(vertex)
    return result


def _validate_stored_vertices(values: Sequence[Vertex], *, allow_bulge: bool) -> List[Vertex]:
    """Validate mutable stored records without accepting tuple shorthand."""
    try:
        vertices = list(values)
    except TypeError as exc:
        raise ValueError("vertices must be a sequence of Vertex records") from exc
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Vertex):
            raise ValueError(f"vertices[{index}] must be a Vertex record")
        _finite_float(vertex.x, f"vertices[{index}].x")
        _finite_float(vertex.y, f"vertices[{index}].y")
        _finite_float(vertex.z, f"vertices[{index}].z")
        bulge = _finite_float(vertex.bulge, f"vertices[{index}].bulge")
        if not allow_bulge and bulge != 0.0:
            raise ValueError(f"Points vertex {index} must have zero bulge")
    return vertices


def _xy_similarity_scale(matrix: Any) -> Optional[float]:
    """Return a non-degenerate XY similarity scale, or ``None`` otherwise."""
    affine = _affine_matrix_or_none(matrix)
    if affine is None:
        return None
    linear = affine[0:2, 0:2]
    magnitude = float(np.max(np.abs(linear)))
    if magnitude == 0.0:
        return None
    normalized = linear / magnitude
    gram = normalized.T @ normalized
    scale_squared = float(np.trace(gram) / 2.0)
    if (not math.isfinite(scale_squared) or scale_squared <= 0.0
            or not np.allclose(
                gram, np.identity(2) * scale_squared,
                rtol=1e-12, atol=1e-12,
            )):
        return None
    scale = magnitude * math.sqrt(scale_squared)
    return scale if math.isfinite(scale) else None

# --- Helper Classes ---

@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box."""
    min_x: float = float('inf')
    min_y: float = float('inf')
    max_x: float = float('-inf')
    max_y: float = float('-inf')

    def is_valid(self) -> bool:
        return self.min_x <= self.max_x and self.min_y <= self.max_y

    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        if not other.is_valid():
            return self
        if not self.is_valid():
            return other
        return BoundingBox(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y)
        )

    @staticmethod
    def from_points(points: Sequence[Tuple[float, float]]) -> 'BoundingBox':
        if not points:
            return BoundingBox() # Invalid box
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return BoundingBox(min_x, min_y, max_x, max_y)


def _affine_matrix_or_none(matrix: Any) -> Optional[np.ndarray]:
    """Return a finite affine 3x3 matrix, or ``None`` for invalid input."""
    try:
        if np.iscomplexobj(matrix):
            return None
        result = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.shape != (3, 3) or not np.isfinite(result).all():
        return None
    # Perspective matrices are not part of the primitive transform contract.
    if not np.array_equal(result[2], (0.0, 0.0, 1.0)):
        return None
    return result


def _angle_is_on_sweep(angle: float, start: float, sweep: float) -> bool:
    """Test directed, potentially wrapping angular interval membership."""
    if sweep >= 0.0:
        distance = (angle - start) % (2.0 * math.pi)
        return distance <= sweep + ARC_ANGLE_TOLERANCE_RADIANS
    distance = (start - angle) % (2.0 * math.pi)
    return distance <= -sweep + ARC_ANGLE_TOLERANCE_RADIANS


def _local_arc_bounding_box(
    center: Tuple[float, float],
    radius: float,
    start: float,
    sweep: float,
    matrix: np.ndarray,
) -> BoundingBox:
    """Exact XY bounds of an affine image of a circular directed sweep."""
    if not (np.isfinite(center).all() and math.isfinite(radius)
            and math.isfinite(start) and math.isfinite(sweep)):
        return BoundingBox()
    affine = _affine_matrix_or_none(matrix)
    if affine is None:
        return BoundingBox()

    two_pi = 2.0 * math.pi
    full = abs(sweep) >= two_pi - math.radians(ARC_SWEEP_TOLERANCE_DEGREES)
    linear = affine[:2, :2]
    world_center = linear @ np.asarray(center, dtype=float) + affine[:2, 2]
    if not np.isfinite(world_center).all():
        return BoundingBox()

    if full:
        # Each world coordinate is u*cos(t) + v*sin(t), whose range is the
        # Euclidean norm of (u,v).  This remains exact under shear/reflection.
        amplitudes = abs(radius) * np.asarray([
            math.hypot(float(row[0]), float(row[1])) for row in linear
        ])
        values = (
            float(world_center[0] - amplitudes[0]),
            float(world_center[1] - amplitudes[1]),
            float(world_center[0] + amplitudes[0]),
            float(world_center[1] + amplitudes[1]),
        )
        return BoundingBox(*values) if all(math.isfinite(value) for value in values) else BoundingBox()

    angles = [start, start + sweep]
    # Derivative extrema for each transformed coordinate.  A zero row has no
    # angular extrema, and endpoint inclusion above still gives its range.
    for row in linear:
        u, v = radius * row[0], radius * row[1]
        if math.hypot(u, v) <= np.finfo(float).eps:
            continue
        critical = math.atan2(v, u)
        for candidate in (critical, critical + math.pi):
            if _angle_is_on_sweep(candidate, start, sweep):
                angles.append(candidate)

    points = []
    for angle in angles:
        local = np.asarray((radius * math.cos(angle), radius * math.sin(angle)))
        point = world_center + linear @ local
        if not np.isfinite(point).all():
            return BoundingBox()
        points.append((float(point[0]), float(point[1])))
    return BoundingBox.from_points(points)

# --- Base Entity Class ---

@dataclass
class CamBamEntity:
    """Base class for all CamBam entities."""
    internal_id: uuid.UUID = field(default_factory=uuid.uuid4, init=False) # Primary key, set on creation
    user_identifier: str = "" # User-friendly ID, must be unique within project

    def __post_init__(self):
        # Ensure user_identifier is set if not provided, using UUID initially
        if not self.user_identifier:
            self.user_identifier = str(self.internal_id)
@dataclass
class Primitive(CamBamEntity, ABC):
    """
    Abstract Base Class for geometric primitives.
    Holds intrinsic geometry and transformation state.
    Classification attributes (groups, description) are stored for the XML <Tag>.
    Layer assignment and parent/child links are managed externally by the project.
    """
    # Intrinsic Attributes
    effective_transform: np.ndarray = field(default_factory=identity_matrix)
    groups: List[str] = field(default_factory=list) # Classification
    description: str = ""                           # Classification
    output_decimals: Optional[int] = 9              # For XML serialization
    local_z_offset: float = 0.0                     # Local Z-only transform

    # Reference back to the project (transient, for context)
    _project_ref: Optional[weakref.ReferenceType] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.local_z_offset = _finite_float(self.local_z_offset, "local_z_offset")
        # Normalize accepted array-like input, but never replace malformed
        # geometry with an identity transform silently.
        valid_transform = _affine_matrix_or_none(self.effective_transform)
        if valid_transform is None:
            raise ValueError("effective_transform must be a finite affine 3x3 matrix")
        self.effective_transform = valid_transform.copy()
        # Ensure groups is a list
        if self.groups is None:
            self.groups = []

    # --- State Management (for pickling) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_project_ref', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._project_ref = None

    # --- Project Context ---
    def get_project(self) -> Optional["CamBamProject"]:
        """Returns the owning project, if linked."""
        if self._project_ref:
            project = self._project_ref()
            if project:
                return project
        # logger.warning(f"Primitive {self.user_identifier} ({self.internal_id}) is not linked to a project.")
        return None

    def set_project_link(self, project: "CamBamProject"):
        """Sets the weak reference to the owning project."""
        self._project_ref = weakref.ref(project)

    # --- Transformations ---
    def get_total_transform(self) -> np.ndarray:
        """Calculates the total transformation matrix by cascading from the root parent."""
        total_tf = self.effective_transform.copy()
        project = self.get_project()
        if project:
            parent_id = project.get_parent_of_primitive(self.internal_id)
            while parent_id:
                parent = project.get_primitive(parent_id)
                if parent:
                    total_tf = parent.effective_transform @ total_tf # Apply parent's transform first
                    parent_id = project.get_parent_of_primitive(parent.internal_id)
                else:
                    logger.warning(f"Parent primitive {parent_id} not found for {self.user_identifier} during transform calculation.")
                    break # Stop climbing if parent is missing
        return total_tf

    def get_total_z_offset(self) -> float:
        """Return this primitive's Z-only transform including all ancestors."""
        total_z = _finite_float(getattr(self, 'local_z_offset', 0.0), "local_z_offset")
        project = self.get_project()
        if not project:
            return total_z

        parent = project.get_parent_of_primitive(self.internal_id)
        visited: Set[uuid.UUID] = set()
        while parent is not None:
            if parent.internal_id in visited:
                raise ValueError("Primitive parent cycle detected while calculating Z offset")
            visited.add(parent.internal_id)
            total_z += _finite_float(
                getattr(parent, 'local_z_offset', 0.0), "parent local_z_offset"
            )
            parent = project.get_parent_of_primitive(parent.internal_id)
        if not math.isfinite(total_z):
            raise ValueError("Total Z offset must be finite")
        return total_z

    def get_total_transform_xyz(self) -> np.ndarray:
        """Return the supported world transform as a 4x4 XYZ affine matrix."""
        xy = _affine_matrix_or_none(self.get_total_transform())
        if xy is None:
            raise ValueError("Expected a finite affine 3x3 XY transform")
        result = np.identity(4, dtype=float)
        result[0:2, 0:2] = xy[0:2, 0:2]
        result[0, 3] = xy[0, 2]
        result[1, 3] = xy[1, 2]
        result[2, 3] = self.get_total_z_offset()
        return result

    # --- Geometry Calculations ---
    def get_absolute_coordinates(self) -> Any:
        """Calculates the primitive's geometry in absolute world coordinates."""
        total_tf = self.get_total_transform()
        return self._calculate_absolute_geometry(total_tf)

    def get_absolute_coordinates_xyz(self) -> Any:
        """Return the shape's world geometry with explicit Z coordinates."""
        transform = _affine_matrix_or_none(self.get_total_transform())
        if transform is None:
            raise ValueError("Expected a finite affine world transform")
        return self._calculate_absolute_geometry_xyz(
            transform, self.get_total_z_offset()
        )

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement XYZ geometry queries"
        )

    def get_bounding_box(self) -> BoundingBox:
        """Calculates the 2D bounding box in absolute world coordinates."""
        # Avoid recalculating absolute geometry if possible
        # This implementation recalculates each time. Caching could be added if needed.
        try:
            abs_coords = self.get_absolute_coordinates()
            return self._calculate_bounding_box(abs_coords)
        except Exception as e:
            logger.error(f"Error calculating bounding box for {self.user_identifier}: {e}")
            return BoundingBox() # Return invalid box on error

    @abstractmethod
    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Any:
        """Subclasses implement this to return absolute geometry based on total transform."""
        pass

    @abstractmethod
    def _calculate_bounding_box(self, absolute_geometry: Any) -> BoundingBox:
        """Subclasses implement this to calculate BB from their absolute geometry."""
        pass

    @abstractmethod
    def get_geometric_center(self) -> Tuple[float, float]:
        """Subclasses implement this to return the geometric center in absolute coordinates."""
        pass

    # --- XML Generation ---
    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """
        Generates the base XML element for the primitive.
        Subclasses will create the specific element type (e.g., 'pline', 'circle')
        and call this method to add common attributes and the matrix.
        """
        # This method now expects the parent UUID to be passed in by the writer
        # It also doesn't return the element directly, but rather configures a passed element.
        # Let's change the pattern: subclasses create their element, then call a helper
        # from this base class to add common stuff.

        # This method won't be called directly. See _add_common_xml_attributes
        raise NotImplementedError("Use specific to_xml_element in subclasses.")

    def _add_common_xml_attributes(self, element: ET.Element, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]):
        """Adds common attributes (ID, Tag, Matrix) to the primitive's XML element."""
        element.set("id", str(xml_primitive_id))

        # Create Tag data
        tag_data = {
            "user_id": self.user_identifier,
            "internal_id": str(self.internal_id),
            "groups": self.groups or [],
            "parent": str(parent_uuid) if parent_uuid else None,
            "description": self.description or ""
        }
        # Remove None values for cleaner JSON
        tag_data = {k: v for k, v in tag_data.items() if v is not None}
        ET.SubElement(element, "Tag").text = json.dumps(tag_data, separators=(',', ':')) # Compact JSON

        # Add Transformation Matrix
        # The matrix stored in XML is the *total* transformation relative to world origin
        total_tf = self.get_total_transform()
        mat_str = to_cambam_matrix_str(
            total_tf,
            output_decimals=self.output_decimals,
            z_offset=self.get_total_z_offset(),
        )
        ET.SubElement(element, "mat", {"m": mat_str})

    def shift_geometry_z(self, dz: float) -> None:
        """Shift intrinsic geometry in Z without changing its transform offset."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement intrinsic Z shifts"
        )

    @abstractmethod
    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix directly to the primitive's relative geometry.
        
        Args:
            transform_to_bake: The transformation matrix to bake into the geometry.
                            If None, the primitive's current effective_transform is used
                            and then reset to identity.
        
        When a specific matrix is provided, that transformation is baked into the actual
        geometry points without modifying the primitive's effective_transform.
        
        When no matrix is provided, the primitive's effective_transform is applied to
        its geometry and then reset to identity.
        """
        pass
