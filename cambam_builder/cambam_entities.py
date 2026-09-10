"""
cambam_entities.py

Defines the core CamBam entity classes (Layer, Part, Primitive, Mop).
Entities primarily hold their intrinsic attributes (geometry, parameters, visual style).
Relationships between entities (layer assignment, parent/child links, MOP assignments)
are managed centrally by the CamBamProject class.
"""

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
    from cambam_project import CamBamProject

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

# --- Layer Entity ---

@dataclass
class Layer(CamBamEntity):
    """Represents a drawing layer with visual properties."""
    color: str = 'Green'
    alpha: float = 1.0
    pen_width: float = 1.0
    visible: bool = True
    locked: bool = False
    # Note: No primitive_ids or _xml_objects_element here. Managed by Project.

    def to_xml_element(self) -> ET.Element:
        """Creates the <layer> XML element (without the <objects> container)."""
        layer_elem = ET.Element("layer", {
            "name": self.user_identifier, # Use user_identifier as CamBam's layer name
            "color": self.color,
            "alpha": str(self.alpha),
            "pen": str(self.pen_width),
            "visible": str(self.visible).lower(),
            "locked": str(self.locked).lower()
        })
        # The <objects> sub-element will be added by the writer based on project registry
        return layer_elem

# --- Part Entity ---

@dataclass
class Part(CamBamEntity):
    """Represents a machining part with stock and default parameters."""
    enabled: bool = True
    stock_thickness: float = 12.5
    stock_width: float = 1220.0
    stock_height: float = 2440.0
    stock_material: str = "MDF"
    stock_color: str = "210,180,140" # RGB string
    machining_origin: Tuple[float, float] = (0.0, 0.0) # XY offset
    default_tool_diameter: Optional[float] = None
    default_spindle_speed: Optional[int] = None
    # Note: No mop_ids or _xml_machineops_element here. Managed by Project.

    def to_xml_element(self) -> ET.Element:
        """Creates the <part> XML element (without the <machineops> container)."""
        part_elem = ET.Element("part", {
            "Name": self.user_identifier, # Use user_identifier as CamBam's part name
            "Enabled": str(self.enabled).lower()
        })
        # The <machineops> sub-element will be added by the writer based on project registry

        # Add stock and other part-level settings directly here
        stock = ET.SubElement(part_elem, "Stock")
        # CamBam stock is defined by PMin(x,y,z) and PMax(x,y,z)
        # We define the stock offset, PMin, as (0,0,-thickness) since it will be aligned to the machine origin, and thickness is negative Z (material surface at Z=0).
        # Then we use the machine origin to offset the stock and machine origin out on the canvas, where primitives pertaining to this stock will be drawn.
        # PMax is the stock width, height, surface (Z=0)
        ET.SubElement(stock, "PMin").text = f"0,0,{-self.stock_thickness}"
        ET.SubElement(stock, "PMax").text = f"{self.stock_width},{self.stock_height},0"
        ET.SubElement(stock, "Material").text = self.stock_material
        ET.SubElement(stock, "Color").text = self.stock_color

        ET.SubElement(part_elem, "MachiningOrigin").text = f"{self.machining_origin[0]},{self.machining_origin[1]}"

        if self.default_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(self.default_tool_diameter)
        if self.default_spindle_speed is not None:
            # CamBam doesn't seem to have a direct default spindle speed at part level in XML?
            # MOPs inherit if not set, but storing it here is useful for the framework.
            # We won't write it to XML unless a specific field is found later.
            pass

        # Add other common Part elements expected by CamBam
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill" # Default, can be overridden by MOPs
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "BasePoint").text = "0,0"
        ET.SubElement(nesting, "NestMethod").text = "None"

        return part_elem

# --- Primitive Base Class and Concrete Classes ---

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
        # Don't pickle weak reference
        if '_project_ref' in state:
            del state['_project_ref']
        # Convert numpy array to list for potentially better pickle compatibility?
        # Or keep as array if pickle handles it reliably. Let's keep it for now.
        # state['effective_transform'] = self.effective_transform.tolist()
        return state

    def __setstate__(self, state):
        # Restore numpy array if it was converted to list
        # if isinstance(state.get('effective_transform'), list):
        #     state['effective_transform'] = np.array(state['effective_transform'])
        fields = getattr(type(self), "__dataclass_fields__", {})
        state.setdefault('local_z_offset', 0.0)
        if 'elevation' in fields:
            state.setdefault('elevation', 0.0)
        if 'xml_p2_elevation' in fields:
            state.setdefault('xml_p2_elevation', None)
        if 'xml_p2_position' in fields:
            state.setdefault('xml_p2_position', None)
        self.__dict__.update(state)
        # Re-initialize transient fields
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

# --- Concrete Primitive Classes ---

@dataclass
class Pline(Primitive):
    # Intrinsic geometry
    vertices: List[Vertex] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.vertices = _normalize_vertices(self.vertices, allow_bulge=True)
        self._validated_vertices()

    def _validated_vertices(self) -> List[Vertex]:
        vertices = _validate_stored_vertices(self.vertices, allow_bulge=True)
        segment_count = max(0, len(vertices) - 1)
        if self.closed and vertices:
            segment_count += 1
        for index in range(segment_count):
            vertex = vertices[index]
            bulge = _finite_float(vertex.bulge, f"vertices[{index}].bulge")
            next_index = (index + 1) % len(vertices)
            if (abs(bulge) > PLINE_BULGE_TOLERANCE
                    and not math.isclose(
                        vertex.z, vertices[next_index].z,
                        rel_tol=0.0, abs_tol=CURVE_POINT_TOLERANCE,
                    )):
                raise ValueError(
                    "Bulged Pline segments require equal endpoint Z values "
                    f"(segment {index} to {next_index})"
                )
        return vertices

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float, float]]:
        # Extract XY for transformation
        vertices = self._validated_vertices()
        rel_pts_xy = [(vertex.x, vertex.y) for vertex in vertices]
        abs_pts_xy = apply_transform(rel_pts_xy, total_transform)

        # Re-attach bulge values (bulge is not transformed by standard matrix)
        abs_pts_with_bulge = []
        for i, (x, y) in enumerate(abs_pts_xy):
            # Handle potential index out of bounds if apply_transform skipped points
            if i < len(vertices):
                abs_pts_with_bulge.append((x, y, vertices[i].bulge))
            else:
                logger.warning(f"Point mismatch after transformation for Pline {self.user_identifier}. Skipping bulge.")

        return abs_pts_with_bulge

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float, float]]:
        vertices = self._validated_vertices()
        absolute_xy_bulge = self._calculate_absolute_geometry(total_transform)
        if (any(abs(p[2]) > PLINE_BULGE_TOLERANCE for p in absolute_xy_bulge)
                and _xy_similarity_scale(total_transform) is None):
            raise ValueError("Bulged Pline XYZ queries require an XY similarity transform")
        orientation = -1.0 if np.linalg.det(total_transform[:2, :2]) < 0.0 else 1.0
        return [
            (x, y, _finite_float(vertices[index].z + total_z_offset, "world Z"),
             bulge * orientation)
            for index, (x, y, bulge) in enumerate(absolute_xy_bulge)
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        vertices = self._validated_vertices()
        self.vertices = [
            Vertex(vertex.x, vertex.y, _finite_float(vertex.z + delta, "shifted vertex Z"),
                   bulge=vertex.bulge)
            for vertex in vertices
        ]

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float, float]]) -> BoundingBox:
        # Bounding box ignores bulge, uses only XY coordinates
        points_xy = [(p[0], p[1]) for p in absolute_geometry]
        if not points_xy:
            return BoundingBox()
        # TODO: Bulges affect the bounding box! This is an approximation.
        # A proper calculation would need to find the extrema of the arcs.
        if any(abs(p[2]) > 1e-6 for p in absolute_geometry):
            logger.debug(f"Bounding box for Pline {self.user_identifier} with bulges is approximate.")
        return BoundingBox.from_points(points_xy)

    def get_bounding_box(self) -> BoundingBox:
        """Return exact bounds for line and bulge segments in world space.

        Bulge arcs are kept in local coordinates and transformed as affine
        parametric curves.  This is necessary because a general affine map
        turns a circular bulge into an ellipse; transforming a sampled or
        endpoint-only absolute representation cannot recover its extrema.
        """
        try:
            matrix = _affine_matrix_or_none(self.get_total_transform())
        except (TypeError, ValueError):
            matrix = None
        if matrix is None or not self.vertices:
            return BoundingBox()
        try:
            vertices = self._validated_vertices()
        except (TypeError, ValueError):
            return BoundingBox()

        local_points = []
        for vertex in vertices:
            try:
                x, y = float(vertex.x), float(vertex.y)
            except (TypeError, ValueError, OverflowError):
                return BoundingBox()
            if not (math.isfinite(x) and math.isfinite(y)):
                return BoundingBox()
            local_points.append((x, y))

        # A point-only/open one-point Pline still has a useful point bound.
        if len(local_points) == 1:
            transformed = matrix[:2, :2] @ np.asarray(local_points[0]) + matrix[:2, 2]
            if not np.isfinite(transformed).all():
                return BoundingBox()
            return BoundingBox.from_points([(float(transformed[0]), float(transformed[1]))])

        segment_count = len(local_points) - 1 + (1 if self.closed else 0)
        bounds = BoundingBox()
        for index in range(segment_count):
            next_index = (index + 1) % len(local_points)
            p0 = local_points[index]
            p1 = local_points[next_index]
            raw_bulge = vertices[index].bulge
            try:
                bulge = float(raw_bulge)
            except (TypeError, ValueError, OverflowError):
                return BoundingBox()
            if not math.isfinite(bulge):
                return BoundingBox()

            point0 = np.asarray(p0, dtype=float)
            point1 = np.asarray(p1, dtype=float)
            # Half-deltas and half-sums avoid overflowing when opposite finite
            # endpoints span more than the largest representable float.
            half_chord = point1 * 0.5 - point0 * 0.5
            half_chord_length = math.hypot(float(half_chord[0]), float(half_chord[1]))
            if (half_chord_length <= CURVE_POINT_TOLERANCE * 0.5
                    or abs(bulge) <= PLINE_BULGE_TOLERANCE):
                segment_points = apply_transform((p0, p1), matrix)
                if len(segment_points) != 2:
                    return BoundingBox()
                segment_box = BoundingBox.from_points(segment_points)
            else:
                # CamBam's bulge is tan(included_angle / 4), with positive
                # values sweeping counter-clockwise from p0 to p1.
                sweep = 4.0 * math.atan(bulge)
                midpoint = point0 * 0.5 + point1 * 0.5
                left_normal = np.asarray(
                    (-half_chord[1], half_chord[0])
                ) / half_chord_length
                inverse_bulge = 1.0 / bulge
                offset = half_chord_length * ((inverse_bulge - bulge) / 2.0)
                center = midpoint + left_normal * offset
                radius = half_chord_length * (
                    (abs(bulge) + abs(inverse_bulge)) / 2.0
                )
                start = math.atan2(p0[1] - center[1], p0[0] - center[0])
                segment_box = _local_arc_bounding_box(
                    (float(center[0]), float(center[1])), radius, start, sweep, matrix
                )
            if not segment_box.is_valid():
                return BoundingBox()
            bounds = bounds.union(segment_box)
        return bounds

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center as geometric center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.vertices:
            # Fallback to first point if bbox is invalid (e.g., single point pline)
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0][0], abs_coords[0][1]
        return (0.0, 0.0) # Default fallback

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to vertices.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        transform_to_apply = affine
        vertices = self._validated_vertices()
        segment_count = max(0, len(vertices) - 1)
        if self.closed and vertices:
            segment_count += 1
        has_bulged_segment = any(
            abs(_finite_float(
                vertices[index].bulge,
                f"vertices[{index}].bulge",
            )) > PLINE_BULGE_TOLERANCE
            for index in range(segment_count)
        )
        if has_bulged_segment and _xy_similarity_scale(transform_to_apply) is None:
            raise ValueError(
                "Bulged Pline geometry can only bake XY similarity transforms"
            )
        reflected = np.linalg.det(transform_to_apply[0:2, 0:2]) < 0.0
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform XY coordinates
            rel_pts_xy = [(vertex.x, vertex.y) for vertex in vertices]
            baked_pts_xy = apply_transform(rel_pts_xy, transform_to_apply)

            # Rebuild relative points with original bulge values
            new_vertices = []
            for i, (x, y) in enumerate(baked_pts_xy):
                if i < len(vertices):
                    bulge = vertices[i].bulge
                    if reflected:
                        bulge = -bulge
                    new_vertices.append(Vertex(x, y, vertices[i].z, bulge=bulge))
                else:
                    logger.warning(f"Point mismatch during baking for Pline {self.user_identifier}. Skipping point.")

            self.vertices = new_vertices
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Pline {self.user_identifier}: {e}")


    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <pline> XML element."""
        pline_elem = ET.Element("pline", {"Closed": str(self.closed).lower()})

        # Add points WITHOUT applying the effective_transform here
        # The matrix added later by _add_common_xml_attributes handles the total transform
        pts_elem = ET.SubElement(pline_elem, "pts")
        vertices = self._validated_vertices()
        for vertex in vertices:
            # CamBam point format: x,y,z (z is usually 0 for 2D)
            x = round(vertex.x, self.output_decimals) if self.output_decimals is not None else vertex.x
            y = round(vertex.y, self.output_decimals) if self.output_decimals is not None else vertex.y
            z = vertex.z
            z = round(z, self.output_decimals) if self.output_decimals is not None else z
            bulge = vertex.bulge
            bulge = round(bulge, self.output_decimals) if self.output_decimals is not None else bulge
            
            ET.SubElement(pts_elem, "p", {"b": str(bulge)}).text = f"{x},{y},{z}"

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(pline_elem, xml_primitive_id, parent_uuid)
        return pline_elem


@dataclass
class Circle(Primitive):
    # Intrinsic geometry
    relative_center: Tuple[float, float] = (0.0, 0.0)
    diameter: float = 1.0
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute center and scaled diameter."""
        abs_center = get_transformed_point(self.relative_center, total_transform)

        # Calculate scaled diameter - average scaling factor from matrix columns
        # Assumes uniform scaling or takes an average for non-uniform
        sx = np.linalg.norm(total_transform[:, 0]) # Length of transformed x-axis vector
        sy = np.linalg.norm(total_transform[:, 1]) # Length of transformed y-axis vector
        avg_scale = (sx + sy) / 2.0
        abs_diameter = self.diameter * avg_scale
        if not math.isclose(sx, sy):
            logger.warning(f"Circle {self.user_identifier} transformed with non-uniform scale ({sx:.3f}, {sy:.3f}). Using average scale for diameter calculation.")

        return {"center": abs_center, "diameter": abs_diameter}

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        if _xy_similarity_scale(total_transform) is None:
            raise ValueError("Circle XYZ queries require an XY similarity transform")
        geometry = self._calculate_absolute_geometry(total_transform)
        center_x, center_y = geometry["center"]
        geometry["center"] = (
            center_x, center_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        cx, cy = absolute_geometry["center"]
        radius = absolute_geometry["diameter"] / 2.0
        return BoundingBox(cx - radius, cy - radius, cx + radius, cy + radius)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Center is simply the transformed relative center
        return get_transformed_point(self.relative_center, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to center and diameter.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        similarity_scale = _xy_similarity_scale(transform_to_apply)
        if similarity_scale is None:
            raise ValueError("Circle geometry can only bake XY similarity transforms")
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)

            # Bake diameter (using average scale factor)
            self.diameter *= similarity_scale
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Circle {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <circle> XML element."""
        cx = round(self.relative_center[0], self.output_decimals) if self.output_decimals is not None else self.relative_center[0]
        cy = round(self.relative_center[1], self.output_decimals) if self.output_decimals is not None else self.relative_center[1]
        cz = _finite_float(self.elevation, "elevation")
        cz = round(cz, self.output_decimals) if self.output_decimals is not None else cz
        c_diam = round(self.diameter, self.output_decimals) if self.output_decimals is not None else self.diameter
        
        circle_elem = ET.Element("circle", {
            "c": f"{cx},{cy},{cz}", # Center (x,y,z)
            "d": str(c_diam) # Diameter
         })
        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(circle_elem, xml_primitive_id, parent_uuid)
        return circle_elem


@dataclass
class Rect(Primitive):
    # Intrinsic geometry defined by corner, width, height
    # Note: CamBam rect XML uses 'p' (corner), 'w', 'h' but seems to store it as a Pline internally.
    # We'll keep w/h definition for ease of use but might need adjustments based on CamBam behavior.
    relative_corner: Tuple[float, float] = (0.0, 0.0) # Bottom-left
    width: float = 1.0
    height: float = 1.0
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _get_relative_corners(self) -> List[Tuple[float, float]]:
        """Returns the four corners in relative coordinates."""
        x0, y0 = self.relative_corner
        return [(x0, y0), (x0 + self.width, y0), (x0 + self.width, y0 + self.height), (x0, y0 + self.height)]

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        """Returns the four corners in absolute coordinates."""
        return apply_transform(self._get_relative_corners(), total_transform)

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float]]:
        z = _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z")
        return [
            (x, y, z)
            for x, y in self._calculate_absolute_geometry(total_transform)
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float]]) -> BoundingBox:
        if not absolute_geometry:
            return BoundingBox()
        return BoundingBox.from_points(absolute_geometry)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Calculate relative center
        rel_cx = self.relative_corner[0] + self.width / 2.0
        rel_cy = self.relative_corner[1] + self.height / 2.0
        # Transform relative center to absolute
        return get_transformed_point((rel_cx, rel_cy), self.get_total_transform())

    def is_rectangular_after_transform(self) -> bool:
        """
        Determines if the rectangle's geometry is still rectangular after applying
        the effective transformation.
        
        Returns:
            True if the transformed corners still form a rectangle, False otherwise
        """
        if np.allclose(self.effective_transform, identity_matrix()):
            return True  # With identity transform, it's definitely rectangular
        
        # Get the rectangle corners
        corners = self._get_relative_corners()
        
        # Transform the corners
        transformed_corners = apply_transform(corners, self.effective_transform)
        
        # Check if the transformed corners still form a rectangle
        # This requires adjacent sides to be perpendicular
        if len(transformed_corners) == 4:
            # Calculate vectors for adjacent sides
            v1 = (transformed_corners[1][0] - transformed_corners[0][0], 
                  transformed_corners[1][1] - transformed_corners[0][1])
            v2 = (transformed_corners[3][0] - transformed_corners[0][0], 
                  transformed_corners[3][1] - transformed_corners[0][1])
            v3 = (transformed_corners[2][0] - transformed_corners[1][0],
                  transformed_corners[2][1] - transformed_corners[1][1])
            v4 = (transformed_corners[2][0] - transformed_corners[3][0],
                  transformed_corners[2][1] - transformed_corners[3][1])
            
            # Calculate dot products to check perpendicularity
            dot1 = v1[0]*v2[0] + v1[1]*v2[1]
            dot2 = v2[0]*v4[0] + v2[1]*v4[1]
            dot3 = v4[0]*v3[0] + v4[1]*v3[1]
            dot4 = v3[0]*v1[0] + v3[1]*v1[1]
            
            # All dot products should be close to 0 for a rectangle
            return (math.isclose(dot1, 0, abs_tol=1e-10) and
                    math.isclose(dot2, 0, abs_tol=1e-10) and
                    math.isclose(dot3, 0, abs_tol=1e-10) and
                    math.isclose(dot4, 0, abs_tol=1e-10))
        
        return False
    
    def to_pline_representation(self) -> Pline:
        """
        Creates a Pline representation of this rectangle, applying any transformations.
        
        Returns:
            A new Pline object representing the same geometry with the transformation applied
        """
        # Get the rectangle corners
        corners = self._get_relative_corners()
        
        # Transform the corners
        transformed_corners = apply_transform(corners, self.effective_transform)
        
        # Create points for Pline (adding the first point again to close the loop if needed)
        pline_points = [Vertex(p[0], p[1], self.elevation) for p in transformed_corners]
        
        # Create a new Pline
        pline = Pline(
            user_identifier=f"{self.user_identifier}_as_pline",
            groups=self.groups.copy() if self.groups else [],
            description=f"Converted from Rect: {self.description}",
            effective_transform=identity_matrix(),  # Use identity since we've already applied the transform
            vertices=pline_points,
            closed=True,
            local_z_offset=self.local_z_offset,
        )
        
        return pline

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """Bake the outline, converting this object to a closed Pline if needed.

        Existing references and all common metadata survive conversion. Rect-only
        geometry attributes cease to exist. An explicit matrix leaves the local
        effective transform intact; an implicit bake resets it to identity.
        """
        matrix = self.effective_transform if transform_to_bake is None else transform_to_bake
        if np.iscomplexobj(matrix):
            raise ValueError("Expected a finite affine 3x3 matrix")
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[2], [0., 0., 1.])):
            raise ValueError("Expected a finite affine 3x3 matrix")

        corners = np.asarray(apply_transform(self._get_relative_corners(), matrix))
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("Rect baking produced invalid corners")
        lower = corners.min(axis=0)
        upper = corners.max(axis=0)
        bounds = np.array([lower, (upper[0], lower[1]), upper, (lower[0], upper[1])])
        # Compare the complete closed outline, not perpendicularity or bounds.
        # Absolute tolerance only: large coordinates must not hide a shear.
        axis_aligned = any(
            np.allclose(corners, np.roll(order, shift, axis=0), rtol=0, atol=1e-12)
            for order in (bounds, bounds[::-1]) for shift in range(4)
        )
        if axis_aligned:
            self.relative_corner = tuple(lower)
            self.width, self.height = upper - lower
        else:
            elevation = _finite_float(self.elevation, "elevation")
            vertices = [Vertex(x, y, elevation) for x, y in corners]
            # Both dataclasses have the same ordinary Python object layout.
            # Assign the class before geometry edits so an unsupported subclass
            # layout fails without discarding the original Rect geometry.
            self.__class__ = Pline
            self.vertices = vertices
            self.closed = True
            del self.relative_corner, self.width, self.height, self.elevation
        if transform_to_bake is None:
            self.effective_transform = identity_matrix()

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """
        Creates an XML element for this Rect.
        If the geometry is not rectangular after transformation, converts to a Pline representation.
        """
        # Check if we need to convert to Pline for XML output
        if not self.is_rectangular_after_transform():
            # Serialize the complete world outline with an identity matrix.
            # ``to_pline_representation`` intentionally bakes only this Rect's
            # local matrix; using it here loses ancestor transforms because the
            # temporary Pline is not linked to a project.
            pline_repr = Pline(
                user_identifier=self.user_identifier,
                groups=self.groups.copy() if self.groups else [],
                description=self.description,
                output_decimals=self.output_decimals,
                effective_transform=identity_matrix(),
                local_z_offset=self.get_total_z_offset(),
                vertices=[Vertex(x, y, self.elevation) for x, y in apply_transform(
                    self._get_relative_corners(), self.get_total_transform())],
                closed=True,
            )
            
            # Get the Pline's XML element, but with our metadata
            pline_elem = pline_repr.to_xml_element(xml_primitive_id, parent_uuid)
            
            # Update the Tag element to maintain our identity (but no rect-specific data)
            tag_data = {
                "user_id": self.user_identifier,
                "internal_id": str(self.internal_id),
                "groups": self.groups,
                "description": self.description
            }
            
            # Add parent reference if provided
            if parent_uuid:
                tag_data["parent"] = str(parent_uuid)
            
            # Update or add the Tag element
            tag_node = pline_elem.find("Tag")
            if tag_node is None:
                tag_node = ET.SubElement(pline_elem, "Tag")
            tag_node.text = json.dumps(tag_data)

            # Log that this Rect was converted to Pline
            logger.info(f"Rect '{self.user_identifier}' converted to Pline for XML output due to non-rectangular geometry.")

            return pline_elem
        
        # Otherwise, create a normal Rect XML element
        x = round(self.relative_corner[0], self.output_decimals) if self.output_decimals is not None else self.relative_corner[0]
        y = round(self.relative_corner[1], self.output_decimals) if self.output_decimals is not None else self.relative_corner[1]
        z = _finite_float(self.elevation, "elevation")
        z = round(z, self.output_decimals) if self.output_decimals is not None else z
        w = round(self.width, self.output_decimals) if self.output_decimals is not None else self.width
        h = round(self.height, self.output_decimals) if self.output_decimals is not None else self.height
        
        rect_elem = ET.Element("rect", {
            "Closed": "true", # Rectangles are implicitly closed
            "p": f"{x},{y},{z}", # Corner (x,y,z)
            "w": str(w),
            "h": str(h)
        })
        
        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(rect_elem, xml_primitive_id, parent_uuid)
        
        return rect_elem


@dataclass
class Arc(Primitive):
    # Intrinsic geometry
    relative_center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    start_angle: float = 0.0 # Degrees
    extent_angle: float = 90.0 # Degrees (sweep angle)
    elevation: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute center, scaled radius, and transformed angles."""
        abs_center = get_transformed_point(self.relative_center, total_transform)

        # Calculate scaled radius (similar to Circle)
        sx = np.linalg.norm(total_transform[:, 0])
        sy = np.linalg.norm(total_transform[:, 1])
        avg_scale = (sx + sy) / 2.0
        abs_radius = self.radius * avg_scale

        # Calculate transformed start angle
        # Get the transformation's rotation component in degrees
        # atan2(m10, m00) gives the rotation angle
        rotation_rad = math.atan2(total_transform[1, 0], total_transform[0, 0])
        rotation_deg = math.degrees(rotation_rad)
        abs_start_angle = (self.start_angle + rotation_deg) % 360

        # Extent angle remains unchanged by rotation/translation/uniform scale
        # Non-uniform scale would distort the arc into an elliptical arc, not handled here.
        abs_extent_angle = self.extent_angle
        if not math.isclose(sx, sy):
            logger.warning(f"Arc {self.user_identifier} transformed with non-uniform scale ({sx:.3f}, {sy:.3f}). Extent angle might be inaccurate.")

        return {
            "center": abs_center,
            "radius": abs_radius,
            "start_angle": abs_start_angle,
            "extent_angle": abs_extent_angle
        }

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        if _xy_similarity_scale(total_transform) is None:
            raise ValueError("Arc XYZ queries require an XY similarity transform")
        geometry = self._calculate_absolute_geometry(total_transform)
        center_x, center_y = geometry["center"]
        geometry["center"] = (
            center_x, center_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        start = math.radians(self.start_angle)
        direction = total_transform[:2, :2] @ np.array([math.cos(start), math.sin(start)])
        geometry["start_angle"] = math.degrees(math.atan2(direction[1], direction[0])) % 360
        geometry["extent_angle"] = self.extent_angle * (
            -1.0 if np.linalg.det(total_transform[:2, :2]) < 0.0 else 1.0
        )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        shifted = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        self.elevation = shifted

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        # Approximate bounding box using center and radius
        # TODO: A precise bounding box needs to consider the actual arc sweep.
        cx, cy = absolute_geometry["center"]
        radius = absolute_geometry["radius"]
        logger.debug(f"Bounding box for Arc {self.user_identifier} is approximate (using full circle).")
        return BoundingBox(cx - radius, cy - radius, cx + radius, cy + radius)

    def get_bounding_box(self) -> BoundingBox:
        """Return exact world bounds of the directed circular sweep."""
        try:
            matrix = _affine_matrix_or_none(self.get_total_transform())
        except (TypeError, ValueError):
            matrix = None
        if matrix is None:
            return BoundingBox()
        try:
            center = (float(self.relative_center[0]), float(self.relative_center[1]))
            radius = float(self.radius)
            start = math.radians(float(self.start_angle))
            sweep = math.radians(float(self.extent_angle))
        except (TypeError, ValueError, OverflowError, IndexError):
            return BoundingBox()
        return _local_arc_bounding_box(center, radius, start, sweep, matrix)

    def get_geometric_center(self) -> Tuple[float, float]:
        # For simplicity, use the transformed center of the arc's circle
        # TODO: A more accurate center would be the midpoint of the arc chord or centroid.
        return get_transformed_point(self.relative_center, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to center, radius, and angles.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        similarity_scale = _xy_similarity_scale(transform_to_apply)
        if similarity_scale is None:
            raise ValueError("Arc geometry can only bake XY similarity transforms")
        transform_to_apply = _affine_matrix_or_none(transform_to_apply)
        assert transform_to_apply is not None
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake center position
            self.relative_center = get_transformed_point(self.relative_center, transform_to_apply)
            
            self.radius *= similarity_scale

            # Transform the actual start direction. This also handles a
            # reflected similarity without guessing its mirror axis.
            start_radians = math.radians(self.start_angle)
            start_vector = np.asarray(
                (math.cos(start_radians), math.sin(start_radians)), dtype=float
            )
            transformed_start = transform_to_apply[0:2, 0:2] @ start_vector
            self.start_angle = math.degrees(math.atan2(
                transformed_start[1], transformed_start[0]
            )) % 360
            if np.linalg.det(transform_to_apply[0:2, 0:2]) < 0:
                self.extent_angle = -self.extent_angle
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Arc {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <arc> XML element."""
        cx = round(self.relative_center[0], self.output_decimals) if self.output_decimals is not None else self.relative_center[0]
        cy = round(self.relative_center[1], self.output_decimals) if self.output_decimals is not None else self.relative_center[1]
        cz = _finite_float(self.elevation, "elevation")
        cz = round(cz, self.output_decimals) if self.output_decimals is not None else cz
        radius = round(self.radius, self.output_decimals) if self.output_decimals is not None else self.radius
        start_angle = round(self.start_angle % 360, self.output_decimals) if self.output_decimals is not None else self.start_angle % 360
        extent_angle = round(self.extent_angle, self.output_decimals) if self.output_decimals is not None else self.extent_angle
        
        arc_elem = ET.Element("arc", {
            "p": f"{cx},{cy},{cz}", # Center (x,y,z)
            "r": str(radius),       # Radius
            "s": str(start_angle),  # Start Angle (degrees)
            "w": str(extent_angle)  # Sweep Angle (degrees)
        })

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(arc_elem, xml_primitive_id, parent_uuid)
        return arc_elem


@dataclass
class Points(Primitive):
    # Intrinsic geometry
    vertices: List[Vertex] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.vertices = _normalize_vertices(self.vertices, allow_bulge=False)

    def _validated_vertices(self) -> List[Vertex]:
        return _validate_stored_vertices(self.vertices, allow_bulge=False)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        return apply_transform(
            [(vertex.x, vertex.y) for vertex in self._validated_vertices()],
            total_transform,
        )

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> List[Tuple[float, float, float]]:
        vertices = self._validated_vertices()
        return [
            (x, y, _finite_float(vertices[index].z + total_z_offset, "world Z"))
            for index, (x, y) in enumerate(
                self._calculate_absolute_geometry(total_transform)
            )
        ]

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        vertices = self._validated_vertices()
        self.vertices = [
            Vertex(vertex.x, vertex.y, _finite_float(vertex.z + delta, "shifted vertex Z"))
            for vertex in vertices
        ]

    def _calculate_bounding_box(self, absolute_geometry: List[Tuple[float, float]]) -> BoundingBox:
        if not absolute_geometry:
            return BoundingBox()
        return BoundingBox.from_points(absolute_geometry)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bounding box center
        bbox = self.get_bounding_box()
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif self.vertices:
            abs_coords = self.get_absolute_coordinates()
            if abs_coords:
                return abs_coords[0] # Fallback to first point
        return (0.0, 0.0)

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to vertices.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        transform_to_apply = affine
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Transform the points
            vertices = self._validated_vertices()
            transformed = apply_transform(
                [(vertex.x, vertex.y) for vertex in vertices], transform_to_apply
            )
            self.vertices = [
                Vertex(x, y, vertices[index].z)
                for index, (x, y) in enumerate(transformed)
            ]
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Points {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <points> XML element."""
        points_elem = ET.Element("points")
        pts_elem = ET.SubElement(points_elem, "pts")
        for vertex in self._validated_vertices():
            x, y, pz = vertex.x, vertex.y, vertex.z
            px = round(x, self.output_decimals) if self.output_decimals is not None else x
            py = round(y, self.output_decimals) if self.output_decimals is not None else y
            pz = round(pz, self.output_decimals) if self.output_decimals is not None else pz
            # CamBam point format: x,y,z (z is usually 0)
            ET.SubElement(pts_elem, "p").text = f"{px},{py},{pz}"

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(points_elem, xml_primitive_id, parent_uuid)
        return points_elem


@dataclass
class Text(Primitive):
    # Intrinsic properties
    text_content: str = "Text"
    relative_position: Tuple[float, float] = (0.0, 0.0) # Anchor point
    # CamBam's MText API documents p2 as currently unused. Preserve the optional
    # serialized value without assigning unverified drawing semantics.
    xml_p2_position: Optional[Tuple[float, float]] = None
    elevation: float = 0.0
    xml_p2_elevation: Optional[float] = None
    height: float = 10.0 # Font height in drawing units
    font: str = 'Arial'
    style: str = '' # e.g., 'bold', 'italic', 'bold,italic'
    line_spacing: float = 1.0 # Multiplier
    align_horizontal: str = 'center' # 'left', 'center', 'right'
    align_vertical: str = 'center' # 'top', 'center', 'bottom'

    def __post_init__(self):
        super().__post_init__()
        self.elevation = _finite_float(self.elevation, "elevation")
        if self.xml_p2_elevation is not None:
            self.xml_p2_elevation = _finite_float(
                self.xml_p2_elevation, "xml_p2_elevation"
            )
        if self.xml_p2_position is not None:
            try:
                if len(self.xml_p2_position) != 2:
                    raise ValueError
                self.xml_p2_position = (
                    _finite_float(self.xml_p2_position[0], "xml_p2_position x"),
                    _finite_float(self.xml_p2_position[1], "xml_p2_position y"),
                )
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    "xml_p2_position must contain exactly two finite coordinates"
                ) from exc

    def _effective_xml_p2_position(self) -> Tuple[float, float]:
        return self.relative_position if self.xml_p2_position is None else self.xml_p2_position

    def _effective_xml_p2_elevation(self) -> float:
        if self.xml_p2_elevation is None:
            return _finite_float(self.elevation, "elevation")
        return _finite_float(self.xml_p2_elevation, "xml_p2_elevation")

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        """Returns absolute anchor position and scaled height."""
        abs_position = get_transformed_point(self.relative_position, total_transform)

        # Calculate scaled height (using average scale factor)
        sx = np.linalg.norm(total_transform[:, 0])
        sy = np.linalg.norm(total_transform[:, 1])
        avg_scale = (sx + sy) / 2.0
        abs_height = self.height * avg_scale

        # Font, style, alignment etc are not directly transformed, but scale affects appearance
        return {
            "position": abs_position,
            "height": abs_height,
            "text": self.text_content,
            # Include other properties if needed for bounding box calculation
        }

    def _calculate_absolute_geometry_xyz(
        self, total_transform: np.ndarray, total_z_offset: float
    ) -> Dict[str, Any]:
        geometry = self._calculate_absolute_geometry(total_transform)
        position_x, position_y = geometry["position"]
        geometry["position"] = (
            position_x, position_y,
            _finite_float(_finite_float(self.elevation, "elevation") + total_z_offset, "world Z"),
        )
        if self.xml_p2_position is None and self.xml_p2_elevation is None:
            geometry["xml_p2"] = None
        else:
            p2_x, p2_y = get_transformed_point(
                self._effective_xml_p2_position(), total_transform
            )
            geometry["xml_p2"] = (
                p2_x, p2_y,
                _finite_float(self._effective_xml_p2_elevation() + total_z_offset,
                              "world XML p2 Z"),
            )
        return geometry

    def shift_geometry_z(self, dz: float) -> None:
        delta = _finite_float(dz, "dz")
        elevation = _finite_float(
            _finite_float(self.elevation, "elevation") + delta,
            "shifted elevation",
        )
        if self.xml_p2_elevation is None:
            p2_elevation = None
        else:
            p2_elevation = _finite_float(
                self._effective_xml_p2_elevation() + delta,
                "shifted xml_p2_elevation",
            )
        self.elevation = elevation
        self.xml_p2_elevation = p2_elevation

    def _calculate_bounding_box(self, absolute_geometry: Dict[str, Any]) -> BoundingBox:
        # Bounding box for text is complex and font-dependent.
        # Provide a rough estimate based on height and text length.
        logger.warning(f"Bounding box for Text primitive '{self.user_identifier}' is approximate.")
        px, py = absolute_geometry["position"]
        h = absolute_geometry["height"]
        # Estimate width: number of chars in longest line * height * aspect ratio (e.g., 0.6)
        longest_line_len = max(len(line) for line in self.text_content.splitlines()) if self.text_content else 0
        est_width = longest_line_len * h * 0.6
        num_lines = self.text_content.count('\n') + 1
        est_total_height = h * (1 + (num_lines - 1) * self.line_spacing)

        # Adjust position based on alignment (relative to estimated box)
        # Horizontal
        if self.align_horizontal == 'left':
            min_x, max_x = px, px + est_width
        elif self.align_horizontal == 'right':
            min_x, max_x = px - est_width, px
        else: # center
            min_x, max_x = px - est_width / 2, px + est_width / 2
        # Vertical
        if self.align_vertical == 'top':
            min_y, max_y = py - est_total_height, py
        elif self.align_vertical == 'bottom':
            min_y, max_y = py, py + est_total_height
        else: # center
            min_y, max_y = py - est_total_height / 2, py + est_total_height / 2

        return BoundingBox(min_x, min_y, max_x, max_y)


    def get_geometric_center(self) -> Tuple[float, float]:
        # Use the transformed anchor position as the 'center'
        # A better center might be the center of the approximate bounding box.
        return get_transformed_point(self.relative_position, self.get_total_transform())

    def bake_geometry(self, transform_to_bake: Optional[np.ndarray] = None) -> None:
        """
        Applies a transformation matrix to position and height.
        
        If transform_to_bake is None, uses and resets the effective_transform.
        """
        # Determine which transformation to apply
        if transform_to_bake is None:
            # Use current effective transform
            transform_to_apply = self.effective_transform
            reset_transform = True
        else:
            # Use provided transform
            transform_to_apply = transform_to_bake
            reset_transform = False

        affine = _affine_matrix_or_none(transform_to_apply)
        if affine is None:
            raise ValueError("Expected a finite affine 3x3 matrix")
        linear = affine[0:2, 0:2]
        scale = float((linear[0, 0] + linear[1, 1]) / 2.0)
        if (scale <= 0.0 or not np.allclose(
                linear, np.identity(2) * scale, rtol=0.0, atol=1e-12)):
            raise ValueError(
                "Text geometry can only bake XY translation and positive uniform scale"
            )
        transform_to_apply = affine
            
        # Skip if identity matrix (nothing to bake)
        if np.array_equal(transform_to_apply, identity_matrix()):
            return
            
        try:
            # Bake position
            self.relative_position = get_transformed_point(self.relative_position, transform_to_apply)
            if self.xml_p2_position is not None:
                self.xml_p2_position = get_transformed_point(
                    self.xml_p2_position, transform_to_apply
                )
            
            # Bake height (using average scale factor)
            self.height *= scale
            
            # Handle text mirroring - affects alignment
            det = np.linalg.det(transform_to_apply[0:2, 0:2])
            if det < 0:  # Mirroring detected
                # Determine axis of mirroring
                sx_sign = np.sign(transform_to_apply[0, 0]) 
                sy_sign = np.sign(transform_to_apply[1, 1])
                
                # For x-mirroring (negative x scale)
                if sx_sign < 0:
                    if self.align_horizontal == 'left':
                        self.align_horizontal = 'right'
                    elif self.align_horizontal == 'right':
                        self.align_horizontal = 'left'
                
                # For y-mirroring (negative y scale)
                if sy_sign < 0:
                    if self.align_vertical == 'top':
                        self.align_vertical = 'bottom'
                    elif self.align_vertical == 'bottom':
                        self.align_vertical = 'top'
            
            # Reset effective transform if using it
            if reset_transform:
                self.effective_transform = identity_matrix()
                
        except Exception as e:
            logger.error(f"Failed to bake Text {self.user_identifier}: {e}")

    def to_xml_element(self, xml_primitive_id: int, parent_uuid: Optional[uuid.UUID]) -> ET.Element:
        """Creates the <text> XML element."""
        # CamBam omits p1 at the default origin and may preserve an optional,
        # currently-unused p2. Align is serialized independently.
        cb_v_align = self.align_vertical
        cb_h_align = self.align_horizontal
        p1x = round(self.relative_position[0], self.output_decimals) if self.output_decimals is not None else self.relative_position[0]
        p1y = round(self.relative_position[1], self.output_decimals) if self.output_decimals is not None else self.relative_position[1]
        p1z = _finite_float(self.elevation, "elevation")
        p2_position = self._effective_xml_p2_position()
        p2x = round(p2_position[0], self.output_decimals) if self.output_decimals is not None else p2_position[0]
        p2y = round(p2_position[1], self.output_decimals) if self.output_decimals is not None else p2_position[1]
        p2z = self._effective_xml_p2_elevation()
        if self.output_decimals is not None:
            p1z = round(p1z, self.output_decimals)
            p2z = round(p2z, self.output_decimals)

        attributes = {
            "Height": str(self.height),
            "Font": self.font,
            "linespace": str(self.line_spacing),
            "align": f"{cb_v_align},{cb_h_align}",
            "style": self.style
        }
        if (p1x, p1y, p1z) != (0, 0, 0):
            attributes["p1"] = f"{p1x},{p1y},{p1z}"
        if self.xml_p2_position is not None or self.xml_p2_elevation is not None:
            attributes["p2"] = f"{p2x},{p2y},{p2z}"
        text_elem = ET.Element("text", attributes)
        # Text content goes inside the element
        text_elem.text = self.text_content

        # Add common ID, Tag (with parent), and Matrix
        self._add_common_xml_attributes(text_elem, xml_primitive_id, parent_uuid)
        return text_elem


# --- MOP Base and Concrete Classes ---

MopType = TypeVar('MopType', bound='Mop')

# Native XML paths for fields represented by the four supported MOP classes.
# Keeping this table beside the entity model gives the reader and writer one
# vocabulary for state tracking while the native XML template retains fields
# that this model does not represent.
MOP_XML_FIELD_PATHS: Dict[str, Tuple[str, ...]] = {
    'target_depth': ('TargetDepth',), 'depth_increment': ('DepthIncrement',),
    'stock_surface': ('StockSurface',), 'roughing_clearance': ('RoughingClearance',),
    'clearance_plane': ('ClearancePlane',), 'spindle_direction': ('SpindleDirection',),
    'spindle_speed': ('SpindleSpeed',), 'velocity_mode': ('VelocityMode',),
    'work_plane': ('WorkPlane',), 'optimisation_mode': ('OptimisationMode',),
    'tool_diameter': ('ToolDiameter',), 'tool_number': ('ToolNumber',),
    'tool_profile': ('ToolProfile',), 'plunge_feedrate': ('PlungeFeedrate',),
    'cut_feedrate': ('CutFeedrate',), 'max_crossover_distance': ('MaxCrossoverDistance',),
    'custom_mop_header': ('CustomMOPHeader',), 'custom_mop_footer': ('CustomMOPFooter',),
    'stepover': ('StepOver',), 'profile_side': ('InsideOutside',),
    'milling_direction': ('MillingDirection',), 'collision_detection': ('CollisionDetection',),
    'corner_overcut': ('CornerOvercut',), 'lead_in_type': ('LeadInMove', 'LeadInType'),
    'lead_in_spiral_angle': ('LeadInMove', 'SpiralAngle'),
    'final_depth_increment': ('FinalDepthIncrement',), 'cut_ordering': ('CutOrdering',),
    'tab_method': ('HoldingTabs', 'TabMethod'), 'tab_width': ('HoldingTabs', 'Width'),
    'tab_height': ('HoldingTabs', 'Height'), 'tab_min_tabs': ('HoldingTabs', 'MinimumTabs'),
    'tab_max_tabs': ('HoldingTabs', 'MaximumTabs'), 'tab_distance': ('HoldingTabs', 'TabDistance'),
    'tab_size_threshold': ('HoldingTabs', 'SizeThreshold'),
    'tab_use_leadins': ('HoldingTabs', 'UseLeadIns'), 'tab_style': ('HoldingTabs', 'TabStyle'),
    'stepover_feedrate': ('StepoverFeedrate',), 'region_fill_style': ('RegionFillStyle',),
    'finish_stepover': ('FinishStepover',),
    'finish_stepover_at_target_depth': ('FinishStepoverAtTargetDepth',),
    'roughing_finishing': ('RoughingFinishing',), 'drilling_method': ('DrillingMethod',),
    'peck_distance': ('PeckDistance',), 'retract_height': ('RetractHeight',),
    'dwell': ('Dwell',), 'hole_diameter': ('HoleDiameter',),
    'drill_lead_out': ('DrillLeadOut',), 'spiral_flat_base': ('SpiralFlatBase',),
    'lead_out_length': ('LeadOutLength',), 'custom_script': ('CustomScript',),
}
MOP_XML_PATH_TO_FIELD = {path: field for field, path in MOP_XML_FIELD_PATHS.items()}

@dataclass
class Mop(CamBamEntity, ABC):
    """
    Abstract Base Class for Machine Operations (MOPs).
    Holds intrinsic machining parameters.
    Targets and part assignment are owned by the project.
    """
    # Intrinsic Attributes
    name: str = "MOP" # User-visible name in CamBam UI MOP tree
    enabled: bool = True
    target_depth: Optional[float] = None # If None, uses Part/Project default (or fails if none set)
    depth_increment: Optional[float] = None # If None, uses TargetDepth (single pass)
    stock_surface: float = 0.0
    roughing_clearance: float = 0.0
    clearance_plane: float = 15.0
    spindle_direction: str = 'CW' # 'CW', 'CCW', 'Off'
    spindle_speed: Optional[int] = None # If None, uses Part/Project default
    velocity_mode: str = 'ExactStop' # 'ExactStop', 'ConstantVelocity'
    work_plane: str = 'XY' # 'XY', 'XZ', 'YZ'
    optimisation_mode: str = 'Standard' # 'Standard', 'Experimental', 'Legacy'
    tool_diameter: Optional[float] = None # If None, uses Part/Project default
    tool_number: int = 0 # If 0, uses current tool
    tool_profile: str = 'EndMill' # 'EndMill', 'Vcutter', 'BallNose', 'Engrave', 'Drill'
    plunge_feedrate: float = 1000.0
    cut_feedrate: Optional[float] = None # If None, calculated based on TargetDepth or uses default
    max_crossover_distance: float = 0.7 # Multiplier of tool diameter
    custom_mop_header: str = ""
    custom_mop_footer: str = ""
    # Note: No part_id or _resolved_xml_primitive_ids here. Managed by Project/Writer.

    def __setattr__(self, name, value):
        # Once imported or explicitly state-edited, an assignment is intentional,
        # even if it repeats a cached Default value or restores an earlier value.
        if name in MOP_XML_FIELD_PATHS and (
                "_xml_template" in self.__dict__
                or "_xml_explicit_parameter_states" in self.__dict__):
            self.__dict__.setdefault("_xml_dirty_parameters", set()).add(name)
            self.__dict__.get("_xml_explicit_parameter_states", set()).discard(name)
        super().__setattr__(name, value)

    def set_parameter_state(self, field_name: str, state: str) -> None:
        """Set a top-level scalar parameter's CamBam inheritance state.

        ``Default`` leaves resolution to CamBam's CAM styles.  The state is
        metadata separate from the Python value so callers may explicitly
        restore inheritance after assigning a value.
        """
        if field_name not in MOP_XML_FIELD_PATHS or not hasattr(self, field_name):
            raise ValueError(f"Unsupported MOP parameter: {field_name}")
        if len(MOP_XML_FIELD_PATHS[field_name]) != 1:
            raise ValueError("Nested parameter inheritance belongs to its native container")
        if state not in ("Default", "Value"):
            raise ValueError("MOP parameter state must be 'Default' or 'Value'")
        states = getattr(self, "_xml_parameter_states", None)
        if states is None:
            states = {}
            self._xml_parameter_states = states
        states[field_name] = state
        explicit = getattr(self, "_xml_explicit_parameter_states", None)
        if explicit is None:
            explicit = set()
            self._xml_explicit_parameter_states = explicit
        explicit.add(field_name)

    def _native_mop_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> Optional[ET.Element]:
        """Clone an imported MOP and patch only fields explicitly changed."""
        template = getattr(self, "_xml_template", None)
        if template is None:
            return None
        root = deepcopy(template)
        root.set("Enabled", str(self.enabled).lower())
        name = root.find("Name")
        if name is None:
            name = ET.Element("Name")
            root.insert(0, name)
        name.text = self.name
        tag = root.find("Tag")
        if tag is None:
            tag = ET.Element("Tag")
            root.insert(1, tag)
        tag.text = json.dumps({"user_id": self.user_identifier, "internal_id": str(self.internal_id)}, separators=(",", ":"))

        primitive = root.find("primitive")
        if primitive is None:
            primitive = ET.Element("primitive")
            root.insert(2 + self._xml_primitive_index, primitive)
        for child in list(primitive):
            primitive.remove(child)
        for pid in sorted(resolved_primitive_xml_ids):
            ET.SubElement(primitive, "prim").text = str(pid)

        baseline = getattr(self, "_xml_parameter_baseline", {})
        states = getattr(self, "_xml_parameter_states", {})
        explicit_states = getattr(self, "_xml_explicit_parameter_states", set())
        for field_name, path in MOP_XML_FIELD_PATHS.items():
            if not hasattr(self, field_name):
                continue
            current = getattr(self, field_name)
            changed = (field_name in getattr(self, "_xml_dirty_parameters", set())
                       or field_name not in baseline or current != baseline[field_name])
            explicit_state = field_name in explicit_states
            if not changed and not explicit_state:
                continue
            parent = root
            for part in path[:-1]:
                child = parent.find(part)
                if child is None:
                    child = ET.SubElement(parent, part)
                parent = child
            element = parent.find(path[-1])
            if element is None:
                element = ET.SubElement(parent, path[-1])
            state = states.get(field_name) if field_name in explicit_states else None
            if state is None:
                state = "Default" if current is None else "Value"
            if len(path) == 1 or "state" in element.attrib or path[0] == "LeadInMove":
                element.set("state", state)
            element.text = self._format_mop_parameter(current)
            # A nested child edit must opt the native container into Value as
            # well, otherwise CamBam can continue inheriting the whole group.
            for depth in range(1, len(path)):
                parent_node = root.find("/".join(path[:depth]))
                if parent_node is not None:
                    parent_node.set("state", "Value")
        return root

    @staticmethod
    def _format_mop_parameter(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _apply_explicit_parameter_states(self, root: ET.Element) -> None:
        """Apply state edits to a newly-created (non-imported) MOP tree."""
        for field_name in getattr(self, "_xml_explicit_parameter_states", set()):
            path = MOP_XML_FIELD_PATHS[field_name]
            parent = root
            for part in path[:-1]:
                child = parent.find(part)
                if child is None:
                    child = ET.SubElement(parent, part)
                parent = child
            element = parent.find(path[-1])
            if element is None:
                element = ET.SubElement(parent, path[-1])
            state = self._xml_parameter_states[field_name]
            element.set("state", state)
            element.text = self._format_mop_parameter(getattr(self, field_name))
            if path[:-1]:
                ancestor = root.find(path[0])
                if ancestor is not None:
                    ancestor.set("state", state)

    @abstractmethod
    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        """
        Generates the MOP's specific XML element (e.g., <profile>, <pocket>).
        Requires the project context to resolve default parameters and the list of
        resolved primitive XML IDs (calculated by the writer).
        """
        pass

    def _get_effective_param(self, param_name: str, project: "CamBamProject") -> Any:
        """Helper to get a parameter value, checking MOP, then Part, then Project defaults."""
        mop_value = getattr(self, param_name, None)
        if mop_value is not None:
            return mop_value

        # Check Part default
        part = project.get_part_of_mop(self.internal_id) if project else None
        if part:
            part_param_map = {
                'tool_diameter': 'default_tool_diameter',
                'spindle_speed': 'default_spindle_speed'
                # Add other mappings if Part gets more defaults
            }
            part_attr = part_param_map.get(param_name)
            if part_attr:
                part_value = getattr(part, part_attr, None)
                if part_value is not None:
                    return part_value

        # Check Project default
        project_param_map = {
            'tool_diameter': 'default_tool_diameter'
            # Add other mappings if Project gets more defaults
        }
        project_attr = project_param_map.get(param_name)
        if project_attr and project:
            project_value = getattr(project, project_attr, None)
            if project_value is not None:
                return project_value

        # No value found anywhere
        # logger.debug(f"Parameter '{param_name}' not resolved for MOP '{self.name}'. Returning None.")
        return None

    def _calculate_effective_cut_feedrate(self, project: "CamBamProject") -> float:
        """Calculates the cut feedrate to use, applying logic if not explicitly set."""
        if self.cut_feedrate is not None:
            return self.cut_feedrate

        # Try simple fallback logic (e.g., based on depth, or just a default)
        # Original logic: 350 * abs(TargetDepth) + 6500
        eff_target_depth = self._get_effective_param('target_depth', project)
        if eff_target_depth is not None and eff_target_depth != 0:
            # Apply the formula (ensure it makes sense, maybe cap it)
            calculated_feedrate = round(350 * abs(eff_target_depth) + 6500, 0)
            feedrate = max(calculated_feedrate, 1000.0) # Ensure minimum feedrate
            logger.debug(f"MOP '{self.name}': Calculated CutFeedrate={feedrate} based on TargetDepth={eff_target_depth}")
            return feedrate
        else:
            # Fallback if TargetDepth isn't available or zero
            default_feedrate = 3000.0
            logger.warning(f"MOP '{self.name}': TargetDepth not set or zero. Using fallback CutFeedrate={default_feedrate}.")
            return default_feedrate

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: "CamBamProject", resolved_primitive_xml_ids: List[int]):
        """Adds common XML sub-elements shared by all MOP types."""
        ET.SubElement(mop_root_elem, "Name").text = self.name # Use MOP's intrinsic name
        ET.SubElement(mop_root_elem, "Tag").text = json.dumps({
            "user_id": self.user_identifier, "internal_id": str(self.internal_id),
        })

        # Resolve parameters using defaults if needed
        eff_target_depth = self._get_effective_param('target_depth', project)
        eff_depth_inc = self.depth_increment if self.depth_increment is not None else abs(eff_target_depth) if eff_target_depth is not None else None
        eff_spindle_speed = self._get_effective_param('spindle_speed', project)
        eff_tool_dia = self._get_effective_param('tool_diameter', project)
        eff_cut_feedrate = self._calculate_effective_cut_feedrate(project)

        # Helper to create elements with state attribute
        def add_param(parent, tag, value, mop_attr, state=None):
            if state is None:
                state = "Value" if getattr(self, mop_attr, None) is not None else "Default"
            # Handle optional values that might resolve to None
            text_value = self._format_mop_parameter(value)
            # A Default parameter must be resolved by CamBam's style system;
            # do not bake project/part effective values into a new MOP.
            if state == "Default" and getattr(self, mop_attr, None) is None:
                text_value = ""
            # Special case: TargetDepth needs a value even if default? Check CamBam output.
            # Assuming empty text is okay for unresolved optional defaults.
            if value is None and tag in ["TargetDepth", "DepthIncrement", "SpindleSpeed", "ToolDiameter"]:
                logger.warning(f"MOP '{self.name}': Parameter '{tag}' resolved to None. Writing empty element.")
                # CamBam might require a default value here, e.g., "0" or "-1"
                # text_value = "0" # Or handle based on specific parameter

            ET.SubElement(parent, tag, {"state": state}).text = text_value

        add_param(mop_root_elem, "TargetDepth", eff_target_depth, 'target_depth', "Value")
        add_param(mop_root_elem, "DepthIncrement", eff_depth_inc, 'depth_increment', "Value")
        add_param(mop_root_elem, "StockSurface", self.stock_surface, 'stock_surface', "Value")
        add_param(mop_root_elem, "RoughingClearance", self.roughing_clearance, 'roughing_clearance', "Value")
        add_param(mop_root_elem, "ClearancePlane", self.clearance_plane, 'clearance_plane', "Value")
        add_param(mop_root_elem, "SpindleDirection", self.spindle_direction, 'spindle_direction', "Value")
        add_param(mop_root_elem, "SpindleSpeed", eff_spindle_speed, 'spindle_speed', "Value")
        ET.SubElement(mop_root_elem, "SpindleRange", {"state": "Value"}).text = "0" # Default value
        add_param(mop_root_elem, "VelocityMode", self.velocity_mode, 'velocity_mode', "Value")
        add_param(mop_root_elem, "WorkPlane", self.work_plane, 'work_plane', "Value")
        add_param(mop_root_elem, "OptimisationMode", self.optimisation_mode, 'optimisation_mode', "Value")
        add_param(mop_root_elem, "ToolDiameter", eff_tool_dia, 'tool_diameter', "Value")
        add_param(mop_root_elem, "ToolNumber", self.tool_number, 'tool_number', "Value")
        add_param(mop_root_elem, "ToolProfile", self.tool_profile, 'tool_profile', "Value")
        add_param(mop_root_elem, "PlungeFeedrate", self.plunge_feedrate, 'plunge_feedrate')
        add_param(mop_root_elem, "CutFeedrate", eff_cut_feedrate, 'cut_feedrate', "Value")
        add_param(mop_root_elem, "MaxCrossoverDistance", self.max_crossover_distance, 'max_crossover_distance', "Value")
        add_param(mop_root_elem, "CustomMOPHeader", self.custom_mop_header, 'custom_mop_header', "Value")
        add_param(mop_root_elem, "CustomMOPFooter", self.custom_mop_footer, 'custom_mop_footer', "Value")

        # Add primitive references (using the resolved XML IDs passed by the writer)
        primitive_container = ET.SubElement(mop_root_elem, "primitive")
        if resolved_primitive_xml_ids:
            for pid in sorted(resolved_primitive_xml_ids): # Sort for consistency
                ET.SubElement(primitive_container, "prim").text = str(pid)
        # else: # CamBam seems to omit the <primitive> tag entirely if empty
            # pass

    def _add_lead_in_out_elements(self, parent_elem: ET.Element, lead_type: str = "Spiral", spiral_angle: float = 30.0, tangent_radius: float = 0.0, feedrate: float = 0.0):
        """Adds LeadInMove and LeadOutMove elements (common pattern)."""
        # Determine state (assume "Value" if explicitly called, could be refined)
        state = "Value"

        lead_in = ET.SubElement(parent_elem, "LeadInMove", {"state": state})
        ET.SubElement(lead_in, "LeadInType", {"state": state}).text = lead_type
        ET.SubElement(lead_in, "SpiralAngle", {"state": state}).text = str(spiral_angle)
        ET.SubElement(lead_in, "TangentRadius", {"state": state}).text = str(tangent_radius)
        ET.SubElement(lead_in, "LeadInFeedrate", {"state": state}).text = str(feedrate) # 0 usually means use CutFeedrate

        # Lead out often mirrors lead in settings in CamBam defaults
        lead_out = ET.SubElement(parent_elem, "LeadOutMove", {"state": state})
        ET.SubElement(lead_out, "LeadInType", {"state": state}).text = lead_type # Yes, uses "LeadInType" tag name
        ET.SubElement(lead_out, "SpiralAngle", {"state": state}).text = str(spiral_angle)
        ET.SubElement(lead_out, "TangentRadius", {"state": state}).text = str(tangent_radius)
        ET.SubElement(lead_out, "LeadInFeedrate", {"state": state}).text = str(feedrate)

# --- Concrete MOP Classes ---
# (Minimal changes: Update to_xml_element signature and call _add_common_mop_elements)

@dataclass
class ProfileMop(Mop):
    # Profile specific parameters
    stepover: float = 0.4 # Tool diameter fraction
    profile_side: str = 'Inside' # 'Inside', 'Outside'
    milling_direction: str = 'Conventional' # 'Conventional', 'Climb'
    collision_detection: bool = True
    corner_overcut: bool = False
    lead_in_type: str = 'Spiral' # 'None', 'Spiral', 'Tangent', 'Ramp'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0 # If > 0, amount for final pass
    cut_ordering: str = 'DepthFirst' # 'DepthFirst', 'LevelFirst'
    # Holding Tabs parameters
    tab_method: str = 'None' # 'None', 'Automatic', 'Manual' (Manual needs points)
    tab_width: float = 6.0
    tab_height: float = 1.5
    tab_min_tabs: int = 3
    tab_max_tabs: int = 3
    tab_distance: float = 40.0 # Approx distance between auto tabs
    tab_size_threshold: float = 4.0 # Min shape size for tabs
    tab_use_leadins: bool = False
    tab_style: str = 'Square' # 'Square', 'Triangle', 'Ramp'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            return native
        mop_elem = ET.Element("profile", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        # Add profile-specific elements
        state = "Value" # Assume explicit value for these for now
        ET.SubElement(mop_elem, "StepOver", {"state": state}).text = str(self.stepover)
        ET.SubElement(mop_elem, "InsideOutside", {"state": state}).text = self.profile_side
        ET.SubElement(mop_elem, "MillingDirection", {"state": state}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": state}).text = str(self.collision_detection).lower()
        ET.SubElement(mop_elem, "CornerOvercut", {"state": state}).text = str(self.corner_overcut).lower()

        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)

        fdi_state = "Value" if self.final_depth_increment is not None else "Default" # Can be optional
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering

        # Holding Tabs
        tabs = ET.SubElement(mop_elem, "HoldingTabs", {"state": state})
        ET.SubElement(tabs, "TabMethod").text = self.tab_method
        if self.tab_method != 'None':
            ET.SubElement(tabs, "Width").text = str(self.tab_width)
            ET.SubElement(tabs, "Height").text = str(self.tab_height)
            ET.SubElement(tabs, "MinimumTabs").text = str(self.tab_min_tabs)
            ET.SubElement(tabs, "MaximumTabs").text = str(self.tab_max_tabs)
            ET.SubElement(tabs, "TabDistance").text = str(self.tab_distance)
            ET.SubElement(tabs, "SizeThreshold").text = str(self.tab_size_threshold)
            ET.SubElement(tabs, "UseLeadIns").text = str(self.tab_use_leadins).lower()
            ET.SubElement(tabs, "TabStyle").text = self.tab_style
            # Manual tabs would need a <points> sub-element here if TabMethod='Manual'

        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem


@dataclass
class PocketMop(Mop):
    # Pocket specific parameters
    stepover: float = 0.4
    stepover_feedrate: str = 'Plunge Feedrate' # Name of feedrate to use for stepover moves
    milling_direction: str = 'Conventional'
    collision_detection: bool = True
    lead_in_type: str = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    region_fill_style: str = 'InsideOutsideOffsets' # 'InsideOutsideOffsets', 'HorizontalScanline', 'VerticalScanline'
    finish_stepover: float = 0.0 # If > 0, performs a finishing pass this far from edge
    finish_stepover_at_target_depth: bool = False # Apply finish pass only at final depth
    roughing_finishing: str = 'Roughing' # 'Roughing', 'Finishing', 'RoughFinish'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            return native
        mop_elem = ET.Element("pocket", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "StepOver", {"state": state}).text = str(self.stepover)
        ET.SubElement(mop_elem, "StepoverFeedrate", {"state": state}).text = self.stepover_feedrate
        ET.SubElement(mop_elem, "MillingDirection", {"state": state}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": state}).text = str(self.collision_detection).lower()

        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)

        fdi_state = "Value" if self.final_depth_increment is not None else "Default"
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering
        ET.SubElement(mop_elem, "RegionFillStyle", {"state": state}).text = self.region_fill_style
        ET.SubElement(mop_elem, "FinishStepover", {"state": state}).text = str(self.finish_stepover)
        ET.SubElement(mop_elem, "FinishStepoverAtTargetDepth", {"state": state}).text = str(self.finish_stepover_at_target_depth).lower()
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": state}).text = self.roughing_finishing
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # Usually calculated unless specified

        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem

@dataclass
class EngraveMop(Mop):
    # Engrave specific parameters
    roughing_finishing: str = 'Roughing' # Seems less relevant for Engrave? Default='Roughing'
    final_depth_increment: Optional[float] = 0.0 # Depth for final pass
    cut_ordering: str = 'DepthFirst'

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            return native
        mop_elem = ET.Element("engrave", {"Enabled": str(self.enabled).lower()})
        # Engrave uses ToolDiameter differently (often for simulation only)
        # We still add common params, including ToolDiameter resolution
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": state}).text = self.roughing_finishing

        fdi_state = "Value" if self.final_depth_increment is not None else "Default"
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": fdi_state}).text = str(self.final_depth_increment if self.final_depth_increment is not None else 0.0)

        ET.SubElement(mop_elem, "CutOrdering", {"state": state}).text = self.cut_ordering
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"}) # Usually follows shape order

        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem


@dataclass
class DrillMop(Mop):
    # Drill specific parameters
    drilling_method: str = 'CannedCycle' # 'CannedCycle', 'SpiralMill_CW', 'SpiralMill_CCW', 'CustomScript'
    # Parameters for CannedCycle
    peck_distance: float = 0.0 # If > 0, enables pecking (G83)
    retract_height: float = 5.0 # R plane for canned cycles
    dwell: float = 0.0 # Dwell time at bottom (ms)
    # Parameters for SpiralMill
    hole_diameter: Optional[float] = None # Required for SpiralMill if not using points
    drill_lead_out: bool = False
    spiral_flat_base: bool = True
    lead_out_length: float = 0.0
    # Parameter for CustomScript
    custom_script: str = ""

    def to_xml_element(self, project: "CamBamProject", resolved_primitive_xml_ids: List[int]) -> ET.Element:
        native = self._native_mop_element(project, resolved_primitive_xml_ids)
        if native is not None:
            return native
        mop_elem = ET.Element("drill", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project, resolved_primitive_xml_ids)

        state = "Value"
        ET.SubElement(mop_elem, "DrillingMethod", {"state": state}).text = self.drilling_method

        # Canned Cycle Params
        ET.SubElement(mop_elem, "PeckDistance", {"state": state}).text = str(self.peck_distance)
        ET.SubElement(mop_elem, "RetractHeight", {"state": state}).text = str(self.retract_height)
        ET.SubElement(mop_elem, "Dwell", {"state": state}).text = str(self.dwell)

        # Spiral Mill Params (conditionally add based on method)
        if self.drilling_method.startswith("SpiralMill"):
            hd_state = "Value" if self.hole_diameter is not None else "Default"
            # HoleDiameter is crucial for spiral milling if source isn't points
            if self.hole_diameter is None:
                logger.warning(f"Drill MOP '{self.name}' uses SpiralMill but HoleDiameter is not set. ToolDiameter will likely be used by CamBam.")
            # Write element even if None, CamBam might use ToolDiameter as fallback
            ET.SubElement(mop_elem, "HoleDiameter", {"state": hd_state}).text = str(self.hole_diameter if self.hole_diameter is not None else "")

            ET.SubElement(mop_elem, "DrillLeadOut", {"state": state}).text = str(self.drill_lead_out).lower()
            ET.SubElement(mop_elem, "SpiralFlatBase", {"state": state}).text = str(self.spiral_flat_base).lower()
            ET.SubElement(mop_elem, "LeadOutLength", {"state": state}).text = str(self.lead_out_length)

        # Custom Script Param
        cs_state = "Value" if self.custom_script else "Default"
        ET.SubElement(mop_elem, "CustomScript", {"state": cs_state}).text = self.custom_script

        # Other common Drill elements
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"})
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = "Roughing" # Drill is typically roughing

        self._apply_explicit_parameter_states(mop_elem)
        return mop_elem
