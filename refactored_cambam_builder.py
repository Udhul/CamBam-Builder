import xml.etree.ElementTree as ET
import os
import pickle
import uuid
import logging
import math
import numpy as np
import weakref
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Tuple, Union, Optional, Set, Any, Sequence, cast, Type, TypeVar
)

logger = logging.getLogger(__name__)
# Example basic config if not configured elsewhere:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Transformation Helpers (NumPy) ---

def identity_matrix() -> np.ndarray:
    """Returns a 3x3 identity matrix."""
    return np.identity(3, dtype=float)

def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """Returns a 3x3 translation matrix."""
    mat = np.identity(3, dtype=float)
    mat[0, 2] = dx
    mat[1, 2] = dy
    return mat

def rotation_matrix_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Returns a 3x3 rotation matrix around point (cx, cy)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Rotation around origin matrix
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    # If rotating around a point other than origin, translate to origin, rotate, translate back
    if not np.isclose(cx, 0.0) or not np.isclose(cy, 0.0):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        # Order: Translate to origin, rotate, translate back
        return from_origin @ rot_mat @ to_origin
    return rot_mat

def scaling_matrix(sx: float, sy: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Returns a 3x3 scaling/mirroring matrix around point (cx, cy)."""
    scale_mat = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)
    # If scaling around a point other than origin, translate to origin, scale, translate back
    if not np.isclose(cx, 0.0) or not np.isclose(cy, 0.0):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        # Order: Translate to origin, scale, translate back
        return from_origin @ scale_mat @ to_origin
    return scale_mat

def apply_transform(points: Sequence[Union[Tuple[float, float], np.ndarray]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    """Applies a 3x3 transformation matrix to a list of 2D points."""
    if not points: 
        return []
    # Convert points to homogeneous coordinates (nx3 matrix)
    points_h = np.ones((len(points), 3), dtype=float)
    for i, p in enumerate(points):
        points_h[i, 0] = p[0]
        points_h[i, 1] = p[1]

    # Apply transformation (matrix multiplication)
    transformed_points_h = (matrix @ points_h.T).T

    # Convert back to 2D points (list of tuples), handling potential perspective division
    result = []
    for row in transformed_points_h:
        w = row[2]
        if np.isclose(w, 0):
            logger.error("Transformation resulted in point at infinity (w=0). Skipping point.")
            continue
        result.append((row[0] / w, row[1] / w))
    if not np.allclose([row[2] for row in transformed_points_h if not np.isclose(row[2], 0)], 1.0):
         logger.warning("Non-affine transformation detected (perspective division != 1).")
    return result

def get_transformed_point(point: Tuple[float, float], matrix: np.ndarray) -> Tuple[float, float]:
    """Applies a 3x3 transformation matrix to a single 2D point."""
    res = apply_transform([point], matrix)
    if not res:
        raise ValueError(f"Transformation resulted in invalid point for input: {point}")
    return res[0]

# Tolerance for checking angles and matrix properties
MATRIX_TOLERANCE = 1e-6

def convert_3x3_to_4x4(mat3: np.ndarray) -> np.ndarray:
    return np.array([
        [mat3[0, 0], mat3[1, 0], 0, 0],
        [mat3[0, 1], mat3[1, 1], 0, 0],
        [0,          0,          1, 0],
        [mat3[0, 2], mat3[1, 2], 0, 1]
    ], dtype=float)

def matrix_to_string(mat: np.ndarray) -> str:
    return " ".join(str(x) for x in mat.flatten())

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
            return BoundingBox()
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return BoundingBox(min_x, min_y, max_x, max_y)

class XmlPrimitiveIdResolver:
    """Helper class to resolve Primitive UUIDs/groups to integer XML IDs."""
    def __init__(self,
                 uuid_to_xml_id: Dict[uuid.UUID, int],
                 primitive_groups: Dict[str, Set[uuid.UUID]],
                 all_primitives: Dict[uuid.UUID, 'Primitive']):
        self._uuid_to_xml_id = uuid_to_xml_id
        self._primitive_groups = primitive_groups
        self._all_primitives = all_primitives

    def resolve(self, pid_source: Union[str, List[uuid.UUID]]) -> List[int]:
        primitive_uuids: Set[uuid.UUID] = set()
        if isinstance(pid_source, str):
            group_uuids = self._primitive_groups.get(pid_source)
            if group_uuids is not None:
                primitive_uuids.update(group_uuids)
            else:
                logger.warning(f"MOP references non-existent group '{pid_source}'.")
        elif isinstance(pid_source, list):
            for uid in pid_source:
                if uid in self._all_primitives:
                    primitive_uuids.add(uid)
                else:
                    logger.warning(f"MOP references non-existent primitive UUID '{uid}'.")
        else:
            logger.warning(f"Invalid pid_source type: {type(pid_source)}. Expected str or List[uuid.UUID].")
            return []
        resolved_ids = []
        for uid in primitive_uuids:
            xml_id = self._uuid_to_xml_id.get(uid)
            if xml_id is not None:
                resolved_ids.append(xml_id)
            else:
                logger.error(f"Primitive UUID '{uid}' found in group/list but not in XML ID map during resolution.")
        return sorted(resolved_ids)

# --- Base Classes ---

Identifiable = Union[str, uuid.UUID, 'CamBamEntity']
T = TypeVar('T', bound='CamBamEntity')

@dataclass
class CamBamEntity:
    internal_id: uuid.UUID = field(default_factory=uuid.uuid4)
    user_identifier: str = ""

    def __post_init__(self):
        if not self.user_identifier:
            self.user_identifier = str(self.internal_id)

@dataclass
class Layer(CamBamEntity):
    """Represents a drawing layer."""
    color: str = 'Green'
    alpha: float = 1.0
    pen_width: float = 1.0
    visible: bool = True
    locked: bool = False
    primitive_ids: Set[uuid.UUID] = field(default_factory=set)
    _xml_objects_element: Optional[ET.Element] = field(default=None, repr=False, init=False)

    def to_xml_element(self) -> ET.Element:
        layer_elem = ET.Element("layer", {
            "name": self.user_identifier,
            "color": self.color,
            "alpha": str(self.alpha),
            "pen": str(self.pen_width),
            "visible": str(self.visible).lower(),
            "locked": str(self.locked).lower()
        })
        self._xml_objects_element = ET.SubElement(layer_elem, "objects")
        return layer_elem

@dataclass
class Part(CamBamEntity):
    """Represents a machining part (container for MOPs and stock definition)."""
    enabled: bool = True
    mop_ids: List[uuid.UUID] = field(default_factory=list)
    stock_thickness: float = 12.5
    stock_width: float = 1220.0
    stock_height: float = 2440.0
    stock_material: str = "MDF"
    stock_color: str = "210,180,140"
    machining_origin: Tuple[float, float] = (0.0, 0.0)
    default_tool_diameter: Optional[float] = None
    default_spindle_speed: Optional[int] = None
    _xml_machineops_element: Optional[ET.Element] = field(default=None, repr=False, init=False)

    def to_xml_element(self) -> ET.Element:
        part_elem = ET.Element("part", {
            "Name": self.user_identifier,
            "Enabled": str(self.enabled).lower()
        })
        self._xml_machineops_element = ET.SubElement(part_elem, "machineops")
        stock = ET.SubElement(part_elem, "Stock")
        ET.SubElement(stock, "PMin").text = f"0,0,{-self.stock_thickness}"
        ET.SubElement(stock, "PMax").text = f"{self.stock_width},{self.stock_height},0"
        ET.SubElement(stock, "Material").text = self.stock_material
        ET.SubElement(stock, "Color").text = self.stock_color
        ET.SubElement(part_elem, "MachiningOrigin").text = f"{self.machining_origin[0]},{self.machining_origin[1]}"
        effective_tool_diameter = self.default_tool_diameter
        if effective_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(effective_tool_diameter)
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill"
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "BasePoint").text = "0,0"
        ET.SubElement(nesting, "NestMethod").text = "None"
        return part_elem

# --- Primitive Base and Concrete Classes ---

@dataclass
class Primitive(CamBamEntity, ABC):
    """Abstract Base Class for geometric primitives with transformations."""
    layer_id: uuid.UUID = field(kw_only=True)
    groups: List[str] = field(default_factory=list)
    tags: str = ""
    # Transformation relative to parent (or global origin if no parent)
    effective_transform: np.ndarray = field(default_factory=identity_matrix)
    parent_primitive_id: Optional[uuid.UUID] = None

    # Cache control flags and cached values
    _abs_coords_dirty: bool = field(default=True, init=False, repr=False)
    _bbox_dirty: bool = field(default=True, init=False, repr=False)
    _total_transform_dirty: bool = field(default=True, init=False, repr=False)
    _cached_total_transform: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_abs_coords: Any = field(default=None, init=False, repr=False)
    _cached_bbox: Optional[BoundingBox] = field(default=None, init=False, repr=False)
    # Store direct children for faster invalidation propagation
    _child_primitive_ids: Set[uuid.UUID] = field(default_factory=set, init=False, repr=False)
    _project_ref: Optional[weakref.ReferenceType] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Ensure transform is a NumPy array, e.g., after pickling
        if not isinstance(self.effective_transform, np.ndarray):
            self.effective_transform = identity_matrix()
        if self.groups is None:
            self.groups = []

    # Remove weak references from state during pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if '_project_ref' in state:
            del state['_project_ref']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._project_ref = None

    def get_project(self) -> 'CamBamProject':
        if self._project_ref is None:
            raise ValueError(f"Primitive {self.user_identifier} is not registered with a project.")
        project = self._project_ref()
        if project is None:
            raise ValueError(f"Primitive {self.user_identifier} project reference is lost.")
        return project

    def _invalidate_caches(self, invalidate_children: bool = True):
        self._total_transform_dirty = True
        self._abs_coords_dirty = True
        self._bbox_dirty = True
        self._cached_total_transform = None
        self._cached_abs_coords = None
        self._cached_bbox = None
        if invalidate_children:
            proj = self.get_project()
            for child_id in self._child_primitive_ids:
                child = proj.get_primitive(child_id)
                if child:
                    child._invalidate_caches(invalidate_children=True)

    def _get_total_transform(self) -> np.ndarray:
        """Calculates and caches the total absolute transformation matrix."""
        if not self._total_transform_dirty and self._cached_total_transform is not None:
            return self._cached_total_transform
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
            parent = self.get_project().get_primitive(self.parent_primitive_id)
            if parent:
                parent_tf = parent._get_total_transform()
            else:
                logger.warning(f"Parent primitive {self.parent_primitive_id} not found for {self.internal_id}.")
        self._cached_total_transform = parent_tf @ self.effective_transform
        self._total_transform_dirty = False
        return self._cached_total_transform

    def get_absolute_coordinates(self) -> Any:
        """Calculates and caches absolute coordinates based on total transform."""
        if not self._abs_coords_dirty and self._cached_abs_coords is not None:
             return self._cached_abs_coords
        total_transform = self._get_total_transform()
        self._cached_abs_coords = self._calculate_absolute_geometry(total_transform)
        self._abs_coords_dirty = False
        return self._cached_abs_coords

    @abstractmethod
    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Any:
        """Subclasses implement this to calculate geometry with the final transform."""
        pass

    def get_bounding_box(self) -> BoundingBox:
        """Calculates and caches the bounding box based on absolute coordinates."""
        if not self._bbox_dirty and self._cached_bbox is not None:
            return self._cached_bbox
        abs_coords = self.get_absolute_coordinates()
        self._cached_bbox = self._calculate_bounding_box(abs_coords)
        self._bbox_dirty = False
        return self._cached_bbox

    @abstractmethod
    def _calculate_bounding_box(self, abs_coords: Any) -> BoundingBox:
        """Subclasses implement this based on absolute coords and the transform."""
        pass

    @abstractmethod
    def get_geometric_center(self) -> Tuple[float, float]:
        """Calculates the geometric center of the primitive in absolute coordinates."""
        pass

    def _set_effective_transform(self, matrix: np.ndarray):
        """Sets the effective transform and invalidates caches recursively."""
        if not np.array_equal(self.effective_transform, matrix):
             self.effective_transform = matrix.copy()
             self._invalidate_caches(invalidate_children=True)

    def _apply_transform_locally(self, local_transform_matrix: np.ndarray):
        """
        Applies a transformation matrix LOCALLY to the primitive's
        effective_transform (post-multiplication) and invalidates caches.
        M_eff_new = M_eff_old @ local_transform_matrix
        """
        new_eff_tf = self.effective_transform @ local_transform_matrix
        self._set_effective_transform(new_eff_tf)

    def _apply_transform_globally(self, global_transform_matrix: np.ndarray):
        """
        Applies a GLOBAL transformation matrix to the primitive, modifying its
        effective_transform relative to its parent.
        M_total_new = global_transform_matrix @ M_total_old
        M_eff_new = inv(M_parent) @ global_transform_matrix @ M_parent @ M_eff_old
        """
        # 1. Get current total and parent transforms
        current_total_tf = self._get_total_transform()
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
             parent = self.get_project().get_primitive(self.parent_primitive_id)
             if parent:
                 parent_tf = parent._get_total_transform()
        # 2. Calculate the change needed in the effective transform
        try:
            inv_parent_tf = np.linalg.inv(parent_tf)
        except np.linalg.LinAlgError:
            logger.error(f"Cannot apply global transform to {self.user_identifier}: Parent transform is singular. Applying locally instead.")
            self._apply_transform_locally(global_transform_matrix)
            return
        
        # This calculates the effective transform delta relative to the parent frame
        # It represents how much the local transform needs to change to achieve the global effect
        transform_delta = inv_parent_tf @ global_transform_matrix @ parent_tf
        # 3. Apply this delta to the current effective transform
        new_eff_tf = self.effective_transform @ transform_delta
        self._set_effective_transform(new_eff_tf)

    def _add_common_xml_attributes(self, element: ET.Element, xml_primitive_id: int):
        """Adds common ID and Tag sub-element."""
        element.set("id", str(xml_primitive_id))
        groups_str = ",".join(self.groups) if self.groups else "[]"
        tag_text = f"{groups_str}\n{self.tags}"
        ET.SubElement(element, "Tag").text = tag_text

    def _add_matrix_xml(self, element: ET.Element):
        """Adds the <mat> sub-element with the total transformation matrix."""
        total_tf = self._get_total_transform()
        mat4 = convert_3x3_to_4x4(total_tf)
        ET.SubElement(element, "mat", {"m": matrix_to_string(mat4)})

    @abstractmethod
    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        """Generates the XML element for the primitive, handling transformations."""
        pass

    @abstractmethod
    def bake_geometry(self, matrix: np.ndarray) -> None:
        """Applies the given transformation matrix to the primitive's geometry."""
        pass

    def apply_bake(self, additional_transform: np.ndarray):
        """Applies the given transformation matrix to the primitive's geometry and updates the effective transform."""
        # Combine the current effective transform with the additional transform.
        combined_transform = self.effective_transform @ additional_transform
        # Bake the geometry using the combined transform.
        self.bake_geometry(combined_transform)
        # Reset this primitive’s effective transform.
        self.effective_transform = identity_matrix()
        self._invalidate_caches(invalidate_children=False)
        proj = self.get_project()
        # Recursively update and bake each child.
        for child_id in self._child_primitive_ids:
            child = proj.get_primitive(child_id)
            if child:
                # Update the child's effective transform to include the parent's baked transform.
                child.effective_transform = combined_transform @ child.effective_transform
                # Recursively bake the child using its (now updated) effective transform.
                child.apply_bake(np.eye(3))

    def bake(self):
        """Applies all accumulated transformations to the geometry and resets the effective transform."""
        # Bake this primitive using its total transformation.
        total_tf = self._get_total_transform()
        self.bake_geometry(total_tf)
        self.effective_transform = identity_matrix()
        self._invalidate_caches(invalidate_children=False)
        proj = self.get_project()
        # Recursively bake all children, incorporating the parent's baked transform.
        for child_id in self._child_primitive_ids:
            child = proj.get_primitive(child_id)
            if child:
                child.effective_transform = total_tf @ child.effective_transform
                child.bake()

# --- Concrete Primitive Classes ---

@dataclass
class Pline(Primitive):
    relative_points: List[Union[Tuple[float, float], Tuple[float, float, float]]] = field(kw_only=True)
    closed: bool = False

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float, float]]:
        rel_pts_xy = [(p[0], p[1]) for p in self.relative_points]
        abs_pts_xy = apply_transform(rel_pts_xy, total_transform)
        abs_pts = []
        for i, (x, y) in enumerate(abs_pts_xy):
            bulge = self.relative_points[i][2] if len(self.relative_points[i]) > 2 else 0.0
            abs_pts.append((x, y, bulge))
        return abs_pts

    def _calculate_bounding_box(self, abs_coords: List[Tuple[float, float, float]]) -> BoundingBox:
        points_xy = [(p[0], p[1]) for p in abs_coords]
        if not points_xy:
            return BoundingBox()
        # TODO: Accurate bbox for transformed bulges (elliptical arcs). Very complex.
        return BoundingBox.from_points(points_xy)

    def get_geometric_center(self) -> Tuple[float, float]:
        abs_coords = self.get_absolute_coordinates()
        points_xy = [(p[0], p[1]) for p in abs_coords]
        if not points_xy:
            return (0.0, 0.0)
        # Use bbox center as approximation for pline center
        bbox = BoundingBox.from_points(points_xy)
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        else:
            return points_xy[0]

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        pline_elem = ET.Element("pline", {"Closed": str(self.closed).lower()})
        self._add_common_xml_attributes(pline_elem, xml_primitive_id)
        pts_elem = ET.SubElement(pline_elem, "pts")
        for pt in self.relative_points:
            bulge = pt[2] if len(pt) > 2 else 0
            ET.SubElement(pts_elem, "p", {"b": str(bulge)}).text = f"{pt[0]},{pt[1]},0"
        self._add_matrix_xml(pline_elem)
        return pline_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        new_points = []
        for pt in self.relative_points:
            x, y = get_transformed_point((pt[0], pt[1]), matrix)
            bulge = pt[2] if len(pt) > 2 else 0
            new_points.append((x, y, bulge))
        self.relative_points = new_points

@dataclass
class Circle(Primitive):
    relative_center: Tuple[float, float] = field(kw_only=True)
    diameter: float = field(kw_only=True)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Tuple[float, float]:
        return get_transformed_point(self.relative_center, total_transform)

    def get_geometric_center(self) -> Tuple[float, float]:
        return self.get_absolute_coordinates()

    def _calculate_bounding_box(self, abs_coords: Tuple[float, float]) -> BoundingBox:
        cx, cy = abs_coords
        scaled_radius = self.diameter / 2.0 * np.linalg.norm(self._get_total_transform()[0, 0:2])
        return BoundingBox(cx - scaled_radius, cy - scaled_radius, cx + scaled_radius, cy + scaled_radius)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
         circle_elem = ET.Element("circle", {
             "c": f"{self.relative_center[0]},{self.relative_center[1]},0",
             "d": str(self.diameter)
         })
         self._add_common_xml_attributes(circle_elem, xml_primitive_id)
         self._add_matrix_xml(circle_elem)
         return circle_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        self.relative_center = get_transformed_point(self.relative_center, matrix)
        sx = np.linalg.norm(matrix[0, 0:2])
        self.diameter *= sx

@dataclass
class Rect(Primitive):
    relative_corner: Tuple[float, float] = field(kw_only=True)
    width: float = field(kw_only=True)
    height: float = field(kw_only=True)

    def _get_relative_corners(self) -> List[Tuple[float, float]]:
        x0, y0 = self.relative_corner
        return [(x0, y0), (x0 + self.width, y0), (x0 + self.width, y0 + self.height), (x0, y0 + self.height)]

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        return apply_transform(self._get_relative_corners(), total_transform)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Calculate center of the relative rectangle and transform it
        rel_cx = self.relative_corner[0] + self.width / 2.0
        rel_cy = self.relative_corner[1] + self.height / 2.0
        return get_transformed_point((rel_cx, rel_cy), self._get_total_transform())

    def _calculate_bounding_box(self, abs_coords: List[Tuple[float, float]]) -> BoundingBox:
        if not abs_coords:
            return BoundingBox()
        return BoundingBox.from_points(abs_coords)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        rect_elem = ET.Element("rect", {
            "Closed": "true",
            "p": f"{self.relative_corner[0]},{self.relative_corner[1]},0",
            "w": str(self.width),
            "h": str(self.height)
        })
        self._add_common_xml_attributes(rect_elem, xml_primitive_id)
        self._add_matrix_xml(rect_elem)
        return rect_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        corners = self._get_relative_corners()
        transformed = apply_transform(corners, matrix)
        xs = [pt[0] for pt in transformed]
        ys = [pt[1] for pt in transformed]
        self.relative_corner = (min(xs), min(ys))
        self.width = max(xs) - min(xs)
        self.height = max(ys) - min(ys)

@dataclass
class Arc(Primitive):
    relative_center: Tuple[float, float] = field(kw_only=True)
    radius: float = field(kw_only=True)
    start_angle: float = field(kw_only=True)
    extent_angle: float = field(kw_only=True)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Dict[str, Any]:
        abs_center = get_transformed_point(self.relative_center, total_transform)
        return {"center": abs_center}

    def get_geometric_center(self) -> Tuple[float, float]:
        return get_transformed_point(self.relative_center, self._get_total_transform())

    def _calculate_bounding_box(self, abs_coords: Dict[str, Any]) -> BoundingBox:
        cx, cy = abs_coords["center"]
        scaled_radius = self.radius * np.linalg.norm(self._get_total_transform()[0, 0:2])
        return BoundingBox(cx - scaled_radius, cy - scaled_radius, cx + scaled_radius, cy + scaled_radius)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
         arc_elem = ET.Element("arc", {
             "p": f"{self.relative_center[0]},{self.relative_center[1]},0",
             "r": str(self.radius),
             "s": str(self.start_angle % 360),
             "w": str(self.extent_angle)
         })
         self._add_common_xml_attributes(arc_elem, xml_primitive_id)
         self._add_matrix_xml(arc_elem)
         return arc_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        self.relative_center = get_transformed_point(self.relative_center, matrix)
        sx = np.linalg.norm(matrix[0, 0:2])
        self.radius *= sx
        cos_angle = matrix[0, 0] / sx
        sin_angle = matrix[1, 0] / sx
        rot_angle = math.degrees(math.atan2(sin_angle, cos_angle))
        self.start_angle = (self.start_angle + rot_angle) % 360

@dataclass
class Points(Primitive):
    relative_points: List[Tuple[float, float]] = field(kw_only=True)

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> List[Tuple[float, float]]:
        return apply_transform(self.relative_points, total_transform)

    def get_geometric_center(self) -> Tuple[float, float]:
        # Use bbox center of absolute points
        abs_coords = self.get_absolute_coordinates()
        if not abs_coords:
            return (0.0, 0.0)
        bbox = BoundingBox.from_points(abs_coords)
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        else:
            return abs_coords[0]

    def _calculate_bounding_box(self, abs_coords: List[Tuple[float, float]]) -> BoundingBox:
        if not abs_coords:
            return BoundingBox()
        return BoundingBox.from_points(abs_coords)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        points_elem = ET.Element("points")
        self._add_common_xml_attributes(points_elem, xml_primitive_id)
        pts_elem = ET.SubElement(points_elem, "pts")
        for x, y in self.relative_points:
            ET.SubElement(pts_elem, "p").text = f"{x},{y}"
        self._add_matrix_xml(points_elem)
        return points_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        self.relative_points = apply_transform(self.relative_points, matrix)

@dataclass
class Text(Primitive):
    text_content: str = field(kw_only=True)
    relative_position: Tuple[float, float] = field(kw_only=True)
    height: float = 100.0
    font: str = 'Arial'
    style: str = ''
    line_spacing: float = 1.0
    align_horizontal: str = 'center'
    align_vertical: str = 'center'

    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Tuple[float, float]:
        return get_transformed_point(self.relative_position, total_transform)

    def get_geometric_center(self) -> Tuple[float, float]:
        return self.get_absolute_coordinates()

    def _calculate_bounding_box(self, abs_coords: Tuple[float, float]) -> BoundingBox:
        logger.warning(f"Bounding box for Text {self.user_identifier} is approximate.")
        px, py = abs_coords
        sx = np.linalg.norm(self._get_total_transform()[0, 0:2])
        sy = np.linalg.norm(self._get_total_transform()[1, 0:2])
        # Estimate based on scaled height and rough aspect ratio, ignore rotation
        scaled_height = self.height * sy
        est_width = scaled_height * len(self.text_content.split('\n')[0]) * 0.6
        return BoundingBox(px - est_width / 2, py - scaled_height / 2, px + est_width / 2, py + scaled_height / 2)

    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        text_elem = ET.Element("text", {
            "p1": f"{self.relative_position[0]},{self.relative_position[1]},0",
            "p2": f"{self.relative_position[0]},{self.relative_position[1]},0",
            "Height": str(self.height),
            "Font": self.font,
            "linespace": str(self.line_spacing),
            "align": f"{self.align_vertical},{self.align_horizontal}",
            "style": self.style
        })
        self._add_common_xml_attributes(text_elem, xml_primitive_id)
        text_elem.text = self.text_content
        self._add_matrix_xml(text_elem)
        return text_elem

    def bake_geometry(self, matrix: np.ndarray) -> None:
        self.relative_position = get_transformed_point(self.relative_position, matrix)
        sx = np.linalg.norm(matrix[0, 0:2])
        sy = np.linalg.norm(matrix[1, 0:2])
        self.height *= sy

# --- MOP Base and Concrete Classes ---

M = TypeVar('M', bound='Mop')

@dataclass
class Mop(CamBamEntity, ABC):
    """Abstract Base Class for Machine Operations."""
    part_id: uuid.UUID = field(kw_only=True)
    name: str = field(kw_only=True)
    pid_source: Union[str, List[uuid.UUID]] = field(kw_only=True)
    enabled: bool = True

    # Common MOP parameters
    target_depth: Optional[float] = None
    depth_increment: Optional[float] = None
    stock_surface: float = 0.0
    roughing_clearance: float = 0.0
    clearance_plane: float = 15.0
    spindle_direction: str = 'CW'
    spindle_speed: Optional[int] = None
    velocity_mode: str = 'ExactStop'
    work_plane: str = 'XY'
    optimisation_mode: str = 'Standard'
    tool_diameter: Optional[float] = None
    tool_number: int = 0
    tool_profile: str = 'EndMill'
    plunge_feedrate: float = 1000.0
    cut_feedrate: Optional[float] = None
    max_crossover_distance: float = 0.7
    custom_mop_header: str = ""
    custom_mop_footer: str = ""
    _resolved_xml_primitive_ids: List[int] = field(default_factory=list, init=False, repr=False)

    def resolve_xml_primitive_ids(self, resolver: XmlPrimitiveIdResolver) -> None:
        self._resolved_xml_primitive_ids = resolver.resolve(self.pid_source)
        if not self._resolved_xml_primitive_ids and self.pid_source:
             logger.warning(f"MOP '{self.name}' ({self.user_identifier}) resolved to zero primitives for source: {self.pid_source}")

    @abstractmethod
    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        pass

    def _get_effective_param(self, param_name: str, project: 'CamBamProject') -> Any:
        mop_value = getattr(self, param_name, None)
        if mop_value is not None:
            return mop_value
        part = project.get_part(self.part_id)
        if part:
            part_param_map = {'tool_diameter': 'default_tool_diameter', 'spindle_speed': 'default_spindle_speed'}
            part_attr = part_param_map.get(param_name)
            if part_attr:
                part_value = getattr(part, part_attr, None)
                if part_value is not None:
                    return part_value
        project_param_map = {'tool_diameter': 'default_tool_diameter'}
        project_attr = project_param_map.get(param_name)
        if project_attr:
            project_value = getattr(project, project_attr, None)
            if project_value is not None:
                return project_value
        return None

    def _calculate_cut_feedrate(self, project: 'CamBamProject') -> float:
        if self.cut_feedrate is not None:
            return self.cut_feedrate
        td = self._get_effective_param('target_depth', project)
        if td is None:
            logger.warning(f"MOP '{self.name}': TargetDepth not set. Using fallback cut feedrate 3000.")
            return 3000.0
        calculated_feedrate = round(350 * abs(td) + 6500, 0)
        return max(calculated_feedrate, 1000.0)

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: 'CamBamProject'):
        ET.SubElement(mop_root_elem, "Name").text = self.name
        td = self._get_effective_param('target_depth', project)
        ET.SubElement(mop_root_elem, "TargetDepth", {"state": "Value" if td is not None else "Default"}).text = str(td) if td is not None else None
        di = self.depth_increment if self.depth_increment is not None else abs(td) if td is not None else None
        ET.SubElement(mop_root_elem, "DepthIncrement", {"state": "Value" if di is not None else "Default"}).text = str(di) if di is not None else None
        ET.SubElement(mop_root_elem, "StockSurface", {"state": "Value"}).text = str(self.stock_surface)
        ET.SubElement(mop_root_elem, "RoughingClearance", {"state": "Value"}).text = str(self.roughing_clearance)
        ET.SubElement(mop_root_elem, "ClearancePlane", {"state": "Value"}).text = str(self.clearance_plane)
        ET.SubElement(mop_root_elem, "SpindleDirection", {"state": "Value"}).text = self.spindle_direction
        ss = self._get_effective_param('spindle_speed', project)
        ET.SubElement(mop_root_elem, "SpindleSpeed", {"state": "Value" if ss is not None else "Default"}).text = str(ss) if ss is not None else None
        ET.SubElement(mop_root_elem, "SpindleRange", {"state": "Value"}).text = "0"
        ET.SubElement(mop_root_elem, "VelocityMode", {"state": "Value"}).text = self.velocity_mode
        ET.SubElement(mop_root_elem, "WorkPlane", {"state": "Value"}).text = self.work_plane
        ET.SubElement(mop_root_elem, "OptimisationMode", {"state": "Value"}).text = self.optimisation_mode
        tool_dia = self._get_effective_param('tool_diameter', project)
        ET.SubElement(mop_root_elem, "ToolDiameter", {"state": "Value" if tool_dia is not None else "Default"}).text = str(tool_dia) if tool_dia is not None else None
        ET.SubElement(mop_root_elem, "ToolNumber", {"state": "Value"}).text = str(self.tool_number)
        ET.SubElement(mop_root_elem, "ToolProfile", {"state": "Value"}).text = self.tool_profile
        ET.SubElement(mop_root_elem, "PlungeFeedrate", {"state": "Value"}).text = str(self.plunge_feedrate)
        cf = self._calculate_cut_feedrate(project)
        ET.SubElement(mop_root_elem, "CutFeedrate", {"state": "Value"}).text = str(cf)
        ET.SubElement(mop_root_elem, "MaxCrossoverDistance", {"state": "Value"}).text = str(self.max_crossover_distance)
        ET.SubElement(mop_root_elem, "CustomMOPHeader", {"state": "Value"}).text = self.custom_mop_header
        ET.SubElement(mop_root_elem, "CustomMOPFooter", {"state": "Value"}).text = self.custom_mop_footer
        primitive_container = ET.SubElement(mop_root_elem, "primitive")
        if self._resolved_xml_primitive_ids:
            for pid in self._resolved_xml_primitive_ids:
                prim_elem = ET.SubElement(primitive_container, "prim")
                prim_elem.text = str(pid)

    def _add_lead_in_out_elements(self, parent_elem: ET.Element, lead_type: str = "Spiral", spiral_angle: float = 30.0, tangent_radius: float = 0.0, feedrate: float = 0.0):
        lead_in = ET.SubElement(parent_elem, "LeadInMove", {"state": "Value"})
        ET.SubElement(lead_in, "LeadInType").text = lead_type
        ET.SubElement(lead_in, "SpiralAngle").text = str(spiral_angle)
        ET.SubElement(lead_in, "TangentRadius").text = str(tangent_radius)
        ET.SubElement(lead_in, "LeadInFeedrate").text = str(feedrate)
        lead_out = ET.SubElement(parent_elem, "LeadOutMove", {"state": "Value"})
        ET.SubElement(lead_out, "LeadInType").text = lead_type
        ET.SubElement(lead_out, "SpiralAngle").text = str(spiral_angle)
        ET.SubElement(lead_out, "TangentRadius").text = str(tangent_radius)
        ET.SubElement(lead_out, "LeadInFeedrate").text = str(feedrate)

@dataclass
class ProfileMop(Mop):
    stepover: float = 0.4
    profile_side: str = 'Inside'
    milling_direction: str = 'Conventional'
    collision_detection: bool = True
    corner_overcut: bool = False
    lead_in_type: str = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    tab_method: str = 'None'
    tab_width: float = 6.0
    tab_height: float = 1.5
    tab_min_tabs: int = 3
    tab_max_tabs: int = 3
    tab_distance: float = 40.0
    tab_size_threshold: float = 4.0
    tab_use_leadins: bool = False
    tab_style: str = 'Square'

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element("profile", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)
        ET.SubElement(mop_elem, "StepOver", {"state": "Value"}).text = str(self.stepover)
        ET.SubElement(mop_elem, "InsideOutside", {"state": "Value"}).text = self.profile_side
        ET.SubElement(mop_elem, "MillingDirection", {"state": "Value"}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": "Value"}).text = str(self.collision_detection).lower()
        ET.SubElement(mop_elem, "CornerOvercut", {"state": "Value"}).text = str(self.corner_overcut).lower()
        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(self.final_depth_increment)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering
        tabs = ET.SubElement(mop_elem, "HoldingTabs", {"state": "Value"})
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
        return mop_elem

@dataclass
class PocketMop(Mop):
    stepover: float = 0.4
    stepover_feedrate: str = 'Plunge Feedrate'
    milling_direction: str = 'Conventional'
    collision_detection: bool = True
    lead_in_type: str = 'Spiral'
    lead_in_spiral_angle: float = 30.0
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'
    region_fill_style: str = 'InsideOutsideOffsets'
    finish_stepover: float = 0.0
    finish_stepover_at_target_depth: bool = False
    roughing_finishing: str = 'Roughing'

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element("pocket", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)
        ET.SubElement(mop_elem, "StepOver", {"state": "Value"}).text = str(self.stepover)
        ET.SubElement(mop_elem, "StepoverFeedrate", {"state": "Value"}).text = self.stepover_feedrate
        ET.SubElement(mop_elem, "MillingDirection", {"state": "Value"}).text = self.milling_direction
        ET.SubElement(mop_elem, "CollisionDetection", {"state": "Value"}).text = str(self.collision_detection).lower()
        self._add_lead_in_out_elements(mop_elem, lead_type=self.lead_in_type, spiral_angle=self.lead_in_spiral_angle)
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(self.final_depth_increment)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering
        ET.SubElement(mop_elem, "RegionFillStyle", {"state": "Value"}).text = self.region_fill_style
        ET.SubElement(mop_elem, "FinishStepover", {"state": "Value"}).text = str(self.finish_stepover)
        ET.SubElement(mop_elem, "FinishStepoverAtTargetDepth", {"state": "Value"}).text = str(self.finish_stepover_at_target_depth).lower()
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = self.roughing_finishing
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"})
        return mop_elem

@dataclass
class EngraveMop(Mop):
    roughing_finishing: str = 'Roughing'
    final_depth_increment: Optional[float] = 0.0
    cut_ordering: str = 'DepthFirst'

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element("engrave", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = self.roughing_finishing
        ET.SubElement(mop_elem, "FinalDepthIncrement", {"state": "Value"}).text = str(self.final_depth_increment)
        ET.SubElement(mop_elem, "CutOrdering", {"state": "Value"}).text = self.cut_ordering
        ET.SubElement(mop_elem, "StartPoint", {"state": "Value"})
        return mop_elem

@dataclass
class DrillMop(Mop):
    drilling_method: str = 'CannedCycle'
    hole_diameter: Optional[float] = None
    drill_lead_out: bool = False
    spiral_flat_base: bool = True
    lead_out_length: float = 0.0
    peck_distance: float = 0.0
    retract_height: float = 5.0
    dwell: float = 0.0
    custom_script: str = ""

    def to_xml_element(self, project: 'CamBamProject') -> ET.Element:
        mop_elem = ET.Element("drill", {"Enabled": str(self.enabled).lower()})
        self._add_common_mop_elements(mop_elem, project)
        ET.SubElement(mop_elem, "DrillingMethod", {"state": "Value"}).text = self.drilling_method
        if self.drilling_method.startswith("SpiralMill"):
            ET.SubElement(mop_elem, "DrillLeadOut", {"state": "Value"}).text = str(self.drill_lead_out).lower()
            ET.SubElement(mop_elem, "SpiralFlatBase", {"state": "Value"}).text = str(self.spiral_flat_base).lower()
            ET.SubElement(mop_elem, "LeadOutLength", {"state": "Value"}).text = str(self.lead_out_length)
            hd = self.hole_diameter
            ET.SubElement(mop_elem, "HoleDiameter", {"state": "Value" if hd is not None else "Default"}).text = str(hd) if hd is not None else None
            if hd is None:
                logger.warning(f"Drill MOP '{self.name}' uses SpiralMill but HoleDiameter is not set.")
        ET.SubElement(mop_elem, "PeckDistance", {"state": "Value"}).text = str(self.peck_distance)
        ET.SubElement(mop_elem, "RetractHeight", {"state": "Value"}).text = str(self.retract_height)
        ET.SubElement(mop_elem, "Dwell", {"state": "Value"}).text = str(self.dwell)
        ET.SubElement(mop_elem, "CustomScript", {"state": "Default" if not self.custom_script else "Value"}).text = self.custom_script
        ET.SubElement(mop_elem, "StartPoint", {"state": "Default"})
        ET.SubElement(mop_elem, "RoughingFinishing", {"state": "Value"}).text = "Roughing"
        return mop_elem

# --- CamBamProject Class ---

T = TypeVar('T', bound=CamBamEntity)

class CamBamProject:
    """
    Manages the creation, manipulation, and generation of a CamBam project file
    using transformations, part-level stock/defaults, and a drawing cursor.
    """
    def __init__(self, project_name: str, default_tool_diameter: float = 6.0):
        self.project_name: str = project_name
        self.default_tool_diameter: float = default_tool_diameter
        self._primitives: Dict[uuid.UUID, Primitive] = {}
        self._layers: Dict[uuid.UUID, Layer] = {}
        self._parts: Dict[uuid.UUID, Part] = {}
        self._mops: Dict[uuid.UUID, Mop] = {}
        self._layer_order: List[uuid.UUID] = []
        self._part_order: List[uuid.UUID] = []
        self._identifier_registry: Dict[str, uuid.UUID] = {}
        self._primitive_groups: Dict[str, Set[uuid.UUID]] = {"unassigned": set()}
        self._bbox_dirty: bool = True
        self._cached_bbox: Optional[BoundingBox] = None
        # Transformation context stack
        self._transform_stack: List[np.ndarray] = [identity_matrix()]
        # Drawing cursor relative to global origin
        self._cursor: Tuple[float, float] = (0.0, 0.0)

        logger.info(f"Initialized CamBamProject: {self.project_name}")

    def _resolve_identifier(self, identifier: Identifiable, entity_type: Optional[Type[T]] = None) -> Optional[uuid.UUID]:
        """Resolves a string identifier, UUID, or object instance to an internal UUID."""
        target_uuid: Optional[uuid.UUID] = None
        if isinstance(identifier, uuid.UUID):
            target_uuid = identifier
        elif isinstance(identifier, str):
            target_uuid = self._identifier_registry.get(identifier)
        elif isinstance(identifier, CamBamEntity):
            target_uuid = identifier.internal_id
        else:
            logger.error(f"Invalid identifier type: {type(identifier)}")
            return None
        if target_uuid is None:
            return None
        if entity_type:
            entity: Optional[CamBamEntity] = None
            if target_uuid in self._primitives: entity = self._primitives[target_uuid]
            elif target_uuid in self._layers: entity = self._layers[target_uuid]
            elif target_uuid in self._parts: entity = self._parts[target_uuid]
            elif target_uuid in self._mops: entity = self._mops[target_uuid]
            else: return None # UUID known but object missing? Should not happen.
            if not isinstance(entity, entity_type):
                logger.debug(f"Identifier '{identifier}' resolved to UUID {target_uuid} but type mismatch (Expected {entity_type.__name__}, Got {type(entity).__name__}).")
                return None # Found, but wrong type
        return target_uuid

    # --- Registration/Unregistration (Modified to handle parent/child links) ---

    def _register_entity(self, entity: T, registry: Dict[uuid.UUID, T]) -> None:
        # Simplified: Assume identifier uniqueness check happens before calling if needed
        if entity.internal_id in registry: return # Avoid re-registering same object
        registry[entity.internal_id] = entity
        if entity.user_identifier:
            # Only add if not already present or points to the same entity
            if entity.user_identifier not in self._identifier_registry or \
               self._identifier_registry[entity.user_identifier] == entity.internal_id:
                self._identifier_registry[entity.user_identifier] = entity.internal_id
            else:
                logger.warning(f"Cannot register identifier '{entity.user_identifier}' as it points to a different entity.")

    def _unregister_entity(self, entity_id: uuid.UUID, registry: Dict[uuid.UUID, T]) -> Optional[T]:
        entity = registry.pop(entity_id, None)
        if entity:
            if entity.user_identifier and entity.user_identifier in self._identifier_registry:
                if self._identifier_registry[entity.user_identifier] == entity_id:
                    self._identifier_registry.pop(entity.user_identifier, None)
            # If Primitive, handle parent/child cleanup
            if isinstance(entity, Primitive):
                # Remove from parent's child list
                if entity.parent_primitive_id:
                    parent = self.get_primitive(entity.parent_primitive_id)
                    if parent:
                        parent._child_primitive_ids.discard(entity_id)
                # Clear parent link for any children of the removed primitive
                for child_id in list(entity._child_primitive_ids):
                    child = self.get_primitive(child_id)
                    if child:
                        child.parent_primitive_id = None
                        # Invalidate child cache as its total transform changes
                        child._invalidate_caches()
        return entity

    def _register_primitive(self, primitive: Primitive):
        """Registers primitive, updates layer/groups, and manages parent/child links."""
        self._register_entity(primitive, self._primitives)
        primitive._project_ref = weakref.ref(self)
        self._bbox_dirty = True
        # Add to Layer
        layer = self.get_layer(primitive.layer_id)
        if layer:
            layer.primitive_ids.add(primitive.internal_id)
        else:
            logger.error(f"Primitive {primitive.user_identifier} references non-existent layer {primitive.layer_id}.")
        # Add to Groups
        target_groups = self._primitive_groups.setdefault("unassigned", set())
        if primitive.groups:
            for group_name in primitive.groups:
                self._primitive_groups.setdefault(group_name, set()).add(primitive.internal_id)
            target_groups.discard(primitive.internal_id)
        else:
            target_groups.add(primitive.internal_id)
        # Update Parent's Child List
        if primitive.parent_primitive_id:
            parent = self.get_primitive(primitive.parent_primitive_id)
            if parent:
                parent._child_primitive_ids.add(primitive.internal_id)
                primitive._invalidate_caches() # Invalidate self due to new parent link
            else: # Parent ID set but parent not found - clear the link
                logger.warning(f"Primitive {primitive.user_identifier} has invalid parent ID {primitive.parent_primitive_id}. Clearing parent link.")
                primitive.parent_primitive_id = None

    def _unregister_primitive(self, primitive_id: uuid.UUID) -> Optional[Primitive]:
        """Unregisters primitive, removing from layer/groups and handling parent/child links."""
        primitive = self.get_primitive(primitive_id)
        if not primitive:
            return None
        # Remove from Groups
        for group_name in list(self._primitive_groups.keys()):
            self._primitive_groups[group_name].discard(primitive_id)
        # Remove from Layer
        layer = self.get_layer(primitive.layer_id)
        if layer:
            layer.primitive_ids.discard(primitive_id)
        # Unregister (handles parent/child cleanup via _unregister_entity)
        unregistered_primitive = self._unregister_entity(primitive_id, self._primitives)
        if unregistered_primitive:
            self._bbox_dirty = True
        return unregistered_primitive

    def _register_mop(self, mop: Mop):
        self._register_entity(mop, self._mops)
        # Ordering is handled by add_mop method

    def _unregister_mop(self, mop_id: uuid.UUID) -> Optional[Mop]:
        mop = self._unregister_entity(mop_id, self._mops)
        if mop:
            part = self.get_part(mop.part_id)
            if part:
                try:
                    part.mop_ids.remove(mop_id)
                except ValueError:
                    pass
        return mop

    def _find_insert_position(self, target_identifier: Optional[Identifiable],
                              order_list: List[uuid.UUID],
                              place_last: bool) -> int:
        if target_identifier is None:
            return len(order_list) if place_last else 0
        target_uuid = self._resolve_identifier(target_identifier)
        if target_uuid is None or target_uuid not in order_list:
            logger.warning(f"Target identifier '{target_identifier}' not found for ordering. Appending.")
            return len(order_list)
        try:
            target_index = order_list.index(target_uuid)
            return target_index + 1 if place_last else target_index
        except ValueError:
            logger.error(f"Target UUID '{target_uuid}' in registry but not order list? Appending.")
            return len(order_list)

    # --- Transformation Context and Cursor Management ---

    @property
    def current_transform(self) -> np.ndarray:
        return self._transform_stack[-1]

    def set_transform(self, matrix: np.ndarray) -> None:
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            raise TypeError("Transform must be a 3x3 NumPy array.")
        if not self._transform_stack:
            self._transform_stack.append(identity_matrix())
        self._transform_stack[-1] = matrix.copy()

    def push_transform(self, matrix: Optional[np.ndarray] = None) -> None:
        if matrix is not None and (not isinstance(matrix, np.ndarray)):
            raise TypeError("Transform must be None or a 3x3 NumPy array.")
        current_top = self.current_transform
        new_top = current_top.copy() if matrix is None else current_top @ matrix
        self._transform_stack.append(new_top)

    def pop_transform(self) -> np.ndarray:
        if len(self._transform_stack) > 1:
            return self._transform_stack.pop()
        else:
            logger.warning("Cannot pop the base identity transform.")
            return self._transform_stack[0]

    def get_entity_by_identifier(self, identifier: str) -> Optional[CamBamEntity]:
        entity_uuid = self._resolve_identifier(identifier)
        if not entity_uuid:
            return None
        return (self._primitives.get(entity_uuid) or
                self._layers.get(entity_uuid) or
                self._parts.get(entity_uuid) or
                self._mops.get(entity_uuid))

    def set_cursor(self, x: float, y: float) -> None:
        """Sets the drawing cursor position."""
        self._cursor = (x, y)
        logger.debug(f"Cursor set to: {self._cursor}")

    def reset_cursor(self) -> None:
        """Resets the drawing cursor to the global origin (0, 0)."""
        self.set_cursor(0.0, 0.0)

    # --- Entity Accessors ---
    def get_primitive(self, primitive_id: uuid.UUID) -> Optional[Primitive]:
        return self._primitives.get(primitive_id)

    def get_layer(self, layer_id: uuid.UUID) -> Optional[Layer]:
        return self._layers.get(layer_id)

    def get_part(self, part_id: uuid.UUID) -> Optional[Part]:
        return self._parts.get(part_id)

    def get_mop(self, mop_id: uuid.UUID) -> Optional[Mop]:
        return self._mops.get(mop_id)

    def get_layer_by_identifier(self, identifier: Identifiable) -> Optional[Layer]:
        layer_uuid = self._resolve_identifier(identifier, Layer)
        return self._layers.get(layer_uuid) if layer_uuid else None

    def get_part_by_identifier(self, identifier: Identifiable) -> Optional[Part]:
        part_uuid = self._resolve_identifier(identifier, Part)
        return self._parts.get(part_uuid) if part_uuid else None

    def get_mop_by_identifier(self, identifier: Identifiable, part_context: Optional[Identifiable] = None) -> Optional[Mop]:
        mop_uuid = self._resolve_identifier(identifier, Mop)
        if not mop_uuid:
            return None
        mop = self._mops.get(mop_uuid)
        if mop and part_context:
            part_uuid = self._resolve_identifier(part_context, Part)
            if mop.part_id != part_uuid:
                return None
        return mop

    def list_primitives(self) -> List[Primitive]:
        return list(self._primitives.values())

    def list_layers(self) -> List[Layer]:
        return [self._layers[uid] for uid in self._layer_order if uid in self._layers]

    def list_parts(self) -> List[Part]:
        return [self._parts[uid] for uid in self._part_order if uid in self._parts]

    def list_mops(self, part_identifier: Optional[Identifiable] = None) -> List[Mop]:
        if part_identifier:
            part = self.get_part_by_identifier(part_identifier)
            if not part:
                return []
            return [self._mops[uid] for uid in part.mop_ids if uid in self._mops]
        else:
            return list(self._mops.values())

    def list_groups(self) -> List[str]:
        return list(self._primitive_groups.keys())

    def get_primitives_in_group(self, group_name: str) -> List[Primitive]:
        uuids = self._primitive_groups.get(group_name, set())
        return [self._primitives[uid] for uid in uuids if uid in self._primitives]

    # --- Entity Creation Methods ---

    def add_layer(self, identifier: str, color: str = 'Green', alpha: float = 1.0, pen_width: float = 1.0,
                  visible: bool = True, locked: bool = False,
                  target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Layer:
        existing_layer = self.get_layer_by_identifier(identifier)
        if existing_layer:
            existing_layer.color, existing_layer.alpha, existing_layer.pen_width = color, alpha, pen_width
            existing_layer.visible, existing_layer.locked = visible, locked
            self._register_entity(existing_layer, self._layers)
            return existing_layer
        else:
            new_layer = Layer(user_identifier=identifier, color=color, alpha=alpha, pen_width=pen_width, visible=visible, locked=locked)
            self._register_entity(new_layer, self._layers)
            insert_pos = self._find_insert_position(target_identifier, self._layer_order, place_last)
            self._layer_order.insert(insert_pos, new_layer.internal_id)
            return new_layer

    def add_part(self, identifier: str, enabled: bool = True,
                 stock_thickness: float = 12.5, stock_width: float = 1220.0, stock_height: float = 2440.0,
                 stock_material: str = "MDF", stock_color: str = "210,180,140",
                 machining_origin: Tuple[float, float] = (0.0, 0.0),
                 default_tool_diameter: Optional[float] = None,
                 default_spindle_speed: Optional[int] = None,
                 target_identifier: Optional[Identifiable] = None, place_last: bool = True) -> Part:
        existing_part = self.get_part_by_identifier(identifier)
        if existing_part:
            existing_part.enabled = enabled
            existing_part.stock_thickness, existing_part.stock_width, existing_part.stock_height = stock_thickness, stock_width, stock_height
            existing_part.stock_material, existing_part.stock_color = stock_material, stock_color
            existing_part.machining_origin = machining_origin
            existing_part.default_tool_diameter = default_tool_diameter
            existing_part.default_spindle_speed = default_spindle_speed
            self._register_entity(existing_part, self._parts)
            return existing_part
        else:
            new_part = Part(user_identifier=identifier, enabled=enabled, stock_thickness=stock_thickness,
                            stock_width=stock_width, stock_height=stock_height, stock_material=stock_material,
                            stock_color=stock_color, machining_origin=machining_origin,
                            default_tool_diameter=default_tool_diameter, default_spindle_speed=default_spindle_speed)
            self._register_entity(new_part, self._parts)
            insert_pos = self._find_insert_position(target_identifier, self._part_order, place_last)
            self._part_order.insert(insert_pos, new_part.internal_id)
            return new_part

    def _add_primitive_internal(self, PrimitiveClass: Type[Primitive],
                                layer_identifier: Identifiable,
                                identifier: Optional[str] = None,
                                groups: Optional[List[str]] = None,
                                tags: str = "",
                                parent_identifier: Optional[Identifiable] = None,
                                **kwargs) -> Primitive:
        layer = self.get_layer_by_identifier(layer_identifier)
        if not layer:
            if isinstance(layer_identifier, str):
                layer = self.add_layer(layer_identifier)
            else:
                raise ValueError(f"Layer identifier '{layer_identifier}' not found.")
        parent_uuid = self._resolve_identifier(parent_identifier, Primitive) if parent_identifier else None
        if parent_identifier and not parent_uuid:
            raise ValueError(f"Parent primitive identifier '{parent_identifier}' not found.")
        prim_id = identifier if identifier else str(uuid.uuid4())
        groups = groups if groups is not None else []
        # Calculate initial transform: Current context combined with cursor offset
        cursor_tf = translation_matrix(self._cursor[0], self._cursor[1])
        initial_eff_tf = self.current_transform @ cursor_tf
        primitive = PrimitiveClass(user_identifier=prim_id, layer_id=layer.internal_id,
                                     groups=groups, tags=tags,
                                     effective_transform=initial_eff_tf,
                                     parent_primitive_id=parent_uuid,
                                     **kwargs)
        self._register_primitive(primitive) # Handles adding to layer/groups/parent
        return primitive

    def add_pline(self, layer: Identifiable, points: List[Union[Tuple[float, float], Tuple[float, float, float]]],
                  closed: bool = False, identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                  tags: str = "", parent: Optional[Identifiable] = None) -> Pline:
        prim = self._add_primitive_internal(Pline, layer, identifier, groups, tags, parent, relative_points=points, closed=closed)
        return cast(Pline, prim)

    def add_circle(self, layer: Identifiable, center: Tuple[float, float], diameter: float,
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, tags: str = "",
                   parent: Optional[Identifiable] = None) -> Circle:
        prim = self._add_primitive_internal(Circle, layer, identifier, groups, tags, parent, relative_center=center, diameter=diameter)
        return cast(Circle, prim)

    def add_rect(self, layer: Identifiable, corner: Tuple[float, float] = (0,0), width: float = 0, height: float = 0,
                 identifier: Optional[str] = None, groups: Optional[List[str]] = None, tags: str = "",
                 parent: Optional[Identifiable] = None) -> Rect:
        prim = self._add_primitive_internal(Rect, layer, identifier, groups, tags, parent, relative_corner=corner, width=width, height=height)
        return cast(Rect, prim)

    def add_arc(self, layer: Identifiable, center: Tuple[float, float], radius: float, start_angle: float, extent_angle: float,
                identifier: Optional[str] = None, groups: Optional[List[str]] = None, tags: str = "",
                parent: Optional[Identifiable] = None) -> Arc:
        prim = self._add_primitive_internal(Arc, layer, identifier, groups, tags, parent, relative_center=center, radius=radius, start_angle=start_angle, extent_angle=extent_angle)
        return cast(Arc, prim)

    def add_points(self, layer: Identifiable, points: List[Tuple[float, float]],
                   identifier: Optional[str] = None, groups: Optional[List[str]] = None, tags: str = "",
                   parent: Optional[Identifiable] = None) -> Points:
        prim = self._add_primitive_internal(Points, layer, identifier, groups, tags, parent, relative_points=points)
        return cast(Points, prim)

    def add_text(self, layer: Identifiable, text: str, position: Tuple[float, float], height: float = 100.0,
                 font: str = 'Arial', style: str = '', line_spacing: float = 1.0, align_horizontal: str = 'center',
                 align_vertical: str = 'center', identifier: Optional[str] = None, groups: Optional[List[str]] = None,
                 tags: str = "", parent: Optional[Identifiable] = None) -> Text:
        prim = self._add_primitive_internal(Text, layer, identifier, groups, tags, parent, text_content=text, relative_position=position, height=height, font=font, style=style, line_spacing=line_spacing, align_horizontal=align_horizontal, align_vertical=align_vertical)
        return cast(Text, prim)

    # --- MOP Creation Methods ---
    def _add_mop_internal(self, MopClass: Type[M], part_identifier: Identifiable,
                          pid_source: Union[str, List[Identifiable]],
                          name: str, identifier: Optional[str] = None,
                          target_mop_identifier: Optional[Identifiable] = None, place_last: bool = True,
                          **kwargs) -> M:
         part = self.get_part_by_identifier(part_identifier)
         if not part:
             raise ValueError(f"Part identifier '{part_identifier}' not found.")
         if isinstance(pid_source, str):
             resolved_pid_source = pid_source
         elif isinstance(pid_source, list):
             resolved_pid_list: List[uuid.UUID] = []
             for item in pid_source:
                 pid_uuid = self._resolve_identifier(item, Primitive)
                 if pid_uuid:
                     resolved_pid_list.append(pid_uuid)
                 else:
                     logger.warning(f"Could not resolve primitive identifier '{item}' for MOP '{name}'. Skipping.")
             resolved_pid_source = resolved_pid_list
         else:
             raise TypeError(f"Invalid pid_source type for MOP '{name}': {type(pid_source)}")
         mop_id = identifier if identifier else f"{name}_{uuid.uuid4().hex[:6]}"
         mop = MopClass(user_identifier=mop_id, part_id=part.internal_id,
                         pid_source=resolved_pid_source,
                         name=name, **kwargs)
         self._register_mop(mop)
         insert_pos = self._find_insert_position(target_mop_identifier, part.mop_ids, place_last)
         part.mop_ids.insert(insert_pos, mop.internal_id)
         return mop

    # Concrete MOP adders
    def add_profile_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new profile', identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> ProfileMop:
         mop = self._add_mop_internal(ProfileMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
         return cast(ProfileMop, mop)

    def add_pocket_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new pocket', identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> PocketMop:
         mop = self._add_mop_internal(PocketMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
         return cast(PocketMop, mop)

    def add_engrave_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new engrave', identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> EngraveMop:
         mop = self._add_mop_internal(EngraveMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
         return cast(EngraveMop, mop)

    def add_drill_mop(self, part: Identifiable, pid_source: Union[str, List[Identifiable]], name: str = 'new drill', identifier: Optional[str] = None, target_mop: Optional[Identifiable] = None, place_last: bool = True, **kwargs) -> DrillMop:
         mop = self._add_mop_internal(DrillMop, part, pid_source, name, identifier, target_mop, place_last, **kwargs)
         return cast(DrillMop, mop)

    # --- Transformation Methods ---

    def transform_primitive_locally(self, primitive_identifier: Identifiable, matrix: np.ndarray, bake: bool = False) -> bool:
         """Applies a transformation matrix LOCALLY (post-multiplied) to a primitive."""
         primitive = self._resolve_primitive(primitive_identifier)
         if not primitive:
             return False
         if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
             logger.error(f"Invalid local transform matrix for {primitive.user_identifier}.")
             return False
         if bake:
             primitive.apply_bake(matrix)
         else:
             primitive._apply_transform_locally(matrix)
         self._bbox_dirty = True
         return True

    def transform_primitive_globally(self, primitive_identifier: Identifiable, matrix: np.ndarray, bake: bool = False) -> bool:
         """Applies a transformation matrix GLOBALLY to a primitive."""
         primitive = self._resolve_primitive(primitive_identifier)
         if not primitive:
             return False
         if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
             logger.error(f"Invalid global transform matrix for {primitive.user_identifier}.")
             return False
         if bake:
             primitive.apply_bake(matrix)
         else:
             primitive._apply_transform_globally(matrix)
         self._bbox_dirty = True
         return True

    def translate_primitive(self, primitive_identifier: Identifiable, dx: float, dy: float, bake: bool = False) -> bool:
         """Translates a primitive globally."""
         return self.transform_primitive_globally(primitive_identifier, translation_matrix(dx, dy), bake=bake)

    def rotate_primitive_deg(self, primitive_identifier: Identifiable, angle_deg: float, cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Rotates a primitive globally around a specified center (cx, cy) or its geometric center if None."""
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}. Rotation failed.")
                return False
        else:
            center_x, center_y = cx, cy
        global_rotation_matrix = rotation_matrix_deg(angle_deg, center_x, center_y)
        return self.transform_primitive_globally(primitive_identifier, global_rotation_matrix, bake=bake)

    def scale_primitive(self, primitive_identifier: Identifiable, sx: float, sy: float, cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """
        Scales/Mirrors a primitive globally around a specified center (cx, cy)
        or its geometric center if None. sx=-1 or sy=-1 performs mirroring.
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        if cx is None or cy is None:
            try:
                center_x, center_y = primitive.get_geometric_center()
            except Exception as e:
                logger.error(f"Could not calculate geometric center for {primitive.user_identifier}: {e}. Scaling failed.")
                return False
        else:
            center_x, center_y = cx, cy
        global_scale_matrix = scaling_matrix(sx, sy, center_x, center_y)
        return self.transform_primitive_globally(primitive_identifier, global_scale_matrix, bake=bake)

    def mirror_primitive(self, primitive_identifier: Identifiable, axis: str = 'x', cx: Optional[float] = None, cy: Optional[float] = None, bake: bool = False) -> bool:
        """Mirrors a primitive across the 'x' or 'y' axis passing through (cx, cy) or its geometric center."""
        sx, sy = 1.0, 1.0
        if axis.lower() == 'x':
            sy = -1.0
        elif axis.lower() == 'y':
            sx = -1.0
        else:
            logger.error("Invalid mirror axis. Use 'x' or 'y'.")
            return False
        return self.scale_primitive(primitive_identifier, sx, sy, cx, cy, bake=bake)

    def align_primitive(self, primitive_identifier: Identifiable, align: str, target: Tuple[float, float], bake: bool = False) -> bool:
        """
        Aligns a primitive's specified point with the target coordinate.
        Supported alignments: 'center', 'lower_left', 'lower_right', 'upper_left', 'upper_right'.
        """
        primitive = self._resolve_primitive(primitive_identifier)
        if not primitive:
            return False
        bbox = primitive.get_bounding_box()
        if align == "center":
            current = ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        elif align == "lower_left":
            current = (bbox.min_x, bbox.min_y)
        elif align == "lower_right":
            current = (bbox.max_x, bbox.min_y)
        elif align == "upper_left":
            current = (bbox.min_x, bbox.max_y)
        elif align == "upper_right":
            current = (bbox.max_x, bbox.max_y)
        else:
            current = ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        translation = translation_matrix(dx, dy)
        return self.transform_primitive_globally(primitive_identifier, translation, bake=bake)

    def _resolve_primitive(self, identifier: Identifiable) -> Optional[Primitive]:
         """Helper to resolve and get a primitive object."""
         primitive_uuid = self._resolve_identifier(identifier, Primitive)
         if not primitive_uuid:
             logger.error(f"Primitive identifier '{identifier}' not found.")
             return None
         primitive = self.get_primitive(primitive_uuid)
         if not primitive:
              logger.error(f"Primitive UUID {primitive_uuid} resolved but object not found.")
              return None
         return primitive

    # --- Removal Methods ---
    def remove_primitive(self, primitive_identifier: Identifiable) -> bool:
        primitive_uuid = self._resolve_identifier(primitive_identifier, Primitive)
        if not primitive_uuid:
            logger.error(f"Cannot remove: Primitive '{primitive_identifier}' not found.")
            return False
        prim = self.get_primitive(primitive_uuid)
        source_groups = set(prim.groups) if prim else set()
        affected_mops = [m.user_identifier for m in self.list_mops() if
                         (isinstance(m.pid_source, list) and primitive_uuid in m.pid_source) or
                         (isinstance(m.pid_source, str) and source_groups and m.pid_source in source_groups and
                          not self._primitive_groups.get(m.pid_source))]
        if affected_mops:
            logger.warning(f"Removing primitive '{primitive_identifier}' referenced by MOPs: {', '.join(affected_mops)}.")
        removed_primitive = self._unregister_primitive(primitive_uuid)
        if removed_primitive:
            logger.info(f"Removed primitive: {removed_primitive.user_identifier}")
        return removed_primitive is not None

    def remove_mop(self, mop_identifier: Identifiable) -> bool:
        mop_uuid = self._resolve_identifier(mop_identifier, Mop)
        if not mop_uuid:
            logger.error(f"Cannot remove: MOP '{mop_identifier}' not found.")
            return False
        removed_mop = self._unregister_mop(mop_uuid)
        if removed_mop:
            logger.info(f"Removed MOP: {removed_mop.user_identifier}")
        return removed_mop is not None

    def remove_layer(self, layer_identifier: Identifiable, transfer_primitives_to: Optional[Identifiable] = None) -> bool:
        layer_uuid = self._resolve_identifier(layer_identifier, Layer)
        if not layer_uuid:
            logger.error(f"Cannot remove: Layer '{layer_identifier}' not found.")
            return False
        layer_to_remove = self.get_layer(layer_uuid)
        if not layer_to_remove:
            return False
        target_layer: Optional[Layer] = None
        if transfer_primitives_to:
            target_layer = self.get_layer_by_identifier(transfer_primitives_to)
            if not target_layer:
                logger.error(f"Target layer '{transfer_primitives_to}' not found.")
                return False
        primitive_ids_to_process = list(layer_to_remove.primitive_ids)
        if target_layer:
            logger.info(f"Transferring {len(primitive_ids_to_process)} prims from '{layer_to_remove.user_identifier}' to '{target_layer.user_identifier}'.")
            for prim_id in primitive_ids_to_process:
                 primitive = self.get_primitive(prim_id)
                 if primitive:
                     primitive.layer_id = target_layer.internal_id
                     target_layer.primitive_ids.add(prim_id)
            layer_to_remove.primitive_ids.clear()
        else:
             logger.warning(f"Removing layer '{layer_to_remove.user_identifier}' and its {len(primitive_ids_to_process)} primitives.")
             for prim_id in primitive_ids_to_process:
                 self.remove_primitive(prim_id)
        removed_layer = self._unregister_entity(layer_uuid, self._layers)
        if removed_layer:
            try:
                self._layer_order.remove(layer_uuid)
            except ValueError:
                pass
            logger.info(f"Removed layer: {removed_layer.user_identifier}")
            return True
        return False

    def remove_part(self, part_identifier: Identifiable, remove_contained_mops: bool = True) -> bool:
        part_uuid = self._resolve_identifier(part_identifier, Part)
        if not part_uuid:
            logger.error(f"Cannot remove: Part '{part_identifier}' not found.")
            return False
        part_to_remove = self.get_part(part_uuid)
        if not part_to_remove:
            return False
        mop_ids_to_process = list(part_to_remove.mop_ids)
        if remove_contained_mops:
            logger.info(f"Removing part '{part_to_remove.user_identifier}' and {len(mop_ids_to_process)} MOPs.")
            for mop_id in mop_ids_to_process:
                self.remove_mop(mop_id)
        elif mop_ids_to_process:
            logger.error(f"Cannot remove part '{part_to_remove.user_identifier}': Contains MOPs.")
            return False
        removed_part = self._unregister_entity(part_uuid, self._parts)
        if removed_part:
            try:
                self._part_order.remove(part_uuid)
            except ValueError:
                pass
            logger.info(f"Removed part: {removed_part.user_identifier}")
            return True
        return False

    # --- Object Transfer ---
    def transfer_primitive_to(self, primitive_identifier: Identifiable,
                                target_project: 'CamBamProject',
                                new_layer_identifier: Optional[Identifiable] = None,
                                new_groups: Optional[List[str]] = None) -> bool:
        primitive_to_transfer = self._resolve_primitive(primitive_identifier)
        if not primitive_to_transfer:
            return False
        source_uuid = primitive_to_transfer.internal_id
        source_groups = set(primitive_to_transfer.groups)
        # Unregister from source *first* to handle links correctly
        primitive_copy_data = self._unregister_primitive(source_uuid)
        if not primitive_copy_data:
            logger.error(f"Transfer failed: unregister")
            return False
        # Warn MOPs
        affected_mops = [m.user_identifier for m in self.list_mops() if
                         (isinstance(m.pid_source, list) and source_uuid in m.pid_source) or
                         (isinstance(m.pid_source, str) and source_groups and m.pid_source in source_groups and
                          not self._primitive_groups.get(m.pid_source))]
        if affected_mops:
            logger.warning(f"Primitive '{primitive_identifier}' transferred. Source MOPs {affected_mops} may be invalid.")
        try:
            copied_primitive = deepcopy(primitive_copy_data)
            copied_primitive.internal_id = uuid.uuid4()
            # Update identifier if needed
            if copied_primitive.user_identifier == str(source_uuid):
                 copied_primitive.user_identifier = str(copied_primitive.internal_id)
            elif copied_primitive.user_identifier in target_project._identifier_registry:
                 new_id = f"{copied_primitive.user_identifier}_{copied_primitive.internal_id.hex[:6]}"
                 logger.warning(f"Identifier '{copied_primitive.user_identifier}' exists in target. Renaming to '{new_id}'.")
                 copied_primitive.user_identifier = new_id
            # Reset parent/child links for the copy
            copied_primitive.parent_primitive_id = None
            copied_primitive._child_primitive_ids = set()
            copied_primitive._invalidate_caches()
            # Resolve target layer
            target_layer = target_project.get_layer_by_identifier(new_layer_identifier) if new_layer_identifier else None
            if not target_layer:
                original_layer = self.get_layer(primitive_copy_data.layer_id)
                default_layer_name = original_layer.user_identifier if original_layer else "Default"
                # Try to find or create layer with same name in target
                target_layer = target_project.get_layer_by_identifier(default_layer_name)
                if not target_layer:
                    target_layer = target_project.add_layer(default_layer_name)
            copied_primitive.layer_id = target_layer.internal_id
            if new_groups is not None:
                copied_primitive.groups = new_groups
            # Register in target
            target_project._register_primitive(copied_primitive)
            logger.info(f"Transferred primitive '{primitive_identifier}' as '{copied_primitive.user_identifier}'.")
            return True
        except Exception as e:
            logger.error(f"Transfer failed processing: {e}", exc_info=True)
            # Revert: Re-register original primitive in source
            logger.info(f"Attempting revert by re-registering '{primitive_copy_data.user_identifier}'.")
            try:
                self._register_primitive(primitive_copy_data)
            except Exception as revert_e:
                logger.error(f"Failed to revert state: {revert_e}", exc_info=True)
            return False


    # --- Bounding Box ---
    def get_bounding_box(self) -> BoundingBox:
        if not self._bbox_dirty and self._cached_bbox is not None:
            return self._cached_bbox
        overall_bbox = BoundingBox()
        for primitive in self._primitives.values():
            try:
                prim_bbox = primitive.get_bounding_box()
                if prim_bbox.is_valid():
                    overall_bbox = overall_bbox.union(prim_bbox)
            except Exception as e:
                logger.error(f"Bbox error for {primitive.user_identifier}: {e}", exc_info=True)
        self._cached_bbox = overall_bbox
        self._bbox_dirty = False
        return overall_bbox

    # --- Persistence ---
    def save_state(self, file_path: str) -> None:
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Project state saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
            raise

    @staticmethod
    def load_state(file_path: str) -> 'CamBamProject':
        try:
            with open(file_path, 'rb') as f:
                loaded_project = pickle.load(f)
            if not isinstance(loaded_project, CamBamProject):
                raise TypeError("Loaded object is not CamBamProject.")
            # Mark caches dirty and reset transient XML elements/child links
            loaded_project._bbox_dirty = True
            loaded_project._cached_bbox = None
            # Rebuild child links and invalidate caches after load
            for prim in loaded_project._primitives.values():
                 prim._invalidate_caches(invalidate_children=False)
                 prim._child_primitive_ids = set()
                 # Ensure transform is numpy array
                 if not isinstance(prim.effective_transform, np.ndarray):
                      prim.effective_transform = identity_matrix()
                 prim._project_ref = weakref.ref(loaded_project)
            # Reset layer/part XML element references
            for layer in loaded_project._layers.values():
                layer._xml_objects_element = None
            for part in loaded_project._parts.values():
                part._xml_machineops_element = None
            logger.info(f"Project state loaded: {file_path}")
            return loaded_project
        except Exception as e:
            logger.error(f"Error loading state: {e}", exc_info=True)
            raise

    # --- XML Generation and Saving ---
    def _build_xml_tree(self) -> ET.ElementTree:
        # 1. Assign XML Primitive IDs
        uuid_to_xml_id: Dict[uuid.UUID, int] = {}
        xml_id_counter = 1
        sorted_primitive_uuids = sorted(self._primitives.keys(), key=lambda u: u.int)
        primitive_build_order: List[uuid.UUID] = []
        for prim_uuid in sorted_primitive_uuids:
            uuid_to_xml_id[prim_uuid] = xml_id_counter
            primitive_build_order.append(prim_uuid)
            xml_id_counter += 1

        # 2. Create Root Element
        root = ET.Element("CADFile", {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
            "Version": "0.9.8.0",
            "Name": self.project_name
        })

        # 3. Add Global Machining Options
        machining_options = ET.SubElement(root, "MachiningOptions")
        stock = ET.SubElement(machining_options, "Stock")
        ET.SubElement(stock, "Material")
        ET.SubElement(stock, "PMin").text = "0,0,0"
        ET.SubElement(stock, "PMax").text = "0,0,0"
        ET.SubElement(stock, "Color").text = "255,165,0"

        # 4. Build Layers structure
        layers_container = ET.SubElement(root, "layers")
        layer_objects_map: Dict[uuid.UUID, ET.Element] = {}
        for layer_uuid in self._layer_order:
            layer = self.get_layer(layer_uuid)
            if layer:
                layer_elem = layer.to_xml_element()
                layers_container.append(layer_elem)
                if layer._xml_objects_element is not None:
                    layer_objects_map[layer_uuid] = layer._xml_objects_element
                else:
                    logger.error(f"Layer {layer.user_identifier} failed <objects> ref.")

        # 5. Build Primitives (placed in correct layer)
        for primitive_uuid in primitive_build_order:
            primitive = self.get_primitive(primitive_uuid)
            if not primitive:
                continue
            xml_primitive_id = uuid_to_xml_id[primitive_uuid]
            try:
                primitive_elem = primitive.to_xml_element(xml_primitive_id)
            except Exception as e:
                logger.error(f"XML build error for primitive {primitive.user_identifier}: {e}", exc_info=True)
                continue
            layer_objects_container = layer_objects_map.get(primitive.layer_id)
            if layer_objects_container is not None:
                layer_objects_container.append(primitive_elem)
            else:
                logger.error(f"Layer {primitive.layer_id} <objects> not found for primitive {primitive.user_identifier}.")
                if layer_objects_map:
                    next(iter(layer_objects_map.values())).append(primitive_elem)

        # 6. Build Parts structure
        parts_container = ET.SubElement(root, "parts")
        part_machineops_map: Dict[uuid.UUID, ET.Element] = {}
        for part_uuid in self._part_order:
             part = self.get_part(part_uuid)
             if part:
                 part_elem = part.to_xml_element()
                 parts_container.append(part_elem)
                 if part._xml_machineops_element is not None:
                     part_machineops_map[part_uuid] = part._xml_machineops_element
                 else:
                     logger.error(f"Part {part.user_identifier} failed <machineops> ref.")

        # 7. Build MOPs (placed in correct part)
        resolver = XmlPrimitiveIdResolver(uuid_to_xml_id, self._primitive_groups, self._primitives)
        for part_uuid in self._part_order:
            part = self.get_part(part_uuid)
            if not part:
                continue
            machineops_container = part_machineops_map.get(part_uuid)
            if machineops_container is None:
                logger.error(f"<machineops> not found for part {part.user_identifier}.")
                continue
            for mop_uuid in part.mop_ids:
                mop = self.get_mop(mop_uuid)
                if not mop:
                    logger.warning(f"MOP {mop_uuid} in part {part.user_identifier} order not found.")
                    continue
                try:
                    mop.resolve_xml_primitive_ids(resolver)
                    mop_elem = mop.to_xml_element(self)
                    machineops_container.append(mop_elem)
                except Exception as e:
                    logger.error(f"XML build error for MOP {mop.user_identifier}: {e}", exc_info=True)
                    continue
        return ET.ElementTree(root)

    def save(self, file_path: str, pretty_print: bool = True) -> None:
        try:
            base, _ = os.path.splitext(file_path)
            output_path = base + '.cb'
            dir_name = os.path.dirname(output_path)
            if dir_name and not os.path.isdir(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            tree = self._build_xml_tree()
            if pretty_print:
                try:
                    ET.indent(tree, space="  ", level=0)
                except AttributeError:
                    logger.warning("XML pretty-printing requires Python 3.9+.")
                except Exception as e:
                    logger.warning(f"XML indenting error: {e}.")
            tree.write(output_path, encoding='utf-8', xml_declaration=True, short_empty_elements=False)
            logger.info(f"CamBam file saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving file {output_path}: {e}", exc_info=True)
            raise

# --- Example Usage ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logger.setLevel(logging.DEBUG) # Show debug messages

    proj = CamBamProject("Project Test with linking and transformations", default_tool_diameter=5.0)

    # Layers
    layer_frame_top = proj.add_layer("Frame Top", color="Cyan")
    layer_frame_bottom = proj.add_layer("Frame Bottom", color="Red")

    # Parts
    part_a = proj.add_part("12.4 | EM6.0", stock_thickness=12.4, machining_origin=(0,0), default_spindle_speed=20000, default_tool_diameter=3.175)
    part_b = proj.add_part("9.3 | EM3.175", stock_thickness=9.3, machining_origin=(1220,0), default_spindle_speed=24000, default_tool_diameter=3.175)

    # Top and Bottom Frame Boards primitives
    # proj.set_cursor(20,20) # Set machine origin offset

    rect_frame_top = proj.add_rect(layer=layer_frame_top, identifier="frame_top", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_top_edge_left = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_right = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_back = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_1 = proj.add_rect(layer=layer_frame_top, identifier="frame_top_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_2 = proj.add_rect(layer=layer_frame_top, identifier="frame_top_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    text_frame_top = proj.add_text(layer=layer_frame_top, identifier="frame_top_id", text="Frame Top Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_top)

    rect_frame_bottom = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_bottom_edge_left = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_right = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_back = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_1 = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_2 = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    # text_frame_bottom_id = proj.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_bottom)

    # Transformations
    # proj.translate_primitive(rect_frame_top, 20, 20)
    proj.translate_primitive(rect_frame_bottom, 300, 0)
    # proj.translate_primitive(rect_frame_bottom, 20+800+30, 20)
    proj.mirror_primitive(rect_frame_bottom, "y", rect_frame_bottom.get_absolute_coordinates())
    rect_frame_bottom.bake()
    text_frame_bottom_id = proj.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=rect_frame_bottom.get_geometric_center(), height=20, groups=["frame_id"], parent=rect_frame_bottom)
    c = rect_frame_bottom.get_absolute_coordinates()[0]
    proj.rotate_primitive_deg(rect_frame_bottom, 90, cx=c[0], cy=c[1])
    proj.align_primitive(rect_frame_bottom, "lower_left", (0,0))


    # MOP
    proj.add_pocket_mop(part=part_a, pid_source="frame_edge_groove", name="frame_edge_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.1)
    proj.add_pocket_mop(part=part_a, pid_source="frame_groove", name="frame_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.2)
    proj.add_profile_mop(part=part_a, pid_source="frame_cutout", name="frame_id", profile_side="Outside", target_depth=-12.9, roughing_clearance=-0.1)
    
    output_dir = "./test_output"
    proj.save(os.path.join(output_dir, "v5.cb"))
    proj.save_state(os.path.join(output_dir, "v5.pkl"))

    # --- Load Test ---
    loaded = CamBamProject.load_state(os.path.join(output_dir, "v5.pkl"))
    print(f"\nLoaded Project '{loaded.project_name}'")
    print(f" Bounding Box: {loaded.get_bounding_box()}")
    print(f" Primitives: {len(loaded.list_primitives())}")