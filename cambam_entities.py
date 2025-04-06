"""
cambam_entities.py

Defines the CamBam entity classes including the base CamBamEntity,
primitive classes (with geometry and transformation methods), and machine
operation (MOP) classes.
"""

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import weakref
import uuid
import logging
from typing import List, Dict, Tuple, Union, Optional, Set, Any, Sequence, Type, TypeVar, TYPE_CHECKING
import numpy as np
import math
import json

from cad_transformations import (
    identity_matrix, translation_matrix, rotation_matrix_deg, scale_matrix,
    mirror_x_matrix, mirror_y_matrix, apply_transform, get_transformed_point,
    to_cambam_matrix
)

if TYPE_CHECKING:
    from cambam_project import CamBamProject

logger = logging.getLogger(__name__)

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
    """
    Helper to resolve primitive UUIDs/groups to integer XML IDs.
    """
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
                logger.error(f"Primitive UUID '{uid}' not found in XML ID map.")
        return sorted(resolved_ids)

# --- Base Entity Classes ---

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
    """Represents a machining part."""
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
        if self.default_tool_diameter is not None:
            ET.SubElement(part_elem, "ToolDiameter").text = str(self.default_tool_diameter)
        ET.SubElement(part_elem, "ToolProfile").text = "EndMill"
        nesting = ET.SubElement(part_elem, "Nesting")
        ET.SubElement(nesting, "BasePoint").text = "0,0"
        ET.SubElement(nesting, "NestMethod").text = "None"
        return part_elem

# --- Primitive Base Class and Concrete Classes ---

@dataclass
class Primitive(CamBamEntity, ABC):
    """
    Abstract Base Class for geometric primitives.
    Assignment properties are decoupled as separate attributes.
    """
    layer_id: uuid.UUID = field(kw_only=True)
    groups: List[str] = field(default_factory=list)
    description: str = ""                  # Free-text description
    assigned_mops: List[str] = field(default_factory=list)  # Mop assignments
    effective_transform: np.ndarray = field(default_factory=identity_matrix)
    parent_primitive_id: Optional[uuid.UUID] = None
    _child_primitive_ids: Set[uuid.UUID] = field(default_factory=set, init=False, repr=False)
    _project_ref: Optional[weakref.ReferenceType] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.effective_transform, np.ndarray):
            self.effective_transform = identity_matrix()
        if self.groups is None:
            self.groups = []

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_project_ref' in state:
            del state['_project_ref']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._project_ref = None

    def get_project(self) -> "CamBamProject":
        if self._project_ref is None:
            raise ValueError(f"Primitive {self.user_identifier} is not linked to a project.")
        project = self._project_ref()
        if project is None:
            raise ValueError(f"Project reference lost for {self.user_identifier}.")
        return project

    def _get_total_transform(self) -> np.ndarray:
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
            parent = self.get_project().get_primitive(self.parent_primitive_id)
            if parent:
                parent_tf = parent._get_total_transform()
            else:
                logger.warning(f"Parent primitive {self.parent_primitive_id} not found for {self.user_identifier}.")
        return parent_tf @ self.effective_transform

    def get_absolute_coordinates(self) -> Any:
        total_tf = self._get_total_transform()
        return self._calculate_absolute_geometry(total_tf)

    def get_bounding_box(self) -> BoundingBox:
        abs_coords = self.get_absolute_coordinates()
        return self._calculate_bounding_box(abs_coords)

    def _set_effective_transform(self, matrix: np.ndarray):
        if not (self.effective_transform == matrix).all():
            self.effective_transform = matrix.copy()

    def _apply_transform_locally(self, local_transform_matrix: np.ndarray):
        new_eff_tf = self.effective_transform @ local_transform_matrix
        self._set_effective_transform(new_eff_tf)

    def _apply_transform_globally(self, global_transform_matrix: np.ndarray):
        current_total_tf = self._get_total_transform()
        parent_tf = identity_matrix()
        if self.parent_primitive_id:
            parent = self.get_project().get_primitive(self.parent_primitive_id)
            if parent:
                parent_tf = parent._get_total_transform()
        try:
            inv_parent_tf = np.linalg.inv(parent_tf)
        except np.linalg.LinAlgError:
            logger.error(f"Cannot apply global transform to {self.user_identifier}: singular parent matrix. Applying locally.")
            self._apply_transform_locally(global_transform_matrix)
            return
        transform_delta = inv_parent_tf @ global_transform_matrix @ parent_tf
        new_eff_tf = self.effective_transform @ transform_delta
        self._set_effective_transform(new_eff_tf)

    def _add_common_xml_attributes(self, element: ET.Element, xml_primitive_id: int):
        element.set("id", str(xml_primitive_id))
        # Save structured classification data
        tag_data = {
            "groups": self.groups,
            "description": self.description,
            "assigned_mops": self.assigned_mops,
            "layer": str(self.layer_id),
            "parent": str(self.parent_primitive_id) if self.parent_primitive_id else None,
        }
        ET.SubElement(element, "Tag").text = json.dumps(tag_data)

    def _add_matrix_xml(self, element: ET.Element):
        total_tf = self._get_total_transform()
        mat_str = to_cambam_matrix(total_tf)
        ET.SubElement(element, "mat", {"m": mat_str})

    @abstractmethod
    def _calculate_absolute_geometry(self, total_transform: np.ndarray) -> Any:
        pass

    @abstractmethod
    def _calculate_bounding_box(self, abs_coords: Any) -> BoundingBox:
        pass

    @abstractmethod
    def get_geometric_center(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def to_xml_element(self, xml_primitive_id: int) -> ET.Element:
        pass

    @abstractmethod
    def bake_geometry(self, matrix: np.ndarray) -> None:
        pass

    def apply_bake(self, additional_transform: np.ndarray):
        combined_transform = self.effective_transform @ additional_transform
        self.bake_geometry(combined_transform)
        self.effective_transform = identity_matrix()
        proj = self.get_project()
        for child_id in self._child_primitive_ids:
            child = proj.get_primitive(child_id)
            if child:
                child.effective_transform = combined_transform @ child.effective_transform
                child.apply_bake(np.eye(3))

    def bake(self):
        total_tf = self._get_total_transform()
        self.bake_geometry(total_tf)
        self.effective_transform = identity_matrix()
        proj = self.get_project()
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
        return BoundingBox.from_points(points_xy)

    def get_geometric_center(self) -> Tuple[float, float]:
        abs_coords = self.get_absolute_coordinates()
        points_xy = [(p[0], p[1]) for p in abs_coords]
        if not points_xy:
            return (0.0, 0.0)
        bbox = BoundingBox.from_points(points_xy)
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
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
        scale = abs(self._get_total_transform())[0, 0]
        scaled_radius = self.diameter / 2.0 * scale
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
        sx = abs((matrix[0,0]**2 + matrix[1,0]**2)**0.5)
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
        scale = abs(self._get_total_transform())[0, 0]
        scaled_radius = self.radius * scale
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
        sx = abs((matrix[0,0]**2 + matrix[1,0]**2)**0.5)
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
        abs_coords = self.get_absolute_coordinates()
        if not abs_coords:
            return (0.0, 0.0)
        bbox = BoundingBox.from_points(abs_coords)
        if bbox.is_valid():
            return ((bbox.min_x + bbox.max_x) / 2, (bbox.min_y + bbox.max_y) / 2)
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
        scale = abs(self._get_total_transform())[0, 0]
        scaled_height = self.height * scale
        est_width = scaled_height * len(self.text_content.split('\n')[0]) * 0.6
        return BoundingBox(px - est_width/2, py - scaled_height/2, px + est_width/2, py + scaled_height/2)

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
        sx = abs((matrix[0,0]**2 + matrix[1,0]**2)**0.5)
        self.height *= sx

# --- MOP Base and Concrete Classes ---

M = TypeVar('M', bound='Mop')

@dataclass
class Mop(CamBamEntity, ABC):
    """Abstract Base Class for machine operations."""
    part_id: uuid.UUID = field(kw_only=True)
    name: str = field(kw_only=True)
    pid_source: Union[str, List[uuid.UUID]] = field(kw_only=True)
    enabled: bool = True
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
    def to_xml_element(self, project: "CamBamProject") -> ET.Element:
        pass

    def _get_effective_param(self, param_name: str, project: "CamBamProject") -> Any:
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

    def _calculate_cut_feedrate(self, project: "CamBamProject") -> float:
        if self.cut_feedrate is not None:
            return self.cut_feedrate
        td = self._get_effective_param('target_depth', project)
        if td is None:
            logger.warning(f"MOP '{self.name}': TargetDepth not set. Using fallback feedrate 3000.")
            return 3000.0
        calculated_feedrate = round(350 * abs(td) + 6500, 0)
        return max(calculated_feedrate, 1000.0)

    def _add_common_mop_elements(self, mop_root_elem: ET.Element, project: "CamBamProject"):
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

    def to_xml_element(self, project: "CamBamProject") -> ET.Element:
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

    def to_xml_element(self, project: "CamBamProject") -> ET.Element:
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

    def to_xml_element(self, project: "CamBamProject") -> ET.Element:
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

    def to_xml_element(self, project: "CamBamProject") -> ET.Element:
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
